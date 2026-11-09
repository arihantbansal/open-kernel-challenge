"""Host-only compatibility and diagnostic tests; never invoke Docker or CUDA tools."""

import ast
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
PRECHECK = TOOLS / "precheck"


def load_precheck(path):
    name = "_precheck_under_test_" + path.name.replace(".", "_")
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and the native consumers require this
    loader.exec_module(module)
    return module


PC = load_precheck(PRECHECK)

PTXAS_LOG = """host warning: unrelated spill text
ptxas info : Compiling entry function '_Z5firstv' for 'sm_90a'
ptxas info : (C7511) wgmma.mma_async instructions are serialized due to insufficient register resources
ptxas info : (C7511) wgmma.mma_async instructions are serialized due to insufficient register resources
ptxas info : Function properties for _Z5firstv
    16 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
ptxas warning : Registers are spilled to local memory in function '_Z5firstv'
ptxas info : Used 128 registers, 1024 bytes smem
ptxas info : Compiling entry function '_Z6secondv' for 'sm_90a'
ptxas info : (C7512) wgmma.mma_async instructions are serialized in the function '_Z5firstv'
ptxas info : Function properties for _Z6secondv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 64 registers
"""

SASS = """Fatbin elf code:
    Function : _Z5firstv
        /*0000*/ HGMMA.64x64x16.F32.BF16 R0, R1, gdesc[UR0], R0;
    Function : _Z6secondv
        /*0000*/ EXIT;
"""


class PrecheckCompatibilityTests(unittest.TestCase):
    def run_cli(self, path, *args):
        # Empty PATH prevents accidentally invoking Docker/compiler executables.
        env = dict(os.environ, PATH="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        return subprocess.run([sys.executable, "-B", str(path), *map(str, args)],
                              env=env, cwd=TOOLS, capture_output=True, text=True, timeout=30, check=False)

    def test_entry_point_is_executable_and_rooted_in_this_checkout(self):
        self.assertTrue(os.access(PRECHECK, os.X_OK))
        self.assertEqual(PC.ROOT, ROOT)
        self.assertEqual(PC.TOOLS_DIR, TOOLS)
        self.assertIsNone(PC.find_problem_dir(ROOT / "nowhere.py", "muse_decode_layer"))

    def test_help_exposes_sass_and_lint_only(self):
        result = self.run_cli(PRECHECK, "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("--sass REGEX", result.stdout)
        self.assertIn("--lint-only", result.stdout)

    def test_lint_only_cli_for_clean_and_rejected_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "submission.py"
            for extra, status in (("", 0), ("import subprocess\n", 1)):
                source.write_text("# okc: problem=muse_attention_decode\n" + extra
                                  + "def kernel_fn(data):\n    return data\n")
                result = self.run_cli(PRECHECK, source, "--lint-only", "--json", "--sass", ".*")
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(result.stderr, "")
                report = json.loads(result.stdout)
                self.assertEqual(report["stages"]["lint"]["status"], "FAIL" if status else "PASS")
                for stage in ("build", "trace", "triton"):
                    self.assertEqual(report["stages"][stage], {"status": "SKIP", "why": "--lint-only"})


class PrecheckDiagnosticTests(unittest.TestCase):
    def test_ptxas_resources_diagnostics_attribution_and_deduplication(self):
        usage = PC.parse_ptxas(PTXAS_LOG)
        self.assertEqual(usage["_Z5firstv"], {"regs": 128, "stack": 16, "spill_st": 8, "spill_ld": 4, "smem": 1024})
        self.assertEqual(usage["_Z6secondv"], {"regs": 64, "stack": 0, "spill_st": 0, "spill_ld": 0, "smem": 0})
        diagnostics = PC.ptxas_diagnostics(PTXAS_LOG)
        self.assertEqual([(d["kernel"], d["code"]) for d in diagnostics],
                         [("_Z5firstv", "C7511"), ("_Z5firstv", "spill"),
                          ("_Z5firstv", "spill"), ("_Z5firstv", "C7512")])

    def test_all_serialization_codes_are_not_filtered_as_info(self):
        for code in range(7510, 7520):
            with self.subTest(code=code):
                log = ("ptxas info : Function properties for triton_kernel\n"
                       f"ptxas info : (C{code}) wgmma.mma_async instructions are serialized\n")
                diagnostics = PC.ptxas_diagnostics(log)
                self.assertEqual(len(diagnostics), 1)
                self.assertEqual(diagnostics[0]["kernel"], "triton_kernel")
                self.assertEqual(diagnostics[0]["code"], f"C{code}")

    def test_cuda_and_triton_report_serialization_without_duplicate_spill_warnings(self):
        result = {
            "builds": [{"line": 17, "kind": "load_inline", "name": "fixture", "ok": True, "seconds": 1,
                        "ptxas": PC.parse_ptxas(PTXAS_LOG), "ptxas_diag": PC.ptxas_diagnostics(PTXAS_LOG),
                        "ptxas_names": {"_Z5firstv": "void first()", "_Z6secondv": "void second()"},
                        "sass": [{"kernel": "first", "path": "/stage/sass/first.sass"}]}],
            "triton": [{"kernel": "triton_kernel", "line": 31, "num_warps": 4,
                        "regs": 128, "spill_st": 4, "spill_ld": 0,
                        "ptxas_diag": [{"kernel": "triton_kernel", "code": "C7514", "text": "ptxas info : C7514 serialized"},
                                       {"kernel": "triton_kernel", "code": "spill", "text": "4 bytes spill stores"}]}],
        }
        report = PC.Report("fixture.py", None)
        PC.merge_container_result(report, result)
        warnings = [(f.stage, f.rule) for f in report.findings if f.severity == "WARN"]
        self.assertCountEqual(warnings, [("build", "ptxas.serialized"), ("build", "ptxas.serialized"),
                                         ("build", "ptxas.spill"), ("triton", "ptxas.serialized"),
                                         ("triton", "triton.spill")])
        spill = next(f for f in report.findings if f.rule == "ptxas.spill")
        self.assertIn("first in 'fixture' spills: 8 B stores / 4 B loads at 128 regs", spill.message)
        self.assertIn("sass first: /stage/sass/first.sass", report.details["sections"][0])
        self.assertEqual(report.stages["build"]["status"], "PASS")

    def test_sass_selection_demangling_and_collision_safe_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "sass"
            names = {"_Z5firstv": "void panel_gemm<(int)32>()", "_Z6secondv": "void second()"}
            with patch.object(PC.subprocess, "run", return_value=SimpleNamespace(stdout=SASS, returncode=0)) as run:
                for pattern in ("_Z5firstv", "panel_gemm<32>", r"panel_gemm<\(int\)32>"):
                    records = PC.dump_sass([Path("fixture.so")], pattern, outdir, names)
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]["kernel"], "panel_gemm<32>")
                    output = Path(records[0]["path"]).read_text()
                    self.assertIn("Function : _Z5firstv", output)
                    self.assertIn("HGMMA.64x64x16", output)
                    self.assertNotIn("EXIT", output)
                self.assertEqual(len(list(outdir.glob("*.sass"))), 3)
                self.assertEqual(PC.dump_sass([Path("fixture.so")], "no_match", outdir, names), [])
            for call in run.call_args_list:
                self.assertEqual(call.args[0], ["/usr/local/cuda/bin/cuobjdump", "-sass", "fixture.so"])

    def test_container_command_alias_sass_path_translation_and_stage_lifetime(self):
        # Execute the real host driver, replacing only external Docker operations.
        for module in (PC,):
            for sass in (None, "panel"):
                with self.subTest(entry=module.__file__, sass=sass), tempfile.TemporaryDirectory() as tmp:
                    source = Path(tmp) / "input.py"
                    source.write_text("def kernel_fn(data):\n    return data\n")
                    args = SimpleNamespace(rebuild_image=False, stage_root=tmp, no_trace=True, quick=False,
                                           case_timeout=120, sass=sass, memory="6g", json=True, keep_stage=False)
                    stages = []

                    def docker_result(command, *, stages=stages, module=module, sass=sass, **kwargs):
                        mount = next(arg for arg in command if arg.endswith(":/stage"))
                        stage = Path(mount.removesuffix(":/stage"))
                        stages.append(stage)
                        self.assertEqual(command[-3:], [f"/tools/{Path(module.__file__).name}", "--in-container", "/stage"])
                        self.assertIn(f"{TOOLS}:/tools:ro", command)
                        self.assertEqual(json.loads((stage / "config.json").read_text())["sass"], sass)
                        build = {"name": "fixture", "kind": "load_inline", "line": 1, "seconds": 1, "ok": True}
                        if sass:
                            (stage / "sass").mkdir()
                            (stage / "sass/panel.sass").write_text(SASS)
                            build["sass"] = [{"kernel": "panel", "path": "/stage/sass/panel.sass"}]
                        (stage / "result.json").write_text(json.dumps({"builds": [build]}))
                        return SimpleNamespace(stdout=io.StringIO(""), wait=lambda: None, returncode=0)

                    report = module.Report(str(source), None)
                    with patch.object(module.shutil, "which", return_value="docker"), \
                         patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                         patch.object(module, "ensure_image", return_value=True), \
                         patch.object(module, "ensure_cutlass_headers", return_value=None), \
                         patch.object(module, "docker_lock", side_effect=contextlib.nullcontext), \
                         patch.object(module.subprocess, "Popen", side_effect=docker_result), \
                         contextlib.redirect_stderr(io.StringIO()) as stderr:
                        module.run_container(report, source, None, args)
                    self.assertEqual(len(stages), 1)
                    self.assertEqual(stages[0].exists(), bool(sass))
                    if sass:
                        self.assertIn(str(stages[0] / "sass/panel.sass"), report.details["sections"][0])
                        self.assertIn("stage kept", stderr.getvalue())
                    self.assertEqual(report.stages["build"]["status"], "PASS")


def trace_report(cases, prepare=None, findings=()):
    report = PC.Report("fixture.py", "muse_decode_layer")
    report.findings.extend(findings)
    PC.merge_container_result(report, {"import": {"ok": True}, "prepare": prepare or {"ok": True}, "cases": cases})
    return report.stages["trace"]


def case(status="ok", stub=None, line=40):
    return {"group": "scored", "case": f"batch {line}", "status": status, "stub": stub, "line": line, "error": "e"}


class TraceVerdictTests(unittest.TestCase):
    def test_judge_only_stubs_pass_with_a_distinct_label(self):
        st = trace_report([case(), case("emulation", "sgl_kernel"), case("emulation", "sglang")])
        self.assertEqual((st["status"], st["stubbed"]), ("PASS-STUBBED", ["sgl_kernel", "sglang"]))
        self.assertEqual(PC.status_label(st), "PASS (stubbed: sgl_kernel, sglang)")
        self.assertEqual(trace_report([case(), case()])["status"], "PASS")

    def test_other_emulation_stops_stay_partial(self):
        for stub in (None, "einops"):  # a meta-tensor limit, or a library the judge image does not have
            with self.subTest(stub=stub):
                st = trace_report([case("emulation", "sgl_kernel"), case("emulation", stub)])
                self.assertEqual(st["status"], "PARTIAL")

    def test_an_error_still_fails(self):
        st = trace_report([case("emulation", "sgl_kernel"), case("raised")])
        self.assertEqual(st["status"], "FAIL")

    def test_prepare_emulation_limit_is_partial_emulation_not_a_kernel_failure(self):
        prep = {"ok": False, "emulation": True, "line": 12, "error": "EmulationLimit: meta tensors cannot run foo"}
        st = trace_report([], prep)
        self.assertEqual((st["status"], st["limit"]), ("PARTIAL-EMULATION", "prepare"))
        self.assertIn("kernel_fn was not traced", st["why"])

    def test_prepare_self_test_comparisons_are_named(self):
        st = trace_report([case(), case("emulation", "sgl_kernel")], {"ok": True, "compared": [321, 648]})
        self.assertEqual((st["status"], st["limit"], st["stubbed"]), ("PARTIAL-EMULATION", "prepare-self-test", ["sgl_kernel"]))
        self.assertIn("321, 648", st["why"])
        # a kernel_fn trace that is itself partial is not upgraded by the self-test path
        st = trace_report([case("emulation"), case("emulation")], {"ok": True, "compared": [321]})
        self.assertEqual(st["status"], "PARTIAL")


TRACE_SCRIPT = r"""
import json, sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import test_precheck as t
PC = t.PC
stage = Path(sys.argv[2])
import torch._dynamo  # initialize CPU torch before installing the compiler fixture
# These fixtures contain no extensions or Triton kernels. Replace only compiler
# boundaries, retaining real import, input generation, prepare, and tracing.
sys.modules["triton"] = types.SimpleNamespace(__version__="fixture")
PC.install_triton_emulation = lambda out: None
PC.install_build_recorder = lambda *args: None
PC.subprocess.run = lambda *args, **kwargs: types.SimpleNamespace(stdout="fixture nvcc")
PC.container_main(stage)
result = json.loads((stage / "result.json").read_text())
report = PC.Report(str(stage / "submission.py"), "muse_decode_layer")
PC.lint(PC.Module(stage / "submission.py"), report, None, stage / "no-scout")
PC.merge_container_result(report, result)
PC.stage_status(report, "lint")
print(json.dumps({"stages": report.stages, "result": result,
                  "findings": [vars(f) for f in report.findings]}))
"""


class TraceExecutionTests(unittest.TestCase):
    def trace(self, source, inputs, reference=None):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            (stage / "submission.py").write_text("# okc: problem=muse_decode_layer\n" + source)
            (stage / "reference.py").write_text(reference or "def generate_input(i, device):\n    return i\n")
            (stage / "config.json").write_text(json.dumps({
                "has_reference": True, "problem": "muse_decode_layer", "no_trace": False,
                "cases": {"scored": [{"i": i} for i in inputs], "tests": [], "nearby": []},
            }))
            result = subprocess.run([sys.executable, "-c", TRACE_SCRIPT, str(TOOLS), str(stage)],
                                    env=dict(os.environ, PATH=""), capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout.splitlines()[-1])
            self.assertNotIn("fatal", report["result"])
            return report

    def test_zero_discovered_cases_cannot_pass(self):
        report = self.trace("def kernel_fn(data):\n    return data\n", [])
        self.assertEqual(report["stages"]["trace"]["status"], "PARTIAL")

    def test_all_input_generators_skipped_cannot_pass(self):
        report = self.trace("def kernel_fn(data):\n    return data\n", range(3),
                            "def generate_input(i, device):\n    raise ValueError('fixture input unavailable')\n")
        self.assertEqual(report["stages"]["trace"]["status"], "PARTIAL")

    def test_one_meta_failure_among_successes_cannot_pass(self):
        source = ("import torch\ndef kernel_fn(data):\n"
                  "    if data == 4:\n        return torch.histogram(torch.ones(4, device='cuda'), bins=2)\n"
                  "    return data\n")
        report = self.trace(source, range(5))
        self.assertEqual([c["status"] for c in report["result"]["cases"]], ["ok"] * 4 + ["emulation"])
        self.assertEqual(report["stages"]["trace"]["status"], "PARTIAL")

    def test_none_typeerror_is_a_failure(self):
        report = self.trace("def kernel_fn(data):\n    state = None\n    return state + 1\n", [0])
        self.assertEqual(report["stages"]["trace"]["status"], "FAIL")
        self.assertFalse(report["result"]["cases"][0]["emulation"])

    def test_generic_meta_and_stub_messages_are_failures(self):
        for message in ("metadata is invalid", "stub configuration is invalid"):
            with self.subTest(message=message):
                report = self.trace(f"def kernel_fn(data):\n    raise RuntimeError({message!r})\n", [0])
                self.assertEqual(report["stages"]["trace"]["status"], "FAIL")

    def test_unrelated_stub_call_does_not_classify_a_generic_error(self):
        source = ("from sgl_kernel.flash_attn import flash_attn_with_kvcache\n"
                  "def kernel_fn(data):\n    out = flash_attn_with_kvcache(data)\n"
                  "    raise ValueError('Stub configuration is invalid')\n")
        report = self.trace(source, [0])
        self.assertEqual(report["stages"]["trace"]["status"], "FAIL")

    def test_trace_reports_case_and_compilation_counts(self):
        report = self.trace("def kernel_fn(data):\n    return data\n", range(3),
                            "def generate_input(i, device):\n"
                            "    if i == 2:\n        raise ValueError('fixture input unavailable')\n    return i\n")
        trace = report["stages"]["trace"]
        self.assertEqual({k: trace.get(k) for k in ("discovered", "reached", "completed", "compiled")},
                         {"discovered": 3, "reached": 2, "completed": 2, "compiled": 0})
        self.assertEqual(trace["status"], "PARTIAL")

    def test_prepare_self_test_and_judge_stub_remain_accepted(self):
        source = ("import torch\nfrom sgl_kernel.flash_attn import flash_attn_with_kvcache\n"
                  "def prepare(weights):\n    a = torch.ones(4, device='cuda')\n"
                  "    assert torch.equal(a, a)\n    return None\n"
                  "def kernel_fn(data, state):\n    out = flash_attn_with_kvcache(data)\n"
                  "    return torch.matmul(out, torch.ones(4, device='cuda'))\n")
        reference = "def generate_weights(seed, device):\n    return None\ndef generate_input(i, device):\n    return i\n"
        report = self.trace(source, [0], reference)
        trace = report["stages"]["trace"]
        self.assertEqual((trace["status"], trace["limit"], trace["stubbed"]),
                         ("PARTIAL-EMULATION", "prepare-self-test", ["sgl_kernel"]))

    def test_meta_scalar_readbacks_are_unknown_not_fabricated_values(self):
        reference = "def generate_weights(seed, device):\n    return None\ndef generate_input(i, device):\n    return i\n"
        for expr in ("a.item()", "int(a)", "float(a)", "bool(a)", "a.__index__()"):
            with self.subTest(expr=expr):
                source = ("import torch\ndef prepare(weights):\n"
                          "    a = torch.zeros((), dtype=torch.int32, device='cuda')\n"
                          f"    return {expr}\n"
                          "def kernel_fn(data, state):\n    return data\n")
                report = self.trace(source, [0], reference)
                self.assertEqual(report["stages"]["trace"]["status"], "PARTIAL-EMULATION")
                self.assertEqual(report["stages"]["trace"]["limit"], "prepare")
                self.assertEqual(report["result"]["cases"], [])


EMULATION_SCRIPT = r"""
import json, sys, torch
sys.path.insert(0, sys.argv[1])
import test_precheck as t
PC = t.PC
sub = sys.argv[2]
PC.Trace.sub_file = sub
missing = PC.install_stub_importer([], {"flashinfer.jit.env": {"CUTLASS_INCLUDE_DIRS": ["/judge/include"]}})
mode = PC.install_gpu_emulation(torch, {})
ns = {}
with mode():
    exec(compile(open(sub).read(), sub, "exec"), ns)
    res = {"dirs": ns["dirs"], "prepare": ns["prepare"]()}
    PC.Trace.in_kernel = True
    for fn in ("in_kernel", "after_stub", "unpack_stub", "none_bug"):
        PC.Trace.events, PC.Trace.stub_calls = [], []
        try:
            ns[fn]()
        except BaseException as e:
            r = PC._exc_record(e, sub)
            res[fn] = [type(e).__name__, r["emulation"], r["stub"]]
res["compared"] = PC.Trace.compared
print(json.dumps(res))
"""

EMULATED_SUBMISSION = """import torch
from flashinfer.jit.env import CUTLASS_INCLUDE_DIRS
from sgl_kernel.flash_attn import flash_attn_with_kvcache
dirs = list(map(str, CUTLASS_INCLUDE_DIRS))
def prepare():
    a = torch.zeros(4, device="cuda")
    return torch.equal(a, a + 1) and a.allclose(a)
def in_kernel():
    a = torch.zeros(4, device="cuda")
    return torch.equal(a, a)
def after_stub():
    out = flash_attn_with_kvcache(torch.zeros(4, device="cuda"))
    return torch.matmul(out, torch.zeros(4, device="cuda"))
def unpack_stub():
    st = {"attn": flash_attn_with_kvcache}
    a, b = st["attn"](1)
def none_bug():
    state = flash_attn_with_kvcache(1) and None
    return state + 1
"""


class EmulationTests(unittest.TestCase):
    def test_stub_values_comparison_shim_and_stub_attribution(self):
        # A subprocess: the emulation patches torch.cuda and installs an import hook for good.
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "submission.py"
            sub.write_text(EMULATED_SUBMISSION)
            r = subprocess.run([sys.executable, "-c", EMULATION_SCRIPT, str(TOOLS), str(sub)],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            res = json.loads(r.stdout.splitlines()[-1])
        self.assertEqual(res["dirs"], ["/judge/include"])  # the judge's include logic sees the mounted headers
        self.assertTrue(res["prepare"])  # prepare's comparisons assumed to hold ...
        self.assertEqual(sorted(set(res["compared"])), [7])  # ... and their line recorded
        self.assertEqual(res["in_kernel"], ["HostSync", False, None])  # in kernel_fn it is a host sync
        self.assertEqual(res["after_stub"], ["TypeError", True, "sgl_kernel"])
        self.assertEqual(res["unpack_stub"], ["ValueError", True, "sgl_kernel"])
        self.assertEqual(res["none_bug"], ["TypeError", False, None])  # the stub call does not explain this None


FAKE_DOCKER = """#!/bin/sh
case "$1" in
  ps) cat "$FAKE_DOCKER_PS" 2>/dev/null ;;
  run) echo "start-${FAKE_TAG:-dock}" >> "$FAKE_DOCKER_LOG"; sleep 1; echo "end-${FAKE_TAG:-dock}" >> "$FAKE_DOCKER_LOG" ;;
esac
"""


class DockLockTests(unittest.TestCase):
    """tools/dock and precheck's docker_lock against a fake docker: at most two dock containers overlap, and a
    precheck's container overlaps none."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "bin").mkdir()
        (self.tmp / "bin" / "docker").write_text(FAKE_DOCKER)
        (self.tmp / "bin" / "docker").chmod(0o755)
        self.log, self.ps = self.tmp / "log", self.tmp / "ps"
        self.env = dict(os.environ, PATH=f"{self.tmp / 'bin'}:{os.environ['PATH']}", OKC_DOCK_LOCK=str(self.tmp / "lock"),
                        FAKE_DOCKER_LOG=str(self.log), FAKE_DOCKER_PS=str(self.ps))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def dock(self):
        return subprocess.Popen([str(TOOLS / "dock"), "img", "true"], env=self.env, cwd=self.tmp,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_dock_and_precheck_serialize(self):
        locker = ("import os, subprocess, sys; sys.path.insert(0, sys.argv[1]); import test_precheck as t\n"
                  "with t.PC.docker_lock(): subprocess.run(['docker', 'run'])")
        procs = [self.dock() for _ in range(3)]
        procs.append(subprocess.Popen([sys.executable, "-c", locker, str(TOOLS)], env=dict(self.env, FAKE_TAG="pc"),
                                      stderr=subprocess.DEVNULL))
        for p in procs:
            self.assertEqual(p.wait(timeout=60), 0)
        events = self.log.read_text().split()
        self.assertEqual(sorted(events), sorted(["start-dock", "end-dock"] * 3 + ["start-pc", "end-pc"]))
        running, alone = 0, None
        for e in events:
            running += 1 if e.startswith("start") else -1
            self.assertLessEqual(running, 2)
            if e == "start-pc":
                self.assertEqual(running, 1)
                alone = True
            elif alone and e.startswith("start"):
                self.fail("a container started while the precheck's ran")
            if e == "end-pc":
                alone = False

    def test_a_killed_holders_container_blocks_the_next_run(self):
        self.ps.write_text("orphan123\n")  # what a SIGKILLed holder leaves running
        p = self.dock()
        time.sleep(2)
        self.assertFalse(self.log.exists())
        self.ps.unlink()
        self.assertEqual(p.wait(timeout=30), 0)
        self.assertEqual(self.log.read_text().split(), ["start-dock", "end-dock"])


if __name__ == "__main__":
    unittest.main()
