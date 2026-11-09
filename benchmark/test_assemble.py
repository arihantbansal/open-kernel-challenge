import pytest

import assemble
from assemble import PatchError

ENTRY = '''# okc: problem=muse_decode_layer
import time


def prepare(weights):
    return {"w": weights}


def kernel_fn(data, state):
    return data
'''

SPEC = dict(name="t", entry="e.py", arms=[], shapes=["d1s"], rounds=2,
            predeclared=dict(shape="d1s", arm="twin", metric="ratio", op="<", value=0.97, text="t"))


def test_carrier_hooks_prepare_and_is_deterministic(tmp_path):
    e = tmp_path / "e.py"
    e.write_text(ENTRY)
    text, prov = assemble.assemble(e, SPEC, "abc")
    assert text.startswith("# okc: problem=muse_decode_layer\n")
    assert assemble.assemble(e, SPEC, "abc")[0] == text
    ns = {"__name__": "carrier"}
    exec(compile(text, "carrier", "exec"), ns)
    calls = []
    ns["ck_run"] = lambda state, t0, fn, spec, prov_, g: calls.append((state, spec["name"], prov_["entry_sha"]))
    assert ns["prepare"]("W") == {"w": "W"} and calls == [({"w": "W"}, "t", assemble.sha(ENTRY)[:16])]
    assert ns["CK_SPEC"]["name"] == "t" and prov["spec_sha"] == assemble.spec_sha(SPEC)[:16]
    assert prov["shapes"] == SPEC["shapes"]


def test_generated_carrier_resolves_and_applies_an_arm():
    spec = dict(SPEC, arms=[dict(name="candidate", apply="h_candidate")])
    text, _ = assemble.assemble_text(ENTRY, assemble.sha(ENTRY), spec)
    ns = {"__name__": "carrier"}
    exec(compile(text, "carrier", "exec"), ns)

    def candidate(state, key):
        old = state["variant"]
        state["variant"] = key
        return lambda: state.update(variant=old)

    ns["h_candidate"] = candidate
    arm, = ns["ck_resolve_arms"](ns["CK_SPEC"], "d1s", None, ns)
    state = dict(variant="entry")
    undo = arm.apply(state)
    assert arm.name == "candidate" and state["variant"] == "d1s"
    undo()
    assert state["variant"] == "entry"


def test_inlining_preserves_multiline_function_decorators():
    src = '''
def multiply(n):
    return lambda fn: lambda x: n * fn(x)

@multiply(
    2
)
@multiply(3)
def scale(x):
    return x + 1
'''
    ns = {}
    exec(assemble.inline_module(src, set(), "decorated.py", []), ns)
    assert ns["scale"](4) == 30


def test_write_carrier_names_by_sha_and_records(tmp_path):
    path, digest = assemble.write_carrier("x = 1\n", "n", tmp_path)
    assert path.name == "n.%s.py" % digest[:8]
    assert (tmp_path / "SHA256SUMS").read_text() == "%s  %s\n" % (digest, path.name)
    assemble.write_carrier("x = 1\n", "n", tmp_path)
    assert (tmp_path / "SHA256SUMS").read_text().count("\n") == 1


def test_top_level_clash_with_the_entry_is_refused(tmp_path):
    e = tmp_path / "e.py"
    e.write_text(ENTRY + "\n\ndef compare():\n    pass\n")
    with pytest.raises(PatchError, match="compare"):
        assemble.assemble(e, SPEC)


@pytest.mark.parametrize("bad", ["import subprocess", "from urllib.request import urlopen", "import reference",
                                 "import os\nos.system('x')"])
def test_check_refuses_banned_use(bad):
    with pytest.raises(PatchError):
        assemble.check("# okc\n" + bad + "\n", "t")


def test_check_size_limit():
    with pytest.raises(PatchError, match="1 MB"):
        assemble.check("x = '%s'\n" % ("a" * (1 << 20)), "t")


def test_entry_must_define_prepare_and_kernel_fn(tmp_path):
    e = tmp_path / "e.py"
    e.write_text("# okc\nx = 1\n")
    with pytest.raises(PatchError, match="prepare"):
        assemble.assemble(e, SPEC)


def test_entry_pin_mismatch(tmp_path):
    e = tmp_path / "e.py"
    e.write_text(ENTRY)
    with pytest.raises(PatchError, match="pin"):
        assemble.assemble(e, dict(SPEC, entry_sha="000000"))


def test_load_spec_validates_rule(tmp_path):
    f = tmp_path / "s.py"
    f.write_text("SPEC = %r" % dict(SPEC, predeclared=dict(SPEC["predeclared"], arm="nope")))
    with pytest.raises(PatchError, match="arm"):
        assemble.load_spec(f)
    f.write_text("SPEC = %r" % SPEC)
    assert assemble.load_spec(f)["name"] == "t"


def test_replace_once():
    assert assemble.replace_once("a b", "a", "c", "x") == "c b"
    with pytest.raises(PatchError):
        assemble.replace_once("a a", "a", "c", "x")


def test_load_spec_accepts_best_arm_only_with_arms(tmp_path):
    f = tmp_path / "s.py"
    rule = dict(SPEC["predeclared"], arm="best", metric="saved_us")
    f.write_text("SPEC = %r" % dict(SPEC, predeclared=rule))
    with pytest.raises(PatchError, match="arm"):
        assemble.load_spec(f)  # no arms to take the best of
    f.write_text("SPEC = %r" % dict(SPEC, predeclared=rule, arms=[dict(name="a"), dict(name="b")]))
    assert assemble.load_spec(f)["predeclared"]["arm"] == "best"
    f.write_text("SPEC = %r" % dict(SPEC, predeclared=rule, arms=[dict(name="best")]))
    with pytest.raises(PatchError, match="arm"):
        assemble.load_spec(f)  # an arm named best would be ambiguous
