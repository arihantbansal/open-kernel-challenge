"""Carrier assembly (CPU): inline the kit's modules flat into an entry file, append the spec, wrap prepare().

Pure Python. No kit module is imported by a carrier: the build parses each module, drops the imports of other kit
modules, hoists the rest, and refuses a top-level name the entry already binds.
"""

import ast
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAX_BYTES = 1 << 20
KIT = ("stats", "budget", "schema", "gpu", "recipes.layer", "recipes.moe", "recipes.attn", "engine", "xray")
BANNED_IMPORTS = {"subprocess", "socket", "urllib", "requests", "http", "ftplib", "reference"}
REQUIRED = ("name", "arms", "shapes", "predeclared")
RULE_KEYS = ("shape", "arm", "metric", "op", "value")


class PatchError(Exception):
    pass


def replace_once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        raise PatchError("%s: expected exactly one occurrence, found %d: %r" % (where, n, old[:90]))
    return text.replace(old, new)


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def check(text, label):
    """The submission rules a carrier must keep: one file under 1 MB, compiles, no network or subprocess use, no
    reference import."""
    if len(text.encode()) > MAX_BYTES:
        raise PatchError("%s: %d bytes, over 1 MB" % (label, len(text.encode())))
    tree = ast.parse(text, label)
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            if m.split(".")[0] in BANNED_IMPORTS:
                raise PatchError("%s: imports %s" % (label, m))
        if isinstance(node, ast.Attribute) and node.attr in ("system", "Popen") and isinstance(node.value, ast.Name) \
                and node.value.id in ("os", "subprocess"):
            raise PatchError("%s: uses %s.%s" % (label, node.value.id, node.attr))
    compile(tree, label, "exec")


def bindings(tree):
    """Top-level names bound by def, class and plain assignment (imports are idempotent and not counted)."""
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    return out


def inline_module(src, taken, label, imports):
    """The module's body as top-level statements. Kit-internal imports and docstrings are dropped, other imports
    are added to `imports` (deduplicated, hoisted by the caller); `taken` grows with the module's own names."""
    tree = ast.parse(src)
    parts = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and getattr(node.test.left, "id", "") == "__name__":
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module in KIT or (node.module or "").startswith("recipes")):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(src, node))
            continue
        # Function/class spans start at def/class, after their decorators.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list:
            start = min(d.lineno for d in node.decorator_list) - 1
            parts.append("\n".join(src.splitlines()[start:node.end_lineno]))
        else:
            parts.append(ast.get_source_segment(src, node))
    clash = sorted(bindings(ast.parse("\n".join(parts))) & taken)
    if clash:
        raise PatchError("%s: top-level names already bound in the entry or an earlier kit module: %s" % (label, clash))
    taken |= bindings(ast.parse("\n".join(parts)))
    return "\n\n\n".join(parts)


def kit_source(entry_text, modules, kit_dir=HERE):
    """-> (kit text, kit sha). Modules in the order given, with all hoisted imports first."""
    taken = bindings(ast.parse(entry_text))
    imports, bodies, digest = [], [], hashlib.sha256()
    for m in modules:
        path = kit_dir / (m.replace(".", "/") + ".py")
        src = path.read_text()
        digest.update(src.encode())
        bodies.append("# ---- benchmark/%s.py ----\n\n%s" % (m.replace(".", "/"), inline_module(src, taken, path.name, imports)))
    head = "\n".join(dict.fromkeys(imports))
    return "# ---- benchmark (inlined flat) ----\n" + head + "\n\n\n" + "\n\n\n".join(bodies), digest.hexdigest()


def load_spec(path):
    ns = {}
    exec(compile(Path(path).read_text(), str(path), "exec"), ns)  # noqa: S102 - a spec is our own checked-in file
    spec = ns["SPEC"]
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise PatchError("spec misses %s" % missing)
    if not spec.get("entry"):
        raise PatchError("spec needs `entry` (a single-file submission)")
    rule = spec["predeclared"]
    if [k for k in RULE_KEYS if k not in rule] or rule["op"] not in ("<", ">") or rule["metric"] not in ("ratio", "saved_us"):
        raise PatchError("predeclared needs shape, arm, metric (ratio|saved_us), op (<|>), value and text: %r" % (rule,))
    names = [a["name"] for a in spec["arms"]] + ["twin"]
    best = rule["arm"] == "best" and bool(spec["arms"])  # the best valid arm (stats.eval_kill); no arm may be named so
    if (rule["arm"] not in names and not best) or len(set(names)) != len(names) or "best" in names:
        raise PatchError("predeclared arm %r is not a unique arm name in %s" % (rule["arm"], names))
    if rule["shape"] not in spec["shapes"]:
        raise PatchError("predeclared shape %r is not in shapes" % rule["shape"])
    return spec


def spec_sha(spec):
    return sha(json.dumps(spec, sort_keys=True, default=str))


def verify_entry(path, text, pinned=None, sums=()):
    """The entry's sha must match the spec's pin and any SHA256SUMS line that names the file."""
    digest = sha(text)
    if pinned and not digest.startswith(pinned):
        raise PatchError("entry sha %s does not start with the spec's pin %s" % (digest[:12], pinned))
    for s in sums:
        f = ROOT / s
        if f.exists():
            for line in f.read_text().splitlines():
                h, _, name = line.partition("  ")
                name = (name.split() or [""])[0]
                if name in (Path(path).name, str(path)) and h != digest:
                    raise PatchError("%s lists %s as %s; the file is %s" % (s, name, h[:12], digest[:12]))
    return digest


GLUE = '''

# ---- benchmark: spec and prepare() hook ----
CK_SPEC = %(spec)s
CK_PROV = %(prov)s
_ck_entry_prepare = prepare


def prepare(weights):
    t_start = time.time()
    state = _ck_entry_prepare(weights)
    if CK_SPEC.get("recipe") == "layer":
        layer_residual_alpha(state, weights)
    ck_run(state, t_start, kernel_fn, CK_SPEC, CK_PROV, globals())
    return state
'''


def assemble(entry_path, spec, git="unknown", kit_dir=HERE):
    """-> (carrier text, provenance dict) for an entry file. The provenance excludes the carrier's own sha;
    the service's content_hash contract is not established."""
    entry = Path(entry_path).read_text()
    return assemble_text(entry, verify_entry(entry_path, entry, spec.get("entry_sha")), spec, git, kit_dir)


def assemble_text(entry, entry_sha, spec, git="unknown", kit_dir=HERE):
    if "\ndef prepare(" not in entry or "\ndef kernel_fn(" not in entry:
        raise PatchError("entry must define module-level prepare(weights) and kernel_fn(data, state)")
    kit, kit_sha = kit_source(entry, list(KIT), kit_dir)
    hooks = ""
    if spec.get("hooks"):
        hooks = "\n\n# ---- spec hooks ----\n" + Path(spec["hooks"]).read_text().rstrip("\n") + "\n"
    prov = dict(name=spec["name"], problem=spec.get("problem", "muse_decode_layer"), entry_sha=entry_sha[:16],
                kit_sha=kit_sha[:16], spec_sha=spec_sha(spec)[:16], git=git, shapes=spec["shapes"],
                predeclared=spec["predeclared"])
    text = entry.rstrip("\n") + "\n\n\n" + kit + hooks + GLUE % dict(spec=repr(spec), prov=repr(prov))
    check(text, spec["name"])
    return text, prov


def write_carrier(text, name, out_dir):
    """Write a carrier and its SHA256SUMS entry. -> (path, full sha)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = sha(text)
    path = out_dir / ("%s.%s.py" % (name, digest[:8]))
    path.write_text(text)
    sums = out_dir / "SHA256SUMS"
    lines = [ln for ln in (sums.read_text().splitlines() if sums.exists() else []) if not ln.endswith("  " + path.name)]
    sums.write_text("\n".join(lines + ["%s  %s" % (digest, path.name)]) + "\n")
    return path, digest
