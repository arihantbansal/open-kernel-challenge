"""Build a single-file performance check from an entry, spec and benchmark kit.

    python benchmark/build.py benchmark/example_aa.py --out /tmp/carriers

The command assembles and validates source on CPU. It does not import the
entry's GPU kernels or launch a challenge job.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import assemble  # noqa: E402


def git_state():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=assemble.ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "benchmark"], cwd=assemble.ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def build(spec_path, out_dir):
    spec = assemble.load_spec(spec_path)
    entry = Path(spec["entry"])
    entry = entry if entry.is_absolute() else assemble.ROOT / entry
    source = entry.read_text()
    assemble.verify_entry(entry, source, spec.get("entry_sha"))
    text, provenance = assemble.assemble_text(source, assemble.sha(source), spec, git_state())
    path, digest = assemble.write_carrier(text, spec["name"], out_dir)
    return path, digest, len(text.encode()), provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, default=assemble.ROOT / "out" / "benchmark")
    args = parser.parse_args(argv)
    path, digest, size, provenance = build(args.spec, args.out)
    print("%s  %d bytes  sha256 %s" % (path, size, digest))
    print("entry %s kit %s spec %s rule: %s" % (
        provenance["entry_sha"][:12], provenance["kit_sha"][:12], provenance["spec_sha"][:12],
        provenance["predeclared"]["text"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
