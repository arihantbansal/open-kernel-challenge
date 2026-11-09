"""Verify saved submissions and reject changed files or unsafe paths."""
import json
from pathlib import Path
import shutil,subprocess,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'tools/verify.py'
@pytest.fixture
def release(tmp_path):
    for name in ('submissions','results'):shutil.copytree(ROOT/name,tmp_path/name)
    return tmp_path

def run(root):return subprocess.run([sys.executable,str(SCRIPT),'--root',str(root)],capture_output=True,text=True,timeout=30)

def test_saved_results_verify():
    r=run(ROOT);assert r.returncode==0,r.stderr
    assert '5 submissions' in r.stdout

def test_changed_kernel_is_rejected(release):
    p=release/'submissions/muse_moe_w4a16.py';p.write_bytes(p.read_bytes()+b'\n# changed\n')
    r=run(release);assert r.returncode==1 and 'source hash mismatch' in r.stderr

def test_failing_saved_case_is_rejected(release):
    p=release/'results/results.json';d=json.loads(p.read_text());d['entries'][0]['benchmarks'][0]['status']='fail';p.write_text(json.dumps(d))
    r=run(release);assert r.returncode==1 and 'failed saved timing case' in r.stderr

def test_source_path_cannot_escape_release(release):
    p=release/'results/results.json';d=json.loads(p.read_text());d['entries'][0]['source']='../../.env';p.write_text(json.dumps(d))
    r=run(release);assert r.returncode==1 and 'unexpected source path' in r.stderr

def test_progress_cannot_inflate_saved_result(release):
    p=release/'results/progress.json';d=json.loads(p.read_text())
    next(e for e in d['milestones'] if e['stage']=='latest')['speedup']=99
    p.write_text(json.dumps(d))
    r=run(release);assert r.returncode==1 and 'progress disagrees with saved result' in r.stderr

def test_measured_hash_must_be_valid(release):
    p=release/'results/results.json';d=json.loads(p.read_text())
    d['entries'][0]['measured_sha256']='missing';p.write_text(json.dumps(d))
    r=run(release);assert r.returncode==1 and 'invalid measured hash' in r.stderr
