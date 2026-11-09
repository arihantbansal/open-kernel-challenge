#!/usr/bin/env python3
"""Verify submission files and saved results."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

PROBLEMS = {
    'muse_decode_layer', 'muse_attention_decode', 'muse_moe_w4a16',
    'muse_bf16_proj', 'muse_prefill_layer',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def verify(root):
    records = json.loads((root / 'results/results.json').read_text())
    require(records['schema_version'] == 1, 'unsupported result schema')
    entries = records['entries']
    require(len(entries) == 5 and {e['problem'] for e in entries} == PROBLEMS,
            'expected one entry per problem')
    count = 0
    for entry in entries:
        problem = entry['problem']
        expected = f'submissions/{problem}.py'
        require(entry['source'] == expected, f'unexpected source path: {problem}')
        path = root / expected
        require(not path.is_symlink(), f'submission is a symlink: {problem}')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == entry['sha256'],
                f'source hash mismatch: {problem}')
        measured = entry['measured_sha256']
        require(isinstance(measured, str) and len(measured) == 64
                and all(c in '0123456789abcdef' for c in measured),
                f'invalid measured hash: {problem}')
        text = raw.decode()
        require(text.splitlines()[0] == f'# okc: problem={problem}',
                f'wrong problem header: {problem}')
        functions = {node.name for node in ast.parse(text).body
                     if isinstance(node, ast.FunctionDef)}
        needed = {'kernel_fn'} if problem == 'muse_attention_decode' else {'prepare', 'kernel_fn'}
        require(needed <= functions, f'missing entrypoint: {problem}')
        summary = entry['summary']
        require(summary['passed'] is True and summary['status'] == 'ok',
                f'failed saved result: {problem}')
        require(type(summary['tests']) is int and summary['tests'] > 0
                and summary['tests_passed'] == summary['tests'],
                f'incomplete saved tests: {problem}')
        require(is_positive(summary['speedup']), f'invalid speedup: {problem}')
        job = entry['job_id']
        require(isinstance(job, str) and len(job) == 16
                and all(c in '0123456789abcdef' for c in job), f'invalid job ID: {problem}')
        require(entry['benchmarks'], f'missing per-case evidence: {problem}')
        for timing in entry['benchmarks']:
            require(timing['status'] == 'pass', f'failed saved timing case: {problem}')
            require(is_positive(timing['baseline_ns']) and is_positive(timing['median_ns']),
                    f'invalid duration: {problem}')
            count += 1
    progress = json.loads((root / 'results/progress.json').read_text())
    require(progress['schema_version'] == 1, 'unsupported progress schema')
    milestones = progress['milestones']
    require(milestones, 'missing progress records')
    latest = {}
    jobs = set()
    for milestone in milestones:
        problem = milestone['problem']
        require(problem in PROBLEMS and milestone['stage'] in ('earlier', 'latest'),
                'invalid progress problem or stage')
        require(milestone['mode'] == 'run' and milestone['passed'] is True,
                'progress must use successful scored runs')
        require(is_positive(milestone['speedup']) and is_positive(milestone['score_us']),
                'invalid progress score')
        measured = milestone['measured_sha256']
        require(isinstance(measured, str) and len(measured) == 64
                and all(c in '0123456789abcdef' for c in measured),
                'invalid progress source hash')
        job = milestone['job_id']
        require(isinstance(job, str) and len(job) == 16
                and all(c in '0123456789abcdef' for c in job) and job not in jobs,
                'invalid or duplicate progress job')
        jobs.add(job)
        if milestone['stage'] == 'latest':
            require(problem not in latest, 'duplicate latest progress record')
            latest[problem] = milestone
    require(set(latest) == PROBLEMS, 'progress must include every latest result')
    for entry in entries:
        milestone = latest[entry['problem']]
        require(milestone['job_id'] == entry['job_id']
                and milestone['measured_sha256'] == entry['measured_sha256']
                and milestone['speedup'] == entry['summary']['speedup']
                and milestone['score_us'] == entry['summary']['score_us']
                and milestone['tests_passed'] == entry['summary']['tests_passed'],
                f"progress disagrees with saved result: {entry['problem']}")
    rank = json.loads((root / 'results/peak-rank.json').read_text())
    require(rank['board'] == 'muse_moe_w4a16' and rank['displayed_rank'] == 1,
            'peak rank must identify the MoE board')
    require(rank['handle'] == 'arihantbansal' and rank['post_url'] ==
            'https://x.com/arihantbansal/status/2113367627251032549',
            'unexpected rank post')
    return len(entries), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        entries, count = verify(args.root)
    except (ValueError, KeyError, OSError, SyntaxError, TypeError) as error:
        parser.exit(1, f'FAIL: {error}\n')
    print(f'PASS: {entries} submissions; {count} saved timing cases; progress and rank records.')


if __name__ == '__main__':
    main()
