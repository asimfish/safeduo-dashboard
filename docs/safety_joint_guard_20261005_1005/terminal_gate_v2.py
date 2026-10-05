"""Record direct evidence for the study-delivery contract; physical approval is separate."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name): return json.loads((HERE / name).read_text())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['pre', 'local', 'delivery']); args = parser.parse_args()
    result = read('results.json'); camera = read('native_visual_verification.json')
    plan = read('holdout_plan_v2.json'); software = read('software_verification.json')
    freeze = read('offline_tool_freeze_v4.json')
    assert result['completed_method_windows'] + result['invalid_method_windows'] == 768
    assert result['analysis_sha256'] == sha(HERE / 'analyze_v4.py') == freeze['source_sha256']['analyze_v4.py']
    assert not result['hardware_approved'] and result['original_negative_endpoint']
    assert camera['status'] == 'PASS_BOUNDED_NATIVE_CAMERA_EVIDENCE' and camera['images'] == 378 and camera['groups'] == 42
    assert camera['verification_source_sha256'] == sha(HERE / 'verify_native_visual_v2.py')
    bank = read('bank_audit.json')
    assert bank['fresh_unique_initials']==192 and bank['old_reference_initials']==384
    assert bank['exact_duplicates_with_old']==0 and bank['selected_before_policy']
    assert all(row['all_selection_refs_valid'] and not row['policy_outcomes_used'] and row['selected_count']==64 for row in bank['rows'])
    commands=read('command_audit.json')
    assert commands['status']=='PASS_ACTUAL_INPUT_UNIQUENESS' and commands['fresh_unique_960_frame_tapes']==192 and commands['exact_repeats_with_prior']==0
    assert commands['verification_source_sha256']==sha(HERE/'audit_commands.py')
    for row in commands['rows']+commands['references']: assert sha(row['path'])==row['sha256']
    assert read('development_equivalence_v2.json')['status'] == 'PASS'
    assert read('fault_verification_v4.json')['status'] == 'PASS_EXPECTED_ABORT'
    assert software['tests'] == 20 and freeze['binding_tests'] == 16
    for name, digest in software['source_sha256'].items(): assert sha(HERE / name) == digest
    for name, digest in freeze['source_sha256'].items(): assert sha(HERE / name) == digest
    for name, digest in plan['source_sha256'].items(): assert sha(Path(plan['cwd']) / name) == digest
    for name, digest in plan['research_source_sha256'].items(): assert sha(name) == digest
    assert sha(plan['checkpoint_path']) == plan['checkpoint_sha256']
    snapshots = read('ASTRA_FRESH_SNAPSHOT_REVIEW.json')
    assert snapshots['status'] == 'PASS_BOUNDED_FRESH_SNAPSHOT_REVIEW' and not snapshots['pending_modes']
    assert snapshots['snapshot_files']==18 and snapshots['env_snapshots']==1152
    assert all(not row['failures'] for row in snapshots['runs'])
    assert 'PASS_BOUNDED_BINDING_REPAIR' in (HERE / 'ASTRA_BINDING_FINAL_REVIEW.md').read_text()
    assert 'PASS_BOUNDED_CAMERA_OBSERVATION' in (HERE / 'ASTRA_CAMERA_OBSERVATION_REVIEW.md').read_text()
    assert (HERE / 'ASTRA_EVIDENCE_REVIEW.md').exists()
    review=read('ASTRA_EVIDENCE_REVIEW.json')
    assert review['status']=='PASS_EVIDENCE_REVIEW' and not review['blocking_findings']
    assert review['parent_result_comparison_completed'] and review['actual_windows']==768 and review['unique_matched_cases']==192
    assert review['final_results_SHA']==sha(HERE/'results.json') and review['final_report_SHA']==sha(HERE/'REPORT.md')
    assert review['review_md_SHA256']==sha(HERE/'ASTRA_EVIDENCE_REVIEW.md')
    assert not review['hardware_approved'] and not review['production_promoted']
    matrix = []
    pending = ['artifact_integrity', 'browser', 'legacy', 'remote']
    contract = read('verification_contract.json')
    for criterion in contract['criteria']:
        key = criterion['id']
        status = 'PASS' if criterion['required'] and key not in pending else 'PENDING'
        if key == 'camera_forward_replay': status = 'FAIL_EXACT_REPLAY_REPORTED'
        if key == 'camera_near_far': status = 'UNABLE_TO_VERIFY_NOT_CLAIMED'
        if key == 'physical_carry': status = 'NOT_RUN_NOT_CLAIMED'
        matrix.append(dict(id=key, required=criterion['required'], status=status, oracle=criterion['oracle']))
    if args.phase in ['local', 'delivery']:
        browser = read('browser.json'); preservation = read('main_preservation.json')
        raw = read('raw_evidence_manifest.json'); legacy = read('legacy_browser.json')
        assert browser['status'] == preservation['status'] == 'PASS'
        assert raw['status'] == 'PASS_TWO_READS_CLOSED_RAW' and legacy['status'] == 'pass'
        assert browser['cases'] == len(result['case_endpoints']) and browser['images'] == 378
        for criterion in matrix:
            if criterion['id'] in ['artifact_integrity', 'browser', 'legacy']: criterion['status'] = 'PASS'
    if args.phase == 'delivery':
        remote = read('remote_verification.json'); live = read('live_browser.json'); legacy_live = read('legacy_live_browser.json')
        assert remote['status'] == live['status'] == 'PASS' and legacy_live['status'] == 'pass'
        assert remote['pages_commit'] == remote['commit']
        assert live['images'] == 378 and live['cases'] == len(result['case_endpoints'])
        for criterion in matrix:
            if criterion['id'] == 'remote': criterion['status'] = 'PASS'
        assert all(c['status'] == 'PASS' for c in matrix if c['required'])
    evidence = {str(p.relative_to(HERE)):sha(p) for p in [
        HERE/'results.json', HERE/'bank_audit.json', HERE/'command_audit.json', HERE/'native_visual_verification.json',
        HERE/'software_verification.json', HERE/'offline_tool_freeze_v4.json',
        HERE/'ASTRA_FRESH_SNAPSHOT_REVIEW.json', HERE/'ASTRA_BINDING_FINAL_REVIEW.md',
        HERE/'ASTRA_CAMERA_OBSERVATION_REVIEW.md', HERE/'ASTRA_EVIDENCE_REVIEW.md',
        HERE/'ASTRA_EVIDENCE_REVIEW.json',HERE/'REPORT.md',HERE/'comparison.png',HERE/'comparison.pdf',HERE/'comparison.svg']}
    receipt = dict(schema='safeduo.joint_guard_delivery_gate.v1',phase=args.phase,
        status='PASS' if args.phase=='delivery' else 'EVIDENCE_READY_UI_PENDING' if args.phase=='pre' else 'PASS_LOCAL_EVIDENCE_REMOTE_PENDING',
        verdict='PASS' if args.phase=='delivery' else None,criteria=matrix,evidence_sha256=evidence,
        candidate_physical_verdict='BLOCKED_UNVALIDATED_WITH_OBSERVED_FAILURES',hardware_approved=False,
        completed_method_windows=result['completed_method_windows'],invalid_method_windows=result['invalid_method_windows'],
        delivery_source_sha256={name:sha(HERE/name) for name in ['terminal_gate_v2.py','build_panel.py','panel_template.html','build_report.py','update_main_panels.py','check_panel_v2.py','legacy_check_panel_v2.py','seal_evidence.py','verify_remote_v2.py']},
        limits=['conditioned risk samples, no IID reliability or full26D volume',
            'full J offline only9 selected snapshots per cell','actual near/far planes unavailable',
            'native equality instantaneous; independent forward replays differ',
            'old carry19/192 retained separately; this study does not approve physical hardware'],
        security='only owned isolated artifacts and authorized static dashboard; no production actor/config or original scientific JSON changes',
        performance='no speed or throughput improvement claimed',compatibility='frozen IsaacLab/CUDA0 conditions only',
        metadata_closure='raw readback is required; final metadata archive/SEALED receipt recorded separately after remote gate')
    (HERE / 'VERIFICATION.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(receipt['status'])


if __name__ == '__main__': main()
