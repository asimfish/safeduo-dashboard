"""Retain failed system LP attempt; bind preselected compatible LP to all4 gates."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n):return json.loads((HERE/n).read_text())
def compatible_binding():
    registration=load('LP_BACKEND_REGISTRATION.json')
    assert registration['status']=='REGISTERED_COMPATIBLE_FIRST_FAILURE_BACKEND_BEFORE_DIAGNOSTIC_OUTCOMES'
    assert registration['geometry_threshold_epsilon_added'] is False and registration['global_package_mutation'] is False
    assert all(sha(p)==d for p,d in registration['sources'].items())
    lp=load('LP_COMPATIBLE_EXECUTION.json')
    assert lp['status']=='PASS_FROZEN_FIRST_FAILURE_COMPATIBLE_BACKEND'
    for field in ('interpreter','numpy','scipy'):assert lp[field]==registration[field],field
    assert lp['source_sha256']==registration['sources'][str(HERE/'failure_audit.py')]
    assert lp['wrapper_sha256']==registration['sources'][str(HERE/'run_lp_compatible.py')]
    assert lp['lp_math_and_options_unchanged'] and lp['geometry_native_float32_threshold_unchanged']
    assert lp['output_root_only_redirected'] and lp['registered_all12conditions_used'] and not lp['global_package_mutation']
    assert datetime.fromisoformat(registration['utc'])<datetime.fromisoformat(lp['utc'])
    src=HERE/'lp_backend_compatible/first_failure_audit.json'
    assert Path(lp['result_path'])==src and sha(src)==lp['result_sha256']
    result=json.loads(src.read_text())
    assert result['status']=='PASS_EXACT_FIRST_FAILURE_BINDING_AND_LP'
    assert result['cases']==lp['cases']==79==len(result['records'])
    assert result['source_sha256']==lp['source_sha256']
    assert all(sha(p)==sha(HERE/'lp_backend_compatible/plans'/p.name) for p in (HERE/'plans').glob('*_plan.json'))
    process=load('LP_COMPATIBLE_PROCESS_EXIT.json')
    assert process['observed_tool_returncode']==0 and process['completion_receipt_sha256']==sha(HERE/'LP_COMPATIBLE_EXECUTION.json')
    assert process['log']==str(HERE/'lp_compatible.log') and process['log_sha256']==sha(HERE/'lp_compatible.log')
    assert (HERE/'lp_compatible.log').read_text().rstrip().endswith(lp['status'])
    return registration,lp,result,src,process
def main():
    original=load('ANALYSIS_FOLLOW_EXECUTION.json');rows={r['name']:r for r in original['jobs']};assert len(rows)==4
    for name in ['run_analysis.py','audit_prediction_h6.py','audit_commands.py']:assert rows[name]['returncode']==0
    assert rows['failure_audit.py']['returncode']!=0 and original['status']=='FAIL'
    log=Path(rows['failure_audit.py']['log']);assert 'ImportError' in log.read_text() and "cannot import name 'Inf'" in log.read_text()
    registration,lp,result,src,process=compatible_binding()
    target=HERE/'first_failure_audit.json';assert not target.exists();shutil.copy2(src,target);assert sha(target)==sha(src)
    effective=[]
    for name in ['run_analysis.py','audit_prediction_h6.py','failure_audit.py','audit_commands.py']:
        if name=='failure_audit.py':
            row=dict(name=name,status='complete_source_bound_compatible_backend',
                original_failed_attempt=rows[name],original_returncode_retained=rows[name]['returncode'],
                actual_entry='run_lp_compatible.py',interpreter=lp['interpreter'],python=lp['python'],numpy=lp['numpy'],scipy=lp['scipy'],
                reproducible_argv=[lp['interpreter'],str(HERE/'run_lp_compatible.py')],
                actual_argv_retention='Exact argv was not recorded by the original compatible receipt; reproducible_argv is not a recovered process argv.',
                log=process['log'],log_sha256=process['log_sha256'],observed_tool_returncode=process['observed_tool_returncode'],
                completion_receipt='LP_COMPATIBLE_EXECUTION.json',completion_receipt_sha256=sha(HERE/'LP_COMPATIBLE_EXECUTION.json'),
                process_exit_receipt_sha256=sha(HERE/'LP_COMPATIBLE_PROCESS_EXIT.json'),source_sha256=lp['source_sha256'],wrapper_sha256=lp['wrapper_sha256'],result_sha256=lp['result_sha256'])
        else:
            row=dict(rows[name]);row['status']='complete'
        effective.append(row)
    receipt=dict(status='PASS_ALL4_SOURCE_BOUND_EFFECTIVE_READBACKS',jobs=effective,
        original_system_attempt_status=original['status'],original_system_lp_failure_retained=True,
        compatible_LP_backend_selected_before_diagnostic_results=True,source_math_or_geometry_threshold_changed=False,global_package_mutation=False,
        original_receipt_sha256=sha(HERE/'ANALYSIS_FOLLOW_EXECUTION.json'),lp_registration_sha256=sha(HERE/'LP_BACKEND_REGISTRATION.json'),compatible_completion_sha256=sha(HERE/'LP_COMPATIBLE_EXECUTION.json'),
        required_result_sha256={n:sha(HERE/n) for n in ['holdout_results.json','offline_analysis_execution.json','H6_PREDICTION_AUDIT.json','first_failure_audit.json','command_audit.json']},utc=datetime.now(timezone.utc).isoformat())
    with (HERE/'ANALYSIS_EFFECTIVE_EXECUTION.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(receipt['status'],flush=True)
if __name__=='__main__':main()
