"""Correct derivative metadata labels; preserve original closure bytes and decisions."""
from pathlib import Path
import copy,hashlib,json,shutil
P=Path(__file__).resolve().parent;H=P.parent
W=Path('/home/liyufeng/safeduo-dashboard-loaded-feedback-20261012');R=W/'docs/safety_loaded_feedback_20261012'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
    original=H/'fourhand_native_probe_v102/LOADED_FEEDBACK_DIAGNOSTIC_DECISION_V1.json'
    reg=read(H/'fourhand_native_probe_v102/REGISTRATION_V1.json');assert reg['first64_physical_prefix_baseline']==101
    result=copy.deepcopy(read(original));assert result['first64_physical_prefix98_byte_identical']
    result['first64_physical_prefix_baseline_version']=101
    result['first64_physical_prefix_required']=result.pop('first64_physical_prefix98_required')
    result['first64_physical_prefix_byte_identical']=result.pop('first64_physical_prefix98_byte_identical')
    result['metadata_correction']={'scope':'Field labels only; actual audit v2 already selected registered101 baseline.',
        'original_preserved':str(original),'original_sha256':sha(original),'original_gate_results_unchanged':True,
        'no_claim_of_physical_prefix_equality_to98':True,'adapter_sha256':sha(Path(__file__))}
    new=original.with_name('LOADED_FEEDBACK_DIAGNOSTIC_DECISION_V2.json');assert not new.exists();write(new,result)
    snapshot=P/'presentation_v1_preserved';assert not snapshot.exists();snapshot.mkdir()
    for f in [R/'index.html',R/'PUBLIC_MANIFEST_V1.json',P/'BUILD_MANIFEST_V1.json',R/'evidence/102_DECISION_V1.json']:shutil.copyfile(f,snapshot/f.name)
    write(R/'evidence/102_DECISION_V2.json',result)
    shutil.copyfile(Path(__file__),R/'evidence/correct_prefix_labels_v2.py')
    matrix=copy.deepcopy(read(H/'full_requirements_v1/EVIDENCE_MATRIX_V29.json'));oldrows=copy.deepcopy(matrix['rows'])
    # JSON object keys become strings after roundtrip.
    matrix['loaded_native_development_evidence']['decisions']['102']=result
    matrix.update(revision=30,metadata_revision_reason='Correct physical prefix102 baseline label to101; all original9 rows and scientific decisions unchanged.')
    assert matrix['rows']==oldrows;out=H/'full_requirements_v1/EVIDENCE_MATRIX_V30.json';assert not out.exists();write(out,matrix);write(R/'EVIDENCE_MATRIX_V30.json',matrix)
    page=(R/'index.html').read_text().replace('evidence/102_DECISION_V1.json','evidence/102_DECISION_V2.json').replace('href="EVIDENCE_MATRIX_V29.json"','href="EVIDENCE_MATRIX_V30.json"').replace('href="PUBLIC_MANIFEST_V1.json"','href="PUBLIC_MANIFEST_V2.json"')
    page=page.replace('复现应先核验','102判定的前64步物理基线为101；旧V1字段名沿用98，V2只修正字段标签，原判据和原始回执保留。<a href="evidence/correct_prefix_labels_v2.py">标签修正源码</a>。复现应先核验')
    (R/'index.html').write_text(page)
    files=[{'path':str(f.relative_to(W)),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(R.rglob('*')) if f.is_file()]
    write(R/'PUBLIC_MANIFEST_V2.json',{'files':files,'manifest_self_excluded':True,'fullSystem0_accepted':False})
    files.append({'path':str((R/'PUBLIC_MANIFEST_V2.json').relative_to(W)),'sha256':sha(R/'PUBLIC_MANIFEST_V2.json'),'bytes':(R/'PUBLIC_MANIFEST_V2.json').stat().st_size})
    files.append({'path':'index.html','sha256':sha(W/'index.html'),'bytes':(W/'index.html').stat().st_size})
    old=read(P/'BUILD_MANIFEST_V1.json');old.update(files=files,metadata_correction_only=True,prior_manifest_sha256=sha(P/'BUILD_MANIFEST_V1.json'));write(P/'BUILD_MANIFEST_V2.json',old)
    print('PREFIX_BASELINE_LABELS_CORRECTED_TO101; ALL_ORIGINAL_THRESHOLDS_RESULTS_AND_FAILURES_PRESERVED',len(files),'FILES')
if __name__=='__main__':main()
