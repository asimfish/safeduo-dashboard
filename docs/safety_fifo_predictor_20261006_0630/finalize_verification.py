"""Promote delivery certificate only after measured software/science/public gates."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
PUBLIC=Path('/home/liyufeng/safeduo-dashboard-feasible-guard-20261005/docs')/HERE.name
def load(n):return json.loads((HERE/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['local','complete']);args=a.parse_args()
    v=load('VERIFICATION.json');assert v['status'].startswith('PASS')
    for n in ['static_verification.json','local_browser.json','root_change_verification.json','raw_evidence_manifest.json','ASSET_PUBLICATION.json']:
        assert load(n)['status'].startswith('PASS'),n
    assert load('legacy_browser.json')['status']=='pass'
    v.update(status='PASS_LOCAL_DELIVERY_PENDING_PUBLIC' if args.stage=='local' else 'PASS',
       physical_safety_status='UNVALIDATED',hardware_approved=False,physical_safety_certified=False,production_promoted=False,
       utc=datetime.now(timezone.utc).isoformat(),public_delivery='PENDING' if args.stage=='local' else 'PASS_EXACT_PUBLIC_BYTES_AND_NATIVE_BROWSER')
    required=['static_verification.json','local_browser.json','legacy_browser.json','root_change_verification.json','raw_evidence_manifest.json','ASSET_PUBLICATION.json']
    if args.stage=='complete':
        extra=['remote_verification_initial.json','live_browser_initial.json','metadata_archive_receipt_initial.json']
        for n in extra:assert load(n)['status'].startswith('PASS'),n
        assert load('legacy_live_browser_initial.json')['status']=='pass'
        required+=extra+['legacy_live_browser_initial.json']
        remote=load('remote_verification_initial.json');v['initial_public_commit']=remote['commit'];v['public_url']=remote['url']
        v['initial_archive']=load('metadata_archive_receipt_initial.json')
    v['delivery_evidence_sha256']={n:sha(HERE/n) for n in required}
    v['limitations']=['conditioned correlated simulation cases; no IID reliability guarantee','nominal empirical predictor is not conservative','actual sphere frustum excludes occlusion/mesh silhouette certification','camera trajectories have separately recorded exact numerical replay status','immutable archived whitespace warnings, when present, are recorded rather than edited']
    (HERE/'VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\n');shutil.copy2(HERE/'VERIFICATION.json',PUBLIC/'VERIFICATION.json')
    assert sha(HERE/'VERIFICATION.json')==sha(PUBLIC/'VERIFICATION.json')
    print(v['status'],flush=True)
if __name__=='__main__':main()
