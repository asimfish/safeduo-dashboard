"""Bounded metadata inventory of earlier task evidence; never upgrades its scope."""
from pathlib import Path
import json,hashlib,datetime
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/liyufeng/safeduo')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def main():
    old=ROOT/'artifacts/forensics/20261004_randomized_safety_campaign_v4'
    gate=json.loads((old/'terminal_quality_gate.json').read_text())
    seal=json.loads((old/'delivery_seal.json').read_text())
    protocol=json.loads((old/'protocol.json').read_text())
    grasp=ROOT/'artifacts/grasp_fix_20260918/full_raw_summary.json'
    g=json.loads(grasp.read_text())
    files=[old/n for n in ('terminal_quality_gate.json','delivery_seal.json','delivery_integrity.json','analysis.json','protocol.json','pressure_analysis.json')]
    files+=[old/'block_00'/n for n in ('pair_contact_manifest.json','support_contact_manifest.json')]
    files+=[grasp,ROOT/'src/safeduo/envs/contact_gated_grasp.py',ROOT/'src/safeduo/envs/partner_contact_view.py',ROOT/'src/safeduo/safety/contact_parity.py']
    x=dict(status='COMPLETE_BOUNDED_METADATA_INVENTORY',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='read existing receipts and source contracts; not reanalysis of all raw physical data, not preregistration of new final trials',
        source_bindings=[bind(p) for p in files],
        historical_two_pair_physical=dict(tasks=seal['physical_complete_tasks'],pass_count=gate['task_pass'],fail_count=gate['task_fail'],
            sphere_violation_env_steps=gate['physical_official_violation_env_steps'],measurement_status=gate['measurement_status'],
            measurement_four_gate_dual_pass=gate['measurement_four_gate_dual_pass'],native_contact_manifests_present=True,
            task=protocol['task'],statistical_scope=protocol['statistical_scope'],approved=False,
            limitation='one two-pair task family, task success failures and uncalibrated measurement; cannot fill eight task families or independent full-scene hazard oracle'),
        earlier_grasp=dict(objects=g['objects'],criteria=g['criteria'],original_task_verdict=g['original_task_verdict'],
            limitation='old explicitly bounded lift/contact/release criteria; not a PASS under newly chosen100mm/2s calibration task'),
        implementation_inventory=dict(partner_contact='exact hand/plank attribution implemented; not all robot/table/payload pairs',
            constraint_grasp='explicit fixed-joint-assisted N=1 support; not friction-only grasp',
            legacy_contact_parity='net-force script cannot attribute partner and is explicitly a skeleton'),
        new_protocol_readiness='no PASS receipts have been registered to the new final protocol; this does not assert that no mechanism or old physical evidence exists')
    with (HERE/'EVIDENCE_INVENTORY.json').open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    for item in x['source_bindings']:assert sha(Path(item['path']))==item['sha256']
    print('INVENTORY',x['historical_two_pair_physical'])
if __name__=='__main__':main()
