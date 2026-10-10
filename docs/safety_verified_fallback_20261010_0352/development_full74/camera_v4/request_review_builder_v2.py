"""Draft builder and exact parent-approved freeze. Import performs no writes."""
import argparse,copy,sys
from repair_common_v2 import *

ATTEMPT=BASE/'env002_FR_native_entry_v4_attempt1'

def source_graph():
    original=read(CANDIDATE/'ORIGINAL_SOURCE_BINDING_V1.json')['frozen500_sources']
    core.verify(original)
    files=dict(original)
    for p,d in read(CANDIDATE/'PRESERVED_V1_INPUT_SHA256.json')['files'].items():
        require(sha(p)==d,'closed V1 source/artifact changed: '+p)
        files[p]=d
    # Include all local transitive runtime modules and immutable candidate files.
    folders=[HERE,CANDIDATE,H/'native_root_restore_equivalence_v1',H/'native_measurement_shutdown_v2',
        H/'parent_runtime_identity_v4',H/'native_lifecycle_repair_v1',H/'native_initialization_diagnostics_v1',
        H/'astra/full74_screen',H/'astra/full74_screen/parent_supervision_v1',
        H/'astra/full74_screen/parent_supervision_v2',H/'astra/full74_screen/parent_supervision_v3']
    for folder in folders:
        for p in folder.glob('*.py'):files[str(p)]=sha(p)
    for p in (H/'astra/full74_camera_angle_oracle_v2').glob('*.py'):files[str(p)]=sha(p)
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).resolve().is_relative_to(H) and Path(path).is_file():files[str(Path(path).resolve())]=sha(path)
    for p in [CANDIDATE/'ORIGINAL_SOURCE_BINDING_V1.json', CANDIDATE/'CAMERA_REPAIR_DIAGNOSTIC_REQUEST_DRAFT_V2.json', CANDIDATE/'PRESERVED_V1_INPUT_SHA256.json',
        Path('/home/liyufeng/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py'),
        Path('/home/liyufeng/miniforge3/envs/safeduo/lib/python3.11/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py')]:
        files[str(p)]=sha(p)
    return dict(schema='safeduo.camera_repair_source_graph.v2',files=files,
        original500=dict(files=original,all_unchanged=True),lifecycle_V2_proof=lifecycle_proof())

def proposed_request(graph,*,cpu_test_only=False,out=None):
    r=read(CANDIDATE/'CAMERA_REPAIR_DIAGNOSTIC_REQUEST_DRAFT_V2.json')
    r['source_files'].update(graph['files'])
    r['input_files'].update({str(p):sha(p) for p in [LIFE_WAIT,LIFE_REPORT,
        V9/'native_process/ACTUAL_WAIT.json',H/'full74_screen_parent_failure_v9_execution.json',
        H/'astra/full74_camera_upgrade_v1/CLOSED_CPU_HANDOFF_V1.json',H/'native_camera_repair_v1/CLOSED_V1_DIRECTION_BUDGET_PROOF_V1.json',
        CANDIDATE/'PRESERVED_V1_INPUT_SHA256.json',
        R/'astra_full74_independent_reader_v1/camera_angle_oracle_v2/READONLY_REVIEW_V2.json']})
    r.update(parent_frozen=False,draft_only=True,cpu_test_only=cpu_test_only,
        entry_protocol='CAMERA_REPAIR_ENTRY_V2_LIFECYCLE_V2',gpu_index=0,
        SSD_operational_reserve_bytes=RESERVE,min_NAS_free_bytes=NAS_FLOOR,
        lifecycle_V2_proof=graph['lifecycle_V2_proof'],source_graph_sha256=digest(graph),
        attempt_root=str(Path(out) if out else ATTEMPT),out=str((Path(out) if out else ATTEMPT)/'native'))
    return r

def frozen_value(draft):
    r=copy.deepcopy(draft);r.update(parent_frozen=True,draft_only=False)
    return r

def prepare():
    graph=source_graph();draft=proposed_request(graph)
    write_exact(HERE/'SOURCE_GRAPH_V2.json',graph)
    write_exact(HERE/'REQUEST_DRAFT_V2.json',draft)
    plan=dict(schema='safeduo.camera_repair_parent_plan.v2',status='HOLD_FOR_INDEPENDENT_PARENT_REVIEW',
        draft=str(HERE/'REQUEST_DRAFT_V2.json'),draft_sha256=digest(draft),
        source_graph=str(HERE/'SOURCE_GRAPH_V2.json'),source_graph_sha256=digest(graph),
        prospective_frozen_request_sha256=digest(frozen_value(draft)),attempt_root=str(ATTEMPT),
        max_wall_seconds=3600,resource=core.LIMITS,SSD_operational_reserve_bytes=RESERVE,
        min_NAS_free_bytes=NAS_FLOOR,case=CASE,
        ordering='V9_NATIVE_MINUS9_OUTER1_REAPED_AND_NO_OWNED_KIT_THEN_SINGLE_CASE_DEV_DIAGNOSTIC',
        subsequent_all32_short_requires='REAL_COUNTEREXAMPLE_DIAGNOSTIC_AND_INDEPENDENT_RAW_AUDIT',
        native_request_frozen=False,native_started=False,independent_raw_acceptance=False)
    write_exact(HERE/'PARENT_PLAN_DRAFT_V2.json',plan)
    approval=dict(schema='safeduo.camera_repair_parent_approval.v2',parent_execute_authorized=False,
        independent_review_status='PENDING',plan_sha256=digest(plan),
        prospective_frozen_request_sha256=plan['prospective_frozen_request_sha256'],
        source_graph_sha256=digest(graph),attempt_root=str(ATTEMPT),
        owned_inventory_complete=False,all_owned_native_launchers_use_serial_lock=False,
        no_other_owned_native_launcher_authorized=False,cpu_test_only=False,
        reviewer_identity='PARENT_MUST_SUPPLY',review_evidence=None)
    write_exact(HERE/'PARENT_APPROVAL_TEMPLATE_V2.json',approval)
    return plan

def review_plan(plan_path,plan_sha,approval_path,approval_sha):
    plan=bound_json(plan_path,plan_sha);approval=bound_json(approval_path,approval_sha)
    require(plan['schema']=='safeduo.camera_repair_parent_plan.v2' and plan['case']==CASE,'plan scope')
    require(plan['attempt_root']==str(ATTEMPT) and plan['max_wall_seconds']==3600 and
            plan['resource']==core.LIMITS and plan['SSD_operational_reserve_bytes']==RESERVE and
            plan['min_NAS_free_bytes']==NAS_FLOOR,'plan budgets/output immutable')
    require(approval['schema']=='safeduo.camera_repair_parent_approval.v2' and
        approval['parent_execute_authorized'] is True and approval['independent_review_status']=='REVIEWED' and
        approval['cpu_test_only'] is False and approval['reviewer_identity']!='PARENT_MUST_SUPPLY','parent approval required')
    for key in ['owned_inventory_complete','all_owned_native_launchers_use_serial_lock','no_other_owned_native_launcher_authorized']:
        require(approval.get(key) is True,'explicit parent serial declaration: '+key)
    require(approval['plan_sha256']==plan_sha and approval['attempt_root']==plan['attempt_root'] and
        approval['prospective_frozen_request_sha256']==plan['prospective_frozen_request_sha256'] and
        approval['source_graph_sha256']==plan['source_graph_sha256'],'approval exact plan/request/source binding')
    evidence=approval['review_evidence'];bound_bytes(evidence['path'],evidence['sha256'])
    graph=bound_json(plan['source_graph'],plan['source_graph_sha256']);core.verify(graph['files'])
    draft=bound_json(plan['draft'],plan['draft_sha256'])
    require(draft==proposed_request(graph),'draft differs from executable closed protocol')
    require(digest(frozen_value(draft))==plan['prospective_frozen_request_sha256'],'prospective frozen bytes')
    return plan,approval,draft

def freeze_after_review(root,draft,plan,approval,approval_path,approval_sha):
    require(root==ATTEMPT and root.is_dir(),'parent-held exclusive attempt')
    r=frozen_value(draft);require(digest(r)==plan['prospective_frozen_request_sha256'],'request exact reviewed bytes')
    request_path=root/'FROZEN_CAMERA_REPAIR_REQUEST_V2.json';write_exact(request_path,r)
    review=dict(schema='safeduo.camera_repair_native_review.v2',status='REVIEWED',request_sha256=digest(r),
        parent_execute_authorized=True,cpu_test_only=False,parent_approval=str(approval_path),
        parent_approval_sha256=approval_sha,source_graph_sha256=plan['source_graph_sha256'])
    review_path=root/'BOUND_PARENT_REVIEW_V2.json';write_exact(review_path,review)
    return request_path,sha(request_path),review_path,sha(review_path)

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--prepare-draft-only',action='store_true',required=True);p.parse_args()
    print(json.dumps(prepare()))
