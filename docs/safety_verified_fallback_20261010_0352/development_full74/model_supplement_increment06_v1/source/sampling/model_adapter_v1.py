"""Future approved MODEL_ONLY; immutable existing FK/frame/calibration, no fit."""
import common_v1 as c
from sampling_v1 import validate_manifest

def modules():
 for name in ('io_v2','prospective_rules_v1','fk_v2','contract_v2','calibration_v2'):
  c.module(c.pinned(c.OLD/(name+'.py')),name)
 return c.sys.modules['contract_v2'],c.sys.modules['calibration_v2']

def authority(r):
 a=c.read(r);proto=c.protocol();dep=c.read(c.ref(c.HERE/'MODEL_DATA_DEPENDENCIES_V1.json'));graph=c.source_graph()
 c.need(a['schema']=='astra.supplemental.parent_MODEL_ONLY_authority.v1' and a['status']=='PARENT_AUTHORIZED_SUPPLEMENTAL_MODEL_ONLY_CPU' and a['stage']=='SCREEN_FIXED_SUPPLEMENT_MODEL_ONLY' and a['draft'] is False,'new real parent MODEL_ONLY authority')
 c.need(a['source_graph']==graph and a['protocol']==c.ref(c.HERE/'PROTOCOL_V1.json') and a['original_cohort']==proto['original_cohort'],'exact new source/protocol and original provenance')
 c.need(a['native_admission'] is False and a['formal_holdout'] is False and a['bank_count']==0 and type(a['max_wall_s']) is int and 0<a['max_wall_s']<=3600,'finite original model CPU ceiling; no native or formal authority')
 peer=c.read(a['independent_source_review']);c.need(peer['source_graph']==graph and peer['protocol']==a['protocol'] and peer['status'] in ('PASS','COND_PASS') and peer['blocking_findings']==[],'distinct review of compatible new input/runner schema')
 m=validate_manifest(c.read(a['supplemental_cohort']));c.need(m['original_reproduction']==a['original_reproduction'],'real exact old8192 input reproduction bound')
 w=c.read(a['input_generation_wait']);c.support().closed_wait(w,0,'astra.full74.owned_cpu_wait.v1')
 c.need(w['argv']==[c.PY,'-B',str(c.HERE/'generate_inputs_v1.py'),'--out',str(c.Path(a['supplemental_cohort']['path']).parent)],'actual owned fixed generator wait')
 release=c.read(a['input_release']);c.need(release['supplemental_cohort']==a['supplemental_cohort'] and release['actual_input_worker_wait']==a['input_generation_wait'] and release['status']=='CLOSED_FIXED_INPUTS_CPU_ONLY_NO_MODEL' and release['all_owned_absent'] is True,'input release before model outcomes')
 for k in ('frame_binding','cal128','cal128_peer','model_replay_wait','model_replay_authority'):
  c.need(a[k]==dep[k],'unchanged actual calibrated model dependency '+k)
 return a

def actual_model(a):
 contract,cal=modules();dep=c.read(c.ref(c.HERE/'MODEL_DATA_DEPENDENCIES_V1.json'))
 b,model=contract.model_inputs(a['frame_binding']);c.need(b['model']==dep['validated_model'],'same validated full74 nonlinear FK; no fit or projection')
 report=c.read(a['cal128']);cal.released(report,a['frame_binding'])
 peer=c.read(a['cal128_peer']);c.need(peer['status'] in ('PASS','COND_PASS') and peer['blocking_findings']==[],'same real128 calibrated MODEL_ONLY peer')
 c.need(peer.get('model_replay',peer.get('calibration_result'))==a['cal128'],'exact observed64+64 replay peer binding')
 wait=c.read(a['model_replay_wait']);c.support().closed_wait(wait,0,'astra.full74.owned_cpu_wait.v1')
 c.need(str(c.OLD/'runner_v2.py') in wait['argv'] and '--authority' in wait['argv'] and wait['argv'][wait['argv'].index('--authority')+1]==a['model_replay_authority']['path'],'real original model calibration command')
 return model
