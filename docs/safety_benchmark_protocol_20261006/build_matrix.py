"""Declared benchmark families; input/runner registrations remain separately gated."""
from pathlib import Path
import json,itertools,hashlib
from gate import registration_gate,COMMON,TASK,ROBUST
HERE=Path(__file__).resolve().parent
ARMS=['F_L','F_R','U_L','U_R'];PAIRS=list(itertools.combinations(ARMS,2))
def main():
    random=[('iid_joint','每步独立关节随机'),('mixed_hold','多保持长度随机'),('smooth_correlated','平滑/相关随机'),('stratified_workspace','广域分层初态'),('synchronized_crossing','四臂相向/交叉/让行'),('boundary_adversarial','开发冻结边界/对抗流')]
    tasks=[('single_pick_place','单臂抓放',[[a] for a in ARMS]),('pair_handover','六臂对交接',PAIRS),('pair_colift','六臂对共抬',PAIRS),
           ('two_pair_transport','两对同步搬运',[[['F_L','F_R'],['U_L','U_R']],[['F_L','U_L'],['F_R','U_R']],[['F_L','U_R'],['F_R','U_L']]]),
           ('four_shared_payload','四臂共享载荷',[ARMS]),('four_cross_transport','四臂相交搬运',[ARMS]),('coordinated_assembly','协作装配',[ARMS]),('dynamic_obstacle','突发障碍与让行',[ARMS])]
    robustness=[('latency','固定延迟/抖动/丢帧'),('perception','噪声/遮挡/时间错位'),('payload','载荷/对象变化'),('physical_parameters','摩擦/阻尼/执行器变化'),('long_horizon','16/60/180s长时与漂移'),('unseen_layout','未见布局/物体')]
    rows=[]
    for key,name in random:rows.append(dict(id=key,name=name,category='random_motion',variants='four arms,6 risk pairs,wide/global and boundary',status='DESIGN_ONLY'))
    for key,name,variants in tasks:rows.append(dict(id=key,name=name,category='real_task',variants=variants,difficulties=['easy','medium','hard'],status='G0_REQUIRED'))
    for key,name in robustness:rows.append(dict(id=key,name=name,category='robustness',status='DOMAIN_AND_G0_REQUIRED'))
    out=dict(schema='safeduo.standard_research_design.v1',rows=rows,
        baselines=['raw_task_policy','target_speed_only','original_backstop','system0_reference','frozen_candidate'],
        final_minimum_independent_seed_blocks=20,cases_per_block=64,
        sample_size_status='power and dependence estimation required on development data before final input registration',
        final_inputs_registered=False,final_protocol_experiments_executed=0,
        primary_comparison='candidate vs system0_reference; full-window event risk plus safe task success/mobility',
        protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest())
    with (HERE/'BENCHMARK_MATRIX.json').open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')
    # Do not silently promote old q-only experiments into complete task evidence.
    readiness={category:registration_gate(category,{}) for category in ('random_motion','real_task','robustness')}
    with (HERE/'READINESS.json').open('x') as f:json.dump(readiness,f,indent=2);f.write('\n')
    print('DECLARED20 FAMILIES; final inputs NOT_REGISTERED; gates NOT_READY')
if __name__=='__main__':main()
