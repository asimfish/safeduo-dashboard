"""Keep initial overlap and prevention outcomes distinct, with matched tapes."""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import copy
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
METHODS={'raw':'无保护','backstop_only':'仅解析兜底','system0':'System 0'}


def summary(episodes):
    safe=[e for e in episodes if not e['initial_violation']]
    return dict(episodes=len(episodes),initial_violations=len(episodes)-len(safe),
                initially_safe=len(safe),violations=sum(e['violation'] for e in episodes),
                safe_initial_violations=sum(e['violation'] for e in safe),
                safe_initial_damaging=sum(e['damaging'] for e in safe),
                by_class={k:sum(e['violation_by_class'][k] for e in safe)
                          for k in ['cross','self_F','self_U','table']},
                min_arm_hand_mm=1000*min(min(e['min_margin_m'][k] for k in ['cross','self_F','self_U']) for e in safe) if safe else None,
                mean_joint_range=float(np.mean([e['joint_range_fraction_mean'] for e in safe])) if safe else None,
                mean_joint_path_rad=float(np.mean([e['measured_joint_path_rad'] for e in safe])) if safe else None,
                output_ratio=sum(e['executed_l2_sum'] for e in safe)/sum(e['command_l2_sum'] for e in safe) if safe else None,
                pair_union_exposed=int((np.array([e['pair_warn_steps'] for e in safe])>0).any(0).sum()) if safe else 0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',required=True)
    args=parser.parse_args()
    manifests=[];all_episodes=[];grouped=defaultdict(list);matched=defaultdict(list)
    latency_audit=[]
    for name in args.runs:
        path=OUT/name
        p=json.loads((path/'protocol.json').read_text())
        assert p['status']=='complete' and p['completed_cells']==len(p['design']),name
        manifests.append(p)
        episodes=json.loads((path/'episodes.json').read_text())
        for e in episodes:
            e={**e,'campaign':name,'duration_s':p['args']['duration_s'],
               'hold_steps':e.get('hold_steps',0)}
            all_episodes.append(e)
            group=(name,e['env_yaml'],e['initial_jitter_rad'],e['actuator_delay_steps'],e['method'],e['hold_steps'])
            grouped[group].append(e)
            # Match methods and candidate YAMLs when physical configuration is frozen.
            key=(e['seed'],e['amp'],e['flow'],e['hold_steps'],e['duration_s'],e['initial_jitter_rad'],e['actuator_delay_steps'],e['env_id'])
            matched[key].append(e)
        for i,cell in enumerate(p['design'],1):
            delay=cell['actuator_delay_steps']
            with np.load(path/f'cell_{i:03d}.npz') as trace:
                assert trace['q'].shape[0]==p['steps']
                if delay:
                    a=trace['actuator_target'];c=trace['controller_target'];q0=trace['q_initial']
                    np.testing.assert_array_equal(a[:delay],np.broadcast_to(q0,a[:delay].shape))
                    np.testing.assert_array_equal(a[delay:],c[:-delay])
                    latency_audit.append(dict(campaign=name,cell_id=i,steps=delay,exact_target_fifo=True))
    assert len({p['checkpoint_sha256'] for p in manifests})==1
    config_ref=None
    for p in manifests:
        cfg=copy.deepcopy(p['resolved_config'])
        cfg['safety'].pop('row_priority_lookahead_s',None)
        for key in ['lookahead_s','self_lookahead_s','table_lookahead_s','predict_backlog','pending_target_steps']:
            cfg['safety']['backstop'].pop(key,None)
        if config_ref is None: config_ref=cfg
        else: assert cfg==config_ref,'unexpected physics/actor/config change between candidates'
    core=['src/safeduo/safety/geometry.py','src/safeduo/safety/backstop.py','src/safeduo/safety/sphere_distance.py','src/safeduo/envs/duo_env.py']
    for key in core:
        assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest()==manifests[-1]['source_sha256'][key],key
    # Historical failed candidates may precede an opt-in source fix. Record,
    # rather than silently erase, each campaign's actual source identity.
    core_sources=[dict(campaign=Path(p['args']['out']).name,
                       sha256={key:p['source_sha256'][key] for key in core}) for p in manifests]
    paired=[]
    for key,rows in matched.items():
        assert len({e['initial_q_sha256'] for e in rows})==1,key
        assert len({e['command_sha256'] for e in rows})==1,key
        assert len({e['initial_violation'] for e in rows})==1,key
        for e in rows:
            paired.append({k:e[k] for k in ['campaign','env_yaml','seed','duration_s','initial_jitter_rad','actuator_delay_steps','hold_steps','env_id','method','initial_q_sha256','command_sha256','initial_violation','violation']})
    result_rows=[]
    for group,rows in grouped.items():
        name,yaml,jitter,delay,method,hold=group
        result_rows.append(dict(campaign=name,candidate=yaml,jitter_rad=jitter,delay_steps=delay,
             delay_ms=delay*rows[0]['dt']*1000,method=method,label=METHODS[method],
             seeds=sorted({e['seed'] for e in rows}),flow=rows[0]['flow'],hold_s=hold*rows[0]['dt'] if hold else None,
             duration_s=next(p['args']['duration_s'] for p in manifests if p['args']['out'].endswith('/'+name)),
             **summary(rows)))
    results=dict(schema='safeduo.perturbation.v1',rows=result_rows,manifests=manifests,core_sources=core_sources,
        latency_audit=latency_audit,method_pairing=True,initial_collision_rejection=False,
        scope='Current measurements; post-safety target delivery delay; arm-joint initial jitter; sphere-layer outcomes',
        limitations=['局部关节初态扰动，不是全空间均匀采样；初态碰撞未筛除，预防结果仅取无违规初态子集。',
            '手指、动力学、传感时间和真实物体任务固定或未测；执行端目标延迟不等于感知延迟。',
            '球距在每个约16.67ms控制周期结束时采样，未证明两个采样点之间不存在短暂碰撞。',
            '窗口和保持动作相关，不提供独立样本安全置信界；实际运动量和输出比例不是任务成功率。'])
    latest_matrix=[e for e in all_episodes if e['campaign']=='pending_final_matrix']
    controls=[e for e in all_episodes if e['campaign']=='stored_verified_matrix' and e['method']=='raw']
    matrix=latest_matrix+controls if latest_matrix else [e for e in all_episodes if 'matrix' in e['campaign']]
    results['matrix_control_reuse']='Reuse the four stored_verified_matrix raw cells; pending-target history only affects safety projection, which RawShim bypasses. Initial q and full command tape hashes matched.' if latest_matrix else None
    results['matrix_headline']=[dict(method=method,label=METHODS[method],**summary([e for e in matrix if e['method']==method]))
                               for method in METHODS if any(e['method']==method for e in matrix)]
    final=[e for e in all_episodes if e['env_yaml']=='duo_env_a31_pending_guard.yaml']
    results['final_protected_total']=summary(final)
    results['final_by_method']={method:summary([e for e in final if e['method']==method])
                              for method in METHODS if any(e['method']==method for e in final)}
    base={(e['actuator_delay_steps'],e['env_id']):e for e in all_episodes if e['campaign']=='pilot_initial_delay'}
    comparisons=[]
    for delay in [0,6]:
        now=[e for e in all_episodes if e['campaign']=='pending_final_pilot' and e['actuator_delay_steps']==delay]
        if now:
            comparisons.append(dict(delay_steps=delay,
                new_failures=[e['env_id'] for e in now if e['violation'] and not base[(delay,e['env_id'])]['violation']],
                corrected_failures=[e['env_id'] for e in now if not e['violation'] and base[(delay,e['env_id'])]['violation']]))
    results['pilot_failure_comparison']=comparisons
    results['candidate_status']='experimental_not_promoted'
    (OUT/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    with (OUT/'paired.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]),lineterminator='\n');writer.writeheader();writer.writerows(paired)
    lines=['# 随机初态与执行端延迟评测','',
        '**最终候选尚未通过安全回归，未推广。** 严格100ms FIFO延迟仍有违规，原初态漫游新增桌面违规；旧bounded_guard保持冻结参考。CPU回归93项通过只证明被断言的软件行为，不替代这些仿真负结果。','',
        '所有配对初态SHA、源命令SHA和初态违规标记逐窗口完全一致；延迟队列按实测发送目标验证FIFO序列。初态碰撞不拒绝，避碰计数只取初态无违规窗口。','',
        '|组/方法|初态±rad|延迟ms|保持/时长s|初态违规/全部|无违规初态后的违规|最小臂/手球距mm|实际关节路径rad|',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for e in result_rows:
        margin='—' if e['min_arm_hand_mm'] is None else f"{e['min_arm_hand_mm']:.3f}"
        path='—' if e['mean_joint_path_rad'] is None else f"{e['mean_joint_path_rad']:.3f}"
        hold='漫游' if e['hold_s'] is None else f"{e['hold_s']:.2f}"
        lines.append(f"|{e['campaign']} · {e['label']}|{e['jitter_rad']}|{e['delay_ms']:.1f}|{hold}/{e['duration_s']}|{e['initial_violations']}/{e['episodes']}|{e['safe_initial_violations']}/{e['initially_safe']}|{margin}|{path}|")
    lines.extend(['','## 对比审计','',
        '无保护矩阵控制来自stored_verified_matrix四个已完成cell，因新历史仅参与RawShim绕过的安全投影而复用；初态和完整命令SHA一致，不增加控制样本。',
        '原失败重播的新失败/修复环境列表：'+json.dumps(comparisons,ensure_ascii=False),
        '最终候选保护结果（异质相关分组，非独立安全置信证明）：'+json.dumps(results['final_protected_total'],ensure_ascii=False),
        '', '## 范围与限制','']+results['limitations']+['',
        '源码/协议见results.json与protocols，逐窗口配对见paired.csv。计数不与上一轮512窗口混算；重复探针和修复重播不增加独立样本数。',''])
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result_rows,ensure_ascii=False))


if __name__=='__main__': main()
