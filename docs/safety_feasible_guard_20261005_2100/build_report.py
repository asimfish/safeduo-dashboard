"""Derive plots and bounded claims from closed fresh evidence, never planned outcomes."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
def read(n):return json.loads((HERE/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    R=read('holdout_results.json');V=read('visual_verification.json');F=read('first_failure_audit.json')
    B=read('bank_audit.json');T=read('command_audit.json');A=read('ASTRA_FINAL_SCORE.json')
    assert R['completed_method_windows']==768 and R['invalid_method_windows']==0 and len(R['cases'])==192
    assert A['status'].startswith('PASS'),A['status']
    modes=['admission_full','admission_scaled_036','joint_reference','joint_repair']
    names=['Full admission','0.36 admission','Joint reference','Joint repair']
    chinese=['全强度预测准入','0.36倍强度预测准入','联合参考','联合重求解']
    totals={t['mode']:t for t in R['totals']};coverage={t['mode']:t['measured'] for t in R['coverage']}
    pathratio=coverage[modes[1]]['mean_joint_path_rad']/coverage[modes[3]]['mean_joint_path_rad']
    rangeratio=coverage[modes[1]]['mean_within_window_joint_range']/coverage[modes[3]]['mean_within_window_joint_range']
    matched=.9<=pathratio<=1.1 and .9<=rangeratio<=1.1
    comparisons=[]
    for a,b in [('joint_reference','joint_repair'),('admission_full','joint_repair'),('admission_scaled_036','joint_repair')]:
        rows=[x for x in R['paired'] if x.get('a')==a and x.get('b')==b]
        comparisons.append(dict(a=a,b=b,**{k:sum(x[k] for x in rows) for k in ['rescued','new_failures','both','neither']}))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    x=np.arange(4);colors=['#55758f','#995fa2','#bc7418','#177565']
    axs[0,0].bar(x-.17,[totals[m]['violations'] for m in modes],.34,color=colors,label='Strict < 0')
    axs[0,0].bar(x+.17,[totals[m]['deep'] for m in modes],.34,color=colors,alpha=.42,label='Deep < -5 mm')
    axs[0,0].set(xticks=x,xticklabels=names,ylabel='Violating cases / 192',title='Observed failures: all 960 frames');axs[0,0].legend()
    for m,n,c in zip(modes,names,colors):
        axs[0,1].scatter(coverage[m]['mean_joint_path_rad'],totals[m]['violations'],color=c,s=65,label=n)
    axs[0,1].set(xlabel='Mean measured joint path (rad)',ylabel='Strict violating cases / 192',title='Safety and delivered movement');axs[0,1].legend()
    p=np.arange(3)
    axs[1,0].bar(p-.17,[r['rescued'] for r in comparisons],.34,color='#177565',label='Rescued')
    axs[1,0].bar(p+.17,[r['new_failures'] for r in comparisons],.34,color='#b85b43',label='New failures')
    axs[1,0].set(xticks=p,xticklabels=['Reference → repair','Full → repair','0.36 → repair'],ylabel='Paired cases',title='Failures introduced remain in the denominator');axs[1,0].legend()
    exp=np.array([[v['exposed'] for v in R['exposure'] if v['mode']==m] for m in modes])
    axs[1,1].imshow(exp,vmin=0,vmax=24,cmap='Blues',aspect='auto')
    for i in range(4):
        for j in range(6):axs[1,1].text(j,i,str(exp[i,j])+'/24',ha='center',va='center',color='white' if exp[i,j]>=18 else 'black')
    axs[1,1].set(xticks=range(6),xticklabels=['FL–FR','FL–UL','FL–UR','FR–UL','FR–UR','UL–UR'],yticks=x,yticklabels=names,title='Actual assigned-pair exposure (≥3 frames <80 mm)')
    fig.suptitle('SafeDuo: 192 fresh conditioned paired cases; four frozen methods\nSimulation evidence only; no hardware or full-space reliability claim',fontsize=13)
    for ext in ['png','pdf','svg']:fig.savefig(HERE/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    table='\n'.join('|'+ ' | '.join([chinese[i],str(totals[m]['violations'])+'/192',str(totals[m]['deep']),f"{totals[m]['min_nonexempt_mm']:.6f}",'/'.join(map(str,totals[m]['class_violations'])),f"{coverage[m]['mean_joint_path_rad']:.6f}",f"{coverage[m]['mean_within_window_joint_range']*100:.6f}%",f"{totals[m]['four_arms_moving_fraction']*100:.6f}%"])+'|' for i,m in enumerate(modes))
    paired='\n'.join('|'+ ' | '.join([r['a']+' → '+r['b'],*[str(r[k]) for k in ['rescued','new_failures','both','neither']]])+'|' for r in comparisons)
    first_summary=[]
    for m in modes:
        records=[r for r in F['records'] if r['mode']==m]
        compliant=sum(all(z['returned_target_safety_residual_m']<=1e-6 and z['returned_target_alpha_residual_rad']<=1e-6 for z in r['joint_set_checks'].values()) for r in records)
        infeasible=sum(any(not z['hard_set_feasible'] for z in r['joint_set_checks'].values()) for r in records)
        first_summary.append(f'- {m}：{len(records)} 个首次失败状态；返回目标的已保存安全/进展残差均≤原容差的案例 {compliant}；至少一个机器人所保存硬线性集合不可行的案例 {infeasible}。范围残差另由全量审计检查。')
    repair=[]
    for r in R['rows']:
        if r['mode']!='joint_repair':continue
        for robot,z in r['repair'].items():
            repair.append(f'- seed {r["seed"]} / {robot}：触发 {z["applied"]["positive_env_steps"]} 环境帧；安全最小松弛严格为正 {z["safety_min_slack_m"]["positive_env_steps"]} 帧，最大 {z["safety_min_slack_m"]["maximum"]*1000:.9f} mm；预算最小松弛严格为正 {z["alpha_min_slack_rad"]["positive_env_steps"]} 帧，最大 {z["alpha_min_slack_rad"]["maximum"]:.9f} rad/步；QP 回退 {z["qp_fallback"]["positive_env_steps"]} 帧。')
    verdict='减少' if totals['joint_repair']['violations']<totals['joint_reference']['violations'] else '增加' if totals['joint_repair']['violations']>totals['joint_reference']['violations'] else '持平'
    visual='\n'.join(f'- {r["mode"]}：数值 {r["numerical_violations"]}/64，视觉 {r["visual_violations"]}/64；逐位跨运行绑定 {r["binding_status"]}，q 首差帧 {r["q_first_difference"]}，最大差 {r["q_max_difference"]:.9f} rad。' for r in V['runs'])
    text=f'''# SafeDuo：联合求解与强度对照的完整新随机实验

新随机实验全部 768 方法窗口完成，无效 0；192 个不同初态/输入组合各重复四种方法，不能称为 768 个独立样本。候选联合重求解的违规案例相对联合参考{verdict}：{totals['joint_reference']['violations']}/192 → {totals['joint_repair']['violations']}/192。仍观察到真实负几何裕度，因此物理安全结论为 **UNVALIDATED**。开发集事先已有反例：参考 6/64、深违规 3；候选 8/64、深违规 6。未把开发集混入新覆盖。

|方法|严格违规|深违规|最差 mm|cross/F 自碰/U 自碰/桌面|平均路径 rad|平均关节范围/软限位|四臂移动帧占比|
|---|---|---|---|---|---|---|---|
{table}

判定读取全部 960 帧原始非豁免几何裕度：<0 违规，<−0.005 m 深违规；没有另加碰撞容差。四臂移动要求第60帧起每臂实际关节位移 L2 >0.001 rad。路径为实际关节绝对增量总和，关节范围为单窗口范围除软限位宽度的平均；不是任务完成指标。

## 配对效果与运动强度

|对照→候选|救回|新增失败|都失败|都无失败|
|---|---|---|---|---|
{paired}

0.36 强度在新结果前依据上一轮比例固定。实测路径比（0.36/候选）{pathratio:.6f}、关节范围比 {rangeratio:.6f}；{'同时满足预注册的两个±10%条件，限于聚合运动量匹配' if matched else '没有同时满足两个±10%条件，不能称运动强度已匹配'}。即使聚合条件满足，也不证明逐案例速度、风险暴露或任务工作量相等。相同初态、未缩放随机输入和同种子GPU内四方法配对均直接检查；无 IID 置信区间或硬件可靠性估计。

## 随机性与覆盖

三个 OS 随机种子组按 DESIGN.json 登记；192 个政策无关合格初态包括六类风险对各24个及全局随机48个。沿用原几何/静置资格门：所有非豁免机器人及原始桌面裕度≥1 mm，指定风险对20–60 mm，零输入1秒无负裕度，漂移≤0.05 rad，末速≤0.1 rad/s。保存所有候选及失败资格记录，没有依据方法结果筛初态。

枚举 {B['prior_compared']} 个已存旧初态，完全重复 {B['exact_duplicates']}，最近 L2 距离 {B['minimum_l2_distance_rad']:.9f} rad；此结论仅限 bank_audit.json 列明的文件。实际新输入 {T['fresh_unique_tapes']} 段，各60步零输入+900步四臂独立混合保持；与列明的旧输入重复 {T['exact_repeats_with_prior']}。旧输入枚举包含保存的前瞻/中止配方，不能把全部旧配方称作已经执行。实测暴露不足仍保留原分母，边际十格和末端10 cm体素不是26维联合空间覆盖率。未穷尽动力学、物体、延迟和现场扰动。

## 联合求解的能力与边界

候选仅在原返回的安全/预算/范围残差超过1e−6时触发。目标范围保持硬约束；先最小化公共进展预算松弛（rad/步），在该最优量附近再最小化公共安全松弛（m），最后用 OSQP 接近原指令。两个量单位不同且按预算优先，不能宣传为无条件最优安全解。SOLVE_ALLOWANCE=2e−8、退出余量2e−7与严格几何端点分别记录，未被用于抹除正松弛或负距离。失败或未满足退出检查的 QP 使用已检查 LP 点，该点未必是最近指令。

GPT‑6 Astra xhigh独立重建18个选定旧状态：13可行、5不可行；其中3个可行集合的旧返回仍超过原容差；66/env23两条cross行各自可行但联合不可行，具有单独双对偶证据。另一轮独立4案例验证了修复和不可行松弛，并检查R19输出可被修正。六个软件回归用例通过，只证明所覆盖软件路径。

候选 diagnostics 的 original_* 字段在修复路径上是修复后的内层记录，不能当作修复前残差；repair_pre_safety_residual 仅在触发位置有修复前含义。

{chr(10).join(repair)}

## 首次真实失败的直接证据

所有 {F['cases']} 个首次失败状态均保存并绑定当步之前 q/qd、issued target、真实六步pending、selected G/h/J、原h及预算/硬范围和实际输出，再绑定下一次实测的全9021距离/豁免。独立数值LP使用行L2归一化、HiGHS容差1e−9，范围仅限所保存的float32约束转binary64；父审计的不可行判断没有额外双证书，不等同精确符号证明。

{chr(10).join(first_summary)}

Astra另外实际检查6个新全强度首失败状态，其中3个返回的已保存约束残差满足原容差，1个严格为零，却在下一帧出现负几何裕度。已保存的线性J增量方向与实测q运动方向可不同；不能把失败唯一归因于PD、队列或几何非线性，也不能用投影通过推导实际安全。全量预测准入仍基于原始proposal，参考proposal漏行是被动诊断，未宣称保守准入。

## 真相机与状态

实际 {V['groups']} 个九视角组、{V['images']} 张1280×720 PNG，包括42个固定组（六风险层+全局、75/480/959三阶段）及 {V['first_failure_groups']} 个各层首次负裕度当步组。真实USD矩阵、光学参数及near/far读取后重新算六面球体视锥；渲染前后全部64环境原生16数组逐位相同。图像绑定自身视觉运行，遮挡消失、全部网格轮廓和未来动力学非干扰均未认证。

{visual}

视觉重放使用同一个新初态bank/输入，但不是新的独立覆盖。若跨运行逐位比较失败，原始差异和视觉失败数保留，禁止将图片绑定成另一次数值运行的精确失败画面。

Astra还实际逐张查看并独立复算参考env000/75和env038/116两组18张原图：目标球六面余量分别至少0.620594756/0.505422926 m，64环境16原生数组逐位相同。多个图存在臂/工具遮挡、总览末端像素较少和局部桌边裁切，不能用照片判毫米间隙或真实接触。独立复核只覆盖这两闭合组，未在该侧线确认首失败时间顺序，整轮时间顺序由父完整960帧审计另验。SDK pos_w过期零值被单列，实际姿态依据USD直接读回，不将SDK值冒称校准。

## 执行和归档

全部12数值进程的协议、960帧、原始source/config/actor SHA、六步实际applied FIFO和独立投影历史都重新核对；预测全9021行按32帧chunk读取、有限性及完整selection union检查。选定9个固定帧和每个首次失败帧保存selected J；生产者检查全J有限性，但没有逐帧归档全9021 J。完整q/qd/曲线、漏行、正松弛和LP回退均保留。

run_analysis.py编译登记的analyze_v2.py与dense_audit.py字节。完整结果与PASS执行回执在14:34:13 UTC写出并完成posthash，但外层工具回收该进程时返回143；原因未确认，不能称CLI干净退出。analysis_completion_readback.json再次核对已写结果/回执和全部编译源SHA。Astra另从原始dense独立评分，最终闭合raw双读清单再对应父行inputSHA；没有从外层退出码推导数据成功。coverage_metrics.py在最初helper哈希表中遗漏；用上一轮Git e8a9db408c1a6d61ff1314477ef92b56c4e73f70的同字节blob执行，明确属于启动后、全面评分前的依赖绑定，不追称预注册。Astra最终独立评分见 ASTRA_FINAL_SCORE.json，公开报告及图像可以逐文件下载；原始大文件留NAS，闭合双读清单及最终归档另存。

首次相机启动因底层取证器需要显式--methods system0而失败（虽解析器默认相同）；有效窗口0、PNG0。原输出/日志/计划留存，新visual_plan_v3只增加该启动参数和新输出目录，冻结的visual_runner_v3及控制策略未改。

独立评分保留两次失败尝试：第一次整体research哈希表比较错误地拒绝三个不同初态库，修正为共有控制/辅助文件同SHA、各块数据文件分别绑定登记SHA；第二次binary64归约与父登记float32算法在三项描述性指标上超出固定比较线。原binary64 FAIL及具体差异均保留，没有放宽物理阈值或比较rtol。随后从相同raw独立重算原生float32归约，全部8个ratio/range描述量精确复现父值，6272个明确映射字段及所有端点/配对通过；最终结论是PASS_ENDPOINTS_AND_NATIVE_FLOAT32_REDUCTION_BINDING，不能称binary64全字段比较PASS。运动强度不匹配和候选变差的结论不受这些数值差异影响。

真实任务/搬运、实时控制deadline及硬件未在本轮验证；上一轮搬运19/192的原分母不变。候选没有提升到生产或硬件。科学证据/面板交付验收与物理安全判定分开，最终状态见VERIFICATION.json。
'''
    (HERE/'REPORT.md').write_text(text)
    (HERE/'REPRODUCE.md').write_text('''# Reproduce the bounded evidence

Use /home/liyufeng/miniforge3/envs/safeduo/bin/python from /home/liyufeng/safeduo. Set PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=src, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. The actor and source digests are in DESIGN.json and the actual holdout_v2/*_plan.json files. No production or actuator writes are part of replay scoring.

The archived outputs are closed. Do not execute simulation plans into those output paths. To rerun simulation, first create a new namespace, preserve these frozen versions and explicitly register new output paths; execute isolated_campaign_v2.py --plan NEW_PLAN --out NEW_OUTPUT_ROOT separately for the three seed blocks. Camera jobs use visual_plan_v3.json, visual_runner_v3.py and --enable_cameras; execute_visual_v2.py refuses existing output folders and checks actual complete protocols. The v1 holdout plans and earlier visual runners remain preserved as unlaunched superseded versions.

For a read-only audit that is safe after sealing, run `/home/liyufeng/miniforge3/envs/safeduo/bin/python /home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100/recheck_closed.py` from the environment above. It prints all12 strict/deep/class counts, actual appliedFIFO6 checks and the primary rescued/new pair counts, and verifies dense bytes against the archived input SHA; it writes no metadata or raw files. Add `--dense` to execute the exact registered fullforecast helper as well. This may take several minutes. The basic read-only command was actually exercised before sealing; no claim the optional all-row replay was run a second time.

The original run_analysis.py, failure_audit.py, audit_banks.py, audit_commands.py, verify_visual_v3.py and astra_final_score.py produce named receipts under H. They were exercised before sealing and their completed receipts remain downloadable. Do not execute these output-writing CLIs inside the sealed directory; a fresh simulation/scoring campaign needs a new namespace and newly registered path/source bindings. Copying a directory alone does not update those absolute source maps or authorize overwriting sealed evidence.

build_report.py derives the three comparison formats and report from verified JSON; build_panel.py copies actual SHA-bound image/state/native files. check_panel.py --url URL checks all 192 cases, four class curves, three actuation curves, 40 filters, four heatmaps, every actual nine-view group, raw links and 390px layout. Browser dependency location and environment are recorded in ENVIRONMENT.json and browser receipts. verify_remote.py checks the exact Pages commit/build and all owned public bytes, not just HTTP availability.

Strict safety endpoints remain original nonexempt geometry <0; deep violations <-0.005 m. LP slack and projection tolerances never replace these endpoints. Rendering comparisons only establish instantaneous native equality and each image's own state; read the separate exact forward replay verdict before cross-run use.
''')
    (HERE/'report_build.json').write_text(json.dumps(dict(status='PASS_DERIVED_REPORT',source_sha256=sha(Path(__file__)),result_sha256=sha(HERE/'holdout_results.json'),visual_sha256=sha(HERE/'visual_verification.json'),path_ratio=pathratio,range_ratio=rangeratio,aggregate_movement_matched=matched,paired=comparisons),indent=2)+'\n')
    print('REPORT_READY',verdict,pathratio,rangeratio,flush=True)
if __name__=='__main__':main()
