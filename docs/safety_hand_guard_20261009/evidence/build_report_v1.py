"""Closed mechanism evidence; keep all earlier public reports byte-identical."""
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parent
H = P.parent
FIX = H/'fixedhand_triplet64_v1'
W = Path('/home/liyufeng/safeduo-dashboard-response-probe-20261009')
SLUG = 'docs/safety_hand_guard_20261009'
D = W/SLUG
METHODS = ('raw','multirow','multirow_hold_fallback')
LABELS = ('Raw random','Multirow proposal','Model-invalid initial hold')
COLORS = ('#bc5968','#27856d','#466bc1')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (P/'PUBLIC_PAYLOAD_MANIFEST.json').exists(), 'Delivery build already attempted'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
    base = subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    result = read(FIX/'FIXEDHAND_TRIPLET64_RESULT_V1.json')
    assert result['all3_native_actual_exit0'] and result['all3_full960_independent_oracles_pass']
    assert read(FIX/'FINISH_EXECUTION_V1.json')['status']=='complete'
    assert read(P/'ISSUE_BRANCH_METADATA_CLARIFICATION_V1.json')['status']=='PASS_ALL480_NATIVE_ISSUE_BRANCHES_EXACT'
    margin = read(H/'HAND_MARGIN64_RESULT_V1.json')
    hold = read(H/'HOLD_FALLBACK64_RESULT_V1.json')
    reg = read(H/'FIXEDHAND_TRIPLET_DEV_REG_V1.json')
    older = read(H/'OLDER_PUBLIC_PAGE_PRESERVATION_V1.json')['files']
    previous = read(H/'PUBLIC_PAYLOAD_MANIFEST.json')['files']
    preserved = {**older, **{k:v for k,v in previous.items() if k!='index.html'}}
    for name, digest in preserved.items():
        assert sha(W/name)==digest, 'Earlier delivery altered '+name
    (P/'OLDER_EVIDENCE_PRESERVATION_V1.json').write_text(json.dumps(dict(files=preserved),indent=2)+'\n')
    D.mkdir(parents=True,exist_ok=False)
    links = []
    def copy(source,name,label=None):
        dest = D/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        if label:
            links.append(f'<li><a href="{html.escape(name)}">{html.escape(label)}</a></li>')
    for name,label in [
        ('HOLD_FALLBACK64_RESULT_V1.json','旧物理设置：64 初态回退结果'),
        ('HOLD_FALLBACK_DEV_REG_V1.json','旧回退预登记'),
        ('HOLD_FALLBACK_ANALYTIC_ORACLE_V1.json','回退分支解析检查'),
        ('HOLD_FALLBACK_FINISH_EXECUTION_V1.json','旧回退实际退出记录'),
        ('HAND_MARGIN64_RESULT_V1.json','4 种手部余量全部 64 初态结果'),
        ('HAND_MARGIN_DIAGNOSTIC_REG_V1.json','手部余量预登记'),
        ('hand_margin64_v1_execution.json','4 组手部实验实际退出记录'),
        ('FIXEDHAND_TRIPLET_DEV_REG_V1.json','新 8 秒三组验证预登记'),
        ('fixedhand_triplet64_v1_execution.json','3 组原生实际退出与资源记录'),
        ('FIXEDHAND_TRIPLET_PLAN_V1.json','509 文件冻结执行计划'),
        ('native_fixedhand_triplet_v1.py','新原生三组运行代码'),
        ('analyse_fixedhand_triplet_v1.py','新三组匹配与余量复算代码'),
        ('analyse_paired128_v1.py','独立原始接触、几何、关节与 FIFO 复算代码'),
        ('native_hand_margin_v1.py','手部余量原生运行代码'),
        ('analyse_hand_margin_v1.py','手部余量独立复算代码'),
        ('native_hold_fallback_v1.py','旧回退原生运行代码'),
        ('analyse_hold_fallback_v1.py','旧回退独立复算代码'),
        ('issue_or_initial_hold_v1.py','原子四臂回退分支代码'),
        ('test_issue_or_initial_hold_v1.py','回退解析检查代码')]:
        copy(H/name,'evidence/'+name,label)
    for name,label in [('FIXEDHAND_TRIPLET64_RESULT_V1.json','新三组全部逐初态结果'),('FINISH_EXECUTION_V1.json','新三组独立复算实际退出')]:
        copy(FIX/name,'evidence/'+name,label)
    for name,label in [('ISSUE_BRANCH_METADATA_CLARIFICATION_V1.json','全部 480 次指令分支核对与字段澄清'),('verify_disposition_v1.py','独立指令分支核对代码'),('localize_joint_v1.py','新三组逐关节事后定位代码'),('build_report_v1.py','报告与图表生成代码')]:
        copy(P/name,'evidence/'+name,label)
    for name,label in [('REGISTRATION_V1.json','后续强随机输入预登记（尚未原生测试）'),('INPUT_PREPARATION_ORACLE_V1.json','后续强随机输入范围检查（不是安全结果）'),('prepare_recipe_v1.py','后续强随机输入生成代码')]:
        copy(H/'strong_random_preparation_v1'/name,'future/'+name,label)
    for job in read(H/'HAND_MARGIN_PLAN_V1.json')['jobs']:
        leaf=Path(job['out']);dest='hand_prefix/'+job['id']+'/'
        for name,label in [('prefix_inputs.npz','全部64初态与参数'),('prefix_result.npz','全部12步9021几何与联合指标'),('point_contact_receipts.json','原始接触块哈希')]:
            copy(leaf/name,dest+name,job['id']+'：'+label)
        for name in ['prefix_protocol.json','actual_solver_roots.json','native_contact_identity.json']:
            copy(leaf/name,dest+name)
        for chunk in read(leaf/'point_contact_receipts.json')['chunks']:
            assert sha(leaf/chunk['path'])==chunk['sha256']
            copy(leaf/chunk['path'],dest+chunk['path'],job['id']+'：12步原始接触点与74关节状态')
    metrics = {}
    physics_dt = None
    for method in METHODS:
        copy(P/f'JOINT_LOCALIZATION_{method}_V1.json',f'evidence/JOINT_LOCALIZATION_{method}_V1.json',method+'：逐关节越界与速度事后定位')
        copy(FIX/f'PAIRED_ORACLE_batch0_{method}_V1.json',f'oracles/{method}.json',method+'：全部 960 步独立复算')
        metric = FIX/f'PAIRED_METRICS_batch0_{method}_V1.npz'
        copy(metric,f'metrics/{method}.npz',method+'：全部 960 步指标')
        with np.load(metric) as z:
            metrics[method]={k:z[k] for k in ['global_input_id','gap_m','force_N','hard_violation_rad','velocity_exceedance_rad_s']}
        leaf = Path(reg['native_namespace'])/f'paired_batch0_{method}_v8'
        receipt=read(leaf/'point_contact_receipts.json')
        with np.load(leaf/receipt['chunks'][0]['path']) as z:
            values=np.unique(z['physics_dt'])
            assert len(values)==1 and values[0]>0
            if physics_dt is None:physics_dt=float(values[0])
            assert float(values[0])==physics_dt
        for name in ['response_protocol.json','actual_solver_roots.json','resolved_native_parameters.npz','point_contact_receipts.json','all_raw_geometry_receipts.json']:
            copy(leaf/name,f'protocols/{method}_{name}')
    # Exportable conventional figures; normalize only SVG whitespace before delivery.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'safeduo-hand-guard-v1'})
    svg_receipts=[]
    def normalized_svg(text):
        root=ET.fromstring(text)
        for node in root.iter():
            for key,value in node.attrib.items():node.set(key,re.sub(r'\s+',' ',value.strip()))
            if node.text is not None:node.text=re.sub(r'\s+',' ',node.text.strip())
            if node.tail is not None:node.tail=re.sub(r'\s+',' ',node.tail.strip())
        return ET.tostring(root)
    def figure(fig,stem):
        for ext in ['png','svg','pdf']:
            path=D/f'{stem}.{ext}'
            opts={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
            fig.savefig(path,dpi=165,**opts)
            if ext=='svg':
                before=path.read_text()
                after='\n'.join(line.rstrip() for line in before.splitlines())+'\n'
                assert normalized_svg(before)==normalized_svg(after)
                path.write_text(after)
                svg_receipts.append(dict(path=f'{stem}.svg',xml_semantics_preserved=True))
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    xs=np.arange(3)
    for ax,data,title in [(axes[0],hold['counts'],'Old physics / hard-boundary hands'),(axes[1],result['counts'],'New physics / 0.03 rad hand margin')]:
        for offset,key,label,color in [(-.25,'primary_failures','Primary failed','#bc5968'),(0,'all74_hard_bad','Hard limit failed','#cd9442'),(.25,'joint_and_primary_pass','Joint + primary passed','#27856d')]:
            bars=ax.bar(xs+offset,[data[m][key] for m in METHODS],.24,label=label,color=color)
            ax.bar_label(bars,padding=2,fontsize=9)
        ax.set_xticks(xs,['Raw','Multirow','Initial hold']);ax.set_ylim(0,70);ax.set_ylabel('Known development states, N=64');ax.set_title(title)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=9);figure(fig,'outcomes')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    ms=[r['hand_margin_rad'] for r in margin['rows']]
    for key,label,color in [('all74_hard_bad','Hard limit failed','#bc5968'),('primary_admitted','Primary admitted','#4f6d92'),('full_prefix_admitted','Joint + primary admitted','#27856d')]:
        axes[0].plot(ms,[r[key] for r in margin['rows']],'-o',label=label,color=color)
    axes[0].set_ylim(0,68);axes[0].set_xlabel('Hand initial + held-target margin (rad)');axes[0].set_ylabel('States, N=64');axes[0].set_title('Only 6 held controls / 12 physics steps');axes[0].legend(fontsize=8)
    newrows=result['states']
    for method,label,color in zip(METHODS,LABELS,COLORS):
        ratio=np.array([s['controlled_path_rad']/r['controlled_path_rad'] for s,r in zip(newrows[method],newrows['raw'])])
        axes[1].plot(np.arange(1,65),np.sort(ratio),label=label,color=color)
    axes[1].set_yscale('symlog',linthresh=.0001);axes[1].set_xlabel('Sorted development states (each method separately)');axes[1].set_ylabel('Path / corresponding new raw path');axes[1].set_title('Motion suppression is not task completion');axes[1].legend(fontsize=8)
    figure(fig,'margin_activity')
    oldreg=read(H/'PAIRED_REGISTRATION_V1.json')
    cases=[]
    for lane in oldreg['camera']['slots']:
        row=result['states']['raw'][lane]
        input_id=row['input_id']
        fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True,layout='constrained')
        for method,label,color in zip(METHODS,LABELS,COLORS):
            x=metrics[method];idx=int(np.flatnonzero(x['global_input_id']==input_id)[0])
            for ax,key,factor in zip(axes,['gap_m','force_N','hard_violation_rad','velocity_exceedance_rad_s'],[1000,1,1,1]):
                ax.plot((np.arange(960)+1)*physics_dt,factor*x[key][:,idx],label=label,color=color,lw=1.1)
        for ax in axes:
            ax.axvline(.1,color='#778da0',ls='--');ax.grid(alpha=.2)
        for ax,key in zip(axes,['Minimum raw gap (mm)','Peak owner normal scalar (N)','Full74 hard breach (rad)','Full74 speed exceedance (rad/s)']):ax.set_ylabel(key)
        axes[0].axhline(0,color='#777',ls=':');axes[1].axhline(.1,color='#777',ls=':');axes[1].set_yscale('symlog',linthresh=.1)
        for ax in axes[2:]:ax.axhline(1e-5,color='#777',ls=':');ax.set_yscale('symlog',linthresh=1e-5)
        for ax,key,floor in [(axes[1],'force_N',.2),(axes[2],'hard_violation_rad',2e-5),(axes[3],'velocity_exceedance_rad_s',2e-5)]:
            ax.set_ylim(0,max(floor,1.15*max(float(metrics[m][key][:,lane].max()) for m in METHODS)))
        axes[0].set_title(f'Input {input_id}, registered cell {row["cell"]}; all 960 recorded physics steps');axes[0].legend(fontsize=8);axes[3].set_xlabel('Native time after reset (s)')
        figure(fig,f'input{input_id:03d}_curves')
        note=f'输入 {input_id} · 组合 {row["cell"]}。'
        for method,label in zip(METHODS,['无保护','多行提案','回退']):
            value=result['states'][method][lane]
            note+=f'{label}：最小间隙 {1000*value["minimum_raw_gap_m"]:.3f} mm，法向峰值 {value["peak_all_arm_scalar_N"]:.3f} N，硬限位越界 {value["supplementary_max_all74_hard_violation_rad"]:.6g} rad；'
        note+=f'回退／无保护路径 {100*result["states"]["multirow_hold_fallback"][lane]["controlled_path_rad"]/row["controlled_path_rad"]:.3f}%，动作抑制不等于任务成功。'
        cases.append(dict(input_id=input_id,cell=row['cell'],plot=f'input{input_id:03d}_curves.png',note=note))
    (P/'SVG_EXPORT_RECEIPT_V1.json').write_text(json.dumps(dict(status='PASS_SVG_WHITESPACE_ONLY_NORMALIZATION',files=svg_receipts),indent=2)+'\n')
    def table(counts,with_prefix=False):
        header='<tr><th>方法</th><th>初态</th>'+('<th>实际等待段主判据合格</th>' if with_prefix else '')+'<th>主判据失败</th><th>几何失败</th><th>接触失败</th><th>74关节越界</th><th>74关节速度超限</th><th>联合通过</th><th>四臂均有运动</th></tr>'
        rows=''
        for m,label in zip(METHODS,['无保护随机','多行提案','模型无效则回到初始目标']):
            c=counts[m];keys=['states']+(['replay_prefix_qualified'] if with_prefix else [])+['primary_failures','geometry_failures','force_failures','all74_hard_bad','all74_speed_bad','joint_and_primary_pass','all4_arms_moved']
            rows+='<tr><th>'+label+'</th>'+''.join(f'<td>{c[k]}</td>' for k in keys)+'</tr>'
        return '<div class="scroll"><table><thead>'+header+'</thead><tbody>'+rows+'</tbody></table></div>'
    margin_table='<div class="scroll"><table><thead><tr><th>余量 rad</th><th>原始初态</th><th>几何／接触准入</th><th>关节越界</th><th>速度超限</th><th>等待段联合准入</th><th>最大越界 rad</th></tr></thead><tbody>'
    for r in margin['rows']:
        margin_table+='<tr>'+''.join(f'<td>{r[k]:.6g}</td>' for k in ['hand_margin_rad','source_states','primary_admitted','all74_hard_bad','all74_speed_bad','full_prefix_admitted','maximum_hard_breach_rad'])+'</tr>'
    margin_table+='</tbody></table></div>'
    extrema='<div class="scroll"><table><thead><tr><th>新设置方法</th><th>最小原始间隙 mm</th><th>峰值接触法向标量 N</th><th>最大硬限位越界 rad</th><th>最大速度超限 rad/s</th></tr></thead><tbody>'
    for method,label in zip(METHODS,['无保护随机','多行提案','模型无效则回到初始目标']):
        c=result['counts'][method]
        extrema+=f'<tr><th>{label}</th><td>{1000*min(r["minimum_raw_gap_m"] for r in result["states"][method]):.3f}</td><td>{c["peak_normal_N"]:.3f}</td><td>{c["max_hard_breach_rad"]:.6g}</td><td>{c["max_speed_exceedance_rad_s"]:.6g}</td></tr>'
    extrema+='</tbody></table></div>'
    csvpath=D/'all64_results.csv'
    perrows=[]
    for lane in range(64):
        r=newrows['raw'][lane];item=dict(input_id=r['input_id'],cell=r['cell'])
        for method in METHODS:
            v=newrows[method][lane]
            item.update({method+'_'+k:v[k] for k in ['replay_prefix_qualified','primary_failure','minimum_raw_gap_m','peak_all_arm_scalar_N','supplementary_max_all74_hard_violation_rad','supplementary_max_all74_velocity_exceedance_rad_s','controlled_path_rad','four_arms_moved']})
            item[method+'_joint_and_primary_pass']=not v['primary_failure'] and v['supplementary_max_all74_hard_violation_rad']<=1e-5 and v['supplementary_max_all74_velocity_exceedance_rad_s']<=1e-5
        perrows.append(item)
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(perrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(perrows)
    pertable='<div class="scroll"><table><thead><tr><th>输入／组合</th><th>无保护联合通过</th><th>多行联合通过</th><th>回退联合通过</th><th>回退／无保护路径</th></tr></thead><tbody>'
    for item in perrows:
        pertable+=f'<tr><th>{item["input_id"]}／{item["cell"]}</th>'+''.join('<td>'+('通过' if item[m+'_joint_and_primary_pass'] else '失败')+'</td>' for m in METHODS)+f'<td>{100*item["multirow_hold_fallback_controlled_path_rad"]/item["raw_controlled_path_rad"]:.3f}%</td></tr>'
    pertable+='</tbody></table></div>'
    newhold=result['counts']['multirow_hold_fallback'];ratio=result['hold_path_ratio_to_new_raw']['median']
    initial=cases[-1]
    options=''.join(f'<option value="{c["input_id"]}"'+(' selected' if c==initial else '')+f'>输入 {c["input_id"]} · 组合 {c["cell"]}</option>' for c in cases)
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SafeDuo · 手部修复与回退机制验证</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f5f8fb;color:#193449;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1160px;margin:auto;padding:26px 20px 70px}}h1{{font-size:clamp(26px,4vw,40px);line-height:1.25}}h2{{font-size:24px;margin-top:40px}}a{{color:#2159a3}}.tag{{font-size:13px;color:#4e677b}}.notice{{padding:18px;border-left:5px solid #b35a45;background:#fff0e8;border-radius:8px}}.card{{background:white;padding:20px;border-radius:12px;margin:18px 0;border:1px solid #dce5ec}}.scroll{{overflow-x:auto;width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}}th,td{{padding:10px;border-bottom:1px solid #dce5ec;text-align:left}}thead{{background:#eef3f8}}img{{max-width:100%;height:auto;display:block}}select{{max-width:100%;padding:9px;font:inherit}}summary{{cursor:pointer;font-weight:600}}li{{overflow-wrap:anywhere}}.downloads{{font-size:14px}}@media(max-width:500px){{main{{padding:18px 14px}}.card{{padding:14px}}}}
    </style></head><body><main><nav><a href="../../#scientific">← 主面板</a> · <a href="../safety_physical_random_128_20261009/">上一轮 128 初态与四视角原生视频</a></nav><p class="tag">2026-10-09 · 原生物理 · 已知开发集 · 64 初态全部保留</p><h1>手部限位修复后，回退机制真的安全了吗</h1>
    <div class="notice"><b>完整 System 0 安全验收未通过。</b><p>新设置下，回退策略的采样联合通过为 {newhold['joint_and_primary_pass']}/64，主判据失败 {newhold['primary_failures']}/64，关节越界 {newhold['all74_hard_bad']}/64。回退／无保护路径长度中位数为 {100*ratio:.3f}%。这些结果只适用于本页固定开发输入与 960 个记录物理步；减少运动不能等同于完成四臂任务。</p></div>
    <h2>1. 完整 8 秒：同初态、同随机目标、同物理设置</h2><p>三种方法均使用 64 次位置求解迭代、0 次速度迭代、0.03 rad 手部余量，运行 480 个控制步／960 个物理步。26 个机械臂关节的初始位置、完整初始速度、根位姿和整条随机目标带相同；全部 74 个关节在前 6 个不可撤销等待步的位置、速度和实际目标逐值一致。每个初态全部计分，未按新结果替换或剔除。</p>{table(result['counts'],True)}
    <p>联合通过要求全部记录步同时满足：9,021 对原始几何间隙 ≥0 m、82 个碰撞刚体接触法向标量 ≤0.1 N、74 关节硬限位越界 ≤10⁻⁵ rad、速度超限 ≤10⁻⁵ rad/s。计数包含等待段。点接触由原始点法向分量独立累加；FIFO、实际求解根参数、手部余量及全部指令分支均核对。</p>
    <figure class="card"><img src="outcomes.png" alt="旧设置和新设置各自的64初态三方法失败与联合通过计数"><figcaption>左右分别是在各自物理设置内配对。跨设置改善同时含求解器与手部初态／目标变化，不能归因于控制器单独改善。</figcaption><p class="downloads"><a href="outcomes.svg">SVG</a> · <a href="outcomes.pdf">PDF</a></p></figure>
    <h2>2. 手部余量：短等待段的独立因素对照</h2><p>此处为原始几何库前 64 个初态，包含物理准入失败项，与完整随机验证的 64 个初态不同。固定 64 次求解迭代，仅同时改变手部初始位置和保持目标到硬限位的余量。机械臂初态、速度、根位姿、增益与阈值保持相同；每组只测前 6 个保持控制步／12 个物理步。零余量基线精确复现；四组全部 64 个初态保留。</p>{margin_table}<p>0.03 rad 是本次所测最小的、能让 30 个几何／接触准入初态全部同时满足关节判据的余量。剩余 3 个关节越界初态属于准入失败项，仍计入分母。该结果本身不证明 8 秒安全、最优余量或生产参数可用。</p>
    <figure class="card"><img src="margin_activity.png" alt="4种余量等待段对照与新三方法逐初态路径比例"><figcaption>右图每种方法分别排序，纵轴为对应初态路径除以新无保护路径；路径下降需要与任务完成一起评估。</figcaption><p class="downloads"><a href="margin_activity.svg">SVG</a> · <a href="margin_activity.pdf">PDF</a></p></figure>
    <h2>3. 旧设置回退：消除接触仍未通过关节验收</h2><p>先前相同随机输入的首批 64 个初态，使用原始 8 次位置求解和硬限位边界手部目标。模型不满足约束时，对该初态四臂 26 个关节一起发回初始目标；6 步 FIFO 保留。旧回退全部 30,720 次初态控制触发回退，主判据失败由 35 降到 0，但关节越界仍为 64/64、联合通过仍为 0/64；路径中位数仅原始随机的 {100*hold['trajectory_path_ratio_to_raw']['median']:.3f}%。此处反映动作抑制，并非协同任务成功。</p>{table(hold['counts'])}
    <h2>4. 真实轨迹指标：预先固定案例</h2><p>沿用旧实验在结果产生前登记的批内槽位 0、23、42、61，覆盖组合 0、5、10、15。三方法曲线展示全部 960 个原生物理步；虚线 0.1 s 表示新发目标最早能越过原有 FIFO。新诊断关闭相机，本页未生成新视频；<a href="../safety_physical_random_128_20261009/">旧设置四视角录像</a>的参数和实验范围保持原标注。</p><div class="card"><label for="case">选择预登记案例 </label><select id="case">{options}</select><p id="case-note">输入 {initial['input_id']} · 组合 {initial['cell']} · 新设置三方法</p><img id="case-plot" src="{initial['plot']}" alt="所选初态三方法全部960物理步的间隙、力、硬限位和速度曲线"><p class="downloads"><a id="case-svg" href="{initial['plot'].replace('.png','.svg')}">SVG</a> · <a id="case-pdf" href="{initial['plot'].replace('.png','.pdf')}">PDF</a></p><noscript>默认曲线与全部结果可直接查看；切换案例需要 JavaScript。</noscript></div>
    <details><summary>展开全部 64 个初态联合判据与路径比例</summary><p><a href="all64_results.csv">下载全部指标 CSV</a>，原始数值与失败分项见独立复算 JSON／NPZ。</p>{pertable}</details>
    <h2>5. 范围与尚未验收的部分</h2><p>本页是已知开发集上的机制定位与回归验证，不是独立盲测。手部余量与求解设置选自已知开发诊断；随机范围继承前轮：26 关节全软限位初态提案，速度最高 50% 限值，局部目标最大 ±0.35 rad。固定根位姿与手部保持目标，没有物体抓取、承载、故障、训练 actor、实际任务成功或真实机器人验证。四臂均有运动只表示测到运动，未证明协同任务完成；未声明连续时间安全、全操作空间覆盖或正式 System 0 验收。</p>
    <details open><summary>可复核证据与代码</summary><ul>{''.join(links)}</ul><p>完整随机输入与原始覆盖账本沿用<a href="../safety_physical_random_128_20261009/">前轮报告</a>。新三组全部原始接触点／几何块保留在实验存储，公开提供逐初态结果、960 步指标、原始块哈希及代码。</p></details></main><script>
    const cases={json.dumps(cases)};const selector=document.querySelector('#case');selector.addEventListener('change',()=>{{const c=cases.find(x=>String(x.input_id)===selector.value);document.querySelector('#case-plot').src=c.plot;document.querySelector('#case-note').textContent=`输入 ${{c.input_id}} · 组合 ${{c.cell}} · 新设置三方法`;document.querySelector('#case-svg').href=c.plot.replace('.png','.svg');document.querySelector('#case-pdf').href=c.plot.replace('.png','.pdf')}});
    </script></body></html>'''
    page=page.replace('<h2>2. 手部余量：短等待段的独立因素对照</h2>',f'<div class="card"><b>同一分母中的极值也保留</b>{extrema}<p>失败次数减少不代表每一种风险都降低；多行提案的接触峰值仍需单独查看。模型无效回退触发 {result["fallback_lane_controls"]:,}/{result["total_lane_controls"]:,} 次初态控制；它的通过计数应与动作抑制一起解释。</p></div><h2>2. 手部余量：短等待段的独立因素对照</h2>',1)
    page=page.replace('<details open><summary>可复核证据与代码</summary>','<h2>6. 后续强随机输入已固定，尚未原生测试</h2><p>已生成 32,000 个新全软限位姿态提案，并固定 16 个速度／指令组合：初始速度比例 0、50%、75%、100%；目标包括局部 ±0.35 rad 和全软限位独立均匀采样，刷新间隔 1、8、32 个控制步。指令与速度全部提前生成、固定种子并记录哈希。原生测试完成数为 0；这部分不是已测安全证据，也没有扩大本轮 64 个初态的统计分母。</p><details open><summary>可复核证据与代码</summary>',1)
    page=page.replace(f'输入 {initial["input_id"]} · 组合 {initial["cell"]} · 新设置三方法</p>',html.escape(initial['note'])+'</p>',1)
    page=page.replace('document.querySelector(\'#case-note\').textContent=`输入 ${c.input_id} · 组合 ${c.cell} · 新设置三方法`',"document.querySelector('#case-note').textContent=c.note",1)
    (D/'index.html').write_text(page)
    mainpath=W/'index.html';mainpage=mainpath.read_text()
    assert 'id="handGuardEvidence"' not in mainpage
    anchor='<div class="sci-note" id="physicalRandom128Evidence">'
    assert mainpage.count(anchor)==1
    card=f'<div class="sci-note" id="handGuardEvidence"><b>10-09 最新：手部限位修复与四臂回退机制验证</b><p>4种余量独立对照＋相同64初态完整8秒三方法。新回退联合通过 {newhold["joint_and_primary_pass"]}/64、主判据失败 {newhold["primary_failures"]}/64、关节越界 {newhold["all74_hard_bad"]}/64；路径中位数为随机基线 {100*ratio:.3f}%。逐物理步曲线、全部64结果与代码可复核；动作抑制不等于任务成功，完整 System 0 未验收。</p><a href="{SLUG}/">打开最新机制实验与完整验收结果 →</a></div>\n'
    mainpath.write_text(mainpage.replace(anchor,card+anchor,1))
    files={str(p.relative_to(W)):sha(p) for p in sorted(D.rglob('*')) if p.is_file()}
    files['index.html']=sha(mainpath)
    manifest=dict(base_commit=base,slug=SLUG,total_files=len(files),total_bytes=sum((W/n).stat().st_size for n in files),files=files,older_evidence_preserved=len(preserved),full_system0_accepted=False)
    (P/'PUBLIC_PAYLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (P/'PUBLIC_CASE_BINDINGS.json').write_text(json.dumps(cases,indent=2)+'\n')
    print('BUILT_CLOSED_MECHANISM_REPORT',len(files),manifest['total_bytes'],flush=True)


if __name__=='__main__':
    main()
