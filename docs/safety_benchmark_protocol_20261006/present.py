"""Export scientific figures, complete pair tables and a reproducible evidence panel."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage import PAIRS,ARM_OF,ARMS
HERE=Path(__file__).resolve().parent
NAMES={'admission_full':'Full admission','admission_scaled_036':'Admission x0.36','joint_reference':'Joint reference','joint_repair':'Joint repair'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,d,name):
    for ext in ('png','pdf','svg'):
        target=d/(name+'.'+ext);fig.savefig(target,dpi=180,bbox_inches='tight')
        if ext=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);args=parser.parse_args()
    d=args.repo/'docs'/HERE.name;d.mkdir(parents=True,exist_ok=True)
    figs=d/'figures';figs.mkdir(exist_ok=True)
    a=json.loads((HERE/'coverage_audit.json').read_text());s=json.loads((HERE/'initial_coverage.json').read_text())
    matrix=json.loads((HERE/'BENCHMARK_MATRIX.json').read_text());inv=json.loads((HERE/'EVIDENCE_INVENTORY.json').read_text())
    assert a['status']=='complete' and s['status']=='complete' and len(matrix['rows'])==20
    protocol_hash=sha(HERE/'PROTOCOL.md');assert protocol_hash==matrix['protocol_sha256']
    supplement={g['mode']:g for g in s['groups']};groups=[]
    pair_rows=[];case_rows=[]
    labels=[f'{ARMS[ARM_OF[j]]}:J{j-sum([7,7,6,6][:ARM_OF[j]])+1}' for j in range(26)]
    for g in a['groups']:
        with np.load(HERE/'histograms'/(g['mode']+'.npz'),allow_pickle=False) as z:
            assert sha(HERE/'histograms'/(g['mode']+'.npz'))==g['histogram_sha256']
            full=z['full_pairs'].tolist();prefix=z['prefix_pairs'].tolist()
        initial=supplement[g['mode']];groups.append({**g,'initial_mean_pair_percent':initial['initial_pairs']['mean_percent'],
            'full_added_pair_percent':initial['mean_full_added_pair_cells'],'prefix_added_pair_percent':initial['mean_prefix_added_pair_cells'],
            'initial_counts':initial['initial_pairs_counts'],'full_counts':full,'prefix_counts':prefix})
        for i,(x,y) in enumerate(PAIRS):
            pair_rows.append(dict(mode=g['mode'],pair_index=i,joint_a=labels[x],joint_b=labels[y],cross_arm=bool(ARM_OF[x]!=ARM_OF[y]),
                initial_percent=initial['initial_pairs']['visited_limit_rectangle_cells'][i],
                full_percent=g['full_pairs']['visited_limit_rectangle_cells'][i],
                prefix_percent=g['nonnegative_prefix_pairs']['visited_limit_rectangle_cells'][i],
                added_full_percent=initial['full_added_pair_cells'][i],added_prefix_percent=initial['nonnegative_prefix_added_pair_cells'][i]))
    for r in a['rows']:
        for e in range(64):
            case_rows.append(dict(mode=r['mode'],command_seed=r['seed'],environment=e,
                full_mean_joint_span_percent=r['normalized_span_percent'][e],prefix_mean_joint_span_percent=r['nonnegative_prefix_span_percent'][e],
                full_joint_l1_path_rad=r['joint_l1_path_rad'][e],prefix_joint_l1_path_rad=r['nonnegative_prefix_joint_l1_path_rad'][e]))
    for name,rows in [('all_joint_pairs.csv',pair_rows),('all_window_motion.csv',case_rows)]:
        with (d/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    # Descriptive figures: no uncertainty bands for this bounded, dependent sample.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    marginal=np.array([g['full_marginal_cells'] for g in groups]).T*10
    im=axes[0].imshow(marginal,aspect='auto',vmin=0,vmax=100,cmap='viridis')
    axes[0].set_yticks(range(26),labels);axes[0].set_xticks(range(4),[NAMES[g['mode']] for g in groups],rotation=30,ha='right')
    axes[0].set_title('Marginal occupancy (pooled192 starts per method)')
    for g in groups:
        axes[1].plot(np.sort(g['nonnegative_prefix_pairs']['visited_limit_rectangle_cells']),label=NAMES[g['mode']])
    axes[1].set(xlabel='Joint pairs sorted within each method (325)',ylabel='Visited10x10 cells (%)',ylim=(0,102),title='Occupancy before first negative sphere gap')
    axes[1].legend(fontsize=8);fig.colorbar(im,ax=axes[0],label='Visited10 marginal cells (%)',shrink=.7)
    fig.suptitle('Soft-limit grid proxies; no certified reachable-area or26D coverage claim')
    save(fig,figs,'coverage_grid')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for g in groups:
        rows=[r for r in a['rows'] if r['mode']==g['mode']]
        for axis,key in zip(axes,['normalized_span_percent','nonnegative_prefix_span_percent']):
            values=np.sort(np.array([r[key] for r in rows]).reshape(-1));axis.plot(values,np.arange(1,193)/192,label=NAMES[g['mode']])
            axis.set(xlabel='Mean26-joint normalized range per window (%)',ylabel='Empirical CDF',xlim=(0,100),ylim=(0,1))
    axes[0].set_title('Complete16s windows');axes[1].set_title('Before first negative sphere gap');axes[1].legend(fontsize=8)
    fig.suptitle('192 paired windows per method; descriptive distribution, no IID confidence bound')
    save(fig,figs,'motion_ranges')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained');x=np.arange(4);names=[NAMES[g['mode']] for g in groups]
    axes[0].bar(x-.18,[g['violations'] for g in groups],width=.36,label='Any signed sphere gap<0')
    axes[0].bar(x+.18,[g['deep'] for g in groups],width=.36,label='Any signed sphere gap<-5mm')
    axes[0].set(ylabel='Affected windows /192',title='Geometric events (not native contact)');axes[0].legend(fontsize=8)
    axes[1].bar(x-.18,[g['mean_normalized_span_percent'] for g in groups],width=.36,label='Complete window')
    axes[1].bar(x+.18,[g['mean_nonnegative_prefix_span_percent'] for g in groups],width=.36,label='Before first negative gap')
    axes[1].set(ylabel='Mean normalized joint span (%)',title='Motion retained');axes[1].legend(fontsize=8)
    for axis in axes:axis.set_xticks(x,names,rotation=20,ha='right')
    fig.suptitle('Report geometry and mobility together; all failures retained')
    save(fig,figs,'events_and_mobility')
    tasks=json.loads((HERE/'PRIOR_TASK_REVIEW.json').read_text())
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');keys=list(tasks['failed_gates']['beam700']);x=np.arange(len(keys))
    for offset,obj in [(-.18,'beam700'),(.18,'beam300')]:ax.bar(x+offset,[tasks['failed_gates'][obj][k] for k in keys],width=.36,label=obj)
    ax.set_xticks(x,['XY placement','Final tilt','Lift','Final height','Sphere event','Reset'],rotation=15)
    ax.set(ylabel='Failed object gate /192 tasks',title='Historical two-pair object task:19 pass,173 fail; failure categories overlap')
    ax.legend();save(fig,figs,'prior_task_failures')
    data=dict(status='NOT_SUFFICIENT_FOR_COMPREHENSIVE_SAFETY_CLAIM',audited_windows=768,new_physical_windows=0,
        old_initials=192,pairs=PAIRS.tolist(),joint_labels=labels,groups=groups,rows=a['rows'],
        initial_states_bitexact=s['actual_initial_states_bitexact_between_methods'],matrix=matrix,inventory=inv,
        readiness=json.loads((HERE/'READINESS.json').read_text()),prior_task_review=tasks,protocol_sha256=protocol_hash)
    (d/'data.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
    for file in HERE.iterdir():
        if file.is_file() and file.suffix in ('.py','.md','.json','.log') and file.name not in ('delivery.json','PUBLIC_MANIFEST.json','verify_local.log','git_diff_check.log'):
            shutil.copy2(file,d/file.name)
    shutil.copytree(HERE/'histograms',d/'histograms',dirs_exist_ok=True)
    shutil.copy2(HERE/'prior_task_cases.csv',d/'prior_task_cases.csv')
    prior=d/'prior_evidence';prior.mkdir(exist_ok=True)
    for item in inv['source_bindings']:
        source=Path(item['path']);assert sha(source)==item['sha256']
        if source.name in ('terminal_quality_gate.json','delivery_seal.json','protocol.json','analysis.json','full_raw_summary.json','pair_contact_manifest.json','support_contact_manifest.json'):
            shutil.copy2(source,prior/source.name)
    (d/'index.html').write_text((HERE/'panel.html').read_text())
    root=args.repo/'index.html';text=root.read_text();marker='<section class="sci-note" id="benchmarkProtocolEvidence">'
    card='''<section class="sci-note" id="benchmarkProtocolEvidence"><h2>完整实验协议与充分性审计</h2><p>已完成768个封存窗口、26个关节与全部325种关节对的覆盖审计；新协议物理试验尚未执行。六类随机、八类物体任务、六类鲁棒性实验均列入矩阵，正式输入仍待准入与统计功效登记。</p><p>目前仍不充分：旧物体任务19/192通过，接触测量校准未过；几何事件、运动范围和任务成功分开报告。</p><p><a href="docs/safety_benchmark_protocol_20261006/">打开规范协议、完整图表、实验矩阵与证据缺口 →</a></p></section>'''
    if marker in text:
        import re;text=re.sub(r'<section class="sci-note" id="benchmarkProtocolEvidence">.*?</section>',card,text,flags=re.S)
    else:
        anchor='  <section class="sci-note" id="queueEnvelopeEvidence">';assert anchor in text;text=text.replace(anchor,card+'\n'+anchor,1)
    root.write_text(text)
    if (HERE/'NATIVE_CALIBRATION_CLOSURE.json').exists():
        import subprocess,sys;subprocess.run([sys.executable,str(HERE/'append_calibration.py'),'--repo',str(args.repo)],check=True)
    print('PRESENTED',len(pair_rows),'pair rows',len(case_rows),'window rows',len(matrix['rows']),'families',d)
if __name__=='__main__':main()
