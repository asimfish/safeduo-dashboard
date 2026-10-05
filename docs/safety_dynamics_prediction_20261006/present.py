"""Presentation only: never alters frozen observer, inputs or metric oracle."""
from pathlib import Path
import argparse,json,hashlib,shutil
from datetime import datetime,timezone
import numpy as np
HERE=Path(__file__).resolve().parent
NAME=HERE.name
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def weighted(rows,key):
    n=sum(r[key]['values'] for r in rows)
    if not n:return dict(values=0,mae_m=None,rmse_m=None,max_m=None)
    return dict(values=n,mae_m=sum(r[key]['mae_m']*r[key]['values'] for r in rows)/n,
                rmse_m=np.sqrt(sum(r[key]['rmse_m']**2*r[key]['values'] for r in rows)/n),max_m=max(r[key]['max_m'] for r in rows))
def build(repo,phase):
    out=repo/'docs'/NAME;out.mkdir(exist_ok=True)
    calibrated=json.loads((HERE/'calibration_results.json').read_text())
    model=json.loads((HERE/'model.json').read_text())
    plans=[json.loads(p.read_text()) for p in sorted((HERE/'registered_retry').glob('block*.json'))]
    payload=dict(status=phase,planned_windows=384,new_commands=192,new_initials=0,updated_utc=datetime.now(timezone.utc).isoformat(),
                 calibration=calibrated,model=model,completed_windows=0,initial_invalid_planned_windows=384,
                 rows=[],groups=[],cases=[],scope='simulation sphere margins; passive predictors; sequential mixed/IID; not certification')
    if phase=='complete':
        result=json.loads((HERE/'results.json').read_text());assert result['status']=='complete' and len(result['rows'])==6
        payload.update(rows=result['rows'],completed_windows=result['method_windows'])
        for mode in ('admission_full','joint_reference'):
            rows=[r for r in result['rows'] if r['mode']==mode];assert len(rows)==3
            predictions=[]
            for name in ('static','target_snap','velocity','empirical'):
                model_rows=[next(p for p in r['predictions'] if p['model']==name) for r in rows]
                predictions.append(dict(model=name,all_nonexempt=weighted(model_rows,'all_nonexempt'),near_pre80mm=weighted(model_rows,'near_pre80mm'),
                    missed_windows=sum(p['missed_windows'] for p in model_rows),missed_env_steps=sum(p['missed_env_steps'] for p in model_rows),
                    missed_negative_rows=sum(p['missed_negative_rows'] for p in model_rows),
                    stable_mask_missed_windows=sum(p['stable_mask_missed_windows'] for p in model_rows)))
            payload['groups'].append(dict(mode=mode,windows=192,violations=sum(r['violations'] for r in rows),deep=sum(r['deep'] for r in rows),
                actual_negative_rows=sum(r['actual_negative_rows'] for r in rows),predictions=predictions))
        case_root=out/'cases';case_root.mkdir(exist_ok=True)
        for block,plan in enumerate(plans):
            arrays={j['mode']:load(Path(plan['output_root'])/j['id']/'cell_001.npz') for j in plan['jobs']}
            initial={j['mode']:load(Path(plan['output_root'])/j['id']/'input_recipe.npz') for j in plan['jobs']}
            for e in range(64):
                case=dict(id=f'b{block}_e{e:02}',block=block,env=e,command_seed=plan['command_seed'],controllers=[])
                for job in plan['jobs']:
                    z=arrays[job['mode']];r=next(r for r in result['rows'] if r['id']==job['id'])
                    q=np.r_[z['q_initial'][None],z['q']][:,e].astype(float)
                    limits=initial[job['mode']]['joint_soft_limits']
                    if limits.ndim==3:limits=limits[e]
                    span=np.ptp(q,axis=0)/(limits[:,1]-limits[:,0])*100
                    margins=z['official_margins'][:,e]
                    assert np.isfinite(margins).all()
                    entry=dict(mode=job['mode'],bad=e in r['bad_case_ids'],
                        model_misses={p['model']:e in p['missed_case_ids'] for p in r['predictions']},
                        normalized_joint_span_percent=float(span.mean()),joint_l1_path_rad=float(np.abs(np.diff(q,axis=0)).sum()),
                        min_margin_m=float(margins.min()),margins=margins.tolist())
                    case['controllers'].append(entry)
                write(case_root/(case['id']+'.json'),case)
                payload['cases'].append({**{k:v for k,v in case.items() if k!='controllers'},
                    'controllers':[{k:v for k,v in c.items() if k!='margins'} for c in case['controllers']]})
        shutil.copy2(HERE/'results.json',out/'results.json')
    else:
        for plan in plans:
            campaign=Path(plan['output_root'])/'campaign.json'
            if campaign.exists():
                state=json.loads(campaign.read_text())
                payload['completed_windows']+=sum(j.get('completed_windows',0) for j in state['jobs'])
    write(out/'data.json',payload)
    for file in HERE.glob('*.py'):shutil.copy2(file,out/file.name)
    for name in ('ADR.md','PLAN.md','CALIBRATION.json','model.json','calibration_results.json','SEAM_FAILURE.md','software_tests_v2.log','seam_regression.log','campaign_completion.json'):
        shutil.copy2(HERE/name,out/name)
    for directory in ('registered','registered_launch','registered_retry'):
        dest=out/directory;dest.mkdir(exist_ok=True)
        for file in (HERE/directory).glob('*.json'):shutil.copy2(file,dest/file.name)
    shutil.copy2(HERE/'panel.html',out/'index.html')
    manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='PUBLIC_MANIFEST.json'}
    write(out/'PUBLIC_MANIFEST.json',dict(files=manifest,status=phase,generated_utc=payload['updated_utc']))
    index=repo/'index.html';html=index.read_text()
    completed=payload['completed_windows']
    card=f'<section class="sci-note" id="dynamicsPassiveEvidence"><h2>实际运动预测：持续随机动作与 IID 对照</h2><p>状态：{"完整结果已复算" if phase=="complete" else "物理实验进行中"}。完成 {completed}/384 方法窗口；192条新随机指令，0个新初态。四种预测只作观察，不参与控制。</p><p>每条轨迹包含450步持续随机动作、450步逐步独立随机指令和60步零指令。历史校准平均误差下降，但存在较大尾部误差；不能当作安全保证。</p><p><a href="docs/{NAME}/">查看协议、误差、危险漏报与逐案例结果 →</a></p></section>'
    import re
    if 'id="dynamicsPassiveEvidence"' in html:
        html,count=re.subn(r'<section class="sci-note" id="dynamicsPassiveEvidence">.*?</section>',lambda _:card,html,flags=re.S);assert count==1
    else:
        marker='<section class="sci-note" id="feasibleGuardEvidence">';assert marker in html;html=html.replace(marker,card+'\n'+marker,1)
    index.write_text(html)
    return out
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--phase',choices=['running','complete'],required=True)
    a=parser.parse_args();print(build(a.repo,a.phase))
