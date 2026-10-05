"""Explicit synthetic final-state UI fixture outside any public dataset."""
from pathlib import Path
import json,shutil
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/nas/data/lyf/double_hand')/HERE.name/'ui_fixture'
def main():
    ROOT.mkdir(exist_ok=False);(ROOT/'cases').mkdir()
    template=json.loads((Path('/home/liyufeng/safeduo-dashboard-dynamics-prediction-20261006')/'docs'/HERE.name/'data.json').read_text())
    template.update(status='complete',completed_windows=384,fixture_only=True,cases=[],groups=[],rows=[])
    for mode in ('admission_full','joint_reference'):
        predictors=[dict(model=n,all_nonexempt=dict(mae_m=.001),near_pre80mm=dict(rmse_m=.001,max_m=.002),missed_windows=1,missed_env_steps=1,stable_mask_missed_windows=1) for n in ('static','target_snap','velocity','empirical')]
        template['groups'].append(dict(mode=mode,windows=192,violations=1,deep=0,predictions=predictors))
    for i in range(192):
        c=dict(id=f'fixture_{i}',command_seed=0,controllers=[])
        for mode in ('admission_full','joint_reference'):
            margins=[[.03]*4 for _ in range(960)]
            if i==0:margins[400][0]=-.002
            c['controllers'].append(dict(mode=mode,bad=i==0,model_misses={n:i==0 for n in ('static','target_snap','velocity','empirical')},normalized_joint_span_percent=10,joint_l1_path_rad=20,min_margin_m=-.002 if i==0 else .03,margins=margins))
        (ROOT/'cases'/(c['id']+'.json')).write_text(json.dumps(c))
        template['cases'].append({**c,'controllers':[{k:v for k,v in row.items() if k!='margins'} for row in c['controllers']]})
    (ROOT/'data.json').write_text(json.dumps(template));shutil.copy2(HERE/'panel.html',ROOT/'index.html')
    print(ROOT)
if __name__=='__main__':main()
