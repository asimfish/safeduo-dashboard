"""Post-run phase descriptions only; never changes frozen task verdicts."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent

def main():
    result=json.loads((R/'native_results.json').read_text())
    with np.load(R/'native_curves.npz',allow_pickle=False) as z:
        t=z['time'];lift=z['lift'];hand=z['hand_normal'];tilt=z['tilt']
        dt=float(t[1]-t[0]);rows=[]
        for c in result['cases']:
            e=c['env'];objects={}
            for oi,o in enumerate(('beam700','beam300')):
                above=lift[:,e,oi]>.05;both=(hand[:,e,oi]>.1).all(-1);carry=(t>=5.7)&(t<11.7)
                ids=np.flatnonzero(above);roll=np.flatnonzero(tilt[:,e,oi]>80)
                late=bool(above.any() and not (above&(t<18.7)).any())
                objects[o]=dict(first_lift50_state_time_s=float(t[ids[0]]) if len(ids) else None,
                    all_lift50_states=int(above.sum()),all_lift50_both_hands_states=int((above&both).sum()),
                    planned_lift_carry_max_lift_m=float(lift[carry,e,oi].max()),
                    planned_lift_carry_two_hand_lift_time_s=float((carry&above&both).sum()*dt),
                    lift50_only_during_clearance_or_later=late,
                    first_tilt80_state_time_s=float(t[roll[0]]) if len(roll) else None,
                    original_object_task_pass=c['objects'][o]['pass_task'])
            rows.append(dict(env=e,binding=c['binding'],layout=c['layout'],original_task_pass=c['pass_task'],objects=objects))
    verdict=dict(status='COMPLETE_POST_RUN_EXPLORATORY_PHASE_DESCRIPTION',scope='descriptive follow-up to unchanged task verdicts, not registered additional acceptance or full grasp/force certification',clock='post-step native state times; lift/carry [5.7,11.7), opening starts15.5, clearance starts18.7 per unchanged task metadata',cases=rows,
        passed_task_but_u_lift_only_late=[c['env'] for c in rows if c['original_task_pass'] and c['objects']['beam300']['lift50_only_during_clearance_or_later']],
        passed_task_but_no_u_dual_hand_lift=[c['env'] for c in rows if c['original_task_pass'] and c['objects']['beam300']['all_lift50_both_hands_states']==0],
        totals={mode:dict(u_planned_lift_carry_with_two_hand_lift_cases=sum(c['objects']['beam300']['planned_lift_carry_two_hand_lift_time_s']>0 for c in rows if c['binding']==b),u_final_tilt_gt80_cases=sum(result['cases'][c['env']]['objects']['beam300']['final_tilt_deg']>80 for c in rows if c['binding']==b)) for mode,b in [('fixed_replay',False),('object_pose_binding',True)]},
        inputs_sha256={name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ('native_results.json','native_curves.npz','task_qualified_two_pair.npz')})
    (R/'PHASE_DIAGNOSIS.json').write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps({k:v for k,v in verdict.items() if k!='cases'},indent=2))
if __name__=='__main__':main()
