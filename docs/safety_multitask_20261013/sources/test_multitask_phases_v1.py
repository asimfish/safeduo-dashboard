import dataclasses
import unittest
import numpy as np
from multitask_phases_v1 import ARMS,PayloadTask,TaskSpec,MultiObjectTasks


def spec(roles=('F_L','F_R'),family='pair_colift',object_id='beam'):
    return TaskSpec('dev_'+object_id,family,object_id,roles,(.2,0,1.02),(.42,0,1.02),(.04,.7,.08),.45)


def observation(s,t,seq,phase='CLOSE'):
    p=np.array(s.initial_position);force=np.zeros(4);opposed=np.zeros(4,bool)
    # Single-holder capacity must also exceed the unchanged1.25mg reserve.
    ids=[ARMS.index(a) for a in s.roles];force[ids]=12.;opposed[ids]=True
    support=s.mass_kg*9.81;open_error=np.ones(4)*.3;mounts=np.zeros((4,3))
    if phase in ['LIFT','CARRY']:p[2]+=.14;support=0.
    if phase in ['CARRY','PLACE','RELEASE','RETREAT']:p[:2]=s.goal_position[:2]
    if phase in ['RELEASE','RETREAT']:force[:]=0;opposed[:]=False;open_error[:]=0
    if phase=='RETREAT':mounts[:,2]=.03
    return dict(object_id=s.object_id,sequence=seq,time_s=t,now_s=t,position=p,quaternion_wxyz=(1.,0,0,0),
        linear_velocity=(0.,0,0),angular_velocity=(0.,0,0),size_m=s.size_m,mass_kg=s.mass_kg,
        hand_force_N=force,opposed=opposed,support_N=support,hand_open_error_rad=open_error,
        hand_speed_rad_s=np.zeros(4),mounts_m=mounts,conservative_mu=.49,physical_observation_valid=True)


class TaskTests(unittest.TestCase):
    def test_original_ordered_stages_each_supported_role_count(self):
        for roles,family in [(('F_L',),'pick_place'),(('F_L','F_R'),'pair_colift'),(ARMS,'four_shared_colift')]:
            s=spec(roles,family);task=PayloadTask(s);t=0.;seq=0
            for phase,duration in [('CLOSE',.3),('LIFT',2.05),('CARRY',.55),('PLACE',.55),('RELEASE',.55),('RETREAT',1.05)]:
                self.assertEqual(task.phase,phase)
                for _ in range(int(duration*100)+1):
                    task.update(observation(s,t,seq,phase));t+=.01;seq+=1
            self.assertEqual(task.phase,'DONE')
            self.assertEqual([x['to'] for x in task.transitions],['LIFT','CARRY','PLACE','RELEASE','RETREAT','DONE'])
            self.assertFalse(task.intent()['native_task_completion_certified'])

    def test_four_holders_cannot_advance_from_only_two_contacts(self):
        s=spec(ARMS,'four_shared_colift');task=PayloadTask(s)
        for seq in range(101):
            o=observation(s,seq*.01,seq);o['hand_force_N'][2:]=0;o['opposed'][2:]=False;task.update(o)
        self.assertEqual(task.phase,'CLOSE')

    def test_contact_without_opposition_or_capacity_cannot_lift(self):
        for missing in ['opposition','capacity']:
            s=spec();task=PayloadTask(s)
            for seq in range(101):
                o=observation(s,seq*.01,seq)
                if missing=='opposition':o['opposed'][:]=False
                else:o['conservative_mu']=.01
                task.update(o)
            self.assertEqual(task.phase,'CLOSE')

    def test_supported_motion_cannot_count_as_airborne_lift(self):
        s=spec();task=PayloadTask(s);task.phase='LIFT'
        for seq in range(251):
            o=observation(s,seq*.01,seq,'LIFT');o['support_N']=s.mass_kg*9.81;task.update(o)
        self.assertEqual(task.phase,'LIFT')

    def test_wrong_object_stale_mass_and_missing_input_fail_closed(self):
        for error in ['object','stale','mass','missing']:
            s=spec();task=PayloadTask(s);o=observation(s,0,0)
            if error=='object':o['object_id']='other'
            elif error=='stale':o['now_s']=.1
            elif error=='mass':o['mass_kg']*=.5
            else:del o['opposed']
            self.assertEqual(task.update(o)['phase'],'FAILED')

    def test_ownership_conflicts_rejected_and_all_objects_required(self):
        with self.assertRaises(ValueError):MultiObjectTasks([spec(),spec(object_id='other')])
        tasks=MultiObjectTasks([spec(('F_L',),'pick_place','a'),spec(('U_R',),'pick_place','b')])
        with self.assertRaises(ValueError):tasks.update({'a':observation(spec(('F_L',),'pick_place','a'),0,0)})
        tasks.tasks['a'].phase='DONE'
        result=tasks.update({s.object_id:observation(s,0,0) for s in [task.spec for task in tasks.tasks.values()]})
        self.assertFalse(result['all_done'])

    def test_40mm_legacy_carry_goal_rejected(self):
        with self.assertRaises(ValueError):PayloadTask(dataclasses.replace(spec(),goal_position=(.2,.04,1.02)))


if __name__=='__main__':unittest.main()
