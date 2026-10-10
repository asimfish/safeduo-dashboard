"""Read-only captured RED replay plus prospective in-memory CPU expectations."""
from evidence_io import cpu_limits
cpu_limits()
import argparse
import ast
import copy
import csv
import io
import json
import math
from pathlib import Path
import resource
import unittest
import numpy as np
from evidence_io import HERE,H,OUTPUT,Evidence,require,sha,write_new,parse_json,contained
from actual_angle_oracle import measured,circular_error,separated,prospective_record,verify_candidate,selected_actual
from camera_optics import frustum

ROOT=None
POINTS=np.array([[-.01,-.01,-.01],[.01,.01,.01]],float)


def camera(angle,center=(0.,0.,0.),radius=1.,height=0.):
    rad=math.radians(angle);world=np.eye(4);world[3,:3]=np.asarray(center)+[radius*math.cos(rad),radius*math.sin(rad),height]
    return {'actual_camera_to_world_row_matrix':world.tolist()}


def record(angle,number=1,requested=None):
    requested=angle if requested is None else requested;r=math.radians(requested)
    c=camera(angle);value=dict(attempt=number,camera=c,world_direction=[math.cos(r),math.sin(r),0.])
    value.update(prospective_record(value,POINTS));return value


class Oracles(unittest.TestCase):
    def test_analytic_cardinal_angles(self):
        for angle in (0.,45.,90.,135.,180.,225.,270.,315.):
            self.assertAlmostEqual(measured(camera(angle),POINTS)['actual_azimuth_deg'],angle,places=12)

    def test_wrap_and_unchanged30_boundary(self):
        self.assertEqual(circular_error(1.,359.),2.);self.assertEqual(circular_error(359.,1.),-2.)
        self.assertTrue(separated([0.,30.,90.]));self.assertFalse(separated([0.,29.999999999,90.]))
        self.assertTrue(separated([350.,20.,80.]));self.assertFalse(separated([350.,19.999999999,80.]))

    def test_identity_threshold_unchanged(self):
        r=record(0.);r['azimuth_deg']=0.000099
        verify_candidate(r,POINTS)
        for bad in (0.000101,1.,180.):
            r['azimuth_deg']=bad
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,'1e-4'):verify_candidate(r,POINTS)

    def test_nominal_pass_actual_below30_must_fail(self):
        rows=[record(0.,1,0.),record(29.99995,2,30.),record(90.,3,90.)]
        self.assertTrue(separated([r['requested_azimuth_deg'] for r in rows]));self.assertFalse(selected_actual(rows,POINTS))
        # Even storing nominal30 would pass1e-4 identity, but actual separation
        # must still fail: the identity tolerance cannot relax the30deg gate.
        rows[1]['azimuth_deg']=30.;verify_candidate(rows[1],POINTS);self.assertFalse(selected_actual(rows,POINTS))

    def test_actual_above30_does_not_use_nominal(self):
        rows=[record(0.,1,0.),record(30.00005,2,29.99995),record(90.,3,90.)]
        self.assertFalse(separated([r['requested_azimuth_deg'] for r in rows]));self.assertTrue(selected_actual(rows,POINTS))

    def test_post_render_matrix_is_authoritative(self):
        r=record(0.);before=copy.deepcopy(r);r['camera']=camera(31.)
        with self.assertRaisesRegex(ValueError,'1e-4'):verify_candidate(r,POINTS)
        prediction=prospective_record(r,POINTS)
        self.assertAlmostEqual(prediction['azimuth_deg'],31.);self.assertEqual(prediction['requested_azimuth_deg'],0.)
        self.assertEqual(before['azimuth_deg'],0.)

    def test_adaptive_target_cannot_fake_view_separation(self):
        rows=[record(0.,i+1,float(i*60)) for i in range(3)]
        for i,r in enumerate(rows):r['target_world_m']=[float(i),float(i*2),0.]
        self.assertTrue(separated([r['requested_azimuth_deg'] for r in rows]));self.assertFalse(selected_actual(rows,POINTS))
        self.assertTrue(all(measured(r['camera'],POINTS)['azimuth_reference_world_m']==[0.,0.,0.] for r in rows))

    def test_moved_reference_rejected(self):
        r=record(40.);r['azimuth_reference_world_m']=[.1,0,0]
        with self.assertRaisesRegex(ValueError,'reference'):verify_candidate(r,POINTS)

    def test_duplicate_attempt_rejected(self):
        with self.assertRaisesRegex(ValueError,'distinct'):selected_actual([record(x,1) for x in (0.,60.,120.)],POINTS)

    def test_nonfinite_reflected_nonrigid_matrix_rejected(self):
        for value in (float('nan'),float('inf'),-1.,2.):
            c=camera(0.);c['actual_camera_to_world_row_matrix'][0][0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):measured(c,POINTS)
        with self.assertRaises(ValueError):measured(camera(0.),[[float('nan'),0,0]])

    def test_vertical_undefined_and_nearvertical_condition(self):
        with self.assertRaisesRegex(ValueError,'vertical'):measured(camera(0.,radius=0.,height=1.),POINTS)
        with self.assertRaisesRegex(ValueError,'vertical'):measured(camera(0.,radius=1e-10,height=1.),POINTS)
        result=measured(camera(45.,radius=1e-8,height=1.),POINTS)
        self.assertAlmostEqual(result['condition'],1e8,delta=1e-7);self.assertAlmostEqual(result['actual_azimuth_deg'],45.)

    def test_translation_invariance_float64(self):
        offset=np.array([16.,-8.,3.]);points=POINTS+offset
        for angle in (17.,72.,251.,358.):
            self.assertAlmostEqual(measured(camera(angle,center=offset),points)['actual_azimuth_deg'],angle,places=11)

    def test_declared_requested_identity_is_separate(self):
        r=record(90.,requested=0.);self.assertEqual(r['requested_azimuth_deg'],0.)
        self.assertEqual(r['actual_azimuth_deg'],90.);verify_candidate(r,POINTS)

    def test_source_boundary_has_no_native_imports(self):
        names=set()
        for path in HERE.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.Import):names.update(x.name.split('.')[0] for x in node.names)
                elif isinstance(node,ast.ImportFrom) and node.module:names.add(node.module.split('.')[0])
        self.assertFalse(names & {'torch','isaaclab','omni','pxr','qualified_views_fullsphere_v1','hand_views_strict_v1'})
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS),(1024**3,1024**3))


def captured_replay(e,a,out):
    e.verify(a['closed_V2']);e.verify(a['closed_angle']);e.verify(a['prospective_static_sources'])
    waitpath=next(p for p in a['closed_V2'] if p.endswith('/ACTUAL_WAIT.json'));w=e.js(waitpath)
    require((w['actual_wait_exit'],w['raw_wait_status'],w['child_pid'],w['child_identity']['startticks'])==(2,512,4191603,172491481),'closed V2 failure identity')
    diagnostic=e.js(next(p for p in a['closed_V2'] if p.endswith('/FAILED_INITIAL_DIAGNOSTIC.json')))
    require(diagnostic['status']=='FAILED_DIAGNOSTIC_INTEGRITY' and diagnostic['camera']['groups_visited']==32,'closed all32 V2 integrity failure')
    original=e.js(next(p for p in a['closed_angle'] if p.endswith('/proofs/ANGLE_DIAGNOSTIC.json')))
    rows=e.js(next(p for p in a['closed_angle'] if p.endswith('/proofs/ALL_HAND_ANGLES.json')))
    batch=Path(a['actual_attempt_root'])/'native/batch_00000';results=[];bykey={};failing=[]
    for row in rows:
        ref=row['attempt_reference'];rec=e.js(batch/ref['path'],ref['sha256']);points=np.asarray(rec['hand_link_positions_world_m'],float)
        planned=float(rec['azimuth_deg']);actual=measured(rec['camera'],points)['actual_azimuth_deg'];frozen=frustum(points,rec['camera'])['azimuth_deg']
        require(abs(circular_error(actual,frozen))<1e-10,'independent oracle/frozen reader exact angle')
        prediction=prospective_record(rec,points);require(abs(circular_error(prediction['azimuth_deg'],frozen))<1e-10,'prospective in-memory expected angle')
        old_rejected=abs(circular_error(planned,frozen))>=1e-4
        result=dict(env_id=row['env_id'],arm=row['arm'],attempt=row['attempt'],old_requested_deg=planned,actual_deg=actual,
            signed_error_deg=circular_error(actual,planned),old_reader_rejects=old_rejected,
            prospective_oracle_prediction=prediction,actual_producer_receipt_unchanged=True)
        results.append(result);bykey[row['env_id'],row['arm'],row['attempt']]=result
        if old_rejected:failing.append(result)
    require(len(results)==732 and len(failing)==16,'exact captured RED denominator')
    olderrors={(g['env_id'],x.split('/')[1],int(x.split('/')[2].split(':')[0])) for g in diagnostic['camera']['groups'] for x in g['integrity_errors']}
    require(olderrors=={(r['env_id'],r['arm'],r['attempt']) for r in failing},'actual V2 errors match16 independent RED cases')
    groups=[]
    for group in original['all_selected_group_separations']:
        angles=[bykey[group['env_id'],group['arm'],i]['actual_deg'] for i in group['selected_attempts']]
        groups.append(dict(env_id=group['env_id'],arm=group['arm'],actual_angles=angles,actual_ge30=separated(angles)))
    require(len(groups)==128 and all(g['actual_ge30'] for g in groups),'128 captured selected trios actual>=30 unchanged')
    source=Path(next(p for p in a['prospective_static_sources'] if p.endswith('/qualified_views_fullsphere_v1.py')))
    tree=ast.parse(e.raw(source));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QualifiedViews')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_attempt')
    nominal=[n for n in ast.walk(method) if isinstance(n,ast.keyword) and n.arg=='azimuth_deg']
    final_reads=[n for n in ast.walk(method) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='_camera']
    writes=[n for n in ast.walk(method) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Constant) and t.slice.value=='azimuth_deg' for t in n.targets)]
    require(len(nominal)==1 and 'direction' in ast.unparse(nominal[0].value) and len(final_reads)==2 and nominal[0].lineno<max(n.lineno for n in final_reads) and not writes,'prospective nominal-only azimuth source reproduction')
    oldoptics=H/'astra/full74_failed_camera_diagnostic_v2/camera_optics.py'
    require(ast.dump(ast.parse(e.raw(oldoptics)),include_attributes=False)==ast.dump(ast.parse(e.raw(HERE/'camera_optics.py')),include_attributes=False),'old optics AST immutable')
    write_new(out/'CAPTURED_ORACLE_REPLAY.json',dict(status='CPU_EXPECTATION_ONLY_NOT_PRODUCER_FIX',captured=results,selected_groups=groups,
        old_rejections_preserved=16,V2_actual_wait_exit=2,prospective_source=dict(path=str(source),sha256=e.hash(source),
        nominal_assignment_line=nominal[0].lineno,final_camera_read_line=max(n.lineno for n in final_reads),actual_angle_update_absent=True),
        native_pass=False,safety_acceptance=False))
    return original,dict(captured_records=732,old_rejections_preserved=16,selected_groups=128,prospective_source_fault_reproduced=True)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--proof-dir',required=True);args=p.parse_args()
    out=contained(args.proof_dir,OUTPUT);out.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Oracles))
    require(result.wasSuccessful(),'independent angle oracle tests failed')
    e=Evidence();a=e.js(HERE/'anchors.json');original,replay=captured_replay(e,a,out);e.recheck()
    write_new(out/'EVIDENCE_LEDGER.json',e.ledger)
    write_new(out/'CPU_TEST_REPORT.json',dict(status='CLOSED_CPU_ANGLE_ORACLES_ONLY',tests_run=result.testsRun,failures=0,errors=0,
        captured_replay=replay,native_pass=False,safety_acceptance=False,original_V2_actual_exit=2,
        max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return 0


if __name__=='__main__':raise SystemExit(main())
