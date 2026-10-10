"""Known CPU fixtures; passing tests never mint native PASS or safety acceptance."""
from evidence_io import cpu_limits
cpu_limits()
import argparse
import ast
import copy
import io
import json
import os
from pathlib import Path
import resource
import types
import unittest
import zipfile
import numpy as np
from evidence_io import (ARMS,H,HERE,SCREEN,OUTPUT,Evidence,require,sha,write_new,
    equal_bits,metadata,contained,parse_json)
from science import (flags,parameters_layout,scalar_contacts,classify,counts,geometry,REASONS)
from camera_adapter import CameraAudit,exact_snapshot,micro_native,supplement
from camera_oracle import mask_counts,token,selected_three,native_and_pixels
from cpu_supervisor import wait_cpu

ROOT=None


def save_npy(path,a):
    with path.open('xb') as f:np.save(f,a,allow_pickle=False)


def layout_fixture():
    return dict(hard=np.tile(np.array([[-1,1]],np.float32),(74,1)),
        soft=np.tile(np.array([[-.5,.5]],np.float32),(26,1)),vmax=np.ones(74,np.float32),
        controlled=np.arange(26,dtype=np.int64),slices=dict(zip(ARMS,((0,19),(19,38),(38,56),(56,74)))))


def row_fixture():
    row={k:np.zeros((32,74),np.float32) for k in ('q74','qd74','target74')}
    row.update(raw9021=np.ones((32,9021),np.float32),scalar82=np.zeros((32,82),np.float64))
    for a in ARMS:
        row[a+'_root_xyzw']=np.zeros((32,7),np.float32);row[a+'_root_velocity']=np.zeros((32,6),np.float32)
        row[a+'_link_xyzw']=np.zeros((32,22 if a.startswith('F') else 19,7),np.float32)
    return row


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/self._testMethodName;self.root.mkdir()

    def test_all509_frozen_inputs_and_camera_anchors(self):
        e=Evidence();anchors=e.js(HERE/'anchors.json');e.verify(anchors['files'])
        b=e.js(SCREEN/'INPUT_BINDING_V1.json');self.assertEqual(len(b['files']),509);self.assertEqual(len(b['original500_sources']),500)
        e.verify(b['files']);e.recheck()
        write_new(self.root/'INPUT_VERIFICATION.json',dict(status='CPU_READ_ONLY_509_AND_ANCHORS',files=e.ledger,native_pass=False))

    def test_frozen_parameter_layout_full74(self):
        e=Evidence();b=e.js(SCREEN/'INPUT_BINDING_V1.json');p=e.npz(b['parameter_path']);v=parameters_layout(p)
        self.assertEqual(v['hard'].shape,(74,2));self.assertEqual(len(v['controlled']),26)
        p['U_R_max_velocity'][31,-1]*=2
        with self.assertRaisesRegex(ValueError,'homogeneous'):parameters_layout(p)

    def test_all74_last_joint_hard_speed(self):
        row=row_fixture();layout=layout_fixture();target=row['target74'].copy()
        row['q74'][31,73]=2;row['qd74'][30,73]=-2
        f=flags(row,layout,target,True)
        self.assertTrue(f[31,1]);self.assertTrue(f[30,2]);self.assertEqual(int(f.sum()),2)

    def test_soft26_original_f32_threshold(self):
        row=row_fixture();l=layout_fixture();t=row['target74'].copy()
        edge=np.float32(.5)+np.float32(1e-5);row['q74'][0,25]=edge
        self.assertFalse(flags(row,l,t,True)[0,3])
        row['q74'][0,25]=np.nextafter(edge,np.float32(1))
        self.assertTrue(flags(row,l,t,True)[0,3])

    def test_raw9021_last_row_and_exemptions(self):
        r=row_fixture();r['raw9021'][12,-1]=-np.nextafter(np.float32(0),np.float32(1));r['exempt9021']=np.ones((32,9021),bool)
        self.assertTrue(flags(r,layout_fixture(),r['target74'].copy(),True)[12,4])

    def test_nonfinite_native_fields_and_targets(self):
        for key in ('q74','qd74','target74','raw9021','scalar82','U_R_link_xyzw','F_L_root_velocity'):
            with self.subTest(key=key):
                r=row_fixture();t=r['target74'].copy();r[key].reshape(32,-1)[7,-1]=np.nan
                self.assertTrue(flags(r,layout_fixture(),t,True)[7,0])

    def test_target_signed_zero_is_bitwise_drift(self):
        r=row_fixture();t=r['target74'].copy();r['target74'][0,73]=-0.
        self.assertTrue(flags(r,layout_fixture(),t,True)[0,6])

    def test_scalar_exact_threshold_freshness(self):
        r=row_fixture();r['scalar82'][0,81]=.1;t=r['target74'].copy()
        self.assertFalse(flags(r,layout_fixture(),t,True)[0,5])
        r['scalar82'][0,81]=np.nextafter(.1,np.inf)
        self.assertTrue(flags(r,layout_fixture(),t,True)[0,5]);self.assertFalse(flags(r,layout_fixture(),t,False)[0,5])

    def point_fixture(self):
        c=np.array([[2,0]],np.int32);s=np.array([[7,99]],np.int32)
        idx=dict(point_indices=np.array([7,8],np.int64),sensor_indices=np.array([0,0],np.int64),partner_indices=np.array([0,0],np.int64))
        f=np.array([[.08],[-.08]],np.float32)
        return c,s,idx,f

    def test_scalar_cancellation_and_holes(self):
        c,s,i,f=self.point_fixture();z=scalar_contacts(c,s,i,f)
        self.assertEqual(z[0,0],2*float(np.float32(.08)));self.assertGreater(z[0,0],.1);self.assertEqual(z[0,1],0)

    def test_contact_overlap_rejected(self):
        c,s,i,f=self.point_fixture();c[0,1]=1;s[0,1]=8
        with self.assertRaisesRegex(ValueError,'overlapping'):scalar_contacts(c,s,i,f)

    def test_contact_capacity_boundary_rejected(self):
        c,s,i,f=self.point_fixture();s[0,0]=262142
        with self.assertRaisesRegex(ValueError,'capacity'):scalar_contacts(c,s,i,f)

    def test_contact_negative_float_uint_overflow(self):
        c,s,i,f=self.point_fixture()
        for bad in (np.array([[-1,0]],np.int32),np.array([[2.,0.]],np.float64),np.array([[2**64-1,0]],np.uint64)):
            with self.subTest(dtype=str(bad.dtype)),self.assertRaises(ValueError):scalar_contacts(bad,s,i,f)

    def test_contact_wrong_owner_or_point_order(self):
        c,s,i,f=self.point_fixture()
        for key in i:
            bad={k:v.copy() for k,v in i.items()};bad[key][0]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):scalar_contacts(c,s,bad,f)

    def test_contact_owned_nonfinite_rejected(self):
        c,s,i,f=self.point_fixture();f[1]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):scalar_contacts(c,s,i,f)

    def test_zero_eligible_valid_full_denominator(self):
        f=np.zeros((13,32,7),bool);f[12,:,4]=True;s=classify(f,native_closed=True,v7_camera=True)
        self.assertTrue(np.all(s==2));ledger=np.r_[s,np.zeros(8160,np.uint8)]
        self.assertEqual(counts(ledger)['UNUSED'],8160);self.assertEqual(counts(ledger)['OBSERVED_REJECT'],32)

    def test_synthetic_never_native_prefix(self):
        f=np.zeros((13,32,7),bool)
        for good in (True,False):self.assertTrue(np.all(classify(f,native_closed=False,v7_camera=good)==4))

    def test_missing_camera_cannot_qualify(self):
        f=np.zeros((13,32,7),bool);f[4,2,3]=True;s=classify(f,native_closed=True,v7_camera=False)
        self.assertEqual(s[2],2);self.assertEqual(np.count_nonzero(s==4),31)

    def test_prefix_missing_last_micro_rejected(self):
        with self.assertRaises(ValueError):classify(np.zeros((12,32,7),bool),native_closed=True,v7_camera=True)

    def test_cifs_metadata_ignores_inode(self):
        a=types.SimpleNamespace(st_mode=0o100600,st_dev=1,st_size=2,st_mtime_ns=3,st_ctime_ns=4,st_ino=100)
        b=copy.copy(a);b.st_ino=999;self.assertEqual(metadata(a),metadata(b));b.st_ctime_ns=5;self.assertNotEqual(metadata(a),metadata(b))

    def test_mutation_same_length_hash_detected(self):
        p=self.root/'bytes.bin';p.write_bytes(b'abcd');e=Evidence();e.hash(p);p.write_bytes(b'abce')
        with self.assertRaisesRegex(ValueError,'changed'):e.recheck()

    def test_symlink_and_escape_rejected(self):
        p=self.root/'x';p.write_bytes(b'x');link=self.root/'link';link.symlink_to(p)
        with self.assertRaises(ValueError):Evidence().hash(link)
        with self.assertRaises(ValueError):contained(self.root/'../outside',self.root)

    def test_duplicate_json_nonfinite_rejected(self):
        for raw in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):parse_json(raw)

    def test_object_array_and_shape_rejected(self):
        p=self.root/'o.npy'
        with p.open('wb') as f:np.save(f,np.array([{}],object),allow_pickle=True)
        with self.assertRaises(ValueError):Evidence().npy(p)
        p=self.root/'n.npy';save_npy(p,np.zeros((32,73),np.float32))
        with self.assertRaisesRegex(ValueError,'shape'):Evidence().npy(p,(32,74),'float32')

    def test_npz_duplicate_members_rejected(self):
        import warnings
        p=self.root/'bad.npz';raw=io.BytesIO();np.save(raw,np.zeros(1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with zipfile.ZipFile(p,'w') as z:z.writestr('x.npy',raw.getvalue());z.writestr('x.npy',raw.getvalue())
        with self.assertRaisesRegex(ValueError,'duplicate'):Evidence().npz(p)

    def mask_fixture(self):
        p='/World/envs/env_0/U_L/index_1';q='/World/envs/env_0/U_L/middle_1'
        inventory={p:dict(env_id=0,arm='U_L',body_name='index_1',hand=True),q:dict(env_id=0,arm='U_L',body_name='middle_1',hand=True)}
        outputs={k:np.zeros((720,1280,1),np.uint32) for k in ('instance_segmentation_fast','semantic_segmentation')}
        outputs['instance_segmentation_fast'].flat[:600]=1;outputs['semantic_segmentation'].flat[:600]=2
        info={'instance_segmentation_fast':{'idToLabels':{'0':'BACKGROUND','1':p+'/mesh'}},
            'semantic_segmentation':{'idToLabels':{'0':{'class':'BACKGROUND'},'2':{'class':token('qv4_hand_',p)}}}}
        return p,q,inventory,outputs,info

    def test_raw_mask_same_rigid_intersection(self):
        p,q,inv,out,info=self.mask_fixture();a=mask_counts(out,info,inv,{}, {},0)
        self.assertEqual(a['per_native_object_pixels'][p],600)
        info['semantic_segmentation']['idToLabels']['2']['class']=token('qv4_hand_',q)
        a=mask_counts(out,info,inv,{}, {},0);self.assertEqual(a['per_native_object_pixels'][p],0);self.assertEqual(a['per_native_object_pixels'][q],0)

    def test_unknown_mask_id_and_class_rejected(self):
        p,q,inv,out,info=self.mask_fixture();out['semantic_segmentation'].flat[0]=7
        with self.assertRaisesRegex(ValueError,'unknown'):mask_counts(out,info,inv,{}, {},0)
        out['semantic_segmentation'].flat[0]=2;info['semantic_segmentation']['idToLabels']['2']['class']='qv4_hand_unregistered'
        with self.assertRaisesRegex(ValueError,'unregistered'):mask_counts(out,info,inv,{}, {},0)

    def test_missing_native27_field_and_last_field_drift(self):
        layout=layout_fixture();row=row_fixture();fields={k:np.repeat(v[None],13,0) for k,v in row.items()}
        fields.update(simulation_time_s=np.arange(13,dtype=np.float64),simulation_step=np.arange(13,dtype=np.int64))
        baseline=micro_native(fields,0,layout,np.zeros((32,3),np.float32));self.assertEqual(len(baseline),27)
        exact_snapshot(baseline,{k:v.copy() for k,v in baseline.items()})
        wrong={k:v.copy() for k,v in baseline.items()};wrong['U_R_native_position_targets'][31,17]=1
        with self.assertRaises(ValueError):exact_snapshot(baseline,wrong)
        wrong=dict(baseline);wrong.pop('environment_origins')
        with self.assertRaises(ValueError):exact_snapshot(baseline,wrong)

    def test_camera_supplement_retains_body_gaps(self):
        inventory={};g={'arms':{},'critical_body_pixels':{}}
        for a in ARMS:
            path=f'/World/envs/env_0/{a}/hand';inventory[path]=dict(env_id=0,arm=a,hand=True,body_name='hand')
            g['arms'][a]={'per_native_body_peak_pixels':{path:64}}
        for a in ('U_L','U_R'):
            for body in ('forearm_link','wrist_2_link'):
                p=f'/World/envs/env_0/{a}/{body}';inventory[p]=dict(env_id=0,arm=a,hand=False,body_name=body);g['critical_body_pixels'][p]=128
        self.assertTrue(supplement(g,inventory,0)['observed_coverage_sufficient'])
        g['arms']['F_R']['per_native_body_peak_pixels']['/World/envs/env_0/F_R/hand']=63
        self.assertFalse(supplement(g,inventory,0)['observed_coverage_sufficient'])

    def test_three_camera_views_angle_bound(self):
        attempts=[dict(rgb=dict(path=str(i),sha256='x'),_independent_azimuth=a) for i,a in enumerate((0.,30.,60.))]
        self.assertEqual(len(selected_three([r['rgb'] for r in attempts],attempts)),3)
        attempts[-1]['_independent_azimuth']=59.9
        with self.assertRaises(ValueError):selected_three([r['rgb'] for r in attempts],attempts)

    def camera_bytes_fixture(self):
        from PIL import Image
        import hashlib
        row=row_fixture();fields={k:np.repeat(v[None],13,0) for k,v in row.items()}
        fields.update(simulation_time_s=np.arange(13,dtype=np.float64),simulation_step=np.arange(13,dtype=np.int64))
        baseline=micro_native(fields,0,layout_fixture(),np.zeros((32,3),np.float32))
        def nz(name,arrays):
            p=self.root/name
            with p.open('xb') as f:np.savez_compressed(f,**arrays)
            return dict(path=name,sha256=sha(p))
        native=nz('native.npz',baseline)
        rgb=np.zeros((720,1280,3),np.uint8);rgb[0,0]=[255,37,11]
        raw=nz('outputs.npz',dict(rgb=rgb,instance_segmentation_fast=np.zeros((720,1280,1),np.uint32),semantic_segmentation=np.zeros((720,1280,1),np.uint32)))
        Image.fromarray(rgb).save(self.root/'rgb.png')
        write_new(self.root/'info.json',{})
        rec=dict(native_before_all64=native,native_after_all64=native,
            held_renders=[dict(index=i,native_before_all64=native,native_after_all64=native) for i in range(3)],
            render_calls=3,sensor_frame_before=[0],sensor_frame_after=[1],sensor_update_dt_s=0.,
            original_outputs=raw,original_info=dict(path='info.json',sha256=sha(self.root/'info.json')),
            rgb=dict(path='rgb.png',sha256=sha(self.root/'rgb.png')),raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        body={a:np.arange(22 if a.startswith('F') else 19) for a in ARMS}
        audit=CameraAudit(self.root,Evidence(),body,{},layout_fixture())
        return rec,baseline,audit,nz,rgb

    def test_raw_rgb_png_pixels_roundtrip_and_mutation(self):
        from PIL import Image
        rec,baseline,audit,nz,rgb=self.camera_bytes_fixture()
        out,info=native_and_pixels(audit,rec,baseline);self.assertTrue(equal_bits(out['rgb'],rgb));self.assertEqual(audit.snapshots,8)
        rgb[0,0]=[0,0,0];Image.fromarray(rgb).save(self.root/'altered.png')
        rec['rgb']=dict(path='altered.png',sha256=sha(self.root/'altered.png'))
        with self.assertRaisesRegex(ValueError,'PNG not exact'):native_and_pixels(audit,rec,baseline)

    def test_each_held_render_native27_and_clock_mutation(self):
        rec,baseline,audit,nz,rgb=self.camera_bytes_fixture()
        for index in range(3):
            bad={k:v.copy() for k,v in baseline.items()};bad['simulation_time_s']=np.array(.008333,np.float64)
            altered=nz('drift_'+str(index)+'.npz',bad);changed=copy.deepcopy(rec)
            changed['held_renders'][index]['native_after_all64']=altered
            with self.subTest(render=index),self.assertRaisesRegex(ValueError,'held27 simulation_time_s'):native_and_pixels(audit,changed,baseline)

    def test_missing_held_render_or_sensor_advance(self):
        rec,baseline,audit,nz,rgb=self.camera_bytes_fixture();changed=copy.deepcopy(rec);changed['held_renders'].pop()
        with self.assertRaisesRegex(ValueError,'three held'):native_and_pixels(audit,changed,baseline)
        rec['sensor_frame_after']=[2]
        with self.assertRaisesRegex(ValueError,'sensor frame'):native_and_pixels(audit,rec,baseline)

    def test_external_unbound_file_denied(self):
        with self.assertRaisesRegex(ValueError,'exact frozen509'):Evidence().hash('/etc/hosts','0'*64)

    def test_geometry_all_rows_and_table_last_row(self):
        r=row_fixture();ids={};slices={};names=[];cursor=0
        for a,n in zip(ARMS,(31,31,40,40)):
            r[a+'_link_xyzw'][...,6]=1;r[a+'_link_xyzw'][...,:3]=np.array([cursor,0,0])
            ids[a]=[0]*n;slices[a]=[cursor,cursor+n];names.extend([a+'/tip']*n);cursor+=n
        pairs=[[0,62]]*8745+[[141,0]]*276
        ident=dict(sphere_offsets=[[0,0,0]]*142,sphere_radii_m=[.1]*142,sphere_body_indices=ids,arm_slices=slices,pair_sphere_idx=pairs,table_slice_start=8745)
        d,c=geometry(r,ident,np.zeros((2,3)),np.ones((2,3)),np.zeros((32,3)))
        self.assertEqual(d.shape,(32,9021));self.assertAlmostEqual(d[0,0],61.8);self.assertAlmostEqual(d[0,-1],100.9)
        ident['pair_sphere_idx'][-1]=[0,0];d,_=geometry(r,ident,np.zeros((2,3)),np.ones((2,3)),np.zeros((32,3)));self.assertAlmostEqual(d[0,-1],-1.1)

    def test_real_owned_timeout_reaped(self):
        w=wait_cpu(HERE/'fixture_child.py',['timeout'],self.root/'timeout',.3)
        self.assertEqual(w['actual_wait_exit'],-9);self.assertEqual(w['raw_wait_status'],9)
        self.assertEqual(w['waited_pid'],w['child_pid']);self.assertEqual(w['signals'][0]['pid'],w['child_pid']);self.assertIn('TimeoutError',w['resource_abort'])

    def test_no_simulator_or_producer_imports(self):
        imports=set()
        for p in HERE.glob('*.py'):
            for n in ast.walk(ast.parse(p.read_text())):
                if isinstance(n,ast.Import):imports.update(x.name.split('.')[0] for x in n.names)
                if isinstance(n,ast.ImportFrom) and n.module:imports.add(n.module.split('.')[0])
        self.assertFalse(imports & {'torch','isaaclab','omni','pxr','native_full74_screen_v2','screen_core_v1','qualified_views_v7'})
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS),(1024**3,1024**3))

