"""Captured-size regression and negative index fixtures; never camera acceptance."""
import copy
import json
import os
from pathlib import Path
import time
import unittest
from evidence_io import HERE,OUTPUT,Evidence,sha,write_new,fingerprint
from camera_index import read_index,metadata_digest,project,MAX_PACKET,MAX_RECORD,CHUNK
ROOT=None
REAL_WORKER_BYTES=26616219
REAL_SHARED_BYTES=26618188


def record(i):
    return dict(env_id=i,step=-1,substep=None,capture_kind='initial',status='failed',reasons=[],arms={},wide_context=None,
        image_count=0,qualified_image_count=0,state=f'qualified_views/initial/capture/env_{i:03d}/state.json',sha256='0'*64,scope='CPU_ONLY_NEVER_NATIVE')


class IndexTests(unittest.TestCase):
    def setUp(self):self.root=ROOT/self._testMethodName;self.root.mkdir()

    def packet(self,records=None,**extra):
        p=self.root/'initial_camera_receipts.json';write_new(p,dict(qualified=False,receipts=[record(i) for i in range(32)] if records is None else records,**extra));return p

    def test_actual_closed_V9_both26MB_indices_project_exact32(self):
        e=Evidence();a=e.js(HERE/'anchors.json');batch=Path(a['actual_attempt_root'])/'native/batch_00000'
        receipt=e.js(batch.parent/'NATIVE_RECEIPT.json');p=batch/'initial_camera_receipts.json'
        self.assertEqual(p.stat().st_size,REAL_WORKER_BYTES)
        with self.assertRaisesRegex(ValueError,'bounded read'):Evidence().js(p)
        worker=read_index(e,p,receipt['artifacts'][str(p)],kind='worker')
        q=batch/Path(worker['receipts'][0]['state']).parent.parent/'receipts.json';self.assertEqual(q.stat().st_size,REAL_SHARED_BYTES)
        shared=read_index(e,q,receipt['artifacts'][str(q)],kind='shared')
        self.assertEqual(worker['receipts'],shared['receipts']);self.assertEqual([r['env_id'] for r in worker['receipts']],list(range(32)))
        self.assertTrue(all(set(r)<{'env_id','step','substep','capture_kind','status','state','sha256','image_count','qualified_image_count','_metadata_sha256','UNUSED'} for r in worker['receipts']))
        self.assertLess(worker['_stream']['peak_buffer_bytes'],MAX_RECORD+CHUNK)
        self.assertFalse(worker['qualified']);e.recheck()
        write_new(self.root/'CAPTURED_INDEX_REPLAY.json',dict(scope='INDEX_PARSE_ONLY_NOT_RAW_CAMERA_AUDIT',old16MiB_reader_rejected=True,
            worker=worker,shared=shared,actual_evidence=e.ledger,native_pass=False,safety_acceptance=False))

    def test_synthetic_exact_actual_size26MB_positive(self):
        p=self.root/'initial_camera_receipts.json';padding=(REAL_WORKER_BYTES-20000)//32
        with p.open('xb') as f:
            f.write(b'{"qualified":false,"receipts":[')
            for i in range(32):
                if i:f.write(b',')
                value=record(i);value['scope']='CPU_fixture'+('x'*padding);f.write(json.dumps(value,separators=(',',':')).encode())
            f.write(b']}');left=REAL_WORKER_BYTES-f.tell();self.assertGreaterEqual(left,0);f.write(b' '*left);f.flush();os.fsync(f.fileno())
        time.sleep(2.2)
        with self.assertRaisesRegex(ValueError,'bounded read'):Evidence().js(p)
        r=read_index(Evidence(),p,kind='worker');self.assertEqual(r['_stream']['file_bytes'],REAL_WORKER_BYTES)
        self.assertEqual(len(r['receipts']),32);self.assertEqual(r['_stream']['retained_full_receipt_objects'],0)
        self.assertLess(len(json.dumps(r)),20000)

    def test_packet_over32MiB_denied_before_read(self):
        p=self.root/'initial_camera_receipts.json'
        with p.open('xb') as f:f.truncate(MAX_PACKET+1)
        e=Evidence()
        with self.assertRaisesRegex(ValueError,'bounded read'):read_index(e,p,kind='worker')
        self.assertEqual(e.ledger,{})

    def test_one_receipt_over4MiB_denied(self):
        records=[record(i) for i in range(32)];records[0]['scope']='x'*(MAX_RECORD+1);p=self.packet(records)
        with self.assertRaisesRegex(ValueError,'4MiB'):read_index(Evidence(),p,kind='worker')

    def test_exact32_denominator_no_selection(self):
        for count in (31,33):
            d=self.root/str(count);d.mkdir();p=d/'initial_camera_receipts.json'
            write_new(p,dict(qualified=False,receipts=[record(i) for i in range(count)]))
            with self.subTest(count=count),self.assertRaisesRegex(ValueError,'32'):read_index(Evidence(),p,kind='worker')

    def test_slot_order_duplicate_and_path_escape_denied(self):
        for field,value in [('env_id',1),('env_id',False),('state','../state.json'),('sha256','x'*64)]:
            r=record(0);r[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):project(r,0)

    def test_nested_metadata_digest_preserves_disagreement(self):
        a=record(0);a['arms']={'F_L':{'group_coverage':{'palm_pixels':2048}}};r=project(a,0)
        self.assertEqual(r['_metadata_sha256'],metadata_digest(a))
        a['arms']['F_L']['group_coverage']['palm_pixels']=2047
        self.assertNotEqual(r['_metadata_sha256'],metadata_digest(a))

    def test_escaped_quotes_unicode_braces_cross_chunks(self):
        records=[record(i) for i in range(32)]
        records[7]['reasons']=['quote"slash\\braces{}[] Unicode 手掌', '\\"\\\\end']
        p=self.packet(records);r=read_index(Evidence(),p,kind='worker',chunk=17)
        self.assertEqual(r['receipts'][7]['_metadata_sha256'],metadata_digest(records[7]))

    def test_duplicate_keys_nonfinite_trailing_and_truncated_denied(self):
        base=json.dumps(dict(qualified=False,receipts=[record(i) for i in range(32)]))
        cases=[base.replace('"qualified": false','"qualified":false,"qualified":false',1),base+' true',base[:-1],
            base.replace('"env_id": 0','"env_id":0,"env_id":0',1),base.replace('"image_count": 0','"image_count":NaN',1),
            base.replace('"qualified": false','"qualified":false,',1)]
        for i,data in enumerate(cases):
            d=self.root/str(i);d.mkdir();p=d/'initial_camera_receipts.json';p.write_text(data)
            with self.subTest(case=i),self.assertRaises(ValueError):read_index(Evidence(),p,kind='worker')

    def test_unrelated_json_retains16MiB_default(self):
        p=self.root/'ordinary.json'
        with p.open('xb') as f:f.truncate(17*1024**2)
        with self.assertRaisesRegex(ValueError,'bounded read'):Evidence().js(p)
        with self.assertRaisesRegex(ValueError,'filename'):read_index(Evidence(),p,kind='worker')

    def test_closed_V1_sources65proof_and_actual_failure2_preserved(self):
        old=Evidence().js(OUTPUT/'V1_PRESERVATION_BEFORE.json')
        for path,expected in old['files'].items():self.assertEqual(fingerprint(path),expected,path)
        link_observations={}
        for path,target in old['links'].items():
            self.assertTrue(Path(path).is_symlink())
            observed=os.readlink(path)
            # CIFS may omit the leading slash for this existing denied fixture.
            # Never follow or rewrite it; all regular bytes/metadata remain exact.
            self.assertEqual(observed.lstrip('/'),target.lstrip('/'))
            with self.assertRaisesRegex(ValueError,'symlink'):Evidence().hash(path)
            link_observations[path]=dict(previous=target,observed=observed,dereferenced=False)
        a=Evidence().js(HERE/'anchors.json');ref=a['preserved_V1_failed_native_reader_wait'];w=Evidence().js(ref['path'],ref['sha256'])
        self.assertEqual((w['actual_wait_exit'],w['raw_wait_status']),(2,512));self.assertEqual(w['child_pid'],4104173)
        self.assertEqual(sha(a['preserved_V1_package']['path']),a['preserved_V1_package']['sha256'])
        write_new(self.root/'PRESERVED_V1.json',dict(regular_files=len(old['files']),denied_symlinks=len(old['links']),
            CPU65_package_preserved=True,actual_native_reader_exit2_preserved=True,
            denied_link_observations=link_observations,native_pass=False))
