"""Exercise the actual offline binding functions on small isolated corruption seams."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent


def functions(filename, names, namespace):
    tree = ast.parse((HERE / filename).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


class ClosedCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name); self.cell = self.root / 'baseline_guard_1'
        self.cell.mkdir(); (self.cell / 'full_forecast').mkdir()
        self.source = self.root / 'analyze_v4.py'; self.source.write_bytes((HERE / 'analyze_v4.py').read_bytes())
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.protocol = dict(status='complete', steps=960, completed_cells=1, args=dict(num_envs=64))
        (self.cell / 'protocol.json').write_text(json.dumps(self.protocol))
        for name in ['cell_001.npz', 'mechanism.npz', 'full_row_identity.json', 'guard_metadata.json', 'project_diagnostics.npz']:
            (self.cell / name).write_bytes(b'original fixture bytes')
        self.receipt = dict(steps=960, rows=9021, chunks=[])
        for i in range(30):
            rel = f'full_forecast/steps_{i*32:04d}_{(i+1)*32:04d}.npz'
            (self.cell / rel).write_bytes(b'original fixture bytes')
            self.receipt['chunks'].append(dict(start=i*32, stop=(i+1)*32, path=rel,
                sha256=hashlib.sha256((self.cell / rel).read_bytes()).hexdigest()))
        (self.cell / 'forecast_receipts.json').write_text(json.dumps(self.receipt))
        self.calls = 0
        self.ns = dict(HERE=self.root, __file__=str(self.source), EXECUTED_SOURCE_SHA256=self.digest,
            json=json, Path=Path, hashlib=hashlib, np=np, load=lambda _: {'sentinel':np.array([1])},
            forecast_audit=self.fake_audit)
        functions('analyze_v4.py', ['sha', 'audit_input_paths', 'check_execution_source', 'forecast_audit_with_receipt'], self.ns)
        self.data = {'sentinel':np.array([1])}

    def tearDown(self): self.tmp.cleanup()

    def fake_audit(self, path, data):
        self.calls += 1
        return dict(marker='audit executed', values_checked=554250240)

    def run_audit(self): return self.ns['forecast_audit_with_receipt'](self.cell, self.data)

    @property
    def cache(self): return self.root / 'closed_cell_audits_v4' / self.digest / (self.cell.name + '.json')

    def mutate_cache(self, transform):
        value = json.loads(self.cache.read_text()); transform(value); self.cache.write_text(json.dumps(value))

    def test_unchanged_complete_inputs_reuse_without_fresh_audit(self):
        first = self.run_audit(); self.assertEqual(first, self.run_audit()); self.assertEqual(self.calls, 1)
        self.assertEqual(len(json.loads(self.cache.read_text())['input_sha256']), 37)

    def test_missing_or_extra_consumed_key_rejected(self):
        self.run_audit(); original = self.cache.read_bytes()
        for transform in [lambda v:v['input_sha256'].pop('project_diagnostics.npz'),
                          lambda v:v['input_sha256'].update({'foreign.npz':'irrelevant'})]:
            self.cache.write_bytes(original); self.mutate_cache(transform)
            with self.assertRaises(AssertionError): self.run_audit()
        self.assertEqual(self.calls, 1)

    def test_cache_identity_and_schema_rejected(self):
        self.run_audit(); original = self.cache.read_bytes()
        for key in ['schema', 'path', 'cell_id', 'analysis_sha256']:
            self.cache.write_bytes(original); self.mutate_cache(lambda v:v.update({key:'wrong'}))
            with self.assertRaises(AssertionError): self.run_audit()

    def test_missing_extra_duplicate_or_foreign_chunk_rejected(self):
        for kind in ['missing', 'extra', 'duplicate', 'foreign']:
            r = copy.deepcopy(self.receipt)
            if kind == 'missing': r['chunks'].pop()
            if kind == 'extra': r['chunks'].append(copy.deepcopy(r['chunks'][0]))
            if kind == 'duplicate': r['chunks'][1] = copy.deepcopy(r['chunks'][0])
            if kind == 'foreign': r['chunks'][1]['path'] = '../foreign.npz'
            (self.cell / 'forecast_receipts.json').write_text(json.dumps(r))
            with self.assertRaises(AssertionError): self.run_audit()
        self.assertEqual(self.calls, 0)

    def test_incomplete_protocol_rejected(self):
        for change in [dict(status='running'), dict(steps=959), dict(completed_cells=0), dict(args=dict(num_envs=63))]:
            (self.cell / 'protocol.json').write_text(json.dumps({**self.protocol, **change}))
            with self.assertRaises(AssertionError): self.run_audit()
        self.assertEqual(self.calls, 0)

    def test_consumed_bytes_changed_after_cache_rejected(self):
        self.run_audit(); (self.cell / 'project_diagnostics.npz').write_bytes(b'changed')
        with self.assertRaises(AssertionError): self.run_audit()

    def test_consumed_bytes_changed_during_audit_rejected(self):
        def mutation(path, data):
            result = self.fake_audit(path, data); (path / 'project_diagnostics.npz').write_bytes(b'changed during audit'); return result
        self.ns['forecast_audit'] = mutation
        with self.assertRaises(AssertionError): self.run_audit()
        self.assertFalse(self.cache.exists())

    def test_source_changed_during_audit_rejected(self):
        def mutation(path, data):
            result = self.fake_audit(path, data); self.source.write_bytes(self.source.read_bytes() + b'\n# changed'); return result
        self.ns['forecast_audit'] = mutation
        with self.assertRaises(AssertionError): self.run_audit()
        self.assertFalse(self.cache.exists())

    def test_stale_caller_dense_array_rejected(self):
        self.data = {'sentinel':np.array([2])}
        with self.assertRaises(AssertionError): self.run_audit()
        self.assertEqual(self.calls, 0)


    def test_cached_stale_caller_dense_array_rejected(self):
        self.run_audit()
        self.data = {'sentinel':np.array([999])}
        with self.assertRaises(AssertionError): self.run_audit()
        self.assertEqual(self.calls, 1)


class ImageBindingTests(unittest.TestCase):
    def setUp(self):
        self.views = {'overview','front','reverse','f_pair','f_opposite_low','f_opposite_high','u_pair','u_opposite_low','u_opposite_high'}
        self.state = dict(env_id=8, step=71, images=[dict(path=f'multiview/env_008/step_0071_{v}.png',sha256='fixture') for v in sorted(self.views)])
        self.receipt = dict(env_id=8, step=71, state='multiview/env_008/step_0071_state.json',images=copy.deepcopy(self.state['images']))
        ns = functions('verify_native_visual_v2.py', ['verify_image_binding'], {})
        self.check = ns['verify_image_binding']

    def test_exact_state_group_accepts(self): self.check(self.state, self.receipt, self.views)

    def test_same_view_names_but_swapped_step_rejected(self):
        for image in self.state['images']: image['path'] = image['path'].replace('0071','0075')
        self.receipt['images'] = copy.deepcopy(self.state['images'])
        with self.assertRaises(AssertionError): self.check(self.state,self.receipt,self.views)

    def test_same_view_names_but_swapped_environment_rejected(self):
        for image in self.state['images']: image['path'] = image['path'].replace('env_008','env_016')
        self.receipt['images'] = copy.deepcopy(self.state['images'])
        with self.assertRaises(AssertionError): self.check(self.state,self.receipt,self.views)

    def test_receipt_state_image_disagreement_rejected(self):
        self.receipt['images'][0]['sha256']='changed'
        with self.assertRaises(AssertionError): self.check(self.state,self.receipt,self.views)

    def test_duplicate_view_rejected(self):
        self.state['images'][-1] = copy.deepcopy(self.state['images'][0]); self.receipt['images'] = copy.deepcopy(self.state['images'])
        with self.assertRaises(AssertionError): self.check(self.state,self.receipt,self.views)

    def test_wrong_state_path_rejected(self):
        self.receipt['state']=self.receipt['state'].replace('0071','0075')
        with self.assertRaises(AssertionError): self.check(self.state,self.receipt,self.views)


if __name__ == '__main__': unittest.main()
