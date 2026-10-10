"""Independent original-URDF residual and native root-reference identity audit."""
from pathlib import Path
import argparse, hashlib, json, math, xml.etree.ElementTree as ET
from audit_native_mimic_window_v5 import main as audit_original_relations
H = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(version):
    p = H/f'fourhand_native_probe_v{version}'
    target = p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json'
    assert not target.exists() and not (p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V5.json').exists()
    reg = json.loads((p/'REGISTRATION_V1.json').read_text())
    for f, digest in reg['source_sha256'].items(): assert sha(f) == digest, f
    raw = Path(json.loads((p/'RESOURCE_EXECUTION_V1.json').read_text())['out'])
    identity = json.loads((raw/'native_mimic_identity.json').read_text())['relations']
    meta = json.loads(Path(reg['coupled_assets']).read_text())
    rows = []
    for arm, asset in meta['assets'].items():
        original = {j.get('name'): (j.find('mimic').get('joint'),
                     float(j.find('mimic').get('multiplier','1')),
                     float(j.find('mimic').get('offset','0')))
                    for j in ET.parse(asset['original_URDF']).findall('joint')
                    if j.find('mimic') is not None and j.get('name') in asset['hand_joint_names']}
        assert len(original) == len(identity[arm]) == 6
        def affine(name, seen=()):
            assert name not in seen
            if name not in original:
                assert name in asset['independent_motor_joint_names']
                return name, 1., 0.
            parent, slope, offset = original[name]
            root, scale, bias = affine(parent, (*seen,name))
            return root, slope*scale, slope*bias+offset
        native = {r['slave']:r for r in identity[arm]}
        assert set(native) == set(original)
        for slave in original:
            root, scale, bias = affine(slave); r = native[slave]
            assert r['master'] == root
            assert abs(r['gearing']+scale) <= 1e-6
            assert abs(r['offset_degrees']+math.degrees(bias)) <= 1e-5
            prefix = '/World/envs/env_0/'+arm+'/'
            assert r['prim'].startswith(prefix) and r['prim'].rsplit('/',1)[1] == slave
            assert r['reference'].startswith(prefix) and r['reference'].rsplit('/',1)[1] == root
            rows.append(dict(arm=arm, slave=slave, root=root, slope=scale, offset_rad=bias,
                             actual_native_identity=r))
    assert len(rows) == 24
    # Reconstruct all 24 ORIGINAL chained relations from every native q/qd;
    # root-reference metadata never replaces the original residual oracle.
    audit_original_relations(version)
    result = json.loads((p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V5.json').read_text())
    result.update(status='CLOSED_INDEPENDENT_ORIGINAL_URDF_AND_NATIVE_ROOT_REFERENCE_WINDOW_AUDIT',
                  native_equivalent_root_reference_identity_verified=True,
                  original_relation_oracle_unchanged=True, native_identity_relations=rows)
    result['source_SHA256'].update({str(Path(__file__)):sha(__file__),
                                   str(p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V5.json'):sha(p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V5.json')})
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'], 'model_pass', result['model_window_pass'], flush=True)

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('version',type=int); main(ap.parse_args().version)
