"""Independent original XML and native-limit oracle for every proposal."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np

H=Path(__file__).resolve().parent
OUT=H/'full_motor_operating_domain_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    report=read(OUT/'FULL_MOTOR_DOMAIN_V1.json')
    for name,digest in report['source_SHA256'].items():assert sha(name)==digest,name
    candidate=OUT/'DEVELOPMENT_FULL50_PROPOSALS_V1.npz'
    assert sha(candidate)==report['proposal_SHA256']
    source=H/'fourhand_native_probe_v31'
    reg=read(source/'REGISTRATION_V1.json')
    meta=read(reg['coupled_assets'])
    nativefile=Path(read(source/'RESOURCE_EXECUTION_V1.json')['out'])/'native_parameters.npz'
    checked=[];relations=0
    with np.load(candidate,allow_pickle=False) as z,np.load(nativefile,allow_pickle=False) as native:
        independent=z['independent_names'].tolist()
        assert len(independent)==len(set(independent))==50
        assert len(z['proposal_id'])==report['total_proposals']==20000
        np.testing.assert_array_equal(z['proposal_id'],np.arange(20000))
        for arm,asset in meta['assets'].items():
            names=native[arm+'_joint_names'].tolist()
            hard=native[arm+'_hard_limits'][0]
            speed=native[arm+'_max_velocity'][0]
            q,v,target=z[arm+'_q'],z[arm+'_qd'],z[arm+'_target']
            assert q.shape==v.shape==target.shape==(20000,len(names))
            assert np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(target).all()
            for values in [q,target]:
                assert np.all(values>=hard[:,0]-1e-12) and np.all(values<=hard[:,1]+1e-12)
            assert np.all(abs(v)<=speed+1e-12)
            motor_names=[names[i] for i in native[arm+'_controlled_indices'].reshape(-1)]+asset['independent_motor_joint_names']
            for name in motor_names:
                index=names.index(name);motor=independent.index(arm+'/'+name)
                for full,key in [(q,'independent_q'),(v,'independent_qd'),(target,'independent_target')]:
                    np.testing.assert_array_equal(full[:,index],z[key][:,motor])
            for joint in ET.parse(asset['original_URDF']).findall('joint'):
                mimic=joint.find('mimic')
                if mimic is None or joint.get('name') not in asset['hand_joint_names']:continue
                slave=names.index(joint.get('name'));master=names.index(mimic.get('joint'))
                multiplier=float(mimic.get('multiplier','1'));offset=float(mimic.get('offset','0'))
                for values in [q,target]:np.testing.assert_allclose(values[:,slave],multiplier*values[:,master]+offset,atol=1e-12,rtol=0)
                np.testing.assert_allclose(v[:,slave],multiplier*v[:,master],atol=1e-12,rtol=0)
                relations+=1
            checked.append(dict(arm=arm,proposals=20000,native_coordinates=len(names),motor_coordinates=len(motor_names)))
    assert relations==24 and sum(row['native_coordinates'] for row in checked)==74
    result=dict(status='PASS_INDEPENDENT_ALL20000_ORIGINAL24_RELATIONS_NATIVE74_LIMITS_AND50_BINDINGS',
                checked=checked,original_relations=relations,proposals=20000,
                geometry_or_native_physics_not_exercised=True,native_trials=0,formal_holdout_count=0,
                fullSystem0_accepted=False,source_SHA256={str(path):sha(path) for path in [Path(__file__),candidate,nativefile,OUT/'FULL_MOTOR_DOMAIN_V1.json']})
    target=OUT/'INDEPENDENT_PROPOSAL_ORACLE_V1.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2)+'\n');print(result['status'],flush=True)


if __name__=='__main__':
    main()
