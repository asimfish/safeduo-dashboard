"""Lossless independent-coordinate view of the existing proposals, no new draws."""
from pathlib import Path
import hashlib
import json
import numpy as np

P = Path(__file__).resolve().parent/'full_motor_operating_domain_v2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    source = P/'DEVELOPMENT_FULL50_PROPOSALS_V2.npz'
    target = P/'DEVELOPMENT_INDEPENDENT50_V2.npz'
    report = P/'INDEPENDENT50_EXPORT_V2.json'
    assert not target.exists() and not report.exists()
    domain = json.loads((P/'FULL_MOTOR_DOMAIN_V2.json').read_text())
    assert sha(source) == domain['proposal_SHA256']
    assert json.loads((P/'INDEPENDENT_PROPOSAL_ORACLE_ACTUAL_EXIT_V2.json').read_text())['actual_outer_exit'] == 0
    with np.load(source, allow_pickle=False) as original:
        keys = [k for k in original.files if k.startswith('independent_') or k in ['proposal_id', 'proposal_stratum', 'mixed_velocity_scale']]
        assert len(original['proposal_id']) == 40000
        np.savez_compressed(target, **{k: original[k] for k in keys}, full_mapped_proposal_sha256=np.asarray(sha(source)))
        with np.load(target, allow_pickle=False) as exported:
            for k in keys:
                assert exported[k].dtype == original[k].dtype
                np.testing.assert_array_equal(exported[k], original[k])
            assert exported['full_mapped_proposal_sha256'].item() == sha(source)
    report.write_text(json.dumps(dict(status='PASS_LOSSLESS_ALL40000_INDEPENDENT50_INPUT_EXPORT',
        original_mapped_archive_sha256=sha(source), export_sha256=sha(target), export_bytes=target.stat().st_size,
        all_exported_arrays_exactly_equal=True, exported_keys=keys, proposal_draws_added=0,
        redundant_native74_arrays_omitted_not_changed=True, reconstruct_with_original24_URDF_relations=True,
        native_trajectories_added=0, formal_holdout_count=0, producer_actual_outer_exit=None, fullSystem0_accepted=False,
        source_sha256={str(f):sha(f) for f in [Path(__file__), source, P/'INDEPENDENT_PROPOSAL_ORACLE_V2.json']}), indent=2)+'\n')
    print('LOSSLESS_INDEPENDENT50_EXPORT_PASS', target.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
