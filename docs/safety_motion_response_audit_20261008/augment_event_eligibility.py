"""Distinguish raw distance crossings from loss of a contact exemption."""
import json
import sys
from pathlib import Path

import numpy as np

from bind_first_failures import sha


def main(directory):
    reg = json.loads((directory / 'REGISTRATION.json').read_text())
    root = Path(reg['root'])
    binding = json.loads((directory / 'FIRST_FAILURE_NATIVE_BINDING.json').read_text())
    forecast = json.loads((root / 'forecast_receipts.json').read_text())
    sources = {}
    for event in binding['events']:
        step = event['first_failure_step']
        entry = next(e for e in forecast['chunks'] if e['start'] <= step < e['stop'])
        path = root / entry['path']
        assert sha(path) == entry['sha256']
        sources[entry['path']] = entry['sha256']
        with np.load(path, allow_pickle=False) as data:
            exempt = np.unpackbits(data['exempt'][step - entry['start']], axis=-1, count=9021).astype(bool)
        for row in event['rows']:
            pre_exempt = bool(exempt[event['env'], row['row']])
            raw_crossing = row['pre_gap_m'] >= 0 and row['post_gap_m'] < 0
            row.update(pre_exempt=pre_exempt, post_exempt=False,
                       raw_zero_crossing_during_this_macro=raw_crossing,
                       exemption_revoked=pre_exempt,
                       event_kind='RAW_DISTANCE_CROSSING' if raw_crossing else
                                  'INITIAL_NEGATIVE' if event['initial_geometry_negative'] else
                                  'NEGATIVE_DISTANCE_EXEMPTION_REVOKED')
            if step > 0 and row['pre_gap_m'] < 0:
                assert pre_exempt, (event['env'], row['row'], 'first-score inconsistency')
    binding.update(status='PASS_NATIVE_CLOCK_AND_FIRST_SCORE_ELIGIBILITY_BINDING',
                   first_failure_scope='First scored nonexempt negative geometry, which can follow exemption revocation without a new raw-zero crossing.',
                   eligibility_analysis_sha256=sha(Path(__file__)),
                   original_native_binding_sha256=sha(directory / 'FIRST_FAILURE_NATIVE_BINDING.json'),
                   eligibility_sources=sources)
    (directory / 'FIRST_SCORE_EVENT_AUDIT.json').write_text(json.dumps(binding, indent=2) + '\n')
    print(binding['status'])
    for event in binding['events']:
        print(event['env'], event['first_failure_step'], [r['event_kind'] for r in event['rows']])


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
