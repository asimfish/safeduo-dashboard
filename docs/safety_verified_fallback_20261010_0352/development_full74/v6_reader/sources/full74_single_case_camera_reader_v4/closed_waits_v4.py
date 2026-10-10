"""Shared closed-parent CPU/native wait evidence, never a native-success gate."""
import os
from evidence_io import require


def closed_wait(wait, expected):
    require(wait['schema'] in ('astra.full74.parent_native_wait.v1', 'astra.full74.owned_cpu_wait.v1'), 'reviewed wait schema')
    for key in ('child_pid', 'waited_pid', 'actual_wait_exit', 'raw_wait_status'):
        require(type(wait[key]) is int, 'typed terminal field ' + key)
    require(wait['waitpid_observed'] is True and wait['child_pid'] == wait['waited_pid'] > 0 and
            wait['wait_mechanism'] == 'os.wait4(owned_pid, WNOHANG)' and
            os.waitstatus_to_exitcode(wait['raw_wait_status']) == wait['actual_wait_exit'] == expected,
            'actual owned wait4 and raw exit agreement')
    require(wait['resource_abort'] is None and wait.get('error') is None and not wait['signals'], 'clean terminal observation')
    require(all(v['live'] is False for v in wait['owned_identities']), 'all recorded owned tasks reaped')
    own = [v for v in wait['owned_identities'] if v['pid'] == wait['child_pid']]
    require(len(own) == 1 and type(own[0]['startticks']) is int and own[0]['startticks'] > 0 and
            own[0]['enrollment']['kind'] == 'direct_fork_handshake' and
            own[0]['enrollment']['parent_pid'] == own[0]['ppid'], 'owned root PID/startticks and direct fork provenance')
    return own[0]
