import os
from evidence_io import require

def terminal_numbers(w):
    require(w['schema']=='astra.full74.parent_native_wait.v1' and w['backend']=='isaac_physx_native','native wait schema/backend')
    for key in ('child_pid','waited_pid','raw_wait_status','actual_wait_exit'):
        require(type(w[key]) is int,'terminal native integer '+key)
    raw=w['raw_wait_status']
    require(w['waitpid_observed'] is True and w['child_pid']==w['waited_pid']>0,'actual waited native PID')
    require(0<=raw<65536 and (os.WIFEXITED(raw) or os.WIFSIGNALED(raw)) and
            os.waitstatus_to_exitcode(raw)==w['actual_wait_exit'],'raw wait/exit mismatch or nonterminal wait')
    require(w['wait_mechanism']=='os.wait4(owned_pid, WNOHANG)','real parent wait4 required')
    own=[x for x in w['owned_identities'] if x['pid']==w['child_pid']]
    require(len(own)==1 and all(x['live'] is False for x in w['owned_identities']),'owned native processes not terminal')
    own=own[0];require(type(own['startticks']) is int and own['startticks']>0 and
        own['enrollment']['kind']=='direct_fork_handshake' and own['enrollment']['parent_pid']==own['ppid']>0,'native enrollment')
    return own

