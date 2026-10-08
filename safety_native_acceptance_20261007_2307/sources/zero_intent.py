"""Registered target-reference conditions; no physical-state or FIFO mutation."""
MODES = ('joint_reference', 'tight_reference', 'zero_inclusive')


def reference_gap(mode):
    if mode not in MODES:
        raise ValueError('unregistered zero-intent comparison mode')
    return .050 if mode == 'joint_reference' else .010
