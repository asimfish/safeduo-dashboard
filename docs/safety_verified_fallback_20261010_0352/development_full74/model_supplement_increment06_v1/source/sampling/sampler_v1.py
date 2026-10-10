"""Same full74 arithmetic; NEW prospective development seeds only."""
from bindings_v1 import *
DEV_SEEDS=tuple(IDENTITY["seeds"])
RESERVED_FRESH_SEEDS=tuple(IDENTITY["reserved_holdout_seeds_excluded"])

def sample_batch(layout, seed, start):
    """Per-proposal RNG identity: independent of batch partition and qualification."""
    require(seed in DEV_SEEDS and seed not in RESERVED_FRESH_SEEDS, 'development identity required')
    require(isinstance(start, int) and start >= 0 and start % N == 0, '32-lane aligned start')
    b, v = layout['bounds'], layout['vmax']
    q, qd = np.empty((N, D), np.float32), np.empty((N, D), np.float32)
    families = np.empty(N, np.uint8)
    for lane in range(N):
        pid = start + lane
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, pid])))
        family, face = pid % 4, pid // 4
        unit = rng.beta(.35, .35, D) if family == 1 else rng.random(D)
        q[lane] = (b[:, 0].astype(np.float64) + unit * (b[:, 1].astype(np.float64) - b[:, 0])).astype(np.float32)
        qd[lane] = (rng.uniform(-1, 1, D) * v.astype(np.float64)).astype(np.float32)
        joint, kind = face % D, (face // D) % 4
        if family == 2:
            edge = b[joint, kind % 2]
            q[lane, joint] = edge if kind < 2 else np.nextafter(edge, b[joint, 1 - kind % 2])
        if family == 3:
            edge = v[joint] * np.float32(-1 if kind % 2 == 0 else 1)
            qd[lane, joint] = edge if kind < 2 else np.nextafter(edge, np.float32(0))
        families[lane] = family
    require(np.all(q >= b[:, 0]) and np.all(q <= b[:, 1]) and np.all(np.abs(qd) <= v), 'sample outside original limits')
    return dict(q74=q, qd74=qd, family=families)
