"""Deterministic six-pair coverage design for the safety battery.

The ordinary uniform command tape is useful as an ablation, but it is not a
coverage design: most windows never bring any arm pair into the warning band.
This module describes the missing experiment as explicit cells so a report can
separate attempted, exposed, near-contact and violating episodes.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass


ARM_KEYS = ("F_L", "F_R", "U_L", "U_R")
ARM_PAIRS = tuple(itertools.combinations(range(len(ARM_KEYS)), 2))
PAIR_NAMES = tuple(f"{ARM_KEYS[i]}-{ARM_KEYS[j]}" for i, j in ARM_PAIRS)
DIRECTIONS = ("approach", "recede", "tangent")

# These are EE-centre target gaps.  The evaluator still uses the sphere
# margin as its safety oracle; the target gap is deliberately kept in the
# metadata so a failed reach is visible rather than silently counted as an
# exposed trial.
DISTANCE_BANDS = (
    ("warn", 0.12),
    ("near", 0.08),
    ("contact", 0.05),
    ("penetration", 0.025),
)


@dataclass(frozen=True)
class CoverageCell:
    """One scheduled episode condition."""

    pair_index: int
    pair: str
    distance_band: str
    target_gap_m: float
    direction: str
    amp: float
    seed: int
    env_slot: int

    def as_dict(self) -> dict:
        return asdict(self)


def _pair_order(seed: int) -> tuple[int, ...]:
    # Keep pair slots fixed.  This makes the coverage guarantee auditable:
    # slot j is always pair j, while seed changes the band/direction cell.
    # Randomising pair order would make a short seed subset accidentally miss
    # a pair x band x direction cell.
    del seed
    return tuple(range(len(PAIR_NAMES)))


def build_schedule(num_envs: int, amp: float, seed: int) -> list[CoverageCell]:
    """Return a balanced, deterministic schedule for one simulator window.

    Every six consecutive slots contain all six arm pairs.  The distance band
    and relative-motion direction advance on different cycles, so repeating
    the window over seeds covers the full pair x band x direction matrix.
    """
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    order = _pair_order(int(seed))
    # Rotate the pair slots between seeds.  The rotation is chosen so the
    # short 32-environment window stays balanced even when it ends mid-block;
    # over seeds 0..3 every pair receives all 12 band/direction combinations.
    shift = (5 * int(seed)) % len(order)
    out = []
    for slot in range(num_envs):
        pair_index = order[(slot + shift) % len(order)]
        combo_index = (int(seed) * len(DIRECTIONS) * len(DISTANCE_BANDS)
                       // 2 + (slot + 5 * int(seed)) // len(order)) % (len(DISTANCE_BANDS) * len(DIRECTIONS))
        band_index = combo_index // len(DIRECTIONS)
        direction_index = combo_index % len(DIRECTIONS)
        band, gap = DISTANCE_BANDS[band_index]
        out.append(CoverageCell(
            pair_index=pair_index,
            pair=PAIR_NAMES[pair_index],
            distance_band=band,
            target_gap_m=gap,
            direction=DIRECTIONS[direction_index],
            amp=float(amp), seed=int(seed), env_slot=slot,
        ))
    return out


def schedule_counts(cells: list[CoverageCell]) -> dict:
    """Count scheduled trials by pair, band and direction."""
    result = {
        "attempted": len(cells),
        "by_pair": {name: 0 for name in PAIR_NAMES},
        "by_distance_band": {name: 0 for name, _ in DISTANCE_BANDS},
        "by_direction": {name: 0 for name in DIRECTIONS},
        "by_pair_band_direction": {},
    }
    for cell in cells:
        result["by_pair"][cell.pair] += 1
        result["by_distance_band"][cell.distance_band] += 1
        result["by_direction"][cell.direction] += 1
        key = f"{cell.pair}|{cell.distance_band}|{cell.direction}"
        result["by_pair_band_direction"][key] = result["by_pair_band_direction"].get(key, 0) + 1
    return result


def missing_schedule_cells(cells: list[CoverageCell]) -> list[dict]:
    """Return pair/band/direction combinations absent from a schedule."""
    observed = {(c.pair, c.distance_band, c.direction) for c in cells}
    return [
        {"pair": pair, "distance_band": band, "direction": direction}
        for pair in PAIR_NAMES
        for band, _ in DISTANCE_BANDS
        for direction in DIRECTIONS
        if (pair, band, direction) not in observed
    ]
