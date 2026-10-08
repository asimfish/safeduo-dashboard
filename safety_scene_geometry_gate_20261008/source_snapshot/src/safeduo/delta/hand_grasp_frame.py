"""Calibration-driven hand grasp frame (R27 S1, 2026-09-05).

Why this exists: the R23 grasp generator assumed the pinch point sits on the
flange axis (``tcp_offset = (0, 0, 0.165)``) and that the thumb closes along
flange +X.  Measuring the real five-finger hands in simulation
(``tools/r27_hand_aperture_calib.py`` -> ``configs/hand_grasp_calib_v7.json``)
showed both assumptions are wrong by several centimetres: the F2 pinch point is
~6 cm off-axis in +X and ~4 cm in Y and the F2 closes along flange Y, the DFX
closes along -X, and the right DFX sits 2 cm further out than the left.  The
old trajectories therefore closed the fingers *beside* the object and relied on
a kinematic attach; this module gives the trajectory designer the measured
pinch point and closing direction for the aperture an object actually needs.

Everything is plain numpy; the Isaac dependency lives only in the calibration
tool that produced the JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIB_DEFAULT = "hand_grasp_calib_v7.json"
F_MAX_DEFAULT = 0.70        # beyond this the fingers collide with each other
                            # (q_err_mean jumps to 0.02-0.1 rad in the sweep)


@dataclass(frozen=True)
class HandGraspFrame:
    """Flange-frame grasp geometry of one hand for one object width.

    pinch_mid     midpoint of the thumb / index pads at ``f_contact`` (flange frame)
    closing_dir   unit vector thumb pad -> index pad at ``f_contact`` (flange frame)
    approach_dir  unit vector: flange +Z (hand extension) with its component
                  along ``closing_dir`` removed -- the direction the pads move
                  onto the object when the wrist descends
    f_contact     HandDriver close fraction at which the pad aperture equals
                  ``width_m + contact_margin`` (pads just touch the object)
    f_squeeze     close fraction to command (contact + extra; the object blocks
                  the fingers so the position error becomes squeeze force)
    """

    arm: str
    width_m: float
    f_contact: float
    f_squeeze: float
    pinch_mid: tuple
    closing_dir: tuple
    approach_dir: tuple
    pad_aperture_m: float
    close_ramp_s: float = 0.6   # R27 S2: finger close ramp for this grasp
    # R27 DFX: thumb squeeze fraction (None = thumb follows the fingers). The
    # RH56DFX pinches thumb-tip against finger-tips and its fingers retract
    # along their own axis once they curl further, so the fingers park at
    # contact and only the thumb keeps travelling.
    f_squeeze_thumb: "float | None" = None

    def meta(self) -> dict:
        return {
            "arm": self.arm, "width_m": round(self.width_m, 4),
            "f_contact": round(self.f_contact, 4), "f_squeeze": round(self.f_squeeze, 4),
            "pinch_mid": [round(float(v), 5) for v in self.pinch_mid],
            "closing_dir": [round(float(v), 5) for v in self.closing_dir],
            "approach_dir": [round(float(v), 5) for v in self.approach_dir],
            "pad_aperture_m": round(self.pad_aperture_m, 4),
            "close_ramp_s": round(self.close_ramp_s, 3),
            "f_squeeze_thumb": (None if self.f_squeeze_thumb is None
                                else round(self.f_squeeze_thumb, 4)),
        }


def load_hand_calib(path: "str | Path | None" = None) -> dict:
    if path is None or str(path) == "v7":
        from safeduo.configs import CONFIG_DIR
        path = CONFIG_DIR / CALIB_DEFAULT
    with open(path) as f:
        return json.load(f)


def pad_curve(calib: dict, arm: str) -> list:
    """[(frac, pad_aperture_thumb_index, thumb_pad(3), index_pad(3), index_knuckle(3))]
    sorted by frac (knuckle = proximal link origin; falls back to the pad
    minus flange +Z for calibration files without it)."""
    rows = []
    for r in calib["hands"][arm]["sweep"]:
        ap = r.get("pad_aperture_thumb_index_m")
        if ap is None:
            continue
        pad = np.asarray(r["index_pad"], dtype=np.float64)
        kn = r.get("index_knuckle")
        kn = (np.asarray(kn, dtype=np.float64) if kn is not None
              else pad - np.array([0.0, 0.0, 0.07]))
        rows.append((float(r["frac"]), float(ap),
                     np.asarray(r["thumb_pad"], dtype=np.float64), pad, kn))
    rows.sort(key=lambda t: t[0])
    return rows


def frame_for_width(calib: dict, arm: str, width_m: float,
                    contact_margin: float = 0.004, squeeze_extra: float = 0.15,
                    f_max: float = F_MAX_DEFAULT,
                    close_ramp_s: float = 0.6,
                    approach_mode: str = "flange_z",
                    thumb_extra: "float | None" = None) -> HandGraspFrame:
    """Interpolate the calibration sweep to the fraction whose pad aperture
    equals ``width_m + contact_margin`` and return the grasp frame there.

    The aperture is monotonically decreasing on the usable part of the sweep;
    only samples up to ``f_max`` are used because beyond it the fingers jam on
    each other and the aperture readings stop being a grasp geometry.
    """
    rows = [r for r in pad_curve(calib, arm) if r[0] <= f_max + 1e-9]
    if len(rows) < 2:
        raise ValueError(f"{arm}: calibration sweep too short")
    target = float(width_m) + float(contact_margin)
    if target > rows[0][1]:
        raise ValueError(f"{arm}: object width {width_m:.3f} m exceeds the fully "
                         f"open pad aperture {rows[0][1]:.3f} m")
    for (f0, a0, t0, i0, k0), (f1, a1, t1, i1, k1) in zip(rows[:-1], rows[1:]):
        if a1 <= target <= a0:
            w = 0.0 if abs(a0 - a1) < 1e-9 else (a0 - target) / (a0 - a1)
            f_c = f0 + w * (f1 - f0)
            thumb = t0 + w * (t1 - t0)
            index = i0 + w * (i1 - i0)
            knuckle = k0 + w * (k1 - k0)
            break
    else:
        # narrower than the smallest usable aperture: pinch at f_max
        f_c, _, thumb, index, knuckle = rows[-1]
    mid = 0.5 * (thumb + index)
    c = index - thumb
    c = c / np.linalg.norm(c)
    if approach_mode == "finger":
        # R27 S2: descend along the FINGER direction (knuckle -> pad) so the
        # curled finger hangs beside the far face instead of sweeping over
        # the object's top edge on its way to the pinch point
        ref = index - knuckle
    elif approach_mode == "flange_z":
        ref = np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError(f"unknown approach_mode {approach_mode!r}")
    a = ref - c * float(ref @ c)
    a = a / np.linalg.norm(a)
    return HandGraspFrame(
        arm=arm, width_m=float(width_m), f_contact=float(f_c),
        f_squeeze=float(min(f_c + squeeze_extra, f_max)),
        pinch_mid=tuple(float(v) for v in mid),
        closing_dir=tuple(float(v) for v in c),
        approach_dir=tuple(float(v) for v in a),
        pad_aperture_m=float(target),
        close_ramp_s=float(close_ramp_s),
        f_squeeze_thumb=(None if thumb_extra is None
                         else float(min(f_c + thumb_extra, 0.85))),
    )


def rot_from_hand_frame(closing_w, approach_w, frame: HandGraspFrame) -> np.ndarray:
    """Flange rotation R (world <- flange) such that the hand's measured closing
    direction maps onto ``closing_w`` and its approach direction onto
    ``approach_w`` (both world unit vectors; closing is re-orthogonalised
    against approach exactly like the legacy generator does)."""
    a_w = np.asarray(approach_w, dtype=np.float64)
    a_w = a_w / np.linalg.norm(a_w)
    c_w = np.asarray(closing_w, dtype=np.float64)
    c_w = c_w - a_w * float(c_w @ a_w)
    n = np.linalg.norm(c_w)
    if n < 1e-9:
        raise ValueError("closing and approach are collinear")
    c_w = c_w / n
    b_w = np.cross(a_w, c_w)
    c_f = np.asarray(frame.closing_dir, dtype=np.float64)
    a_f = np.asarray(frame.approach_dir, dtype=np.float64)
    a_f = a_f - c_f * float(a_f @ c_f)
    a_f = a_f / np.linalg.norm(a_f)
    b_f = np.cross(a_f, c_f)
    basis_w = np.stack([c_w, b_w, a_w], axis=1)
    basis_f = np.stack([c_f, b_f, a_f], axis=1)
    return basis_w @ basis_f.T


# ---------------------------------------------------------------------------
# R27 box co-lift (2026-09-06): palm-press frame. A 15x20x10 cm box exceeds
# any pinch aperture, so the two F hands clamp it between their palms; what
# the designer needs is where the palm surface is in the flange frame and
# which way it faces.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PalmFrame:
    """palm_center  point on the palm surface (flange frame)
    palm_normal  unit vector pointing out of the palm surface, i.e. the
                 direction the fingers curl towards (flange frame)
    finger_dir   unit vector knuckle -> open fingertip (flange frame)"""

    arm: str
    palm_center: tuple
    palm_normal: tuple
    finger_dir: tuple

    def meta(self) -> dict:
        return {"arm": self.arm,
                "palm_center": [round(float(v), 5) for v in self.palm_center],
                "palm_normal": [round(float(v), 5) for v in self.palm_normal],
                "finger_dir": [round(float(v), 5) for v in self.finger_dir]}


def palm_frame(calib: dict, arm: str, surface_offset: float = 0.02,
               wrist_back: float = 0.035) -> PalmFrame:
    """Palm surface from the aperture sweep: finger_dir = knuckle -> open index
    pad; palm_normal = the open->half-closed pad displacement with its
    finger_dir component removed (fingers curl onto the palm); palm_center =
    the index/middle knuckle midpoint moved ``wrist_back`` towards the wrist
    and ``surface_offset`` out along the normal (palm thickness)."""
    rows = {round(float(r["frac"]), 2): r for r in calib["hands"][arm]["sweep"]}
    r0 = rows[min(rows)]
    r_half = rows[min(rows, key=lambda f: abs(f - 0.5))]
    ik = np.asarray(r0["index_knuckle"], dtype=np.float64)
    mk = np.asarray(r0.get("middle_knuckle", r0["index_knuckle"]), dtype=np.float64)
    pad0 = np.asarray(r0["index_pad"], dtype=np.float64)
    pad_half = np.asarray(r_half["index_pad"], dtype=np.float64)
    f_dir = pad0 - ik
    f_dir = f_dir / np.linalg.norm(f_dir)
    n = pad_half - pad0
    n = n - f_dir * float(n @ f_dir)
    n = n / np.linalg.norm(n)
    center = 0.5 * (ik + mk) - wrist_back * f_dir + surface_offset * n
    return PalmFrame(arm, tuple(center.tolist()), tuple(n.tolist()), tuple(f_dir.tolist()))


def rot_from_dirs(primary_f, secondary_f, primary_w, secondary_w) -> np.ndarray:
    """Flange rotation R (world <- flange) mapping the flange-frame ``primary``
    direction exactly onto its world counterpart and the ``secondary`` one as
    closely as the right angle allows (re-orthogonalised on both sides)."""
    def _basis(p, s):
        p = np.asarray(p, dtype=np.float64)
        p = p / np.linalg.norm(p)
        s = np.asarray(s, dtype=np.float64)
        s = s - p * float(s @ p)
        ns = np.linalg.norm(s)
        if ns < 1e-9:
            raise ValueError("primary and secondary directions are collinear")
        s = s / ns
        return np.stack([p, s, np.cross(p, s)], axis=1)
    return _basis(primary_w, secondary_w) @ _basis(primary_f, secondary_f).T


def flange_for_palm(pf: PalmFrame, palm_point_w, palm_normal_w, finger_dir_w) -> tuple:
    """(R, t) of the flange that puts the palm surface centre at
    ``palm_point_w`` facing ``palm_normal_w`` with the fingers along
    ``finger_dir_w``."""
    R = rot_from_dirs(pf.palm_normal, pf.finger_dir, palm_normal_w, finger_dir_w)
    t = np.asarray(palm_point_w, dtype=np.float64) - R @ np.asarray(pf.palm_center, dtype=np.float64)
    return R, t
