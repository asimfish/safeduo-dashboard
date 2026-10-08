"""Object-pose task success criteria (R17 S9, 2026-08-20).

Verdicts read the OBJECT pose trace, not EE waypoints (the owner requirement:
a task only counts when the object actually rose / landed in the target zone
/ ended up following the receiving hand). All traces are (T, 3) world
positions on a uniform control grid (dt seconds/step, env-local frame);
numpy arrays or torch tensors accepted.

Three judges, one per S9 family. Each returns
    {"success": bool, "reason": str, ...metrics...}
with every threshold surfaced in the result for the report tables. Params
default to the S9 choreography numbers but are plain dataclasses so the
handover direction / place zone are reusable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected (T,3) trace, got {a.shape}")
    return a


# ---------------------------------------------------------------------------
# grasp: lift the object >= lift_m and KEEP it up
# ---------------------------------------------------------------------------


@dataclass
class GraspParams:
    lift_m: float = 0.10        # required rise above the initial object z
    hold_ratio: float = 0.8     # held-frame threshold = hold_ratio * lift_m
    hold_frac: float = 0.8      # fraction of post-crossing frames staying held


def judge_grasp(obj_pos, params: "GraspParams | None" = None) -> dict:
    p = params or GraspParams()
    obj = _np(obj_pos)
    lift = obj[:, 2] - obj[0, 2]
    max_lift = float(lift.max())
    res = {"max_lift_m": round(max_lift, 4),
           "final_lift_m": round(float(lift[-1]), 4),
           "lift_required_m": p.lift_m}
    idx = np.flatnonzero(lift >= p.lift_m)
    if idx.size == 0:
        return {"success": False,
                "reason": f"never lifted {p.lift_m:.2f} m "
                          f"(max {max_lift:.3f})", **res}
    tail = lift[idx[0]:]
    held = float((tail >= p.hold_ratio * p.lift_m).mean())
    res["held_frac"] = round(held, 4)
    if held < p.hold_frac:
        return {"success": False,
                "reason": f"lift not sustained (held {held:.2f} "
                          f"< {p.hold_frac})", **res}
    return {"success": True, "reason": "lifted and held", **res}


# ---------------------------------------------------------------------------
# pick_place: lift, move >= min_disp, come to rest inside the place zone
# ---------------------------------------------------------------------------


@dataclass
class PlaceParams:
    place_xy: tuple = (0.30, -0.45)
    radius_m: float = 0.10      # place-zone radius around place_xy
    min_disp_m: float = 0.20    # start->final planar displacement
    min_lift_m: float = 0.05    # must have been picked up at some point
    rest_z: float = 0.825       # expected resting center z (table + half)
    rest_z_tol_m: float = 0.04
    settle_s: float = 0.5       # final window that must be quasi-static
    settle_move_m: float = 0.01


def judge_pick_place(obj_pos, dt: float,
                     params: "PlaceParams | None" = None) -> dict:
    p = params or PlaceParams()
    obj = _np(obj_pos)
    lift = obj[:, 2] - obj[0, 2]
    disp = float(np.linalg.norm(obj[-1, :2] - obj[0, :2]))
    err = float(np.linalg.norm(obj[-1, :2] - np.asarray(p.place_xy)))
    k = max(1, int(round(p.settle_s / dt)))
    settle = float(np.linalg.norm(obj[-1] - obj[-min(k, len(obj))], axis=-1))
    res = {"max_lift_m": round(float(lift.max()), 4),
           "displacement_m": round(disp, 4),
           "place_err_m": round(err, 4),
           "final_z": round(float(obj[-1, 2]), 4),
           "settle_move_m": round(settle, 5),
           "place_xy": tuple(p.place_xy), "radius_m": p.radius_m,
           "min_disp_m": p.min_disp_m}
    if float(lift.max()) < p.min_lift_m:
        return {"success": False, "reason": "object never picked up", **res}
    if disp < p.min_disp_m:
        return {"success": False,
                "reason": f"moved {disp:.3f} < {p.min_disp_m:.2f} m", **res}
    if err > p.radius_m:
        return {"success": False,
                "reason": f"final {err:.3f} m from place target "
                          f"(> {p.radius_m:.2f})", **res}
    if abs(obj[-1, 2] - p.rest_z) > p.rest_z_tol_m:
        return {"success": False,
                "reason": f"not resting on table (z {obj[-1, 2]:.3f} vs "
                          f"{p.rest_z:.3f}+-{p.rest_z_tol_m})", **res}
    if settle > p.settle_move_m:
        return {"success": False,
                "reason": f"still moving at the end ({settle:.3f} m in "
                          f"last {p.settle_s}s)", **res}
    return {"success": True, "reason": "placed in zone at rest", **res}


# ---------------------------------------------------------------------------
# handover: after the transfer the object follows the RECEIVER hand
# ---------------------------------------------------------------------------


@dataclass
class HandoverParams:
    follow_dist_m: float = 0.45   # object-to-receiver-flange leash
    follow_frac: float = 0.8      # fraction of post-transfer frames on leash
    min_travel_m: float = 0.10    # object planar travel with the receiver
    receiver_x_sign: float = -1.0  # receiver side of the seam (U = -x)
    seam_margin_m: float = 0.02   # final obj x must be this far on that side
    min_pre_lift_m: float = 0.05  # giver must have picked the object up


def judge_handover(obj_pos, ee_receiver, dt: float, transfer_t: float,
                   params: "HandoverParams | None" = None) -> dict:
    """transfer_t = time (s) at which the receiver attach event fired
    (meta object_events); post-transfer window starts one control step
    later so the attach frame itself is not judged."""
    p = params or HandoverParams()
    obj, ee_r = _np(obj_pos), _np(ee_receiver)
    if obj.shape != ee_r.shape:
        raise ValueError("obj / receiver traces length mismatch")
    k = min(len(obj) - 1, int(round(transfer_t / dt)) + 1)
    pre_lift = float((obj[:k, 2] - obj[0, 2]).max()) if k > 1 else 0.0
    d_recv = np.linalg.norm(obj[k:] - ee_r[k:], axis=-1)
    follow = float((d_recv <= p.follow_dist_m).mean()) if d_recv.size else 0.0
    travel = float(np.linalg.norm(obj[-1, :2] - obj[k, :2]))
    side = float(obj[-1, 0] * p.receiver_x_sign)
    res = {"transfer_t": round(float(transfer_t), 3),
           "pre_lift_m": round(pre_lift, 4),
           "follow_frac": round(follow, 4),
           "follow_dist_max_m": round(float(d_recv.max()), 4) if d_recv.size else None,
           "post_travel_m": round(travel, 4),
           "final_x": round(float(obj[-1, 0]), 4),
           "follow_dist_m": p.follow_dist_m, "min_travel_m": p.min_travel_m}
    if pre_lift < p.min_pre_lift_m:
        return {"success": False, "reason": "giver never lifted the object",
                **res}
    if follow < p.follow_frac:
        return {"success": False,
                "reason": f"object not following receiver "
                          f"({follow:.2f} < {p.follow_frac})", **res}
    if travel < p.min_travel_m:
        return {"success": False,
                "reason": f"object did not travel with receiver "
                          f"({travel:.3f} < {p.min_travel_m:.2f} m)", **res}
    if side < p.seam_margin_m:
        return {"success": False,
                "reason": f"object did not end on receiver side "
                          f"(x={obj[-1, 0]:+.3f})", **res}
    return {"success": True, "reason": "transferred and carried by receiver",
            **res}


# ---------------------------------------------------------------------------
# dispatch from the npz meta success block (task_record_s9 writes it)
# ---------------------------------------------------------------------------


def judge_from_meta(success_cfg: dict, obj_pos, dt: float,
                    ee_receiver=None) -> dict:
    kind = success_cfg["kind"]
    if kind == "grasp":
        return judge_grasp(obj_pos, GraspParams(
            lift_m=float(success_cfg.get("lift_m", 0.10))))
    if kind == "pick_place":
        return judge_pick_place(obj_pos, dt, PlaceParams(
            place_xy=tuple(success_cfg["place_xy"]),
            radius_m=float(success_cfg.get("radius_m", 0.10)),
            min_disp_m=float(success_cfg.get("min_disp_m", 0.20)),
            rest_z=float(success_cfg.get("rest_z", 0.825))))
    if kind == "handover":
        if ee_receiver is None:
            raise ValueError("handover verdict needs the receiver EE trace")
        return judge_handover(obj_pos, ee_receiver, dt,
                              float(success_cfg["transfer_t"]),
                              HandoverParams(
                                  follow_dist_m=float(
                                      success_cfg.get("follow_dist_m", 0.45)),
                                  min_travel_m=float(
                                      success_cfg.get("min_travel_m", 0.10)),
                                  # R24: relay_chain 终点收方是 F_R（x>0 侧）
                                  # —— 缺省 -1.0 保持 legacy handover 行为
                                  receiver_x_sign=float(
                                      success_cfg.get("receiver_x_sign", -1.0))))
    raise ValueError(f"unknown success kind {kind!r}")
