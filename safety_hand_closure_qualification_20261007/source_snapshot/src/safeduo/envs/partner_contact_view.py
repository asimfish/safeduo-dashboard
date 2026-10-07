"""Exact one-to-one PhysX hand/object contact attribution for D-032.

Each registered hand rigid body owns a separate filtered ``RigidContactView``
whose only filter is the dynamic plank.  Detailed contact buffers are sliced by
their reported count/start pair; net sensor force is intentionally never used
because it cannot establish partner attribution.

Isaac/Omni imports are lazy.  The collector can therefore be contract-tested on
CPU by injecting a simulation-view-compatible object.
"""

from __future__ import annotations

import importlib
import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from safeduo.envs.contact_gated_grasp import Contact


OBJECT_BODY_PATH = "/World/envs/env_0/Obj_plank"
EXACT_HAND_BODY_PATHS: dict[str, str] = {
    "F_L": "/World/envs/env_0/F_L/f2_hand/left_hand_base",
    "F_R": "/World/envs/env_0/F_R/f2_hand/right_hand_base",
    "U_L": "/World/envs/env_0/U_L/f2_hand/left_hand_base",
    "U_R": "/World/envs/env_0/U_R/f2_hand/right_hand_base",
}

Vector3 = tuple[float, float, float]


class PartnerContactContractError(RuntimeError):
    """A created PhysX view does not have the exact registered identity."""


class PartnerContactDataError(RuntimeError):
    """Detailed contact data cannot safely be attributed to one exact pair."""

    def __init__(
        self,
        message: str,
        *,
        evidence: "PartnerContactEvidence | None" = None,
    ) -> None:
        super().__init__(message)
        self.evidence = evidence


class RigidContactViewLike(Protocol):
    sensor_count: int
    filter_count: int
    sensor_paths: Any
    filter_paths: Any
    max_contact_data_count: int

    def check(self) -> bool: ...

    def get_contact_data(self, dt: float) -> tuple[Any, ...]: ...


class SimulationViewLike(Protocol):
    def create_rigid_contact_view(
        self,
        pattern: str,
        filter_patterns: list[str],
        max_contact_data_count: int,
    ) -> RigidContactViewLike: ...


@dataclass(frozen=True)
class PartnerContactEvidence:
    """JSON-sidecar-ready evidence for one hand/plank pair and one step."""

    arm_id: str
    hand_body_path: str
    object_body_path: str
    sensor_count: int
    filter_count: int
    max_contact_data_count: int
    contact_count: int
    start_index: int
    normal_forces_n: tuple[float, ...]
    contact_points_m: tuple[Vector3, ...]
    contact_normals: tuple[Vector3, ...]
    separations_m: tuple[float, ...]
    normal_force_n: float
    impulse_ns: float
    penetration_m: float
    overflow: bool
    dropped_contact_count: int | None

    def to_sidecar(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "hand_body_path": self.hand_body_path,
            "object_body_path": self.object_body_path,
            "sensor_count": self.sensor_count,
            "filter_count": self.filter_count,
            "max_contact_data_count": self.max_contact_data_count,
            "contact_count": self.contact_count,
            "start_index": self.start_index,
            "normal_forces_n": list(self.normal_forces_n),
            "contact_points_m": [list(point) for point in self.contact_points_m],
            "contact_normals": [list(normal) for normal in self.contact_normals],
            "separations_m": list(self.separations_m),
            "normal_force_n": self.normal_force_n,
            "impulse_ns": self.impulse_ns,
            "penetration_m": self.penetration_m,
            "overflow": self.overflow,
            "dropped_contact_count": self.dropped_contact_count,
        }


@dataclass(frozen=True)
class PartnerContactFrame:
    """Attributed contacts plus complete per-pair evidence for one physics step."""

    physics_dt_s: float
    contacts: tuple[Contact, ...]
    pair_evidence: tuple[PartnerContactEvidence, ...]

    @property
    def evidence_by_arm(self) -> dict[str, PartnerContactEvidence]:
        return {item.arm_id: item for item in self.pair_evidence}

    def to_sidecar(self) -> dict[str, Any]:
        return {
            "physics_dt_s": self.physics_dt_s,
            "contacts": [
                {
                    "body0_path": contact.body0_path,
                    "body1_path": contact.body1_path,
                    "normal_force_n": contact.normal_force_n,
                    "impulse_ns": contact.impulse_ns,
                    "penetration_m": contact.penetration_m,
                }
                for contact in self.contacts
            ],
            "pairs": [item.to_sidecar() for item in self.pair_evidence],
        }


def _import_omni_physics_tensors() -> Any:
    """Import the Isaac tensor API only at the explicit production boundary."""

    return importlib.import_module("omni.physics.tensors")


def _as_python(value: Any) -> Any:
    """Convert numpy/torch/warp-style tensor wrappers without importing them."""

    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        return tolist()
    return current


def _flatten(value: Any) -> list[Any]:
    value = _as_python(value)
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


def _one_integer(value: Any, *, name: str) -> int:
    flattened = _flatten(value)
    if len(flattened) != 1:
        raise PartnerContactDataError(f"{name} must contain exactly one value")
    scalar = flattened[0]
    if isinstance(scalar, bool) or not isinstance(scalar, numbers.Integral):
        raise PartnerContactDataError(f"{name} must be an integer")
    return int(scalar)


def _scalar_buffer(value: Any, *, name: str, capacity: int) -> list[float]:
    raw = _as_python(value)
    if not isinstance(raw, (list, tuple)) or len(raw) != capacity:
        raise PartnerContactDataError(
            f"{name} buffer length does not match max_contact_data_count"
        )
    result: list[float] = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise PartnerContactDataError(f"{name} entries must be scalar")
            item = item[0]
        try:
            result.append(float(item))
        except (TypeError, ValueError) as exc:
            raise PartnerContactDataError(f"{name} entries must be numeric") from exc
    return result


def _vector3_buffer(value: Any, *, name: str, capacity: int) -> list[Vector3]:
    raw = _as_python(value)
    if not isinstance(raw, (list, tuple)) or len(raw) != capacity:
        raise PartnerContactDataError(
            f"{name} buffer length does not match max_contact_data_count"
        )
    result: list[Vector3] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise PartnerContactDataError(f"{name} entries must contain three values")
        try:
            vector = (float(item[0]), float(item[1]), float(item[2]))
        except (TypeError, ValueError) as exc:
            raise PartnerContactDataError(f"{name} entries must be numeric") from exc
        result.append(vector)
    return result


def _path_list(value: Any) -> list[Any]:
    return _flatten(value)


def _is_exact_integer(value: Any, expected: int) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Integral)
        and int(value) == expected
    )


class ExactPartnerContactCollector:
    """Collect exact D-032 hand/plank contact data for one environment."""

    def __init__(
        self,
        simulation_view: SimulationViewLike,
        *,
        environment_count: int = 1,
        max_contact_data_count: int = 64,
        hand_body_paths: Mapping[str, str] = EXACT_HAND_BODY_PATHS,
        object_body_path: str = OBJECT_BODY_PATH,
    ) -> None:
        if environment_count != 1:
            raise ValueError("exact partner contact collection currently supports N=1 only")
        if (
            isinstance(max_contact_data_count, bool)
            or not isinstance(max_contact_data_count, numbers.Integral)
            or max_contact_data_count <= 0
        ):
            raise ValueError("max_contact_data_count must be a positive integer")
        if dict(hand_body_paths) != EXACT_HAND_BODY_PATHS:
            raise ValueError("hand_body_paths must equal the exact D-032 hand body paths")
        if object_body_path != OBJECT_BODY_PATH:
            raise ValueError("object_body_path must equal the exact D-032 object body path")

        self.environment_count = environment_count
        self.max_contact_data_count = int(max_contact_data_count)
        self.object_body_path = object_body_path
        self._hand_body_paths = dict(hand_body_paths)
        self._views: dict[str, RigidContactViewLike] = {}
        self.last_pair_evidence: tuple[PartnerContactEvidence, ...] = ()

        for arm_id, hand_path in self._hand_body_paths.items():
            view = simulation_view.create_rigid_contact_view(
                hand_path,
                filter_patterns=[self.object_body_path],
                max_contact_data_count=self.max_contact_data_count,
            )
            self._validate_view_identity(arm_id, hand_path, view)
            self._views[arm_id] = view

    @classmethod
    def from_isaac(
        cls,
        *,
        backend: str = "torch",
        environment_count: int = 1,
        max_contact_data_count: int = 64,
    ) -> "ExactPartnerContactCollector":
        """Create the production simulation view through a lazy Omni import."""

        tensors = _import_omni_physics_tensors()
        simulation_view = tensors.create_simulation_view(backend)
        return cls(
            simulation_view,
            environment_count=environment_count,
            max_contact_data_count=max_contact_data_count,
        )

    @property
    def arm_ids(self) -> tuple[str, ...]:
        return tuple(self._hand_body_paths)

    @property
    def hand_body_paths(self) -> dict[str, str]:
        return dict(self._hand_body_paths)

    def _validate_view_identity(
        self,
        arm_id: str,
        hand_path: str,
        view: RigidContactViewLike,
    ) -> None:
        try:
            valid = bool(view.check())
        except Exception as exc:
            raise PartnerContactContractError(
                f"{arm_id} RigidContactView validity check failed"
            ) from exc
        if not valid:
            raise PartnerContactContractError(f"{arm_id} RigidContactView is invalid")
        if not _is_exact_integer(view.sensor_count, 1):
            raise PartnerContactContractError(
                f"{arm_id} sensor_count must equal 1; got {view.sensor_count!r}"
            )
        if not _is_exact_integer(view.filter_count, 1):
            raise PartnerContactContractError(
                f"{arm_id} filter_count must equal 1; got {view.filter_count!r}"
            )
        if not _is_exact_integer(
            view.max_contact_data_count, self.max_contact_data_count
        ):
            raise PartnerContactContractError(
                f"{arm_id} max_contact_data_count mismatch: "
                f"{view.max_contact_data_count!r}"
            )
        if _path_list(view.sensor_paths) != [hand_path]:
            raise PartnerContactContractError(
                f"{arm_id} sensor path must be exactly {hand_path!r}"
            )
        if _path_list(view.filter_paths) != [self.object_body_path]:
            raise PartnerContactContractError(
                f"{arm_id} filter path must be exactly {self.object_body_path!r}"
            )

    def collect(self, *, physics_dt_s: float) -> PartnerContactFrame:
        """Read one detailed-contact frame, rejecting ambiguous data wholesale."""

        dt = float(physics_dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("physics_dt_s must be finite and positive")

        contacts: list[Contact] = []
        evidence_items: list[PartnerContactEvidence] = []
        self.last_pair_evidence = ()
        for arm_id, hand_path in self._hand_body_paths.items():
            view = self._views[arm_id]
            try:
                raw_data = view.get_contact_data(dt)
            except Exception as exc:
                raise PartnerContactDataError(
                    f"{arm_id} get_contact_data failed"
                ) from exc
            try:
                contact, evidence = self._aggregate_pair(
                    arm_id=arm_id,
                    hand_path=hand_path,
                    raw_data=raw_data,
                    dt=dt,
                )
            except PartnerContactDataError as exc:
                if exc.evidence is not None:
                    evidence_items.append(exc.evidence)
                self.last_pair_evidence = tuple(evidence_items)
                raise
            evidence_items.append(evidence)
            if contact is not None:
                contacts.append(contact)

        self.last_pair_evidence = tuple(evidence_items)
        return PartnerContactFrame(dt, tuple(contacts), self.last_pair_evidence)

    def _aggregate_pair(
        self,
        *,
        arm_id: str,
        hand_path: str,
        raw_data: Any,
        dt: float,
    ) -> tuple[Contact | None, PartnerContactEvidence]:
        if not isinstance(raw_data, (tuple, list)) or len(raw_data) != 6:
            raise PartnerContactDataError(
                f"{arm_id} get_contact_data must return six arrays"
            )
        force_raw, point_raw, normal_raw, separation_raw, count_raw, start_raw = (
            raw_data
        )
        count = _one_integer(count_raw, name="contact_count")
        start = _one_integer(start_raw, name="start_index")
        if count < 0:
            raise PartnerContactDataError("contact_count must be non-negative")
        if start < 0:
            raise PartnerContactDataError("start_index must be non-negative")

        capacity = self.max_contact_data_count
        forces = _scalar_buffer(force_raw, name="force", capacity=capacity)
        points = _vector3_buffer(point_raw, name="point", capacity=capacity)
        normals = _vector3_buffer(normal_raw, name="normal", capacity=capacity)
        separations = _scalar_buffer(
            separation_raw, name="separation", capacity=capacity
        )
        end = start + count
        if start > capacity or end > capacity:
            raise PartnerContactDataError(
                f"{arm_id} contact slice exceeds buffer bounds: "
                f"start={start}, count={count}, capacity={capacity}"
            )

        active_forces = tuple(forces[start:end])
        active_points = tuple(points[start:end])
        active_normals = tuple(normals[start:end])
        active_separations = tuple(separations[start:end])
        active_scalars = (*active_forces, *active_separations)
        active_vectors = (*active_points, *active_normals)
        if not all(math.isfinite(value) for value in active_scalars) or not all(
            math.isfinite(value) for vector in active_vectors for value in vector
        ):
            raise PartnerContactDataError(
                f"{arm_id} active contact data must be finite"
            )
        if any(force < 0.0 for force in active_forces):
            raise PartnerContactDataError(
                f"{arm_id} normal forces must be non-negative"
            )

        normal_force = math.fsum(active_forces)
        impulse = normal_force * dt
        penetration = max(
            (max(0.0, -separation) for separation in active_separations),
            default=0.0,
        )
        overflow = count > 0 and end == capacity
        evidence = PartnerContactEvidence(
            arm_id=arm_id,
            hand_body_path=hand_path,
            object_body_path=self.object_body_path,
            sensor_count=1,
            filter_count=1,
            max_contact_data_count=capacity,
            contact_count=count,
            start_index=start,
            normal_forces_n=active_forces,
            contact_points_m=active_points,
            contact_normals=active_normals,
            separations_m=active_separations,
            normal_force_n=normal_force,
            impulse_ns=impulse,
            penetration_m=penetration,
            overflow=overflow,
            dropped_contact_count=None if overflow else 0,
        )
        if overflow:
            raise PartnerContactDataError(
                f"{arm_id} contact buffer reached capacity; overflow/truncation "
                "cannot be excluded",
                evidence=evidence,
            )

        contact = None
        if count:
            contact = Contact(
                body0_path=hand_path,
                body1_path=self.object_body_path,
                normal_force_n=normal_force,
                impulse_ns=impulse,
                penetration_m=penetration,
            )
        return contact, evidence


__all__ = [
    "EXACT_HAND_BODY_PATHS",
    "OBJECT_BODY_PATH",
    "ExactPartnerContactCollector",
    "PartnerContactContractError",
    "PartnerContactDataError",
    "PartnerContactEvidence",
    "PartnerContactFrame",
]
