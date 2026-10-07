"""Contact-gated, constraint-assisted physical grasp support for an N=1 slice.

This module deliberately owns no object reset or trajectory-follow behavior.  A
dynamic object remains under PhysX integration for the entire episode.  Once an
assigned rigid hand body has sustained attributable contact with that same
object, the hand is closed, the configured grasp frames are close, and no
illegal/deep contact is reported, a fixed USD physics joint may be enabled.

The resulting claim is *constraint-assisted physical grasp*.  It must not be
reported as a friction-only grasp.  The first implementation accepts only one
simulation environment and the ``fixed`` strategy.  A compliant D6 strategy is
an explicit future extension and is rejected instead of silently degrading.

Isaac/USD imports are lazy.  The gate and controller can therefore be tested on
CPU with a tiny runtime double, while :class:`UsdFixedJointRuntime` is the
production boundary that authors ``UsdPhysics.FixedJoint`` and ``jointEnabled``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


Vector3 = tuple[float, float, float]
QuaternionWxyz = tuple[float, float, float, float]


def _finite_tuple(
    values: tuple[float, ...], *, length: int, name: str
) -> tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    out = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in out):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _normalize_quaternion(quaternion: QuaternionWxyz) -> QuaternionWxyz:
    q = _finite_tuple(quaternion, length=4, name="orientation_wxyz")
    norm = math.sqrt(sum(value * value for value in q))
    if norm <= 1.0e-12:
        raise ValueError("orientation_wxyz must have non-zero norm")
    return tuple(value / norm for value in q)  # type: ignore[return-value]


def _quat_conjugate(q: QuaternionWxyz) -> QuaternionWxyz:
    return (q[0], -q[1], -q[2], -q[3])


def _quat_multiply(a: QuaternionWxyz, b: QuaternionWxyz) -> QuaternionWxyz:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate(q: QuaternionWxyz, vector: Vector3) -> Vector3:
    rotated = _quat_multiply(
        _quat_multiply(q, (0.0, vector[0], vector[1], vector[2])),
        _quat_conjugate(q),
    )
    return (rotated[1], rotated[2], rotated[3])


@dataclass(frozen=True)
class Pose:
    """Rigid pose with a world/local position and a normalized wxyz quaternion."""

    position: Vector3
    orientation_wxyz: QuaternionWxyz = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        position = _finite_tuple(self.position, length=3, name="position")
        orientation = _normalize_quaternion(self.orientation_wxyz)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "orientation_wxyz", orientation)

    def distance_to(self, other: "Pose") -> float:
        return math.dist(self.position, other.position)


def relative_pose(parent_world: Pose, child_world: Pose) -> Pose:
    """Return ``child_world`` expressed in ``parent_world`` coordinates."""

    parent_inverse = _quat_conjugate(parent_world.orientation_wxyz)
    delta = tuple(
        child - parent
        for child, parent in zip(child_world.position, parent_world.position)
    )
    local_position = _quat_rotate(parent_inverse, delta)  # type: ignore[arg-type]
    local_orientation = _quat_multiply(parent_inverse, child_world.orientation_wxyz)
    return Pose(local_position, local_orientation)


@dataclass(frozen=True)
class Contact:
    """One PhysX contact contribution with explicit rigid-body attribution."""

    body0_path: str
    body1_path: str
    normal_force_n: float
    impulse_ns: float
    penetration_m: float = 0.0

    def __post_init__(self) -> None:
        if not self.body0_path or not self.body1_path:
            raise ValueError("contact endpoints must be non-empty prim paths")
        for name in ("normal_force_n", "impulse_ns", "penetration_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    def matches(self, body_a: str, body_b: str) -> bool:
        return {self.body0_path, self.body1_path} == {body_a, body_b}


@dataclass(frozen=True)
class GraspGateConfig:
    """Immutable per-arm gate and joint assignment for one environment."""

    arm_id: str
    hand_body_path: str
    object_body_path: str
    joint_prim_path: str
    environment_count: int = 1
    constraint_mode: str = "fixed"
    required_contact_steps: int = 3
    required_contact_duration_s: float = 0.03
    min_normal_force_n: float = 0.0
    min_impulse_ns: float = 0.0
    max_grasp_frame_distance_m: float = 0.02
    max_penetration_m: float = 0.003

    def __post_init__(self) -> None:
        if self.environment_count != 1:
            raise ValueError("contact-gated grasp currently supports N=1 only")
        if self.constraint_mode != "fixed":
            raise ValueError(
                "only fixed constraint-assisted grasp is implemented; "
                f"got {self.constraint_mode!r}"
            )
        if not self.arm_id:
            raise ValueError("arm_id must be non-empty")
        for name in ("hand_body_path", "object_body_path", "joint_prim_path"):
            path = getattr(self, name)
            if not isinstance(path, str) or not path.startswith("/"):
                raise ValueError(f"{name} must be an absolute USD prim path")
        if self.hand_body_path == self.object_body_path:
            raise ValueError("hand and object body paths must differ")
        if self.required_contact_steps < 1:
            raise ValueError("required_contact_steps must be >= 1")
        for name in (
            "required_contact_duration_s",
            "min_normal_force_n",
            "min_impulse_ns",
            "max_grasp_frame_distance_m",
            "max_penetration_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GraspObservation:
    """All gate evidence for one simulation step and one assigned arm."""

    step: int
    dt_s: float
    hand_closed: bool
    hand_grasp_frame_world: Pose
    object_grasp_frame_world: Pose
    contacts: tuple[Contact, ...]
    illegal_contact: bool = False

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        dt = float(self.dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        object.__setattr__(self, "dt_s", dt)
        object.__setattr__(self, "contacts", tuple(self.contacts))


@dataclass(frozen=True)
class FixedJointSpec:
    """The only runtime authoring request emitted by this first version."""

    arm_id: str
    hand_body_path: str
    object_body_path: str
    joint_prim_path: str
    constraint_mode: str = "fixed"


@dataclass(frozen=True)
class FixedJointEvidence:
    """Auditable result returned after the runtime enables the joint."""

    joint_prim_path: str
    body0_path: str
    body1_path: str
    local_frame0: Pose
    local_frame1: Pose
    joint_enabled: bool


class GraspState(Enum):
    WAITING = "waiting"
    ENABLING = "enabling"
    CONSTRAINED = "constrained"
    RELEASED = "released"
    FAILED = "failed"


@dataclass(frozen=True)
class GraspEvent:
    """Sidecar-ready transition evidence for one contact/joint event."""

    step: int
    arm_id: str
    kind: str
    state: str
    reason: str
    contact_endpoints: tuple[str, str]
    normal_force_n: float
    impulse_ns: float
    contact_duration_s: float
    joint_prim_path: str
    body_targets: tuple[str, str]
    local_frame0: Pose | None
    local_frame1: Pose | None
    constraint_mode: str

    def as_dict(self) -> dict[str, object]:
        def pose_dict(pose: Pose | None) -> dict[str, object] | None:
            if pose is None:
                return None
            return {
                "position": pose.position,
                "orientation_wxyz": pose.orientation_wxyz,
            }

        return {
            "step": self.step,
            "arm_id": self.arm_id,
            "kind": self.kind,
            "state": self.state,
            "reason": self.reason,
            "contact_endpoints": self.contact_endpoints,
            "normal_force_n": self.normal_force_n,
            "impulse_ns": self.impulse_ns,
            "contact_duration_s": self.contact_duration_s,
            "joint_prim_path": self.joint_prim_path,
            "body_targets": self.body_targets,
            "local_frame0": pose_dict(self.local_frame0),
            "local_frame1": pose_dict(self.local_frame1),
            "constraint_mode": self.constraint_mode,
        }


class FixedJointRuntime(Protocol):
    """Narrow seam implemented by the lazy USD runtime and CPU test doubles."""

    def enable_fixed(self, spec: FixedJointSpec) -> FixedJointEvidence: ...

    def disable(self, joint_prim_path: str) -> None: ...


@dataclass(frozen=True)
class _ContactSummary:
    endpoints: tuple[str, str]
    normal_force_n: float
    impulse_ns: float
    max_penetration_m: float


class ContactGatedConstraintGrasp:
    """Per-arm, single-episode gate with one-shot connect/release semantics."""

    def __init__(self, config: GraspGateConfig, runtime: FixedJointRuntime):
        self.config = config
        self.runtime = runtime
        self.state = GraspState.WAITING
        self.contact_steps = 0
        self.contact_duration_s = 0.0
        self.events: list[GraspEvent] = []
        self._last_step: int | None = None
        self._last_summary = _ContactSummary(
            (config.hand_body_path, config.object_body_path), 0.0, 0.0, 0.0
        )
        self._joint_evidence: FixedJointEvidence | None = None

    def _reset_streak(self) -> None:
        self.contact_steps = 0
        self.contact_duration_s = 0.0

    def _matching_contact(
        self, observation: GraspObservation
    ) -> _ContactSummary | None:
        matching = [
            contact
            for contact in observation.contacts
            if contact.matches(self.config.hand_body_path, self.config.object_body_path)
        ]
        if not matching:
            return None
        return _ContactSummary(
            endpoints=(self.config.hand_body_path, self.config.object_body_path),
            normal_force_n=sum(contact.normal_force_n for contact in matching),
            impulse_ns=sum(contact.impulse_ns for contact in matching),
            max_penetration_m=max(contact.penetration_m for contact in matching),
        )

    def _record(
        self,
        *,
        step: int,
        kind: str,
        reason: str,
        state: GraspState,
        summary: _ContactSummary | None = None,
        evidence: FixedJointEvidence | None = None,
    ) -> GraspEvent:
        summary = summary or self._last_summary
        evidence = evidence or self._joint_evidence
        event = GraspEvent(
            step=step,
            arm_id=self.config.arm_id,
            kind=kind,
            state=state.value,
            reason=reason,
            contact_endpoints=summary.endpoints,
            normal_force_n=summary.normal_force_n,
            impulse_ns=summary.impulse_ns,
            contact_duration_s=self.contact_duration_s,
            joint_prim_path=self.config.joint_prim_path,
            body_targets=(
                self.config.hand_body_path,
                self.config.object_body_path,
            ),
            local_frame0=evidence.local_frame0 if evidence else None,
            local_frame1=evidence.local_frame1 if evidence else None,
            constraint_mode=self.config.constraint_mode,
        )
        self.events.append(event)
        return event

    def _fail_closed(
        self,
        *,
        step: int,
        reason: str,
        summary: _ContactSummary | None = None,
        cleanup_joint: bool = False,
    ) -> GraspEvent:
        if cleanup_joint:
            try:
                self.runtime.disable(self.config.joint_prim_path)
            except Exception as exc:  # cleanup is best effort; lockout is unconditional
                reason = f"{reason};disable_cleanup_failed:{type(exc).__name__}:{exc}"
        self.state = GraspState.FAILED
        return self._record(
            step=step,
            kind="failed_closed",
            reason=reason,
            state=self.state,
            summary=summary,
        )

    def observe(self, observation: GraspObservation) -> GraspEvent | None:
        """Consume one sample; enable at most one joint in this controller's life."""

        if self.state is not GraspState.WAITING:
            return None

        step_is_contiguous = (
            self._last_step is None or observation.step == self._last_step + 1
        )
        self._last_step = observation.step
        if not step_is_contiguous:
            self._reset_streak()

        summary = self._matching_contact(observation)
        if summary is not None:
            self._last_summary = summary

        if observation.illegal_contact:
            self._reset_streak()
            return self._fail_closed(
                step=observation.step,
                reason="illegal_contact",
                summary=summary,
            )
        if (
            summary is not None
            and summary.max_penetration_m > self.config.max_penetration_m
        ):
            self._reset_streak()
            return self._fail_closed(
                step=observation.step,
                reason="partner_contact_penetration_exceeded",
                summary=summary,
            )

        frame_distance = observation.hand_grasp_frame_world.distance_to(
            observation.object_grasp_frame_world
        )
        gates_pass = (
            observation.hand_closed
            and summary is not None
            and summary.normal_force_n >= self.config.min_normal_force_n
            and summary.impulse_ns >= self.config.min_impulse_ns
            and frame_distance <= self.config.max_grasp_frame_distance_m
        )
        if not gates_pass:
            self._reset_streak()
            return None

        self.contact_steps += 1
        self.contact_duration_s += observation.dt_s
        enough_steps = self.contact_steps >= self.config.required_contact_steps
        enough_duration = (
            self.contact_duration_s + 1.0e-12 >= self.config.required_contact_duration_s
        )
        if not (enough_steps and enough_duration):
            return None

        spec = FixedJointSpec(
            arm_id=self.config.arm_id,
            hand_body_path=self.config.hand_body_path,
            object_body_path=self.config.object_body_path,
            joint_prim_path=self.config.joint_prim_path,
            constraint_mode=self.config.constraint_mode,
        )
        self.state = GraspState.ENABLING
        try:
            evidence = self.runtime.enable_fixed(spec)
            self._validate_evidence(spec, evidence)
        except Exception as exc:
            return self._fail_closed(
                step=observation.step,
                reason=f"joint_enable_failed:{type(exc).__name__}:{exc}",
                summary=summary,
                cleanup_joint=True,
            )

        self._joint_evidence = evidence
        self.state = GraspState.CONSTRAINED
        return self._record(
            step=observation.step,
            kind="constraint_enabled",
            reason="contact_gate_satisfied",
            state=self.state,
            summary=summary,
            evidence=evidence,
        )

    @staticmethod
    def _validate_evidence(spec: FixedJointSpec, evidence: FixedJointEvidence) -> None:
        if not evidence.joint_enabled:
            raise RuntimeError("runtime returned joint_enabled=false")
        if evidence.joint_prim_path != spec.joint_prim_path:
            raise RuntimeError("runtime returned the wrong joint prim")
        if (evidence.body0_path, evidence.body1_path) != (
            spec.hand_body_path,
            spec.object_body_path,
        ):
            raise RuntimeError("runtime returned the wrong body targets")

    def release(
        self, *, step: int, reason: str = "release_requested"
    ) -> GraspEvent | None:
        """Disable an enabled joint and permanently lock this episode instance."""

        if step < 0:
            raise ValueError("step must be non-negative")
        if self.state in (GraspState.RELEASED, GraspState.FAILED):
            return None
        if self.state is GraspState.WAITING:
            self._reset_streak()
            self.state = GraspState.RELEASED
            return self._record(
                step=step,
                kind="released_without_constraint",
                reason=reason,
                state=self.state,
            )
        if self.state is not GraspState.CONSTRAINED:
            return self._fail_closed(
                step=step,
                reason=f"release_from_invalid_state:{self.state.value}",
                cleanup_joint=True,
            )

        try:
            self.runtime.disable(self.config.joint_prim_path)
        except Exception as exc:
            self.state = GraspState.FAILED
            return self._record(
                step=step,
                kind="failed_closed",
                reason=f"joint_disable_failed:{type(exc).__name__}:{exc}",
                state=self.state,
            )

        self.state = GraspState.RELEASED
        return self._record(
            step=step,
            kind="constraint_disabled",
            reason=reason,
            state=self.state,
        )


def _load_pxr():
    """Import USD only inside runtime calls, never during module import."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    return Gf, Usd, UsdGeom, UsdPhysics


def _pose_from_gf_matrix(matrix) -> Pose:
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return Pose(
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (
            float(rotation.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def _prim_is_valid(prim) -> bool:
    return prim is not None and bool(prim) and prim.IsValid()


class UsdFixedJointRuntime:
    """Lazy USD authoring adapter; joint activation is the final operation."""

    def __init__(self, stage):
        if stage is None:
            raise ValueError("stage must not be None")
        self._stage = stage

    def enable_fixed(self, spec: FixedJointSpec) -> FixedJointEvidence:
        if spec.constraint_mode != "fixed":
            raise ValueError("UsdFixedJointRuntime supports fixed joints only")

        Gf, Usd, UsdGeom, UsdPhysics = _load_pxr()
        stage = self._stage
        existing = stage.GetPrimAtPath(spec.joint_prim_path)
        if _prim_is_valid(existing):
            raise RuntimeError(f"joint prim already exists: {spec.joint_prim_path}")

        body0 = stage.GetPrimAtPath(spec.hand_body_path)
        body1 = stage.GetPrimAtPath(spec.object_body_path)
        if not _prim_is_valid(body0) or not _prim_is_valid(body1):
            raise RuntimeError("fixed-joint body target is missing")
        if not body0.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"body0 is not a rigid body: {spec.hand_body_path}")
        if not body1.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"body1 is not a rigid body: {spec.object_body_path}")

        # Derive both local frames from current world transforms.  Choosing the
        # current object-body origin as the common joint frame preserves the
        # exact current hand/object relative pose and introduces no snap.
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        body0_world = _pose_from_gf_matrix(cache.GetLocalToWorldTransform(body0))
        body1_world = _pose_from_gf_matrix(cache.GetLocalToWorldTransform(body1))
        joint_world = body1_world
        local0 = relative_pose(body0_world, joint_world)
        local1 = relative_pose(body1_world, joint_world)

        joint = None
        try:
            joint = UsdPhysics.FixedJoint.Define(stage, spec.joint_prim_path)
            enabled_attr = joint.CreateJointEnabledAttr()
            enabled_attr.Set(False)
            joint.GetBody0Rel().SetTargets([body0.GetPath()])
            joint.GetBody1Rel().SetTargets([body1.GetPath()])
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*local0.position))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(*local0.orientation_wxyz))
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(*local1.position))
            joint.GetLocalRot1Attr().Set(Gf.Quatf(*local1.orientation_wxyz))
            enabled_attr.Set(True)
        except Exception:
            if joint is not None:
                try:
                    joint.CreateJointEnabledAttr().Set(False)
                except Exception:
                    pass
            raise

        return FixedJointEvidence(
            joint_prim_path=spec.joint_prim_path,
            body0_path=spec.hand_body_path,
            body1_path=spec.object_body_path,
            local_frame0=local0,
            local_frame1=local1,
            joint_enabled=True,
        )

    def disable(self, joint_prim_path: str) -> None:
        _, _, _, UsdPhysics = _load_pxr()
        joint = UsdPhysics.Joint.Get(self._stage, joint_prim_path)
        if joint is None or not _prim_is_valid(joint.GetPrim()):
            raise RuntimeError(f"joint prim is missing: {joint_prim_path}")
        joint.CreateJointEnabledAttr().Set(False)


__all__ = [
    "Contact",
    "ContactGatedConstraintGrasp",
    "FixedJointEvidence",
    "FixedJointSpec",
    "GraspEvent",
    "GraspGateConfig",
    "GraspObservation",
    "GraspState",
    "Pose",
    "UsdFixedJointRuntime",
    "relative_pose",
]
