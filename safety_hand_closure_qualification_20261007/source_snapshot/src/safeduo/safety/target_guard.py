"""D-033 pure-Torch event-triggered per-arm target-rebase candidate.

The guard is deliberately not wired into :mod:`safeduo.envs.duo_env`.  It
evaluates all 16 four-arm rebase masks against the complete active row set.
For a selected arm the next position target is ``clamp(q + exec, soft_limits)``;
for every other arm it is the legacy
``clamp(old_target + exec, soft_limits)``.  The smallest feasible mask wins,
then prior-latch overlap, registered cross priority, and numeric mask code
break ties.  Mask ``1111`` additionally needs one independently attributable
trigger row per arm; a single four-arm row cannot authorize an all-arm hold.

Row convention follows ``ConstraintRows``: ``J @ delta`` is positive when a
pair opens.  A candidate is feasible when its real, soft-limit-clamped target
error satisfies ``J @ error >= -gamma * (d - d_min) * dt`` on every valid,
non-exempt row within ``residual_tol``.  A velocity-box-clipped view is retained
only as a diagnostic; it never grants feasibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

from safeduo.baselines.base import ConstraintRows
from safeduo.safety.types import ARM_KEYS, ARMS_OF_ROBOT, CLASS_CROSS


INFEASIBLE_NONE = 0
INFEASIBLE_NO_FULL_ROW_MASK = 1
INFEASIBLE_ALL_FOUR_EVIDENCE = 2
INFEASIBLE_NONFINITE_DERIVED = 3
INFEASIBLE_UNATTRIBUTED_TRIGGER = 4

_N_ARMS = len(ARM_KEYS)
_N_MASKS = 1 << _N_ARMS


@dataclass(frozen=True)
class TargetGuardConfig:
    """Configuration for the offline D-033 target-rebase candidate."""

    enabled: bool = False
    gamma: float = 4.0
    d_min: float = 0.03
    vmax: float = 1.5
    residual_tol: float = 5e-4
    closing_tol: float = 1e-6
    clear_margin: float = 5e-3

    def __post_init__(self) -> None:
        for name in (
            "gamma",
            "d_min",
            "vmax",
            "residual_tol",
            "closing_tol",
            "clear_margin",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if self.vmax <= 0.0:
            raise ValueError("vmax must be positive")
        if self.residual_tol < 0.0:
            raise ValueError("residual_tol must be non-negative")
        if self.closing_tol < 0.0:
            raise ValueError("closing_tol must be non-negative")
        if self.clear_margin < 0.0:
            raise ValueError("clear_margin must be non-negative")


@dataclass
class TargetGuardDecision:
    """Diagnostics shared by executable and fail-closed guard decisions."""

    held_arm: torch.Tensor
    trigger_pair_ids: torch.Tensor
    selected_mask: torch.Tensor
    infeasible: torch.Tensor
    info: dict[str, torch.Tensor]


@dataclass
class TargetGuardOutput(TargetGuardDecision):
    """An executable target update; every batch row passed the guard."""

    next_target: dict[str, torch.Tensor]
    must_terminate: ClassVar[Literal[False]] = False


@dataclass
class TargetGuardEmergency(TargetGuardDecision):
    """Fail-closed result with no target authorized for execution.

    A production caller must terminate the affected rollout or enter its typed
    emergency path.  This candidate deliberately exposes no ``next_target`` so
    an infeasible legacy target cannot be consumed by accident.
    """

    must_terminate: ClassVar[Literal[True]] = True


class TargetGuardContractError(RuntimeError):
    """Typed fail-closed programming-contract breach with no numeric decision."""

    must_terminate: ClassVar[Literal[True]] = True

    __slots__ = ("_contract_code", "_field", "_expected", "_observed")

    def __init__(
        self,
        *,
        contract_code: Literal["dtype", "device", "lifecycle_signature"],
        field: str,
        expected: str,
        observed: str,
    ) -> None:
        if contract_code not in ("dtype", "device", "lifecycle_signature"):
            raise ValueError(f"unsupported target-guard contract code: {contract_code!r}")
        for name, value in (
            ("field", field),
            ("expected", expected),
            ("observed", observed),
        ):
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be a non-empty native str")
        self._contract_code = contract_code
        self._field = field
        self._expected = expected
        self._observed = observed
        super().__init__(
            f"D-033 target guard {contract_code} contract violation: "
            f"{field} is {observed}, expected {expected}"
        )

    @property
    def contract_code(self) -> Literal["dtype", "device", "lifecycle_signature"]:
        return self._contract_code

    @property
    def field(self) -> str:
        return self._field

    @property
    def expected(self) -> str:
        return self._expected

    @property
    def observed(self) -> str:
        return self._observed


class TargetGuardBatchAbort(RuntimeError):
    """Typed pre-commit abort carrying the non-executable batch decision."""

    def __init__(self, emergency: TargetGuardEmergency):
        super().__init__("D-033 target guard rejected the batch before target-buffer commit")
        self.emergency = emergency


class TargetRebaseGuard:
    """Stateful four-arm mask search with per-arm clear hysteresis."""

    def __init__(self, cfg: TargetGuardConfig | None = None):
        self.cfg = cfg or TargetGuardConfig()
        self._held_arm: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear all latches, or only the selected environment rows."""
        if self._held_arm is None:
            return
        if env_ids is None:
            self._held_arm.zero_()
            return
        reset_ids = env_ids.to(device=self._held_arm.device)
        self._held_arm[reset_ids] = False

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")

    @staticmethod
    def _limited_target(
        value: torch.Tensor,
        limits: torch.Tensor | None,
    ) -> torch.Tensor:
        if limits is None:
            return value
        return value.clamp(limits[..., 0], limits[..., 1])

    @staticmethod
    def _require_reference_dtype(
        name: str,
        value: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        if value.dtype != reference.dtype:
            raise TargetGuardContractError(
                contract_code="dtype",
                field=name,
                observed=str(value.dtype),
                expected=str(reference.dtype),
            )

    @staticmethod
    def _require_reference_device(
        name: str,
        value: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        if value.device != reference.device:
            raise TargetGuardContractError(
                contract_code="device",
                field=name,
                observed=str(value.device),
                expected=str(reference.device),
            )

    @staticmethod
    def _validate_arm_inputs(
        q: dict[str, torch.Tensor],
        qd: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        exec_delta: dict[str, torch.Tensor],
        soft_limits: dict[str, torch.Tensor] | None,
        *,
        validate_values: bool = True,
    ) -> tuple[int, torch.device, torch.dtype]:
        missing = [
            arm
            for arm in ARM_KEYS
            if arm not in q or arm not in qd or arm not in target or arm not in exec_delta
        ]
        if missing:
            raise ValueError(f"missing arm tensors: {missing}")
        reference = q[ARM_KEYS[0]]
        if reference.ndim != 2:
            raise ValueError("arm tensors must have shape (N, dof)")
        n = reference.shape[0]
        for arm in ARM_KEYS:
            shape = q[arm].shape
            if q[arm].ndim != 2:
                raise ValueError(f"q[{arm}] must have shape (N, dof)")
            TargetRebaseGuard._require_reference_device(f"q[{arm}]", q[arm], reference)
            TargetRebaseGuard._require_reference_dtype(f"q[{arm}]", q[arm], reference)
            for name, mapping in (
                ("qd", qd),
                ("target", target),
                ("exec_delta", exec_delta),
            ):
                if mapping[arm].shape != shape:
                    raise ValueError(
                        f"{name}[{arm}] shape {tuple(mapping[arm].shape)} != {tuple(shape)}"
                    )
                TargetRebaseGuard._require_reference_device(
                    f"{name}[{arm}]", mapping[arm], reference
                )
                TargetRebaseGuard._require_reference_dtype(
                    f"{name}[{arm}]", mapping[arm], reference
                )
            if validate_values:
                for name, mapping in (
                    ("q", q),
                    ("qd", qd),
                    ("target", target),
                    ("exec_delta", exec_delta),
                ):
                    TargetRebaseGuard._require_finite(f"{name}[{arm}]", mapping[arm])
            if soft_limits is not None:
                if arm not in soft_limits:
                    raise ValueError(f"missing soft limits for {arm}")
                limits = soft_limits[arm]
                if limits.shape[-2:] != (*shape[1:], 2):
                    raise ValueError(
                        f"soft_limits[{arm}] tail shape {tuple(limits.shape[-2:])} "
                        f"!= {(shape[1], 2)}"
                    )
                if limits.ndim != 3 or limits.shape[0] not in (1, n):
                    raise ValueError(f"soft_limits[{arm}] must have shape (1|N, dof, 2)")
                TargetRebaseGuard._require_reference_device(
                    f"soft_limits[{arm}]", limits, reference
                )
                TargetRebaseGuard._require_reference_dtype(f"soft_limits[{arm}]", limits, reference)
                if validate_values:
                    TargetRebaseGuard._require_finite(f"soft_limits[{arm}]", limits)
        return n, reference.device, reference.dtype

    @staticmethod
    def _production_contract_ok(
        *,
        n: int,
        device: torch.device,
        q: dict[str, torch.Tensor],
        qd: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        exec_delta: dict[str, torch.Tensor],
        soft_limits: dict[str, torch.Tensor] | None,
        rows: ConstraintRows,
        priority_p: torch.Tensor,
    ) -> torch.Tensor:
        """Per-environment numeric contract without a device-to-host read."""

        checks: list[torch.Tensor] = []
        for arm in ARM_KEYS:
            for mapping in (q, qd, target, exec_delta):
                checks.append(torch.isfinite(mapping[arm]).reshape(n, -1).all(dim=-1))
            if soft_limits is not None:
                limits = soft_limits[arm]
                limits_ok = torch.isfinite(limits).reshape(limits.shape[0], -1).all(dim=-1) & (
                    limits[..., 0] <= limits[..., 1]
                ).all(dim=-1)
                if limits.shape[0] == 1:
                    limits_ok = limits_ok.expand(n)
                checks.append(limits_ok)

        for value in (rows.d, rows.cls, rows.J["F"], rows.J["U"]):
            checks.append(torch.isfinite(value).reshape(n, -1).all(dim=-1).to(device=device))
        if rows.d_min is not None:
            checks.append(torch.isfinite(rows.d_min).reshape(n, -1).all(dim=-1).to(device=device))
        checks.append(torch.isfinite(priority_p) & (priority_p >= -1.0) & (priority_p <= 1.0))
        return torch.stack([check.to(device=device) for check in checks]).all(dim=0)

    def _previous_state(self, n: int, device: torch.device) -> torch.Tensor:
        """Read compatible latch state without mutating it.

        A new batch shape/device receives a temporary zero prior.  The state is
        committed only after an executable decision, so every emergency path
        is atomic with respect to the previous latch.
        """
        if (
            self._held_arm is None
            or self._held_arm.shape != (n, _N_ARMS)
            or self._held_arm.device != device
        ):
            return torch.zeros(n, _N_ARMS, dtype=torch.bool, device=device)
        return self._held_arm

    @staticmethod
    def _stack_candidates(
        values: dict[str, torch.Tensor],
        robot: str,
    ) -> torch.Tensor:
        return torch.cat([values[arm] for arm in ARMS_OF_ROBOT[robot]], dim=-1)

    @staticmethod
    def _choose_mask(
        feasible: torch.Tensor,
        mask_bits: torch.Tensor,
        previous: torch.Tensor,
        priority_p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lexicographic choice: cardinality, old-latch overlap, p, mask code."""
        n, n_masks = feasible.shape
        device = feasible.device
        codes = torch.arange(n_masks, dtype=torch.int64, device=device)
        cardinality = mask_bits.sum(dim=-1)
        card = torch.where(
            feasible, cardinality[None, :], torch.full((n, n_masks), 99, device=device)
        )
        min_card = card.amin(dim=-1)
        tied = feasible & (cardinality[None, :] == min_card[:, None])

        overlap = (mask_bits[None, :, :] & previous[:, None, :]).sum(dim=-1)
        best_overlap = torch.where(tied, overlap, torch.full_like(overlap, -1)).amax(dim=-1)
        tied &= overlap == best_overlap[:, None]

        # p=+1 means U yields to F, hence prefer masks holding U arms.
        balance = mask_bits[:, 2:].sum(dim=-1).to(priority_p.dtype) - mask_bits[:, :2].sum(
            dim=-1
        ).to(priority_p.dtype)
        priority_score = priority_p[:, None] * balance[None, :]
        best_priority = torch.where(
            tied, priority_score, torch.full_like(priority_score, -torch.inf)
        ).amax(dim=-1)
        tied &= priority_score == best_priority[:, None]

        selected = torch.where(
            tied, codes[None, :], torch.full((n, n_masks), n_masks, device=device)
        ).amin(dim=-1)
        any_feasible = feasible.any(dim=-1)
        return torch.where(any_feasible, selected, torch.zeros_like(selected)), any_feasible

    @staticmethod
    def _batch_requires_abort(infeasible: torch.Tensor) -> bool:
        """The single explicit production batch gate (device to host)."""

        return bool(infeasible.any().item())

    def apply(
        self,
        *,
        q: dict[str, torch.Tensor],
        qd: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        exec_delta: dict[str, torch.Tensor],
        rows: ConstraintRows,
        dt: float,
        pair_ids: torch.Tensor | None = None,
        exempt: torch.Tensor | None = None,
        soft_limits: dict[str, torch.Tensor] | None = None,
        priority_p: torch.Tensor | None = None,
        _production: bool = False,
    ) -> TargetGuardOutput | TargetGuardEmergency:
        """Return the next targets selected by the exact 16-mask search.

        ``soft_limits`` should be the actual per-environment joint soft limits.
        It is optional only for isolated tests whose legacy path has no clamp.
        A no-feasible-mask batch returns :class:`TargetGuardEmergency`, which
        has no executable target.  The future production caller must terminate
        or enter an emergency path before this candidate can be wired.
        """
        # Exact default-off bypass.  This intentionally precedes dt, tensor,
        # row, and priority validation: the production integration must branch
        # outside the guard as well, so disabled mode cannot add a device
        # reduction, Python tensor truth test, state mutation, or diagnostics
        # dependency to the legacy target write.
        if not self.cfg.enabled:
            reference = target[ARM_KEYS[0]]
            n = reference.shape[0]
            device = reference.device
            dtype = reference.dtype
            legacy_target = {
                arm: self._limited_target(
                    target[arm] + exec_delta[arm],
                    None if soft_limits is None else soft_limits[arm],
                )
                for arm in ARM_KEYS
            }
            zero_arm = torch.zeros(n, _N_ARMS, dtype=torch.bool, device=device)
            zero_env = torch.zeros(n, dtype=torch.bool, device=device)
            zero_code = torch.zeros(n, dtype=torch.int64, device=device)
            return TargetGuardOutput(
                next_target=legacy_target,
                held_arm=zero_arm,
                trigger_pair_ids=torch.empty(n, 0, dtype=torch.int64, device=device),
                selected_mask=zero_code,
                infeasible=zero_env,
                info={
                    "row_residual": torch.empty(n, 0, dtype=dtype, device=device),
                    "residual_tol": torch.tensor(self.cfg.residual_tol, dtype=dtype, device=device),
                    "infeasible_reason": zero_code.clone(),
                },
            )

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        n, device, dtype = self._validate_arm_inputs(
            q,
            qd,
            target,
            exec_delta,
            soft_limits,
            validate_values=not _production,
        )
        if rows.d.ndim != 2 or rows.d.shape[0] != n:
            raise ValueError("rows.d must have shape (N, M)")
        m = rows.d.shape[1]
        expected_nm = (n, m)
        if (
            rows.valid.shape != expected_nm
            or rows.cls.shape != expected_nm
            or rows.arm_mask.shape != (n, m, _N_ARMS)
        ):
            raise ValueError("rows valid/arm_mask shapes do not match rows.d")
        if rows.valid.dtype != torch.bool:
            raise ValueError("valid must have dtype torch.bool")
        if rows.arm_mask.dtype != torch.bool:
            raise ValueError("arm_mask must have dtype torch.bool")
        if rows.J["F"].shape[:2] != expected_nm or rows.J["U"].shape[:2] != expected_nm:
            raise ValueError("rows.J shapes do not match rows.d")
        if not _production:
            self._require_finite("rows.d", rows.d)
            self._require_finite("rows.cls", rows.cls)
            self._require_finite("rows.J[F]", rows.J["F"])
            self._require_finite("rows.J[U]", rows.J["U"])
        if rows.d_min is not None:
            if rows.d_min.shape != expected_nm:
                raise ValueError("rows.d_min must have shape (N, M)")
            if not _production:
                self._require_finite("rows.d_min", rows.d_min)

        legacy_sum = {arm: target[arm] + exec_delta[arm] for arm in ARM_KEYS}
        legacy_target = {
            arm: self._limited_target(
                legacy_sum[arm],
                None if soft_limits is None else soft_limits[arm],
            )
            for arm in ARM_KEYS
        }
        if pair_ids is None:
            pair_ids = torch.full(expected_nm, -1, dtype=torch.int64, device=device)
        else:
            if pair_ids.shape != expected_nm:
                raise ValueError("pair_ids shape must match rows.d")
            pair_ids = pair_ids.to(device=device, dtype=torch.int64)
        if exempt is None:
            exempt = torch.zeros(expected_nm, dtype=torch.bool, device=device)
        elif exempt.shape != expected_nm:
            raise ValueError("exempt shape must match rows.d")
        elif exempt.dtype != torch.bool:
            raise ValueError("exempt must have dtype torch.bool")
        else:
            exempt = exempt.to(device=device)
        if priority_p is None:
            priority_p = torch.zeros(n, dtype=dtype, device=device)
        elif priority_p.shape != (n,):
            raise ValueError("priority_p must have shape (N,)")
        else:
            priority_p = priority_p.to(device=device, dtype=dtype)
        if not _production:
            self._require_finite("priority_p", priority_p)
            if bool(((priority_p < -1.0) | (priority_p > 1.0)).any()):
                raise ValueError("priority_p must stay in [-1, 1]")

        production_contract_ok = (
            self._production_contract_ok(
                n=n,
                device=device,
                q=q,
                qd=qd,
                target=target,
                exec_delta=exec_delta,
                soft_limits=soft_limits,
                rows=rows,
                priority_p=priority_p,
            )
            if _production
            else torch.ones(n, dtype=torch.bool, device=device)
        )

        held_previous = self._previous_state(n, device)
        valid = rows.valid.to(device=device)
        eligible = valid & ~exempt
        d_min = (
            rows.d_min.to(device=device, dtype=dtype)
            if rows.d_min is not None
            else torch.full_like(rows.d, self.cfg.d_min, dtype=dtype, device=device)
        )
        d = rows.d.to(device=device, dtype=dtype)
        box = self.cfg.vmax * dt

        current_target = {
            arm: self._limited_target(
                target[arm], None if soft_limits is None else soft_limits[arm]
            )
            for arm in ARM_KEYS
        }
        backlog_error = {arm: current_target[arm] - q[arm] for arm in ARM_KEYS}
        backlog_eff = {arm: backlog_error[arm].clamp(-box, box) for arm in ARM_KEYS}
        j_backlog = torch.zeros_like(d)
        j_qd_step = torch.zeros_like(d)
        for robot in ("F", "U"):
            jacobian = rows.J[robot].to(device=device, dtype=dtype)
            j_backlog = j_backlog + torch.einsum(
                "nmd,nd->nm",
                jacobian,
                self._stack_candidates(backlog_eff, robot),
            )
            j_qd_step = j_qd_step + dt * torch.einsum(
                "nmd,nd->nm",
                jacobian,
                self._stack_candidates(qd, robot),
            )
        backlog_closing = j_backlog < -self.cfg.closing_tol
        qd_closing = j_qd_step < -self.cfg.closing_tol
        inside_dmin = d < d_min
        rebase_sum = {arm: q[arm] + exec_delta[arm] for arm in ARM_KEYS}
        rebase_target = {
            arm: self._limited_target(
                rebase_sum[arm],
                None if soft_limits is None else soft_limits[arm],
            )
            for arm in ARM_KEYS
        }
        mask_codes = torch.arange(_N_MASKS, dtype=torch.int64, device=device)
        mask_bits = (
            mask_codes.unsqueeze(-1)
            .bitwise_right_shift(torch.arange(_N_ARMS, device=device))
            .bitwise_and(1)
            .to(torch.bool)
        )
        candidate_error: dict[str, torch.Tensor] = {}
        candidate_error_clipped: dict[str, torch.Tensor] = {}
        for arm_index, arm in enumerate(ARM_KEYS):
            actual = torch.where(
                mask_bits[None, :, arm_index, None],
                rebase_target[arm][:, None, :],
                legacy_target[arm][:, None, :],
            )
            candidate_error[arm] = actual - q[arm][:, None, :]
            candidate_error_clipped[arm] = candidate_error[arm].clamp(-box, box)

        target_demand = torch.zeros(n, _N_MASKS, m, dtype=dtype, device=device)
        target_demand_clipped = torch.zeros_like(target_demand)
        for robot in ("F", "U"):
            jacobian = rows.J[robot].to(device=device, dtype=dtype)
            target_demand = target_demand + torch.einsum(
                "nmd,nkd->nkm",
                jacobian,
                self._stack_candidates(candidate_error, robot),
            )
            target_demand_clipped = target_demand_clipped + torch.einsum(
                "nmd,nkd->nkm",
                jacobian,
                self._stack_candidates(candidate_error_clipped, robot),
            )
        cap = self.cfg.gamma * (d - d_min) * dt
        raw_row_residual = -target_demand - cap[:, None, :]
        legacy_sum_finite = torch.stack(
            [torch.isfinite(legacy_sum[arm]).all(dim=-1) for arm in ARM_KEYS],
            dim=-1,
        ).all(dim=-1)
        rebase_sum_finite = torch.stack(
            [torch.isfinite(rebase_sum[arm]).all(dim=-1) for arm in ARM_KEYS],
            dim=-1,
        ).all(dim=-1)
        derived_values = [
            *legacy_sum.values(),
            *legacy_target.values(),
            *current_target.values(),
            *backlog_error.values(),
            *backlog_eff.values(),
            j_backlog,
            j_qd_step,
            *rebase_sum.values(),
            *rebase_target.values(),
            *candidate_error.values(),
            *candidate_error_clipped.values(),
            target_demand,
            target_demand_clipped,
            cap,
            raw_row_residual,
        ]
        derived_finite = (
            torch.stack(
                [torch.isfinite(value).reshape(n, -1).all(dim=-1) for value in derived_values],
                dim=0,
            ).all(dim=0)
            & production_contract_ok
        )
        row_residual_all = torch.where(
            torch.isfinite(raw_row_residual),
            raw_row_residual.clamp_min(0.0),
            torch.full_like(raw_row_residual, torch.inf),
        )
        row_residual_all = torch.where(
            eligible[:, None, :], row_residual_all, torch.zeros_like(row_residual_all)
        )
        if m:
            candidate_max_residual = row_residual_all.amax(dim=-1)
        else:
            candidate_max_residual = torch.zeros(n, _N_MASKS, dtype=dtype, device=device)

        # The actual legacy target violating a full row is sufficient to
        # trigger.  qd/current clipped backlog remain diagnostics and clear
        # hysteresis signals; they cannot hide stored target demand.
        trigger_row = eligible & (row_residual_all[:, 0] > self.cfg.residual_tol)
        arm_mask = rows.arm_mask.to(device=device)
        trigger_arm = (trigger_row.unsqueeze(-1) & arm_mask).any(dim=1)
        unattributed_trigger_row = trigger_row & ~arm_mask.any(dim=-1)
        unattributed_trigger = unattributed_trigger_row.any(dim=-1)
        clear_row = (
            (d >= d_min + self.cfg.clear_margin)
            & (row_residual_all[:, 0] <= self.cfg.residual_tol)
            & ~backlog_closing
            & ~qd_closing
        )
        uncleared_row = eligible & ~clear_row
        uncleared_arm = (uncleared_row.unsqueeze(-1) & arm_mask).any(dim=1)
        retained_latch = held_previous & uncleared_arm
        policy_emergency = ~derived_finite | unattributed_trigger
        need_guard = trigger_arm.any(dim=-1) | retained_latch.any(dim=-1) | policy_emergency

        # A four-arm hold needs four independent pieces of trigger evidence.
        # "Independent" is deliberately strict and auditable: a qualifying
        # row's arm_mask has exactly one bit, and every arm has such a row.
        single_arm_row = arm_mask.sum(dim=-1) == 1
        independently_triggered_arm = (
            trigger_row.unsqueeze(-1) & single_arm_row.unsqueeze(-1) & arm_mask
        ).any(dim=1)
        all_four_independent_evidence = independently_triggered_arm.all(dim=-1)
        # G3 authorization is a per-step proof obligation.  Prior held state
        # may affect hysteresis and deterministic selection, but can never
        # inherit authority for mask 1111.
        all_four_authorized = all_four_independent_evidence

        includes_latch = (mask_bits[None, :, :] | ~retained_latch[:, None, :]).all(dim=-1)
        candidate_full_row_feasible = (
            (candidate_max_residual <= self.cfg.residual_tol)
            & includes_latch
            & derived_finite[:, None]
        )
        candidate_feasible = candidate_full_row_feasible & (
            (mask_codes != _N_MASKS - 1)[None, :] | all_four_authorized[:, None]
        )
        candidate_feasible &= ~policy_emergency[:, None]
        allowed_arm = trigger_arm | retained_latch
        restricted_mask = (~mask_bits[None, :, :] | allowed_arm[:, None, :]).all(dim=-1)
        candidate_feasible_restricted = candidate_feasible & restricted_mask
        has_cross_trigger = (trigger_row & (rows.cls.to(device=device) == CLASS_CROSS)).any(dim=-1)
        registered_priority = torch.where(
            has_cross_trigger, priority_p, torch.zeros_like(priority_p)
        )
        restricted_code, restricted_ok = self._choose_mask(
            candidate_feasible_restricted, mask_bits, held_previous, registered_priority
        )
        fallback_code, fallback_ok = self._choose_mask(
            candidate_feasible, mask_bits, held_previous, registered_priority
        )
        feasible_code = torch.where(restricted_ok, restricted_code, fallback_code)
        any_feasible = restricted_ok | fallback_ok
        used_fallback = need_guard & ~restricted_ok & fallback_ok
        selected_mask = torch.where(
            need_guard & any_feasible, feasible_code, torch.zeros_like(feasible_code)
        )
        infeasible = need_guard & ~any_feasible
        all_four_policy_blocked = (
            infeasible & candidate_full_row_feasible[:, _N_MASKS - 1] & ~all_four_authorized
        )
        held_arm = mask_bits[selected_mask]

        arange_n = torch.arange(n, device=device)
        row_residual = row_residual_all[arange_n, selected_mask]
        selected_demand = target_demand[arange_n, selected_mask]
        selected_demand_clipped = target_demand_clipped[arange_n, selected_mask]
        trigger_pair_ids = torch.where(trigger_row, pair_ids, torch.full_like(pair_ids, -1))
        infeasible_reason = torch.where(
            ~derived_finite,
            torch.full_like(selected_mask, INFEASIBLE_NONFINITE_DERIVED),
            torch.where(
                unattributed_trigger,
                torch.full_like(selected_mask, INFEASIBLE_UNATTRIBUTED_TRIGGER),
                torch.where(
                    all_four_policy_blocked,
                    torch.full_like(selected_mask, INFEASIBLE_ALL_FOUR_EVIDENCE),
                    torch.where(
                        infeasible,
                        torch.full_like(selected_mask, INFEASIBLE_NO_FULL_ROW_MASK),
                        torch.full_like(selected_mask, INFEASIBLE_NONE),
                    ),
                ),
            ),
        )

        info: dict[str, torch.Tensor] = {
            "trigger_row": trigger_row,
            "triggered_arm": trigger_arm,
            "unattributed_trigger_row": unattributed_trigger_row,
            "unattributed_trigger": unattributed_trigger,
            "trigger_pair_ids": trigger_pair_ids,
            "backlog_closing": backlog_closing,
            "qd_closing": qd_closing,
            "inside_dmin": inside_dmin,
            "retained_latch": retained_latch,
            "selected_mask": selected_mask,
            "candidate_mask_bits": mask_bits,
            "candidate_full_row_feasible": candidate_full_row_feasible,
            "candidate_feasible": candidate_feasible,
            "candidate_feasible_restricted": candidate_feasible_restricted,
            "candidate_max_residual": candidate_max_residual,
            "target_demand": selected_demand,
            "target_demand_legacy": target_demand[:, 0],
            "target_demand_clipped": selected_demand_clipped,
            "target_demand_legacy_clipped": target_demand_clipped[:, 0],
            "row_residual": row_residual,
            "row_residual_legacy": row_residual_all[:, 0],
            "residual_tol": torch.tensor(self.cfg.residual_tol, dtype=dtype, device=device),
            "infeasible": infeasible,
            "infeasible_reason": infeasible_reason,
            "all_arms_held": held_arm.all(dim=-1),
            "independently_triggered_arm": independently_triggered_arm,
            "all_four_independent_evidence": all_four_independent_evidence,
            "all_four_authorized": all_four_authorized,
            "all_four_policy_blocked": all_four_policy_blocked,
            "used_fallback": used_fallback,
            "registered_priority_p": registered_priority,
            "derived_finite": derived_finite,
            "legacy_sum_finite": legacy_sum_finite,
            "rebase_sum_finite": rebase_sum_finite,
        }

        # A batch containing any infeasible environment authorizes no target at
        # all.  The explicit emergency sum type prevents accidental execution
        # of the unsafe mask-0 legacy target; a future caller must terminate or
        # enter its emergency path.  State is intentionally left unchanged.
        batch_requires_abort = (
            self._batch_requires_abort(infeasible) if _production else bool(infeasible.any())
        )
        if batch_requires_abort:
            return TargetGuardEmergency(
                held_arm=held_arm,
                trigger_pair_ids=trigger_pair_ids,
                selected_mask=selected_mask,
                infeasible=infeasible,
                info=info,
            )

        next_target = {
            arm: torch.where(held_arm[:, arm_index, None], rebase_target[arm], legacy_target[arm])
            for arm_index, arm in enumerate(ARM_KEYS)
        }
        self._held_arm = held_arm.detach().clone()
        return TargetGuardOutput(
            next_target=next_target,
            held_arm=held_arm,
            trigger_pair_ids=trigger_pair_ids,
            selected_mask=selected_mask,
            infeasible=infeasible,
            info=info,
        )

    def apply_production(
        self,
        *,
        q: dict[str, torch.Tensor],
        qd: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        exec_delta: dict[str, torch.Tensor],
        rows: ConstraintRows,
        dt: float,
        pair_ids: torch.Tensor | None = None,
        exempt: torch.Tensor | None = None,
        soft_limits: dict[str, torch.Tensor] | None = None,
        priority_p: torch.Tensor | None = None,
    ) -> TargetGuardOutput | TargetGuardEmergency:
        """Evaluate an enabled batch with one explicit host-side abort gate."""

        if not self.cfg.enabled:
            raise RuntimeError("production target guard must be enabled")
        return self.apply(
            q=q,
            qd=qd,
            target=target,
            exec_delta=exec_delta,
            rows=rows,
            dt=dt,
            pair_ids=pair_ids,
            exempt=exempt,
            soft_limits=soft_limits,
            priority_p=priority_p,
            _production=True,
        )


class TargetGuardRuntime:
    """Production lifecycle: fixed batch identity and permanent fail-closed poison."""

    def __init__(self, guard: TargetRebaseGuard):
        if not guard.cfg.enabled:
            raise ValueError("TargetGuardRuntime requires an enabled guard")
        self.guard = guard
        self._signature: tuple[int, torch.device, torch.dtype] | None = None
        self._abort: TargetGuardBatchAbort | TargetGuardContractError | None = None

    @property
    def poisoned(self) -> bool:
        return self._abort is not None

    @property
    def poison_error(self) -> TargetGuardBatchAbort | TargetGuardContractError | None:
        """Return the exact process-fatal object retained by this runtime."""

        return self._abort

    @staticmethod
    def _format_signature(signature: tuple[int, torch.device, torch.dtype]) -> str:
        n, device, dtype = signature
        return f"N={n},device={device},dtype={dtype}"

    def _check_signature(self, q: dict[str, torch.Tensor]) -> None:
        reference = q.get(ARM_KEYS[0])
        if not isinstance(reference, torch.Tensor) or reference.ndim != 2:
            return
        signature = (reference.shape[0], reference.device, reference.dtype)
        if self._signature is None:
            self._signature = signature
        elif signature != self._signature:
            raise TargetGuardContractError(
                contract_code="lifecycle_signature",
                field="batch_signature",
                expected=self._format_signature(self._signature),
                observed=self._format_signature(signature),
            )

    def apply(self, **kwargs: object) -> TargetGuardOutput:
        """Return an authorized output or raise and permanently poison runtime."""

        if self._abort is not None:
            raise self._abort
        try:
            q = kwargs.get("q")
            if isinstance(q, dict):
                self._check_signature(q)
            decision = self.guard.apply_production(**kwargs)  # type: ignore[arg-type]
        except TargetGuardContractError as error:
            self._abort = error
            raise
        if isinstance(decision, TargetGuardEmergency):
            abort = TargetGuardBatchAbort(decision)
            self._abort = abort
            raise abort
        return decision

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset selected latches; poison is process-fatal and never cleared."""

        if self._abort is not None:
            raise self._abort
        self.guard.reset(env_ids)
