"""Independent scalar CPU oracle; no imports from any candidate implementation.

Fractions retain the exact binary input values. Enumerated endpoints, not a
vectorized tensor reduction, define the nominal frozen-J contract. Neither this
oracle nor its gap output is a physical feasibility/safety certificate.
"""
from fractions import Fraction
import math
import numpy as np


def rational(value):
    if not math.isfinite(float(value)):
        raise ValueError('nonfinite scalar')
    return Fraction.from_float(float(value))


def adaptive_gap(risk):
    """Unit map: metres in, radians out; constants are contract decimals."""
    risk = risk if isinstance(risk, Fraction) else rational(risk)
    low, width = Fraction(1, 100), Fraction(4, 100)
    weight = min(Fraction(1), max(Fraction(0), (risk-low)/width))
    return low + width*weight


def risk_oracle(distance, dmin, jacobian, q, qd, pending, exempt, dt):
    """Return each environment's exact minimum and its row/endpoint witness.

    Normalized interface: distance/dmin/exempt[N,R], J[N,R,D], q/qd[N,D],
    pending[6,N,D]. An empty eligible set is explicitly undefined (None),
    allowing the later registered interface to distinguish it from corruption.
    All supplied numerical entries must be finite, even when a row is exempt.
    """
    arrays = [np.asarray(x) for x in [distance,dmin,jacobian,q,qd,pending]]
    distance,dmin,jacobian,q,qd,pending = arrays
    if q.ndim != 2 or distance.ndim != 2:
        raise ValueError('q and distance must be matrices')
    n, dof = q.shape
    if distance.shape[0] != n:
        raise ValueError('environment mismatch')
    rows = distance.shape[1]
    shape_contract = [(dmin,(n,rows)),(jacobian,(n,rows,dof)),(qd,(n,dof)),(pending,(6,n,dof))]
    if any(a.shape != shape for a,shape in shape_contract):
        raise ValueError('exact shape mismatch; no broadcasting')
    exempt = np.asarray(exempt)
    if exempt.shape != (n,rows) or exempt.dtype.kind != 'b':
        raise ValueError('exact boolean exemption required')
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('nonfinite input, including exempt rows')
    dt = rational(dt)
    if dt <= 0:
        raise ValueError('dt must be positive')
    result = []
    for env in range(n):
        endpoints = [('measured', [Fraction(0)]*dof)]
        for slot in range(6):
            endpoints.append((f'pending_{slot}', [rational(pending[slot,env,j])-rational(q[env,j]) for j in range(dof)]))
        endpoints.append(('velocity_18dt', [rational(qd[env,j])*18*dt for j in range(dof)]))
        witnesses = []
        for row in range(rows):
            if exempt[env,row]:
                continue
            margin = rational(distance[env,row])-rational(dmin[env,row])
            for endpoint, displacement in endpoints:
                projected = margin
                for joint in range(dof):
                    projected += rational(jacobian[env,row,joint])*displacement[joint]
                witnesses.append((projected,row,endpoint))
        if not witnesses:
            result.append(dict(risk=None,gap=None,row=None,endpoint=None,eligible_rows=0))
            continue
        risk,row,endpoint = min(witnesses)
        result.append(dict(risk=risk,gap=adaptive_gap(risk),row=row,endpoint=endpoint,
                           eligible_rows=int((~exempt[env]).sum())))
    return result


def reference_interval_oracle(target, measured, rate_box, gap, lower=None, upper=None):
    """Geometric breakpoint oracle for a scalar target-increment interval.

    For disjoint intervals choose the allowed rate endpoint with least distance
    to the reference interval. This retains rate and never snaps stored targets.
    """
    target,measured,rate_box,gap = map(rational,[target,measured,rate_box,gap])
    if rate_box < 0 or gap < 0:
        raise ValueError('negative interval radius')
    ref_left,ref_right = measured-gap,measured+gap
    points = {-rate_box,rate_box}
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise ValueError('both original soft-limit endpoints required')
        lower,upper = rational(lower),rational(upper)
        if lower > upper:
            raise ValueError('reversed original limits')
        points.update([lower-target,upper-target])
    for boundary in [ref_left,ref_right]:
        increment = boundary-target
        if -rate_box <= increment <= rate_box:
            points.add(increment)
    points = {p for p in points if -rate_box <= p <= rate_box and
              (lower is None or lower <= target+p <= upper)}
    if not points:
        raise ValueError('original rate/soft-limit intersection empty')
    feasible = [p for p in points if ref_left <= target+p <= ref_right]
    if feasible:
        return min(feasible),max(feasible),True
    def violation(increment):
        location = target+increment
        if location < ref_left:
            return ref_left-location
        return location-ref_right
    best = min(points,key=lambda p:(violation(p),abs(p)))
    return best,best,False


def fixtures():
    """Minimal exact-input counterexamples fixed before candidate inspection."""
    q=np.zeros((1,26),dtype=np.float64)
    base=dict(distance=np.array([[.0625]]),dmin=np.array([[.03125]]),
              jacobian=np.zeros((1,1,26)),q=q,qd=q.copy(),
              pending=np.repeat(q[None],6,axis=0),exempt=np.zeros((1,1),bool),dt=.0625)
    def copy():
        return {k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base.items()}
    velocity=copy();velocity['jacobian'][0,0,0]=-1;velocity['qd'][0,0]=.125
    intermediate=copy();intermediate['jacobian'][0,0,0]=-1;intermediate['pending'][2,0,0]=.125
    joint=copy();joint['jacobian'][0,0,0]=joint['jacobian'][0,0,7]=joint['jacobian'][0,0,14]=joint['jacobian'][0,0,20]=-1
    joint['pending'][0,0,[0,7,14,20]]=.015625
    return dict(velocity_without_target_debt=velocity,intermediate_pending_hazard=intermediate,
                four_arm_joint_closing=joint)
