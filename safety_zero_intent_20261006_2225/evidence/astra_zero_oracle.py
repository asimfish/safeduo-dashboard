"""Independent scalar breakpoint oracle, derived from sealed reviewer source."""
from fractions import Fraction
import math

def rational(value):
    if not math.isfinite(float(value)):
        raise ValueError('nonfinite scalar')
    return Fraction.from_float(float(value))

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


def zero_interval_oracle(target, measured, rate_box, gap, lower, upper):
    # Candidate contract rejects rather than holding a target outside limits.
    target, lower, upper = map(rational, (target, lower, upper))
    if not lower <= target <= upper:
        raise ValueError('zero outside original interval')
    left, right, reachable = reference_interval_oracle(target, measured, rate_box, gap, lower, upper)
    return min(left, Fraction(0)), max(right, Fraction(0))
