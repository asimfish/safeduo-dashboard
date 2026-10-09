"""Recorded scalar contact strata; zero measured scalar is NOT no-contact proof."""
import numpy as np
from native_teacherforced_cpu_v1 import require


def contact_strata(scalar):
    s=np.asarray(scalar)
    require(s.ndim==2 and len(s)%2==0 and np.isfinite(s).all() and (s>=0).all(),'scalar micro x lane nonnegative even length')
    positive=s>0
    history=np.maximum.accumulate(positive,axis=0)
    macro=positive.reshape(-1,2,s.shape[1]).any(1)
    previous=np.zeros_like(macro); previous[1:]=positive[1:-1:2]
    linked=np.repeat(macro|previous,2,axis=0)
    return dict(all=np.ones_like(positive),
        event_scalar_zero=s==0,event_scalar_positive=positive,event_scalar_alarm_gt_0p1N=s>.1,
        recorded_contact_free_proxy=~linked,contact_linked_macro_or_preboundary=linked,
        no_positive_scalar_observed_so_far=~history,post_contact_quiet=(~linked)&history)


STRATA_DEFINITION={
    'event_scalar_zero':'At this recorded micro, max of all82 enabled owner-partner SUMABS scalar forces equals0.',
    'event_scalar_positive':'At this recorded micro, one or more recorded owner-partner scalar forces is >0.',
    'event_scalar_alarm_gt_0p1N':'Existing native physical scalar threshold >0.1N; not a new contact threshold.',
    'recorded_contact_free_proxy':'Both micros in macro and prior macro final boundary have zero recorded scalar; prior boundary for control0 is unrecorded, not certified zero.',
    'contact_linked_macro_or_preboundary':'Some scalar>0 in either current macro micro or previous macro final boundary. This is association, not causal attribution; sub0 can be linked to later sub1 contact.',
    'no_positive_scalar_observed_so_far':'All observed scalars through this micro are zero; initial pre-control contact remains unmeasured.',
    'post_contact_quiet':'Current macro/prior boundary quiet but positive scalar occurred earlier in retained history.',
    'qualification':'All masks retain original sample indices. Strata overlap except the declared proxy/linked partition. Zero force does not prove no contact constraints, hidden impulse history, friction or hard limits.'}


def error_statistics(x):
    x=np.asarray(x,dtype=np.float64)
    if x.size==0:return dict(rmse=None,mae=None,max_abs=None,signed_mean=None,empirical_p95_abs=None)
    return dict(rmse=float(np.sqrt(np.mean(x*x))),mae=float(np.mean(abs(x))),
                max_abs=float(np.max(abs(x))),signed_mean=float(np.mean(x)),empirical_p95_abs=float(np.quantile(abs(x),.95)))


def original_delay6(initial26,request26):
    q=np.asarray(initial26); r=np.asarray(request26)
    require(r.ndim==3 and r.shape[-1]==26 and q.shape==r.shape[1:],'original26 delayed request shape')
    require(r.dtype==q.dtype==np.float32,'exact recorded request dtype')
    return np.concatenate([np.repeat(q[None],6,axis=0),r],axis=0)[:len(r)]
