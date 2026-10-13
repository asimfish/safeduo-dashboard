"""Opposite contact faces from the object's native point-contact view.

The view reports outward object-side normals. Compare the *difference* between
two contact positions along those normals, independent of world origin or the
payload's length. This is descriptive opposition, not friction force closure.
"""
import numpy as np


def opposed_contact(bodies, normals, points, forces):
    n=np.asarray(normals,float);p=np.asarray(points,float);f=np.asarray(forces,float)
    if n.shape != (len(bodies),3) or p.shape != n.shape or f.shape != (len(bodies),):
        raise ValueError('one native normal, point and scalar force per body required')
    if not all(np.isfinite(x).all() for x in [n,p,f]):
        raise ValueError('nonfinite native contact')
    norm=np.linalg.norm(n,axis=1)
    eligible=np.flatnonzero((np.abs(f)>.1)&(np.abs(norm-1.)<=.02))
    for ii,i in enumerate(eligible):
        for j in eligible[ii+1:]:
            if bodies[i] == bodies[j]:continue
            ni=n[i]/norm[i];nj=n[j]/norm[j];separation=p[i]-p[j]
            if ni@nj <= -.5 and separation@ni >= .002 and separation@nj <= -.002:
                return True
    return False
