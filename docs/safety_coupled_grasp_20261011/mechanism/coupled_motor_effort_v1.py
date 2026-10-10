"""Map generalized efforts onto the independent URDF motors by virtual work."""
import numpy as np
def motor_effort_projection(names,relations):
    names=tuple(names);by_slave={r['slave']:r for r in relations}
    assert len(by_slave)==len(relations) and len(set(names))==len(names)
    assert all(r['slave'] in names and r['master'] in names for r in relations)
    cache={}
    def affine(n,seen):
        assert n not in seen,'Cyclic mimic relation'
        if n in cache:return cache[n]
        if n not in by_slave:result=(n,1.)
        else:
            r=by_slave[n];root,scale=affine(r['master'],seen|{n});result=(root,scale*float(r['urdf_multiplier']))
        assert np.isfinite(result[1]) and result[1]!=0
        cache[n]=result;return result
    M=np.zeros((len(names),len(names)))
    for i,n in enumerate(names):
        root,scale=affine(n,set());M[names.index(root),i]=scale
    assert all(not M[names.index(n)].any() for n in by_slave)
    return M
