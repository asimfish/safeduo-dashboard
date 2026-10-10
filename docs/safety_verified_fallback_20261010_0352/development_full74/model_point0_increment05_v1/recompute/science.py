"""Independent arithmetic; never imports a native producer or simulator."""
import hashlib
import numpy as np
from evidence_io import ARMS, require, exact

REASONS = ('nonfinite', 'hard74', 'velocity74', 'soft26', 'raw9021', 'point_scalar82', 'held_target74')
STATES = ('UNUSED', 'IN_PROGRESS_OR_INCOMPLETE', 'OBSERVED_REJECT', 'PREFIX_ELIGIBLE_INITIAL_CONTACT_UNKNOWN', 'UNKNOWN_MEASUREMENT')
NAMESPACE = 'astra.full74.screen.DEVELOPMENT.20261010.v1'
SEEDS = [int.from_bytes(hashlib.sha256(f'{NAMESPACE}/{i}'.encode()).digest()[:16], 'big') for i in range(4)]


def parameters_layout(p):
    hard, soft, vmax, columns, names, slices = [], [], [], [], [], {}
    offset = 0
    for arm, width, controlled in zip(ARMS, (19, 19, 18, 18), (7, 7, 6, 6)):
        n, ids = p[arm+'_native_joint_names'], p[arm+'_controlled_joint_indices']
        require(n.shape == (width,) and len(set(n.tolist())) == width, 'joint identity')
        require(ids.shape == (controlled,) and ids.dtype == np.int64 and len(set(ids.tolist())) == controlled and np.all((ids>=0)&(ids<width)), 'controlled identity')
        for key, shape in [('hard_limits',(32,width,2)), ('max_velocity',(32,width)), ('soft_limits',(32,controlled,2))]:
            x = p[arm+'_'+key]
            require(x.shape == shape and x.dtype == np.float32 and np.isfinite(x).all(), 'parameter layout '+key)
            exact(x, np.broadcast_to(x[0], shape), 'homogeneous parameters')
        hard.append(p[arm+'_hard_limits'][0]); soft.append(p[arm+'_soft_limits'][0]); vmax.append(p[arm+'_max_velocity'][0])
        columns.extend((offset+ids).tolist()); names.extend(arm+'/'+str(v) for v in n)
        slices[arm] = (offset, offset+width); offset += width
    h, s, v = np.concatenate(hard), np.concatenate(soft), np.concatenate(vmax)
    c = np.array(columns, dtype=np.int64); b = h.copy()
    b[c,0] = np.maximum(h[c,0], s[:,0]); b[c,1] = np.minimum(h[c,1], s[:,1])
    require(np.all(b[:,0]<b[:,1]) and np.all(v>0), 'invalid bounds')
    return dict(hard=h, soft=s, vmax=v, controlled=c, bounds=b, names=np.array(names), slices=slices)


def flags(row, layout, target, fresh):
    for key, shape in [('q74',(32,74)), ('qd74',(32,74)), ('target74',(32,74)), ('raw9021',(32,9021)), ('scalar82',(32,82))]:
        require(row[key].shape == shape and row[key].dtype.kind == 'f', 'measurement shape '+key)
    finite = np.ones(32, bool)
    for key, x in row.items():
        if key in ('q74','qd74','target74','raw9021','scalar82') or key.endswith(('_root_xyzw','_root_velocity','_link_xyzw')):
            finite &= np.isfinite(x).reshape(32,-1).all(1)
    q, v, h, s, c = row['q74'], row['qd74'], layout['hard'], layout['soft'], layout['controlled']
    require(target.dtype == row['target74'].dtype and target.shape == (32,74), 'target layout')
    drift = np.frombuffer(row['target74'].tobytes(),np.uint8).reshape(32,-1) != np.frombuffer(target.tobytes(),np.uint8).reshape(32,-1)
    return np.column_stack((~finite, ((q<h[:,0]-1e-5)|(q>h[:,1]+1e-5)).any(1),
        (np.abs(v)>layout['vmax']*1.0001).any(1), ((q[:,c]<s[:,0]-1e-5)|(q[:,c]>s[:,1]+1e-5)).any(1),
        (row['raw9021']<0).any(1), (row['scalar82']>.1).any(1)&bool(fresh), drift.any(1)))


def scalar_contacts(counts, starts, indices, forces, capacity=262144):
    require(counts.ndim == 2 and counts.shape == starts.shape and counts.dtype.kind in 'iu' and starts.dtype.kind in 'iu' and counts.dtype.itemsize in (4,8) and starts.dtype.itemsize in (4,8), 'contact integer matrix')
    intervals=[]; total=0
    for pair,(c,s) in enumerate(zip(counts.flat,starts.flat)):
        c,s=int(c),int(s)
        require(c>=0 and 0<=s<=capacity and c<=capacity-s, 'negative/out-of-capacity contact')
        if c:
            require(s+c<capacity, 'active contact touches capacity')
            intervals.append((s,s+c,pair)); total+=c
    require(total<capacity, 'total capacity touched')
    previous=0
    for start,end,_ in sorted(intervals):
        require(start>=previous, 'overlapping point ownership'); previous=end
    for key in ('point_indices','sensor_indices','partner_indices'):
        require(indices[key].shape==(total,) and indices[key].dtype==np.int64, 'point index layout')
    pos=0
    for start,end,pair in intervals:
        sl=slice(pos,pos+end-start); sensor,partner=divmod(pair,counts.shape[1])
        exact(indices['point_indices'][sl],np.arange(start,end,dtype=np.int64),'native point indices')
        require(np.all(indices['sensor_indices'][sl]==sensor) and np.all(indices['partner_indices'][sl]==partner),'native point ownership')
        pos+=end-start
    require(forces.shape==(total,1) and forces.dtype in (np.float32,np.float64) and np.isfinite(forces).all(),'finite owned normal forces')
    sums=np.zeros(counts.shape,np.float64)
    np.add.at(sums,(indices['sensor_indices'],indices['partner_indices']),np.abs(forces[:,0].astype(np.float64)))
    require(np.isfinite(sums).all(),'nonfinite scalar sum')
    return sums


def geometry(row, identity, table_centers, table_half, origins):
    """Rebuild every sphere/pair/table row from measured link poses, in CPU float64.

    Comparison tolerance is a numerical cross-backend diagnostic, never a
    relaxed zero-gap predicate or a certified physical/global error bound.
    """
    centers=[]
    offsets=np.asarray(identity['sphere_offsets'],np.float64)
    radii=np.asarray(identity['sphere_radii_m'],np.float64)
    for arm in ARMS:
        lo,hi=identity['arm_slices'][arm]
        pose=row[arm+'_link_xyzw'][:,identity['sphere_body_indices'][arm]].astype(np.float64)
        require(np.isfinite(pose).all(),'nonfinite geometry pose')
        q=pose[...,3:]; require(np.all(np.abs(np.linalg.norm(q,axis=-1)-1)<1e-4),'native geometry quaternion')
        v=np.broadcast_to(offsets[lo:hi],pose[...,:3].shape)
        uv=np.cross(q[...,:3],v)
        centers.append(pose[...,:3]+v+2*(q[...,3:]*uv+np.cross(q[...,:3],uv)))
    centers=np.concatenate(centers,axis=1)
    pairs=np.asarray(identity['pair_sphere_idx'],np.int64); cut=identity['table_slice_start']
    require(pairs.shape==(9021,2) and cut==8745 and centers.shape==(32,142,3),'global geometry identity')
    a,b=pairs[:cut].T
    sphere=np.linalg.norm(centers[:,a]-centers[:,b],axis=-1)-(radii[a]+radii[b])
    a,b=pairs[cut:].T
    delta=np.abs(centers[:,a]-(origins[:,None,:]+table_centers[b]))-table_half[b]
    table=np.linalg.norm(np.maximum(delta,0),axis=-1)+np.minimum(delta.max(-1),0)-radii[a]
    return np.concatenate((sphere,table),axis=1), centers


def classify(all_flags, *, native_closed, v7_camera):
    require(all_flags.shape==(13,32,7) and all_flags.dtype==bool,'13x32x7 predicate denominator')
    result=np.full(32,4,np.uint8)
    if native_closed:
        bad=all_flags.any(axis=(0,2)); result[bad]=2
        if v7_camera: result[~bad]=3
    return result


def counts(a):
    require(np.all(a<5),'unknown ledger state')
    return {name:int(np.count_nonzero(a==i)) for i,name in enumerate(STATES)}
