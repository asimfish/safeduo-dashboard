import time,statistics
import numpy as np
import torch

def snapshot(view,dt):
    # Preserve legacy getter ordering and clone before another accessor reuses it.
    cp=lambda x:x.detach().clone().cpu().numpy()
    net=cp(view.get_net_contact_forces(dt));matrix=cp(view.get_contact_force_matrix(dt))
    normal=[cp(x) for x in view.get_contact_data(dt)];friction=[cp(x) for x in view.get_friction_data(dt)]
    return dict(net=net,matrix=matrix,normal=normal,friction=friction)

class BatchContactProbe:
    def __init__(self,sv,hands,objects,capacity,dt):
        self.dt=dt;self.rows=[];self.groups=[];self.capacity=capacity
        for arm in ['F_L','F_R','U_L','U_R']:
            vv=[(v,p) for e,a,v,p in hands if a==arm];self._group(sv,'hand_'+arm,vv,capacity)
        for obj in ['beam700','beam300']:
            vv=[(v,list(v.sensor_paths)) for e,o,v,p in objects if o==obj];self._group(sv,'object_'+obj,vv,capacity)

    def _group(self,sv,name,legacy,capacity):
        paths=[s for v,p in legacy for s in v.sensor_paths];filters=[list(f) for v,p in legacy for f in v.filter_paths]
        view=sv.create_rigid_contact_view(paths,filter_patterns=filters,max_contact_data_count=capacity*len(legacy))
        assert view.check() and set(view.sensor_paths)==set(paths)
        rowmap={path:i for i,path in enumerate(view.sensor_paths)}
        actual=list(view.filter_paths)
        for v,p in legacy:
            for path,ff in zip(v.sensor_paths,v.filter_paths):assert list(actual[rowmap[path]])==list(ff)
        self.groups.append((name,legacy,view,rowmap))

    def measure(self,step):
        # Timing covers the same native-contact reads including CPU copies, GPU synchronized.
        torch.cuda.synchronize();st=time.perf_counter()
        old={name:[snapshot(v,self.dt) for v,p in legacy] for name,legacy,batch,mapper in self.groups}
        torch.cuda.synchronize();old_s=time.perf_counter()-st
        torch.cuda.synchronize();st=time.perf_counter()
        new={name:snapshot(batch,self.dt) for name,legacy,batch,mapper in self.groups}
        torch.cuda.synchronize();new_s=time.perf_counter()-st
        count_error=force_error=point_error=0.;used=0
        for name,legacy,batch,mapper in self.groups:
            now=new[name]
            for (v,p),prev in zip(legacy,old[name]):
                ids=[mapper[path] for path in v.sensor_paths]
                for key in ['net','matrix']:
                    er=float(abs(prev[key]-now[key][ids]).max());force_error=max(force_error,er);assert er<=1e-5,(name,step,key,er)
                for kind in ['normal','friction']:
                    aa=prev[kind];bb=now[kind];ac,ast=aa[-2:];bc=bb[-2][ids];bst=bb[-1][ids]
                    assert np.array_equal(ac,bc),(name,kind,'count');used+=int(ac.sum())
                    # Contact ordering is not assumed. Compare matched original point tuples.
                    for si,fi in zip(*np.nonzero(ac)):
                        n=int(ac[si,fi]);start=int(ast[si,fi]);bs=int(bst[si,fi]);a=np.concatenate([x[start:start+n].reshape(n,-1) for x in aa[:-2]],axis=1);b=np.concatenate([x[bs:bs+n].reshape(n,-1) for x in bb[:-2]],axis=1)
                        a=a[np.lexsort(a.T[::-1])];b=b[np.lexsort(b.T[::-1])];er=float(abs(a-b).max());point_error=max(point_error,er);assert er<=1e-5,(name,kind,step,'points',er)
        self.rows.append(dict(step=step,legacy_read_s=old_s,batch_read_s=new_s,max_matrix_or_net_error_n=force_error,max_point_tuple_error=point_error,used_normal_and_friction_points=used));print('BATCH_CONTACT_PARITY',self.rows[-1],flush=True)

    def results(self):
        measured=self.rows[3:]
        return dict(status='PASS_SAME_FRAME_NATIVE_CONTACT_PARITY',warmup_samples=3,measured_samples=len(measured),legacy_median_s=statistics.median(x['legacy_read_s'] for x in measured),batch_median_s=statistics.median(x['batch_read_s'] for x in measured),rows=self.rows,scope='Same native states and same sensor/filter identities; no end-to-end throughput claim.')
