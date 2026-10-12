"""Passive live-stage cooking representation and native offset readback.

No setter, cache release, physics step, collision filter, or USD edit occurs.
The requested representation still needs a coordinate/active-shape binding;
it is not automatically a native collision or future-motion certificate.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np

def capture(env, out, extraction_path, identity):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools, PhysxSchema
    import omni.usd
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from native_state_capture import array
    stage=omni.usd.get_context().get_stage()
    stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    assert stage_id>=0
    expected=json.loads(Path(extraction_path).read_text())['records']
    paths=sorted(set(r['native_collider'] for r in expected))
    extras=[r['path'] for r in identity['inventory'] if r['collision_enabled'] and r['rigid_owner']
            and any('/'+a+'/' in r['rigid_owner'] for a in ['U_L','U_R'])
            and r['rigid_owner'].rsplit('/',1)[-1] in ['forearm_link','wrist_2_link']]
    paths=sorted(set(paths+extras))
    inventory={r['path']:r for r in identity['inventory']};data={};rows=[];started=time.monotonic()
    cooking=get_physx_cooking_interface();cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    for i,path in enumerate(paths):
        assert time.monotonic()-started<300, 'Finite passive cooking-capture bound'
        assert inventory[path]['collision_enabled'] and inventory[path]['rigid_owner']
        prim=stage.GetPrimAtPath(path);owner=stage.GetPrimAtPath(inventory[path]['rigid_owner'])
        assert prim.IsValid() and owner.IsValid()
        # USD Gf matrices act on row vectors; retain this convention explicitly.
        relative=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(owner).GetInverse())
        row=dict(collider=path,owner=str(owner.GetPath()),primitive_type=prim.GetTypeName(),
                 collider_to_owner_row_vector_matrix=relative.tolist(),convexes=[])
        api=PhysxSchema.PhysxCollisionAPI(prim)
        for name,getter in [('schema_contact_offset',api.GetContactOffsetAttr),('schema_rest_offset',api.GetRestOffsetAttr)]:
            v=getter().Get();row[name]=float(v) if v is not None and np.isfinite(v) else None
        answer={}
        def callback(result,convexes):
            answer['result']=str(result);answer['valid']=result==PhysxCollisionRepresentationResult.RESULT_VALID
            # Copy within the callback; the binding data need not outlive it.
            answer['convexes']=[dict(vertices=np.array([[v.x,v.y,v.z] for v in c.vertices],dtype=np.float64),
                indices=np.asarray(c.indices,dtype=np.int64),planes=np.array([[p.plane.x,p.plane.y,p.plane.z,p.plane.w] for p in c.polygons]),
                polygon_counts=np.asarray([p.num_vertices for p in c.polygons],dtype=np.int64),
                polygon_starts=np.asarray([p.index_base for p in c.polygons],dtype=np.int64)) for c in convexes]
        cooking.request_convex_collision_representation(stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
            run_asynchronously=False,on_result=callback)
        row.update(callback_observed=bool(answer),result=answer.get('result','NO_SYNCHRONOUS_CALLBACK'),valid=answer.get('valid',False))
        for j,c in enumerate(answer.get('convexes',[])):
            assert c['vertices'].ndim==2 and c['vertices'].shape[1]==3 and np.isfinite(c['vertices']).all()
            prefix=f'collider_{i:03d}_convex_{j:03d}'
            for k,v in c.items():data[prefix+'_'+k]=v
            row['convexes'].append(dict(array_prefix=prefix,vertices=len(c['vertices']),indices=len(c['indices']),polygons=len(c['planes'])))
        rows.append(row)
        if (i+1)%25==0:print('PASSIVE_COOK_REPRESENTATION',i+1,'/',len(paths),'valid',sum(r['valid'] for r in rows),flush=True)
    offsets={}
    for a in ['F_L','F_R','U_L','U_R']:
        view=env._arms[a].root_physx_view
        contact=array(view.get_contact_offsets());rest=array(view.get_rest_offsets())
        assert contact.shape==rest.shape==(1,view.max_shapes) and np.isfinite(contact).all() and np.isfinite(rest).all()
        data[a+'_native_contact_offsets_m']=contact;data[a+'_native_rest_offsets_m']=rest
        offsets[a]=dict(max_shapes=view.max_shapes,contact_min_m=float(contact.min()),contact_max_m=float(contact.max()),
            rest_min_m=float(rest.min()),rest_max_m=float(rest.max()),per_shape_owner_index_mapping_certified=False)
    target=Path(out)/'native_cooking_representations.npz';np.savez_compressed(target,**data)
    result=dict(status='CLOSED_PASSIVE_LIVE_STAGE_REPRESENTATION_AND_OFFSET_READBACK',requested_colliders=len(rows),
        valid_representation_colliders=sum(r['valid'] for r in rows),rows=rows,actual_native_offset_ranges=offsets,
        npz_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),elapsed_s=time.monotonic()-started,
        row_vector_matrix_convention=True,live_stage_cooking_request=True,physics_or_USD_mutations=False,
        active_native_shape_identity_or_coordinate_certificate=False,future_loaded_geometry_certificate=False,fullSystem0_accepted=False)
    (Path(out)/'native_cooking_representations.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
