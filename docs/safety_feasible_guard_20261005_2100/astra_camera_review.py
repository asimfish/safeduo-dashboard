#!/usr/bin/env python3
"""Two closed camera groups only; no cell NPZ/protocol/receipt consumption.

Independent NumPy affine-camera and normalized-plane arithmetic. Actual source
readback provenance is producer evidence; offline snapshots are independently
compared. Human observations below were recorded after view_image of all18 PNG.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image

H = Path('/home/liyufeng/safeduo/artifacts/safety_feasible_guard_20261005_2100')
RAW = Path('/mnt/nas/data/lyf/double_hand/safety_feasible_guard_20261005_2100')
ROOT = RAW / 'visual_retry_1' / 'joint_reference'
VIEWS = ('overview', 'front', 'reverse', 'f_pair', 'f_opposite_low', 'f_opposite_high',
         'u_pair', 'u_opposite_low', 'u_opposite_high')
GROUPS = (('scheduled', 0, 75), ('first_failure', 38, 116))
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
PLAN_FIELDS = ('cwd', 'sources', 'original_source_sha256', 'actor_sha256', 'slots', 'steps',
               'expected_scheduled_groups', 'expected_images_range', 'first_failure_groups_max',
               'first_failure_capture_rule')
HUMAN = {
    'scheduled/0/75': {
        'overview': '四个基座、四臂和桌面可辨；主体较小，手指细节不足以判断毫米间隙。',
        'front': 'F双臂投影部分重叠，U前臂/手部也有重叠；桌面边界清楚。',
        'reverse': '反侧补出另一侧关节，但两U臂局部重叠；F侧远处指尖尺寸较小。',
        'f_pair': 'F两臂可辨；同桌U臂位于近景，视觉上较大，不能把标签当作无前景遮挡证明。',
        'f_opposite_low': '前后F肩肘/手部有重叠，U局部也重叠；桌面右侧超出画面。',
        'f_opposite_high': '俯视有助分开F前后臂，桌面下边裁切；末端细节仍有限。',
        'u_pair': 'U两臂和大部分桌面可辨，U腕/末端相互重叠；下方桌边裁切。',
        'u_opposite_low': '灰色F前景遮住部分白色U臂/末端；四个基座及桌面仍可辨。',
        'u_opposite_high': '俯视补足白色U臂姿态，但另一臂/自身壳体后的表面不可见。',
    },
    'first_failure/38/116': {
        'overview': '四个基座及桌面可辨，低伸展灰色臂和上方伸展白色臂可定位；末端太小，无法读出负裕度。',
        'front': '灰色前臂低伸展和白色臂向桌心伸展较清楚；两白色臂和手部局部重叠。',
        'reverse': '反侧可见低伸展灰色臂，中央多个手/腕投影靠近并遮挡，不能判定真实接触。',
        'f_pair': 'F低伸展姿态可辨，白色U前景遮挡部分F腕/工具邻近区域。',
        'f_opposite_low': '低侧更清楚显示灰色肩肘与手，但两灰色手/前臂投影重叠；桌后侧/底面不可见。',
        'f_opposite_high': '俯视补出灰色低伸展臂和两个末端；桌面下边裁切，不能测接触深度。',
        'u_pair': '白色U两臂及朝桌心的腕较清楚；局部手部重叠，桌面下角裁切。',
        'u_opposite_low': '四臂可辨但灰色近景遮住白色U臂/中央手部，单图不能显示全部最近间隙。',
        'u_opposite_high': '上方反侧补足四臂相对位置及低伸展灰色臂；中央手/腕仍局部重叠。',
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ClosedInputs:
    def __init__(self):
        self.files = {}

    def read(self, path):
        p = Path(path).resolve(strict=True)
        data = p.read_bytes()
        entry = dict(path=str(p), bytes=len(data), sha256_before=sha(data))
        if str(p) in self.files:
            require(self.files[str(p)]['sha256_before'] == entry['sha256_before'], 'changed between consumers')
        self.files[str(p)] = entry
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def npz(self, path):
        with np.load(io.BytesIO(self.read(path)), allow_pickle=False) as z:
            return {k:z[k].copy() for k in z.files}

    def finish(self):
        for entry in self.files.values():
            entry['sha256_after'] = sha(Path(entry['path']).read_bytes())
            entry['unchanged'] = entry['sha256_before'] == entry['sha256_after']
            require(entry['unchanged'], f'closed input mutated: {entry["path"]}')
        return list(self.files.values())


def close(a, b, tol, context):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), f'invalid {context}')
    error = float(np.max(np.abs(a-b)))
    require(error <= tol, f'{context} arithmetic difference {error} > {tol}')
    return error


def canonical_child(relative):
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'noncanonical relative path')
    p = (ROOT / relative).resolve(strict=True)
    require(p.is_relative_to(ROOT.resolve()), 'input escaped owned visual root')
    return p


def quaternion_matrix(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def camera_math(view, centers, radii, sphere_arm_ids, width, height):
    W = np.asarray(view['actual_camera_to_world_row_matrix'], np.float64)
    require(W.shape == (4,4) and np.isfinite(W).all(), 'nonfinite/invalid affine camera')
    require(np.array_equal(W[:3,3], np.zeros(3)) and W[3,3] == 1., 'camera is not affine')
    basis, eye = W[:3,:3], W[3,:3]
    orthogonal_error = close(basis@basis.T, np.eye(3), 1e-6, 'camera orthonormal rotation')
    require(abs(np.linalg.det(basis)-1) < 1e-6, 'camera handedness/scaling error')
    position_error = close(eye, view['actual_position_world_m'], 1e-8, 'saved camera translation')
    q = np.asarray(view['actual_quat_opengl_wxyz'], np.float64)
    require(abs(np.dot(q,q)-1.) < 1e-6, 'camera quaternion not unit')
    rotation_error = close(basis.T, quaternion_matrix(q), 1e-6, 'row USD matrix versus OpenGL quaternion')
    o = view['actual_usd_optics']
    require(o['focal_length'] > 0 and o['horizontal_aperture'] > 0 and o['vertical_aperture'] > 0,
            'nonpositive pinhole optics')
    require(o['horizontal_aperture_offset'] == 0 and o['vertical_aperture_offset'] == 0,
            'this oracle requires observed zero aperture offsets')
    K = np.array([[width*o['focal_length']/o['horizontal_aperture'],0,width/2],
                  [0,height*o['focal_length']/o['vertical_aperture'],height/2],[0,0,1.]], np.float64)
    saved_K = np.asarray(view['actual_intrinsic_matrix'], np.float64)
    intrinsic_error = close(saved_K, K, 2e-4, 'USD optics independently reconstructed K')
    # Use reconstructed K and orthonormal affine basis, not the producer's
    # saved K + inverse-matrix path. USD's row vectors are world camera axes.
    mask = sphere_arm_ids<2 if view['_name'].startswith('f_') else sphere_arm_ids>=2 if view['_name'].startswith('u_') else np.ones(len(radii),bool)
    camera_xyz_gl = (centers[mask] - eye) @ basis.T
    xyz = camera_xyz_gl * np.array([1.,1.,-1.])
    depths = xyz[:,2]
    require((depths>0).all(), 'target centers behind actual camera')
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    pixels = np.column_stack((fx*xyz[:,0]/depths+cx, cy-fy*xyz[:,1]/depths))
    require((pixels[:,0]>=0).all() and (pixels[:,0]<width).all() and
            (pixels[:,1]>=0).all() and (pixels[:,1]<height).all(), 'target center outside image')
    projection = view['observed_center_projection']
    require(projection['count']==len(depths) and projection['all_inside'] is True, 'wrong center group/flag')
    depth_error = close([depths.min()], [projection['depth_min_m']], 1e-7, 'projection depth')
    # This comparison tolerance covers float32 K storage only, not visibility
    # or physical geometry. Preserve errors and actual positive plane margins.
    pixel_error = max(close(pixels.min(0), projection['pixel_min'], 1e-4, 'min pixels'),
                      close(pixels.max(0), projection['pixel_max'], 1e-4, 'max pixels'))
    near, far = np.asarray(view['actual_clipping_range_m'], np.float64)
    require(np.isfinite([near,far]).all() and 0<near<far, 'invalid actual near/far readback')
    require('USD /World/viz_cam' in view['actual_pose_source'] and
            'USD /World/viz_cam clippingRange' in view['actual_clipping_source'], 'unexpected readback provenance')
    # Six inward-facing half-spaces in right/up/forward camera coordinates.
    # Normalize normals, subtract each sphere radius in metres.
    normals = np.array([[fx,0,cx],[-fx,0,width-cx],[0,-fy,cy],[0,fy,height-cy],
                        [0,0,1],[0,0,-1]], np.float64)
    offsets = np.array([0,0,0,0,-near,far], np.float64)
    norm = np.linalg.norm(normals,axis=1)
    margins = xyz @ (normals/norm[:,None]).T + offsets/norm - radii[mask,None]
    names = ['left','right','top','bottom','near','far']
    saved = view['observed_sphere_frustum']
    require(saved['plane_names'] == names and saved['spheres']==len(depths), 'frustum identity mismatch')
    plane_error = close(margins.min(0), saved['minimum_by_plane_m'], 1e-7, 'six-plane sphere margins')
    close([margins.min()], [saved['minimum_plane_margin_m']], 1e-7, 'overall sphere margin')
    require((margins>=.05).all() and saved['all_contained'] is True, 'represented sphere envelope not at least50mm within frustum')
    projected_extent = pixels.max(0)-pixels.min(0)
    return dict(spheres=int(mask.sum()), reconstructed_K=K.tolist(), intrinsic_max_abs_error_px=intrinsic_error,
        matrix_orthogonality_error=orthogonal_error, translation_error_m=position_error,
        quaternion_rotation_error=rotation_error, projection_error_px=pixel_error, depth_error_m=depth_error,
        plane_margin_reconstruction_error_m=plane_error, minimum_by_plane_m=margins.min(0).tolist(),
        minimum_plane_margin_m=float(margins.min()), center_depth_min_m=float(depths.min()),
        center_depth_max_m=float(depths.max()), sphere_center_pixel_extent=projected_extent.tolist(),
        clipping_range_m=[float(near),float(far)],
        sdk_position_matches_actual=bool(view['sdk_position_matches_actual']),
        scope='saved actual USD readback is producer evidence; independent offline six-plane math for target-group spheres')


def group_audit(inputs, identity, kind, env, step):
    relative = f'multiview/{kind}/env_{env:03d}/step_{step:04d}_state.json'
    path = canonical_child(relative)
    s = inputs.json(path)
    require(s['env_id']==env and s['step']==step and s['capture_kind']==kind and
            s['visual_mode']=='joint_reference' and s['environment_count']==64, 'wrong state identity')
    require(set(s['views'])==set(VIEWS) and len(s['images'])==9, 'view set not exactly nine')
    require(s['first_negative_capture']==(kind=='first_failure'), 'wrong first-failure label')
    expected_bf = f'native_render_{kind}_step_{step:04d}_before.npz'
    expected_af = f'native_render_{kind}_step_{step:04d}_after_env_{env:03d}.npz'
    require(s['fresh_native_snapshot']==expected_bf and s['fresh_native_after_snapshot']==expected_af,
            'native snapshot path does not match state identity')
    bf, af = canonical_child(expected_bf), canonical_child(expected_af)
    before, after = inputs.npz(bf), inputs.npz(af)
    require(inputs.files[str(bf)]['sha256_before']==s['fresh_native_before_sha256'] and
            inputs.files[str(af)]['sha256_before']==s['fresh_native_after_sha256'], 'native SHA mismatch')
    keys = {a+'_'+k for a in ARMS for k in ('q','qd','root','root_vel')}
    require(set(before)==keys and set(after)==keys, 'not exact all64 16-array snapshot set')
    native_arrays, scalar_count = [], 0
    for a in ARMS:
        joints = len(s['fresh_native_all_joint_names'][a])
        for key, width in (('q',joints),('qd',joints),('root',7),('root_vel',6)):
            name=a+'_'+key;b,c=before[name],after[name]
            require(b.shape==(64,width) and b.dtype==np.float32 and np.isfinite(b).all(), f'invalid native {name}')
            require(c.dtype==b.dtype and c.shape==b.shape and b.tobytes()==c.tobytes() and np.array_equal(b,c), f'native changed: {name}')
            selected = np.asarray(s['fresh_native_selected'][a][key],b.dtype)
            require(selected.shape==(width,) and b[env].tobytes()==selected.tobytes(), f'state/native selected binding {name}')
            scalar_count += b.size
            native_arrays.append(dict(key=name,shape=list(b.shape),dtype=str(b.dtype),bitwise_exact=True))
        idx=np.asarray(s['fresh_native_controlled_joint_indices'][a],np.int64)
        require(idx.shape==((7,) if a.startswith('F') else (6,)) and len(set(idx.tolist()))==len(idx) and
                (idx>=0).all() and (idx<joints).all(), 'actual controlled joint indices invalid')
        for key in ('q','qd'):
            selected = np.asarray(s['arms'][a][key],np.float32)
            require(selected.tobytes()==before[a+'_'+key][env,idx].tobytes(), f'controlled state/native {a}:{key}')
    require(s['fresh_native_physx_unchanged_all_64'] is True, 'producer native invariant false')
    centers=np.asarray(s['sphere_centers_world_m'],np.float64);radii=np.asarray(s['sphere_radii_m'],np.float64)
    require(centers.shape==(142,3) and radii.shape==(142,) and np.isfinite(centers).all() and (radii>0).all(),
            'invalid represented sphere geometry')
    require(np.array_equal(radii,np.asarray(identity['sphere_radii_m'],np.float64)), 'radius identity mismatch')
    sphere_arms=np.asarray(identity['sphere_arm_id'],np.int64)
    require(sphere_arms.shape==(142,) and set(sphere_arms)=={0,1,2,3}, 'sphere group identity invalid')
    # Per-state geometry summary only. No chronology assertion from two photos.
    d=np.asarray(s['full_distances_m'],np.float32);exempt=np.asarray(s['full_exempt'],bool)
    require(d.shape==(9021,) and exempt.shape==(9021,) and np.isfinite(d).all(), 'invalid saved full geometry')
    relevant=np.where(exempt,np.inf,d)
    if kind=='first_failure':require((relevant<0).any(),'first-failure state has no negative nonexempt sphere margin')
    images={}
    for item in s['images']:
        name=Path(item['path']).stem.removeprefix(f'step_{step:04d}_')
        require(name in VIEWS and name not in images and
                item['path']==f'multiview/{kind}/env_{env:03d}/step_{step:04d}_{name}.png', 'image/state set not canonical bijection')
        images[name]=item
    per_view, observed = {}, []
    for name in VIEWS:
        item=images[name];ip=canonical_child(item['path']);data=inputs.read(ip)
        require(sha(data)==item['sha256'], 'image SHA does not match own state')
        with Image.open(io.BytesIO(data)) as im: im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load();require(im.size==(1280,720) and im.mode=='RGB', 'PNG dimensions/mode wrong')
        v=dict(s['views'][name]);v['_name']=name
        per_view[name]=camera_math(v,centers,radii,sphere_arms,1280,720)
        close(np.asarray(v['eye'])+np.asarray(s['origin_world_m']),v['actual_position_world_m'],1e-5,'intended eye + origin versus actual USD position')
        per_view[name]['image_path']=str(ip);per_view[name]['human_observation']=HUMAN[f'{kind}/{env}/{step}'][name]
        observed.append(str(ip))
    nearest=int(relevant.argmin())
    return dict(state_path=str(path),state_sha256=inputs.files[str(path)]['sha256_before'],
        capture_kind=kind,env=env,step=step,risk_stratum=s['risk_stratum'],state_time_s=s['state_time_s'],
        native_before=str(bf),native_after=str(af),native_array_checks=16,native_scalar_checks=scalar_count,
        all64_native_16_arrays_bitwise_exact=True,native_array_details=native_arrays,
        controlled_q_qd_and_selected_full_q_qd_root_rootvel_bound=True,
        min_saved_nonexempt_geometry_m=float(relevant[nearest]),min_saved_row=nearest,
        negative_saved_nonexempt_rows=int((relevant<0).sum()),
        first_in_time_verification='not performed; only this saved state checked, no unfinished dense consumed',
        views=per_view,observed_image_files=observed)


def supersession(inputs):
    a=inputs.json(H/'visual_plan_v2.json');b=inputs.json(H/'visual_plan_v3.json')
    require(all(a[k]==b[k] for k in PLAN_FIELDS), 'unexpected frozen visual design/source change')
    require(len(a['jobs'])==2 and len(b['jobs'])==2,'visual job set changed')
    for old,new in zip(a['jobs'],b['jobs']):
        x,y=old['argv'][:],new['argv'][:]
        i=y.index('--methods');require(y[i+1]=='system0','new explicit method differs from old default');del y[i:i+2]
        target=y[y.index('--out')+1]
        require(Path(target)==RAW/'visual_retry_1'/old['mode'],'wrong new visual output namespace')
        y[y.index('--out')+1]=x[x.index('--out')+1]
        require(x==y and old['env']==new['env'] and old['mode']==new['mode'],'unexpected launch delta')
    for name in ('visual_runner_v3.py','guard_runner_v2.py','joint_repair.py','reference_envelope.py',
                 'full_finite_guard.py','projection_diagnostics.py'):
        p=H/name;require(sha(inputs.read(p))==a['sources'][str(p)]==b['sources'][str(p)],'frozen control/observer SHA mismatch')
    # Old attempt is closed; no running retry protocol/receipt is read or hashed.
    old_path=RAW/'visual'/'joint_reference'/'visual_protocol.json'
    old=inputs.json(old_path);execution=inputs.json(H/'visual_execution.json')
    require(old['status']=='failed' and old.get('completed_windows',0)==0 and
            execution['status']=='failed' and execution['jobs'][0]['exit_code']==0, 'old failure scope differs')
    old_images=list(old_path.parent.rglob('*.png'));require(not old_images,'old failed attempt has unexpected PNG')
    return dict(old_attempt_status='failed',old_kit_exit_code=0,old_completed_windows=0,old_PNG=0,
                old_error=old.get('error'),old_path=str(old_path),
                only_execution_delta='explicit --methods system0 plus new output namespace; frozen control/source bytes identical',
                current_capture_scope='two closed joint_reference groups only; whole retry completion pending')


def markdown(result):
    lines=['# Astra：两个闭合九视角组的有界独立复核','',
           f'结论：**{result["status"]}**。只审 joint_reference 的 env000/step75（scheduled）和 env038/step116（first_failure），18张未经编辑原图已逐张用 view_image 实际查看。未读 cell_001、整轮运行协议或仍变化的 capture receipt，未执行最终 dense scorer。', '',
           '每组16个 native before/after 数组含全部64环境，q、qd、root transform、root velocity 的 dtype、shape、字节及数值逐一相同；selected 全关节读回和实际 controlled indices 下的 q/qd 与该组 state 精确绑定。这个结论限于保存的渲染边界瞬时状态，不证明之后960步轨迹不受影响，也没有把图绑定到独立数值运行。', '',
           '实际USD相机矩阵、光学属性和clippingRange是 producer保存的直接读回证据。本侧线用正交基变换、由USD光学值重建的K和归一化六面半空间独立算术复核；没有启动Kit再读USD。SDK pos_w仍是过期零值，与actualUSD分开记录，不冒称有效校准读回。', '',
           '总览/front/reverse核验142个球；F角度只核62个F球，U角度只核80个U球。球半径在六面法向距离中扣除，所有所审目标球距六面均≥50mm。这个包络只覆盖模型所列球，不认证所有网格、桌边、手指轮廓或无遮挡。像素/K/法向残差比较容差只是保存float32及独立算术核对，未用于放宽物理或包络判据。', '',
           '|组|native标量相同比较|最小目标球六面余量m|保存的最小非豁免几何m|', '|---|---:|---:|---:|']
    for g in result['groups']:
        minimum=min(v['minimum_plane_margin_m'] for v in g['views'].values())
        lines.append(f'|{g["capture_kind"]} env{g["env"]:03d} step{g["step"]}|{g["native_scalar_checks"]}|{minimum:.9f}|{g["min_saved_nonexempt_geometry_m"]:.12f}|')
    lines += ['', 'env038/116保存了真实负的非豁免球几何裕度；这不是照片可直接识别的穿透/真实接触证据。本次未读前序全stream，不能独立确认“首次”时间顺序，更不能由两状态判定唯一失败机制或成功率。','',
              '两组图的灰色F与白色U臂和桌面总体可辨，未见邻槽绿架横贯前景；多个方向仍存在臂间、工具和自身壳体遮挡。总览中手/腕占画面较小，白色壳体较亮，不能解析毫米级间隙；部分pair/俯视图桌边裁切。对侧角度补充可见性，不等同完整网格轮廓证明。','',
              '|组/视角|实际原图观察|','|---|---|']
    for g in result['groups']:
        for name,v in g['views'].items():
            lines.append(f'|env{g["env"]:03d}/{g["step"]}/{name}|{v["human_observation"]}|')
    lines += ['', '旧launch保持失败：Kit exit0，但visual_protocol failed，0有效window/PNG；错误为显式argv缺少--methods。v2→v3计划除了显式原默认system0和新visual_retry_1输出路径，actor、种子、环境、视角/stage及控制/observer source SHA均不变。两组成功capture不替代整轮retry闭合验收。','',
              '与各照片同组的state/native/PNG文件，以及必要的已闭合共享球identity、冻结source/计划和旧失败元数据均记录before/after SHA；未哈希新运行的整体protocol/receipt。全部具体路径、算术误差和逐数组结果见JSON。','',
              f'执行oracle SHA256：`{result["executed_source_sha256"]}`。', '',
              '物理策略仍 **BLOCKED**；hardware/production approval均false。最终12条件评分、整轮相机/数值跨run比较和最终科学报告审阅仍待后续材料。', '']
    return '\n'.join(lines)


def main():
    out_json=H/'ASTRA_CAMERA_REVIEW.json';out_md=H/'ASTRA_CAMERA_REVIEW.md'
    require(not out_json.exists() and not out_md.exists(),'refuse overwrite owned completed camera review')
    inputs=ClosedInputs();identity=inputs.json(ROOT/'full_row_identity.json')
    result=dict(schema='astra.closed_two_group_camera_review.v1',
        status='PASS_BOUNDED_NATIVE_SIX_PLANE_CAMERA_REVIEW',utc=datetime.now(timezone.utc).isoformat(),
        executed_source_sha256=ASTRA_EXECUTED_SOURCE_SHA256,
        supersession=supersession(inputs),groups=[],blocking_findings=[],
        physical_strategy_status='BLOCKED',safety_strategy_approved=False,hardware_authorized=False,
        production_promoted=False,final_dense_score_executed=False,
        limitations=['offline actualUSD/readback arithmetic, no separate liveUSD measurement',
                     'native equality at saved render boundaries only; forward replay not established',
                     'cache all64 invariant and all64 sphere-after invariant remain producer assertions',
                     'model target spheres, not complete mesh/occlusion/force/continuous-time evidence',
                     'first-failure chronology not independently reconstructed without full stream'])
    for kind,env,step in GROUPS:result['groups'].append(group_audit(inputs,identity,kind,env,step))
    result['observed_image_files']=[p for g in result['groups'] for p in g['observed_image_files']]
    result['view_image_count']=len(result['observed_image_files'])
    result['checks']=dict(closed_groups=2,actual_images_viewed=18,images_sha_bound_and_decoded=18,
                         native_array_comparisons=32,native_scalar_comparisons=sum(g['native_scalar_checks'] for g in result['groups']),
                         independent_six_plane_view_checks=18,
                         minimum_sphere_plane_clearance_m=min(v['minimum_plane_margin_m'] for g in result['groups'] for v in g['views'].values()),
                         max_K_reconstruction_error_px=max(v['intrinsic_max_abs_error_px'] for g in result['groups'] for v in g['views'].values()),
                         max_plane_reconstruction_error_m=max(v['plane_margin_reconstruction_error_m'] for g in result['groups'] for v in g['views'].values()))
    result['consumed_files']=inputs.finish()
    require(sha(Path(__file__).read_bytes())==ASTRA_EXECUTED_SOURCE_SHA256,'oracle source changed')
    text=markdown(result)
    with out_md.open('x') as f:f.write(text)
    result['review_md_sha256']=sha(out_md.read_bytes())
    with out_json.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    require(json.loads(out_json.read_text())==result,'JSON readback mismatch')
    print(json.dumps(dict(status=result['status'],checks=result['checks'],review_md_sha256=result['review_md_sha256']),indent=2))


if __name__=='__main__' and not globals().get('ASTRA_SOURCE_COMPILED'):
    path=Path(__file__).resolve();data=path.read_bytes()
    scope=dict(__name__='__main__',__file__=str(path),ASTRA_SOURCE_COMPILED=True,
               ASTRA_EXECUTED_SOURCE_SHA256=sha(data))
    exec(compile(data,str(path),'exec'),scope)
elif __name__=='__main__':main()
