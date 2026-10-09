"""Held-physics, mask-qualified original camera captures.

Parent wraps super()._setup_scene() in qualified_camera_factory(), then calls
configure_qualified_camera(self) before play. After env creation/reset, detach
the shared env._viz_cam from scene.sensors and own the physics hold. This module
never steps physics or starts Isaac. All64 labels are authored before play in a
dedicated anonymous layer removed by close(). Segmentation data types reach the
original Camera constructor; class semanticTypes reach annotators before attach.

Three accepted images/arm, 24 total candidates/arm, three held renders per
rendered candidate (two warmups, then final), one dt=0 sensor refresh. Every
held render has independent exact all64 native before/after checks. Geometry is only a
conservative rejection/framing proxy. Qualification means observed_visible_hand
pixels from matching actual instance paths AND semantic hand labels, never
full mesh coverage, absence of occlusion, or physical safety certification.
"""
from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from itertools import combinations
import json
from contextlib import contextmanager
from pathlib import Path
import re

import numpy as np

import hand_views as v

CAMERA_DATA_TYPES = ("rgb", "instance_segmentation_fast", "semantic_segmentation")
ARM_KEYS = v.ARM_KEYS
WIDTH, HEIGHT = 1280, 720
MIN_HAND_PIXELS = 200  # Strictly greater than this number.
MIN_FRUSTUM_M = .02
IMAGES_PER_ARM, MAX_ATTEMPTS_PER_ARM = 3, 24
MAX_DEINSTANCE_ROUNDS = 32
MIN_AZIMUTH_SEPARATION_DEG = 30.
HAND_SEMANTIC_TYPE = "class"
HAND_LABEL_PREFIX = "qv4_hand_"
RENDERS_PER_CANDIDATE = 3
SCOPE = ("observed_visible_hand: >200 actual same-native-rigid instance/semantic pixels; "
         "six-plane .06m native-link spheres with .02m margin are framing only. "
         "Table ray/AABB is diagnostic only and never rejects a candidate. "
         "24 bounded above/horizontal/below candidates, selected after v3 visual failures. "
         "No fullmesh, occlusion-free, contact, "
         "or safety certification; parent owns physical-event binding.")


def _configure_cfg(cfg):
    cfg.data_types = list(CAMERA_DATA_TYPES)
    cfg.colorize_instance_segmentation = False
    cfg.colorize_semantic_segmentation = False
    cfg.semantic_filter = "*:*"
    cfg.update_period = 0.
    cfg.width, cfg.height = WIDTH, HEIGHT
    cfg.spawn.clipping_range = (.001, 1e5)


@contextmanager
def _annotator_types(registry):
    """Installed Replicator requires these parameters BEFORE attach()."""
    descriptor = inspect.getattr_static(registry, "get_annotator")
    original = registry.get_annotator

    def get(name, init_params=None, *args, **kwargs):
        if name in CAMERA_DATA_TYPES[1:]:
            init_params = dict(init_params or {}, semanticTypes=[HAND_SEMANTIC_TYPE], colorize=False)
        return original(name, init_params, *args, **kwargs)

    registry.get_annotator = staticmethod(get)
    try:
        yield
    finally:
        registry.get_annotator = descriptor


def _camera_type(original):
    class VisibilityCamera(original):
        def __init__(self, cfg):
            cfg = cfg.copy()
            _configure_cfg(cfg)
            # Preserve the installed Camera constructor, including its 4.5-only
            # instanceability workaround. It now sees segmentation from entry.
            super().__init__(cfg)
            self._qualified_v4_constructor = True

        def _initialize_impl(self):
            import omni.replicator.core as rep
            with _annotator_types(rep.AnnotatorRegistry):
                super()._initialize_impl()
            self._qualified_v4_annotator_types = [HAND_SEMANTIC_TYPE]

    return VisibilityCamera


@contextmanager
def qualified_camera_factory():
    """Wrap parent super()._setup_scene(); patches its actual local-import site.

    Use serially during scene construction. The installed module is restored
    even on error. No library source file or physics property is patched.
    """
    import isaaclab.sensors as sensors
    original = sensors.Camera
    sensors.Camera = _camera_type(original)
    try:
        yield
    finally:
        sensors.Camera = original


def hand_label(path):
    return HAND_LABEL_PREFIX + hashlib.sha256(path.encode()).hexdigest()


@lru_cache(maxsize=2)
def _token_paths(paths):
    result = {hand_label(path): path for path in paths}
    if len(result) != len(paths):
        raise ValueError("semantic token collision")
    return result


class SemanticOverlay:
    """Local modern-USD label opinions for every clone, removed as one layer."""
    def __init__(self, stage, paths):
        from pxr import Sdf, Usd, UsdSemantics
        self.stage = stage
        self.paths = sorted(paths)
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("empty or duplicate semantic hand paths")
        # Capture every original value BEFORE authoring env0: inherit arcs in
        # copy_from_source=False would otherwise expose env0's new label.
        originals = {}
        for path in self.paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or prim.IsInstanceProxy():
                raise ValueError("missing or uneditable hand rigid prim: " + path)
            api = UsdSemantics.LabelsAPI.Get(prim, HAND_SEMANTIC_TYPE)
            values = list(api.GetLabelsAttr().Get() or []) if api else []
            if any(label.startswith(HAND_LABEL_PREFIX) for label in values):
                raise ValueError("another v4 semantic owner already exists")
            originals[path] = values
        self.layer = Sdf.Layer.CreateAnonymous("qualified_views_v4_semantics")
        session = stage.GetSessionLayer()
        session.subLayerPaths = [self.layer.identifier, *session.subLayerPaths]
        self.closed = False
        try:
            with Usd.EditContext(stage, self.layer):
                self._deinstance_descendants()
                for path in self.paths:
                    prim = stage.GetPrimAtPath(path)
                    api = UsdSemantics.LabelsAPI.Apply(prim, HAND_SEMANTIC_TYPE)
                    api.CreateLabelsAttr(originals[path] + [hand_label(path)])
            self.verify()
        except Exception:
            self.close()
            raise

    def verify(self):
        from pxr import UsdSemantics
        for path in self.paths:
            prim = self.stage.GetPrimAtPath(path)
            values = UsdSemantics.LabelsAPI.Get(prim, HAND_SEMANTIC_TYPE).GetLabelsAttr().Get()
            owned = [x for x in values if x.startswith(HAND_LABEL_PREFIX)]
            if owned != [hand_label(path)]:
                raise ValueError("clone semantic ownership mismatch: " + path)
        remaining = [str(p.GetPath()) for p in self._descendants()
                     if p.IsInstance() or p.IsInstanceProxy()]
        if remaining:
            raise ValueError("hand render hierarchy still has instance/proxy descendants: " + repr(remaining))

    def _descendants(self):
        from pxr import Usd
        roots = set(self.paths)
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            parent = str(prim.GetPath()).rsplit("/", 1)[0]
            while parent:
                if parent in roots:
                    yield prim
                    break
                parent = parent.rsplit("/", 1)[0]

    def _deinstance_descendants(self):
        """Composition changes expose nested roots; obtain fresh prims each round.

        Only instanceable metadata is authored here, in the owned anonymous
        layer. Never edit prototypes, native hand roots, physics attributes,
        materials, transforms, or target-environment visibility.
        """
        from pxr import UsdGeom
        self.instanceability_overrides, self.instanceability_rounds = [], []
        before_paths = sorted(str(p.GetPath()) for p in self._descendants())
        for number in range(MAX_DEINSTANCE_ROUNDS):
            candidates = [str(p.GetPath()) for p in self._descendants()
                          if p.IsInstance() and not p.IsInstanceProxy()]
            if not candidates:
                break
            record = dict(round=number, overrides=[])
            for path in candidates:
                prim = self.stage.GetPrimAtPath(path)
                prior = dict(path=path, is_instance=bool(prim.IsInstance()),
                             is_instance_proxy=bool(prim.IsInstanceProxy()),
                             is_instanceable=bool(prim.IsInstanceable()))
                if not prim.SetInstanceable(False):
                    raise ValueError("cannot deinstance editable hand descendant: " + path)
                prior["after_is_instance"] = bool(self.stage.GetPrimAtPath(path).IsInstance())
                record["overrides"].append(prior)
                self.instanceability_overrides.append(path)
            self.instanceability_rounds.append(record)
        descendants = list(self._descendants())
        after_paths = sorted(str(p.GetPath()) for p in descendants)
        remaining_instances = [str(p.GetPath()) for p in descendants if p.IsInstance()]
        remaining_proxies = [str(p.GetPath()) for p in descendants if p.IsInstanceProxy()]
        if before_paths != after_paths or remaining_instances or remaining_proxies:
            raise ValueError("deinstance incomplete or changed actual prim paths: " + repr(
                dict(remaining_instances=remaining_instances, remaining_proxies=remaining_proxies)))
        mesh_owners = {}
        roots = set(self.paths)
        for prim in descendants:
            if prim.IsA(UsdGeom.Mesh):
                path = str(prim.GetPath())
                owner = path.rsplit("/", 1)[0]
                while owner and owner not in roots:
                    owner = owner.rsplit("/", 1)[0]
                if not owner:
                    raise ValueError("render mesh has no exact native hand ancestor: " + path)
                mesh_owners[path] = owner
        self.instanceability_audit = dict(rounds=self.instanceability_rounds,
            max_rounds=MAX_DEINSTANCE_ROUNDS, descendant_paths=after_paths,
            actual_paths_unchanged=True, remaining_instance_paths=remaining_instances,
            remaining_proxy_paths=remaining_proxies, render_mesh_native_hand_owners=mesh_owners,
            scope="only hand-descendant instanceable metadata; numerical physics footprint not certified")

    def close(self):
        if not self.closed:
            session = self.stage.GetSessionLayer()
            session.subLayerPaths = [p for p in session.subLayerPaths if p != self.layer.identifier]
            self.closed = True


def _stage_hand_paths(stage):
    from pxr import Usd, UsdPhysics
    paths = []
    counts = {(slot, arm): 0 for slot in range(64) for arm in ARM_KEYS}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        match = re.match(r"^/World/envs/env_(\d+)/(F_L|F_R|U_L|U_R)/", path)
        if match and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            slot, arm = int(match[1]), match[2]
            if slot < 64 and v._hand_path(path, arm):
                paths.append(path)
                counts[(slot, arm)] += 1
    if any(n == 0 for n in counts.values()):
        raise ValueError("pre-play USD hands must cover every arm in all64 clones")
    return paths


def configure_qualified_camera(env):
    """Call after DuoEnv._setup_scene(), BEFORE sim play/initialization.

    DuoEnv constructs CameraCfg inline: cfg.scene.viz_cam does not exist.
    Requires qualified_camera_factory() around super()._setup_scene(). Labels
    all64 cloned hand prims before play; native inventory is cross-checked later.
    """
    camera = env._viz_cam
    if camera is None or camera.is_initialized:
        raise ValueError("configure existing camera before sim play only")
    if not getattr(camera, "_qualified_v4_constructor", False):
        raise ValueError("v4 requires qualified_camera_factory around super()._setup_scene()")
    if (camera.cfg.width, camera.cfg.height) != (WIDTH, HEIGHT):
        raise ValueError("original camera resolution must be 1280x720")
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera.cfg.prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise ValueError("original USD camera prim missing")
    UsdGeom.Camera(prim).GetClippingRangeAttr().Set((.001, 1e5))
    if hasattr(env, "_qualified_v4_semantic_overlay"):
        raise ValueError("semantic overlay already configured")
    env._qualified_v4_semantic_overlay = SemanticOverlay(stage, _stage_hand_paths(stage))
    return camera


def segment_aabb(eye, target, lower, upper):
    """Closed eye-to-target segment intersects a conservative world AABB."""
    eye, target, lower, upper = (np.asarray(x, dtype=np.float64) for x in
                                (eye, target, lower, upper))
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (eye, target, lower, upper)):
        raise ValueError("invalid ray/AABB")
    if np.any(lower >= upper):
        raise ValueError("empty AABB")
    lo, hi = 0., 1.
    for axis, delta in enumerate(target - eye):
        if abs(delta) <= 1e-14:
            if eye[axis] < lower[axis] or eye[axis] > upper[axis]:
                return False
        else:
            a, b = (lower[axis] - eye[axis]) / delta, (upper[axis] - eye[axis]) / delta
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
            if lo > hi:
                return False
    return True


def table_rejections(eye, points, tables):
    return [dict(path=t["path"], blocked_link_indices=[i for i, p in enumerate(points)
                 if segment_aabb(eye, p, t["lower_world_m"], t["upper_world_m"])])
            for t in tables]


def native_inventory(env):
    """Exact native link order, all 64 slots; inherited hand inventory rule."""
    if env.num_envs != 64:
        raise ValueError("requires all64 native environments")
    result = {}
    for arm in ARM_KEYS:
        actor = env._arms[arm]
        names, paths = list(actor.body_names), actor.root_physx_view.link_paths
        if not names or len(set(names)) != len(names) or len(paths) != 64:
            raise ValueError("invalid native body inventory: " + arm)
        for slot, row in enumerate(paths):
            if len(row) != len(names):
                raise ValueError("link_paths/body_names length mismatch")
            hand_count = 0
            for i, path in enumerate(row):
                path = str(path)
                if (not path.startswith(f"/World/envs/env_{slot}/{arm}/") or
                        path.rsplit("/", 1)[-1] != names[i] or path in result):
                    raise ValueError("native path/name/slot mismatch")
                hand = v._hand_path(path, arm)
                hand_count += int(hand)
                result[path] = dict(env_id=slot, arm=arm, hand=hand,
                                    body_index=i, body_name=names[i])
            if not hand_count:
                raise ValueError("empty native hand inventory")
    return result


def path_identity(path, inventory):
    """Longest exact native ancestor; arm substring/name guesses cannot qualify."""
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    probe = path.rstrip("/")
    while probe:
        if probe in inventory:
            return dict(inventory[probe], rigid_path=probe, renderer_path=path)
        probe = probe.rsplit("/", 1)[0]
    return None


def _id_map(info):
    if not isinstance(info, dict) or not isinstance(info.get("idToLabels"), dict):
        raise ValueError("missing idToLabels")
    labels = {}
    for key, value in info["idToLabels"].items():
        if isinstance(key, bool) or not re.fullmatch(r"[0-9]+", str(key)):
            raise ValueError("non-integer segmentation ID (colorized output?)")
        idx = int(key)
        if idx in labels:
            raise ValueError("duplicate normalized segmentation ID")
        labels[idx] = value
    return labels


def _label_identity(label, datatype, inventory):
    if datatype == "instance_segmentation_fast":
        if isinstance(label, str) and label in ("BACKGROUND", "UNLABELLED"):
            return None
        if not isinstance(label, str) or not label.startswith("/"):
            raise ValueError("instance label is not a prim path")
        return path_identity(label, inventory)
    if not isinstance(label, dict):
        raise ValueError("semantic label must be a type-to-label dict")
    if HAND_SEMANTIC_TYPE not in label:
        return None
    value = label[HAND_SEMANTIC_TYPE]
    if not isinstance(value, str):
        raise ValueError("invalid qualified_hand label")
    tokens = [p.strip() for p in value.split(",") if p.strip().startswith(HAND_LABEL_PREFIX)]
    if not tokens:
        return None
    mapping = _token_paths(tuple(sorted(p for p, item in inventory.items() if item["hand"])))
    if any(token not in mapping for token in tokens):
        raise ValueError("unverified semantic hand token")
    paths = [mapping[token] for token in tokens]
    identities = [path_identity(p, inventory) for p in paths]
    if any(i is None or not i["hand"] or p != i["rigid_path"]
           for p, i in zip(paths, identities)):
        raise ValueError("unverified semantic hand path")
    owners = {(i["env_id"], i["arm"]) for i in identities}
    if len(owners) != 1:
        raise ValueError("ambiguous inherited semantic hand ownership")
    if len(set(paths)) != 1:
        raise ValueError("ambiguous semantic native hand rigid link")
    return dict(identities[0], semantic_rigid_paths=paths)


def mask_evidence(raw, info, datatype, inventory, slot, arm):
    """Unknown pixel IDs fail closed; preserve original IDs separately on disk."""
    raw = np.asarray(raw)
    if raw.shape != (HEIGHT, WIDTH, 1) or raw.dtype.kind not in "iu" or raw.dtype.itemsize != 4:
        raise ValueError("expected original 720x1280x1 int32/uint32 IDs")
    ids = raw[..., 0]
    # IsaacLab may expose the uint32 bit pattern in int32 storage.
    if ids.dtype.kind == "i":
        ids = ids.view(np.dtype(ids.dtype.str.replace("i", "u")))
    labels = _id_map(info)
    present, counts = np.unique(ids, return_counts=True)
    unknown = [int(i) for i in present if int(i) not in labels]
    mapping, selected = {}, []
    for idx, label in labels.items():
        identity = _label_identity(label, datatype, inventory)
        mapping[str(idx)] = dict(label=label, identity=identity)
        if identity and identity["hand"] and (identity["env_id"], identity["arm"]) == (slot, arm):
            selected.append(idx)
    mask = np.isin(ids, selected)
    foreign_ids = [int(idx) for idx in present if str(int(idx)) in mapping and mapping[str(int(idx))]["identity"]
        and mapping[str(int(idx))]["identity"]["hand"]
        and mapping[str(int(idx))]["identity"]["env_id"] != slot]
    foreign_pixels = int(np.isin(ids, foreign_ids).sum())
    return mask, dict(datatype=datatype, unknown_ids=unknown, id_mapping=mapping,
                      selected_hand_ids=selected, hand_pixels=int(mask.sum()),
                      observed_id_pixel_counts={str(int(i)): int(n) for i, n in zip(present, counts)},
                      mapping_valid=not unknown and not foreign_pixels, foreign_hand_pixels=foreign_pixels)


def qualify_masks(outputs, info, inventory, slot, arm):
    masks, streams, errors = [], {}, []
    for name in CAMERA_DATA_TYPES[1:]:
        try:
            mask, evidence = mask_evidence(outputs[name], info[name], name, inventory, slot, arm)
            masks.append(mask)
            streams[name] = evidence
            if not evidence["mapping_valid"]:
                errors.append(name + ":unknown_ids_or_foreign_hand_pixels")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(name + ":" + str(exc))
    count, same_arm_count = 0, 0
    if len(masks) == 2:
        same_arm_count = int(np.count_nonzero(masks[0] & masks[1]))
        ids = {}
        for name in CAMERA_DATA_TYPES[1:]:
            values = np.asarray(outputs[name])[..., 0]
            ids[name] = values.view(np.dtype(values.dtype.str.replace("i", "u"))) if values.dtype.kind == "i" else values
        instance, semantic = (streams[name] for name in CAMERA_DATA_TYPES[1:])
        same_rigid = np.zeros((HEIGHT, WIDTH), dtype=bool)
        for idx in instance["selected_hand_ids"]:
            rigid = instance["id_mapping"][str(idx)]["identity"]["rigid_path"]
            semantic_ids = [sid for sid in semantic["selected_hand_ids"]
                if semantic["id_mapping"][str(sid)]["identity"]["rigid_path"] == rigid]
            if semantic_ids:
                same_rigid |= ((ids["instance_segmentation_fast"] == idx) &
                               np.isin(ids["semantic_segmentation"], semantic_ids))
        count = int(np.count_nonzero(same_rigid))
    if count <= MIN_HAND_PIXELS:
        errors.append("insufficient_observed_hand_pixels")
    return dict(status="qualified" if not errors else "failed", reasons=errors,
                observed_hand_pixels=count, same_arm_intersection_pixels=same_arm_count,
                pixel_identity_contract="instance native rigid ancestor equals unique semantic native rigid path",
                required_pixels_strictly_greater_than=MIN_HAND_PIXELS,
                streams=streams, scope="observed_visible_hand")


def world_directions():
    """Fixed world hemispheres; 8 above, 8 horizontal, 8 below, at most 24."""
    for elevation in (55, 0, -45):
        for azimuth in (0, 120, 240, 60, 180, 300, 90, 270):
            az, el = np.deg2rad([azimuth, elevation])
            yield np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


def qualified_triplet(records):
    for triple in combinations(records, IMAGES_PER_ARM):
        if azimuth_separated([r["azimuth_deg"] for r in triple]):
            return list(triple)
    return None


def azimuth_separated(angles):
    return any(abs((a-b+180) % 360-180) >= MIN_AZIMUTH_SEPARATION_DEG
               for i, a in enumerate(angles) for b in angles[i+1:])


def camera_geometry(points, camera):
    world = np.asarray(camera["actual_camera_to_world_row_matrix"])
    K = np.asarray(camera["actual_intrinsic_matrix"])
    clip = camera["actual_clipping_range_m"]
    margins = v.sphere_frustum_margins(points, v.RADIUS_M, world, K, clip)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    normals = np.array([[fx, 0, -cx], [-fx, 0, -(WIDTH-cx)],
                        [0, -fy, -cy], [0, fy, -(HEIGHT-cy)], [0, 0, -1], [0, 0, 1.]])
    scale = np.linalg.norm(normals, axis=1)
    normals = (normals / scale[:, None]) @ world[:3, :3]
    offsets = np.array([0., 0., 0., 0., -clip[0], clip[1]]) / scale - normals @ world[3, :3]
    return dict(plane_names=list(v.PLANE_NAMES), sphere_radius_m=v.RADIUS_M,
                plane_equation="inward unit normal dot world_point + offset; subtract sphere radius",
                actual_world_planes=np.column_stack((normals, offsets)).tolist(),
                plane_margins_m=margins.tolist(), minimum_by_plane_m=margins.min(0).tolist(),
                minimum_plane_margin_m=float(margins.min()), required_minimum_m=MIN_FRUSTUM_M,
                framing_pass=bool(np.all(margins >= MIN_FRUSTUM_M)), visibility_claim=False)


def native_evidence(before, after):
    try:
        return dict(status="pass", **v.assert_native_bitwise(before, after))
    except ValueError as exc:
        return dict(status="failed", all_64_bitwise_equal=False, error=str(exc))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(vv) for k, vv in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(vv) for vv in value]
    return value


class QualifiedViews:
    def __init__(self, env, out):
        self.env, self.out = env, Path(out)
        self._closed = False
        self._sequence = 0
        self._stage = None
        self._labels = []
        self._labeled_slots = set()
        self._inventory = None
        self._denied = None
        self._semantic_setup = None

    def _write(self, path, writer):
        with path.open("xb") as stream:
            writer(stream)
        return dict(path=str(path.relative_to(self.out)), sha256=v.sha256(path))

    def _json(self, path, value):
        data = (json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n").encode()
        return self._write(path, lambda f: f.write(data))

    def _npz(self, path, values):
        return self._write(path, lambda f: np.savez_compressed(f, **values))

    def _png(self, path, pixels):
        from PIL import Image
        return self._write(path, lambda f: Image.fromarray(pixels).save(f, format="PNG"))

    def _prepare(self):
        camera = self.env._viz_cam
        v.validate_camera_contract(camera, self.env.sim.render_mode.name)
        if any(sensor is camera for sensor in self.env.scene.sensors.values()):
            raise ValueError("shared camera must be detached from scene.sensors")
        if not set(CAMERA_DATA_TYPES).issubset(camera.cfg.data_types):
            raise ValueError("required camera data_types missing; configure before env creation")
        if camera.cfg.colorize_instance_segmentation or camera.cfg.colorize_semantic_segmentation:
            raise ValueError("segmentation must retain uncolorized original IDs")
        if camera.cfg.semantic_filter != "*:*":
            raise ValueError("semantic_filter must include qualified_hand via '*:*'")
        if self._stage is None:
            import omni.usd
            self._stage = omni.usd.get_context().get_stage()
        if self._inventory is None:
            self._inventory = native_inventory(self.env)
        overlay = self.env._qualified_v4_semantic_overlay
        expected = {p for p, meta in self._inventory.items() if meta["hand"]}
        if set(overlay.paths) != expected:
            raise ValueError("pre-play USD semantic inventory differs from actual all64 native hands")
        if getattr(camera, "_qualified_v4_annotator_types", None) != [HAND_SEMANTIC_TYPE]:
            raise ValueError("annotator semanticTypes not configured before attach")
        overlay.verify()
        from importlib.metadata import version
        self._semantic_setup = dict(schema="safeduo.qualified_semantics.v4",
            all64_native_hand_path_count=len(expected), annotator_semanticTypes=[HAND_SEMANTIC_TYPE],
            semantic_label_prefix=HAND_LABEL_PREFIX, api="UsdSemantics.LabelsAPI",
            local_override_layer=overlay.layer.identifier, labels_authored_before_play=True,
            all64_preplay_labels_match_native_inventory=True, isaacsim_package_version=version("isaacsim"),
            segmentation_passed_to_original_camera_constructor=True,
            instanceability_behavior="Owned session layer disables hand-descendant instanceability before play; physics properties and rigid owners retained.",
            hand_descendant_instanceability_overrides=overlay.instanceability_overrides,
            hand_descendant_instanceability_audit=overlay.instanceability_audit,
            semantic_token_to_native_path={hand_label(p):p for p in sorted(expected)})

    def _tag_slot(self, slot):
        # Every clone was labelled before play; never author env0 during capture.
        if self._semantic_setup is None:
            raise ValueError("missing verified all64 pre-play semantic setup")
        self._labeled_slots.add(slot)

    def _render_candidate(self, path, baseline, rec):
        rec["held_renders"] = []
        for index in range(RENDERS_PER_CANDIDATE):
            before = v.native_snapshot(self.env)
            precheck = native_evidence(baseline, before)
            if precheck["status"] != "pass":
                self._denied = precheck["error"]
                rec["reasons"].append("native_changed_before_held_render")
                break
            row = dict(index=index, warmup=index < RENDERS_PER_CANDIDATE-1,
                       native_before_all64=self._npz(path.with_suffix(f".render{index}.before.npz"), before))
            rec["held_renders"].append(row)
            try:
                self.env.sim.render()
                rec["rendered"] = True
            finally:
                after = v.native_snapshot(self.env)
                row["native_after_all64"] = self._npz(path.with_suffix(f".render{index}.after.npz"), after)
                row["native_invariant"] = native_evidence(before, after)
                if row["native_invariant"]["status"] != "pass":
                    self._denied = row["native_invariant"]["error"]
                    rec["reasons"].append("native_changed_during_held_render")
            if self._denied:
                break
        rec["render_calls"] = len(rec["held_renders"])

    def _visibility(self, slot):
        from pxr import UsdGeom
        saved = []
        try:
            for i in range(64):
                prim = self._stage.GetPrimAtPath(f"/World/envs/env_{i}")
                if not prim.IsValid():
                    raise ValueError("missing env visibility prim")
                attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
                saved.append((attr, attr.HasAuthoredValueOpinion(), attr.Get()))
                attr.Set(UsdGeom.Tokens.inherited if i == slot else UsdGeom.Tokens.invisible)
        except Exception:
            self._restore_visibility(saved)
            raise
        return saved

    @staticmethod
    def _restore_visibility(saved):
        for attr, authored, value in saved:
            attr.Set(value) if authored else attr.Clear()

    def _tables(self, slot):
        from pxr import Usd, UsdGeom
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,
                                 UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=False)
        result = []
        for name in ("TableF", "TableU"):
            path = f"/World/envs/env_{slot}/{name}"
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise ValueError("actual table prim missing: " + path)
            bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            low, high = np.asarray(bound.GetMin()), np.asarray(bound.GetMax())
            if not np.isfinite([low, high]).all() or np.any(low >= high):
                raise ValueError("invalid actual USD table bound")
            result.append(dict(path=path, lower_world_m=low.tolist(), upper_world_m=high.tolist(),
                               source="actual USD BBoxCache world aligned bound; rejection proxy only"))
        return result

    def _pose(self, eye, target):
        import torch
        self.env._viz_cam.set_world_poses_from_view(
            torch.as_tensor(eye, device=self.env.device, dtype=torch.float32)[None],
            torch.as_tensor(target, device=self.env.device, dtype=torch.float32)[None])

    def _camera(self):
        return v._read_camera(self.env, self._stage)

    def _outputs(self):
        data = self.env._viz_cam.data
        outputs = {name: v._array(data.output[name][0]) for name in CAMERA_DATA_TYPES if name in data.output}
        info = _jsonable(data.info[0])
        return outputs, info

    def _attempt(self, root, baseline, points, tables, slot, arm, number, direction):
        path = root / f"{arm}_attempt_{number:02d}"
        rec = dict(arm=arm, attempt=number, status="rejected", reasons=[], rendered=False,
                   scope=SCOPE, sensor_update_dt_s=0., world_direction=direction.tolist())
        before = v.native_snapshot(self.env)
        rec["native_before_all64"] = self._npz(path.with_suffix(".before.npz"), before)
        invariant = native_evidence(baseline, before)
        rec["native_baseline_before"] = invariant
        if invariant["status"] != "pass":
            self._denied = invariant["error"]
            rec["reasons"].append("native_changed_before_render")
        try:
            if self._denied:
                raise ValueError("native capture denied: " + self._denied)
            target = (points.min(0) + points.max(0)) / 2
            distance = .5
            # USD-only analytic pose fit: these operations never render candidates.
            for fit in range(3):
                eye = target + distance * direction
                self._pose(eye, target)
                camera = self._camera()
                v._check_view_pose(camera, eye, target)
                geometry = camera_geometry(points, camera)
                if geometry["framing_pass"]:
                    break
                distance += v.fit_distance_delta(points, v.RADIUS_M,
                    camera["actual_camera_to_world_row_matrix"], camera["actual_intrinsic_matrix"],
                    camera["actual_clipping_range_m"], clearance=MIN_FRUSTUM_M + 1e-4,
                    minimum_delta=.01-distance)
            rec.update(camera=camera, geometry=geometry, fit_operations=fit+1,
                       requested_eye_world_m=eye.tolist(), target_world_m=target.tolist(),
                       hand_link_positions_world_m=points.tolist(), distance_m=distance,
                       azimuth_deg=float(np.degrees(np.arctan2(direction[1], direction[0])) % 360))
            blocked = table_rejections(eye, points, tables)
            rec["table_ray_aabb"] = blocked
            rec["table_aabb_use"] = "diagnostic_only"
            if not geometry["framing_pass"]:
                rec["reasons"].append("six_plane_framing_failed")
            if not rec["reasons"]:
                frame_before = v._array(self.env._viz_cam.frame)
                rec["sensor_frame_before"] = frame_before.tolist()
                self._render_candidate(path, baseline, rec)
                self.env._viz_cam.update(0., force_recompute=True)
                frame_after = v._array(self.env._viz_cam.frame)
                rec["sensor_frame_after"] = frame_after.tolist()
                # Save outputs BEFORE any qualification check can reject the attempt.
                outputs, info = self._outputs()
                rec["original_outputs"] = self._npz(path.with_suffix(".outputs.npz"), outputs)
                rec["original_info"] = self._json(path.with_suffix(".info.json"), info)
                rgb = outputs.get("rgb")
                if rgb is not None and rgb.dtype == np.uint8 and rgb.shape in (
                        (HEIGHT, WIDTH, 3), (HEIGHT, WIDTH, 4)):
                    rec["rgb"] = self._png(path.with_suffix(".png"), rgb)
                    rec.update(resolution=[WIDTH, HEIGHT], channels=rgb.shape[2],
                               raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                else:
                    rec["reasons"].append("original_rgb_shape_or_dtype_failed")
                evidence = qualify_masks(outputs, info, self._inventory, slot, arm)
                rec["mask_evidence"] = self._json(path.with_suffix(".mask.json"), evidence)
                # Global renderer mappings can grow to all64. Keep them on disk,
                # not in the in-memory receipts for every candidate/slot.
                rec["mask"] = {k: val for k, val in evidence.items() if k != "streams"}
                rec["reasons"].extend(evidence["reasons"])
                del outputs, info, evidence
                v.assert_sensor_refresh(frame_before, frame_after)
                rec["sensor_refresh_exactly_one"] = True
                camera = self._camera()
                rec["camera"] = camera
                v._check_view_pose(camera, eye, target)
                rec["geometry"] = camera_geometry(points, camera)
                if not rec["geometry"]["framing_pass"]:
                    rec["reasons"].append("post_render_six_plane_framing_failed")
        except Exception as exc:
            rec["reasons"].append(type(exc).__name__ + ":" + str(exc))
        finally:
            try:
                after = v.native_snapshot(self.env)
                rec["native_after_all64"] = self._npz(path.with_suffix(".after.npz"), after)
                rec["native_invariant"] = native_evidence(before, after)
                rec["native_baseline_after"] = native_evidence(baseline, after)
                for key in ("native_invariant", "native_baseline_after"):
                    if rec[key]["status"] != "pass":
                        self._denied = rec[key]["error"]
                        rec["reasons"].append("native_changed_deny")
            except Exception as exc:
                self._denied = "native after evidence unavailable: " + str(exc)
                rec["reasons"].append(self._denied)
        if rec["rendered"] and not rec["reasons"]:
            rec["status"] = "qualified"
        rec["attempt_record"] = self._json(path.with_suffix(".attempt.json"), rec)
        return rec

    def capture(self, step: int, slots: list[int], kind: str, substep: int | None = None):
        if isinstance(step, bool) or not isinstance(step, int) or step < -1:
            raise ValueError("step must be integer >=-1")
        if step == -1 and (kind != "initial" or substep is not None):
            raise ValueError("step -1 is reserved for initial capture")
        if substep is not None and (isinstance(substep, bool) or not isinstance(substep, int) or substep < 0):
            raise ValueError("invalid actual substep")
        if not isinstance(kind, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", kind):
            raise ValueError("unsafe capture kind")
        slots = list(slots)
        if (len(slots) != len(set(slots)) or any(isinstance(s, bool) or
                not isinstance(s, (int, np.integer)) or not 0 <= s < 64 for s in slots)):
            raise ValueError("unique native slot IDs in [0,64) required")
        if not slots:
            return []
        root = self.out / "qualified_views" / kind / (
            f"step_{step:04d}_sub_{substep if substep is not None else 'none'}_capture_{self._sequence:06d}")
        self._sequence += 1
        root.mkdir(parents=True, exist_ok=False)
        baseline, baseline_ref, setup_error = None, None, None
        states = []
        try:
            baseline = v.native_snapshot(self.env)
            baseline_ref = self._npz(root / "native_before_all64.npz", baseline)
            if self._closed or self._denied:
                raise ValueError("capture closed/denied: " + str(self._denied))
            self._prepare()
            mapping_ref = self._json(root / "native_path_mapping.json", self._inventory)
            semantic_ref = self._json(root / "semantic_setup.json", self._semantic_setup)
        except Exception as exc:
            setup_error = type(exc).__name__ + ":" + str(exc)
            mapping_ref = None
            semantic_ref = None
        for slot in map(int, slots):
            slot_root = root / f"env_{slot:03d}"
            slot_root.mkdir()
            state = dict(schema="safeduo.qualified_views.v1", env_id=slot, step=step,
                         substep=substep, capture_kind=kind, status="failed", reasons=[], arms={},
                         attempts=[], native_before_all64=baseline_ref, native_path_mapping=mapping_ref,
                         semantic_setup=semantic_ref,
                         required_images_per_arm=IMAGES_PER_ARM, max_attempts_per_arm=MAX_ATTEMPTS_PER_ARM,
                         environment_count=64, scope=SCOPE, physics_advanced_by_collector=False)
            if baseline is not None:
                state.update(simulation_time_s=float(baseline["simulation_time_s"]),
                             simulation_time_step_index=int(baseline["simulation_time_step_index"]))
            saved = []
            try:
                if setup_error or self._denied:
                    raise ValueError(setup_error or self._denied)
                self._tag_slot(slot)
                saved = self._visibility(slot)
                tables = self._tables(slot)
                state["actual_table_aabbs"] = tables
                for arm in ARM_KEYS:
                    rows = [dict(meta, rigid_path=p) for p, meta in self._inventory.items()
                            if meta["env_id"] == slot and meta["arm"] == arm and meta["hand"]]
                    points = baseline[arm + "_native_link_transforms_xyzw"][slot,
                              [row["body_index"] for row in rows], :3].astype(np.float64)
                    accepted = []
                    attempts = []
                    for number, direction in enumerate(world_directions(), 1):
                        if number > MAX_ATTEMPTS_PER_ARM:
                            raise ValueError("candidate generator exceeded fixed attempt bound")
                        if self._denied:
                            break
                        record = self._attempt(slot_root, baseline, points, tables, slot, arm, number, direction)
                        state["attempts"].append(record)
                        attempts.append(record["attempt_record"])
                        if record["status"] == "qualified":
                            accepted.append(record)
                        triplet = qualified_triplet(accepted)
                        if triplet is not None:
                            accepted = triplet
                            break
                    angles = [r["azimuth_deg"] for r in accepted]
                    passed = len(accepted) == IMAGES_PER_ARM and azimuth_separated(angles) and not self._denied
                    state["arms"][arm] = dict(status="qualified" if passed else "failed",
                        qualified_images=[r["rgb"] for r in accepted], qualified_count=len(accepted),
                        distinct_azimuths_deg=angles, azimuth_requirement_met=azimuth_separated(angles),
                        attempts=attempts, hand_native_paths=rows,
                        failure_reason=None if passed else "insufficient_views_or_azimuths_or_native_denied")
            except Exception as exc:
                state["reasons"].append(type(exc).__name__ + ":" + str(exc))
            finally:
                try:
                    self._restore_visibility(saved)
                    state["visibility_restored"] = True
                except Exception as exc:
                    state["reasons"].append("visibility_restore_failed:" + str(exc))
            for arm in ARM_KEYS:
                if arm not in state["arms"]:
                    state["arms"][arm] = dict(status="failed", qualified_count=0, qualified_images=[],
                                              failure_reason="capture_unavailable", attempts=[])
            states.append((slot_root, state))
        final_ref, invariant = None, dict(status="failed", all_64_bitwise_equal=False)
        try:
            final = v.native_snapshot(self.env)
            final_ref = self._npz(root / "native_final_all64.npz", final)
            if baseline is not None:
                invariant = native_evidence(baseline, final)
            if invariant["status"] != "pass":
                self._denied = invariant.get("error", "native baseline unavailable")
        except Exception as exc:
            invariant["error"] = str(exc)
            self._denied = "native final unavailable: " + str(exc)
        receipts = []
        for slot_root, state in states:
            state.update(native_final_all64=final_ref, native_invariant=invariant,
                         source_sha256=v.sha256(__file__), inherited_hand_views_sha256=v.sha256(v.__file__))
            if invariant["status"] != "pass" or self._denied:
                state["reasons"].append("native_changed_or_unverifiable_deny")
                for arm_state in state["arms"].values():
                    arm_state["status"] = "failed"
                    arm_state["failure_reason"] = "native_changed_or_unverifiable_deny"
                    arm_state["provisional_images"] = arm_state.pop("qualified_images")
                    arm_state["provisional_count"] = arm_state["qualified_count"]
                    arm_state["qualified_images"], arm_state["qualified_count"] = [], 0
            if not state["reasons"] and all(a["status"] == "qualified" for a in state["arms"].values()):
                state["status"] = "qualified"
            else:
                state["reasons"].append("incomplete_qualified_hand_coverage")
            state["image_count"] = sum("rgb" in r for r in state["attempts"])
            state["qualified_image_count"] = sum(a["qualified_count"] for a in state["arms"].values())
            ref = self._json(slot_root / "state.json", state)
            receipts.append(dict(env_id=state["env_id"], step=step, substep=substep, capture_kind=kind,
                status=state["status"], reasons=state["reasons"], state=ref["path"], sha256=ref["sha256"],
                image_count=state["image_count"], qualified_image_count=state["qualified_image_count"],
                arms=state["arms"], scope=SCOPE))
        self._json(root / "receipts.json", dict(receipts=receipts, native_invariant=invariant,
                   native_before_all64=baseline_ref, native_final_all64=final_ref, scope=SCOPE))
        return receipts

    def close(self):
        """Release owned semantic metadata; shared camera lifetime belongs to parent."""
        if self._closed:
            return dict(status="closed", errors=[])
        errors = []
        try:
            self.env._qualified_v4_semantic_overlay.close()
        except Exception as exc:
            errors.append(str(exc))
        self._closed = True
        return dict(status="closed" if not errors else "failed", errors=errors)
