"""接触语义引擎：消费 B 的 contact_semantics.yaml（owner: Agent A）。

消费顺序（与 B 的契约，STATUS_B @A #3）：
1. link 全局名 "{arm_id}/{local_link}" 用 link_classes 正则分类（首条命中）；
2. 同臂邻接 / 手内部 / 手-腕安装豁免（adjacency 节）先行剔除；
3. 其余对按 pair_rules 顺序取第一条命中 -> verdict；
4. conditional_exempt（near_table）给出运行时双条件（高度+接近速度）豁免参数，
   由 SphereDistanceModule 每步算 mask 并覆写 d_min / VIOLATION 判定。

本模块只做静态裁决（建索引时一次），运行时逻辑在 sphere_distance.py。
纯 python/yaml，无 torch/isaac 依赖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

_ROBOT_CLASSES = ("arm_base", "arm_link", "hand_palm", "hand_finger")


@dataclass
class PairVerdict:
    keep: bool                 # 是否进入危险对集合（产生距离行）
    category: str = ""         # cross / self_F / self_U / table（min_margin 分桶）
    d_min: float = 0.0         # velocity-damper 完全刹停边界
    conditional: bool = False  # True = near_table 运行时豁免候选（手部近自桌）
    verdict: str = ""          # allow/forbid/soft/conditional_exempt（记录用）


@dataclass
class NearTableExemption:
    height_max: float
    closing_max: float
    d_min_override: float
    violation_on_contact: bool


class ContactSemantics:
    def __init__(self, yaml_path: str):
        with open(yaml_path) as f:
            y = yaml.safe_load(f)
        th = y["thresholds"]
        self.d_warn: float = float(th["d_warn"])
        self.d_soft: float = float(th["d_soft"])
        self.d_min: dict = {k: float(v) for k, v in th["d_min"].items()}
        # v6 per-link table d_min（Round 154 ②治根 / A10 §4-2）：结构性悬停
        # link（JAKA link2 肩球恒 17.6mm < table d_min 20mm）的对桌行永久
        # cap<0，是 backstop 泵病理的激励源——此类 link 用独立刹停边界。
        # 键 = link 局部语义名（如 "link2"），值 = d_min（米）；来源为 yaml
        # thresholds.d_min_per_link（可选节），duo_env 侧 safety.d_min_per_link
        # 建对前再覆写（与 d_min_override 同模式，随 pair 表烘焙）。
        self.d_min_per_link: dict = {
            str(k): float(v) for k, v in (th.get("d_min_per_link") or {}).items()}
        self._classes = [(c["class"], re.compile(c["pattern"])) for c in y["link_classes"]]
        self._rules = y["pair_rules"]
        adj = y["adjacency"]
        self._adj = {robot: {frozenset(p) for p in pairs}
                     for robot, pairs in adj.items() if isinstance(pairs, list)}
        self._hand_internal_all = adj.get("hand_internal") == "all"
        self._hand_wrist = {r: set(v) for r, v in adj.get("hand_wrist", {}).items()}
        # v3 换装：adjacency/hand_wrist 的机器人键按 YAML robot_keys 解析
        # （缺省 = v0 硬编码 {F: fr3, U: ur5e}；JAKA 换装文件写 {F: fr3, U: jaka_zu7}，
        # 否则 U 臂邻接豁免全部失配 -> 出生即幽灵违规）
        rk = y.get("robot_keys", {"F": "fr3", "U": "ur5e"})
        self._robot_key = {str(k): str(v) for k, v in rk.items()}
        ex = y["exemptions"]["near_table"]
        self.near_table = NearTableExemption(
            height_max=float(ex["height_above_table_max"]),
            closing_max=float(ex["closing_speed_max"]),
            d_min_override=float(ex["override"]["d_min"]),
            violation_on_contact=bool(ex["override"]["violation_on_contact"]),
        )

    # ---- 分类 ----

    def classify(self, qualified: str) -> str | None:
        for cls, pat in self._classes:
            if pat.search(qualified):
                return cls
        return None

    @staticmethod
    def _arm_of(qualified: str) -> str | None:
        head = qualified.split("/")[0]
        return head if head in ("F_L", "F_R", "U_L", "U_R") else None

    @staticmethod
    def _robot_of_arm(arm: str) -> str:
        return arm[0]  # "F" / "U"

    @staticmethod
    def own_table(arm: str) -> str:
        return "table_F" if arm[0] == "F" else "table_U"

    @staticmethod
    def _local(qualified: str) -> str:
        return qualified.split("/", 1)[1] if "/" in qualified else qualified

    def _is_hand(self, cls: str | None) -> bool:
        return cls in ("hand_palm", "hand_finger")

    def _table_dmin(self, qa: str, qb: str) -> float:
        """link×table/ground 行的 d_min：per-link 覆写优先（v6，Round 154 ②）。"""
        link_q = qb if (qa.startswith("table") or qa == "ground") else qa
        return self.d_min_per_link.get(self._local(link_q), self.d_min["table"])

    # ---- 邻接豁免（先行）----

    def _adjacent_exempt(self, qa: str, qb: str, ca: str | None, cb: str | None) -> bool:
        arm_a, arm_b = self._arm_of(qa), self._arm_of(qb)
        if arm_a is None or arm_b is None or arm_a != arm_b:
            return False
        la, lb = self._local(qa), self._local(qb)
        hand_a, hand_b = self._is_hand(ca), self._is_hand(cb)
        if hand_a and hand_b:
            return self._hand_internal_all  # 同手内部全豁免
        if hand_a != hand_b:  # 手 <-> 载体臂末端安装豁免
            arm_local = lb if hand_a else la
            return arm_local in self._hand_wrist.get(self._robot_key[arm_a[0]], set())
        return frozenset((la, lb)) in self._adj.get(self._robot_key[arm_a[0]], set())

    # ---- pair_rules 引擎（首条命中）----

    def _class_member(self, cls: str | None, group: str) -> bool:
        if group == "any_robot_link":
            return cls in _ROBOT_CLASSES
        return cls == group

    def _rule_matches(self, rule: dict, qa: str, qb: str, ca, cb) -> bool:
        g1, g2 = rule["pair"]
        hit = ((self._class_member(ca, g1) and self._class_member(cb, g2))
               or (self._class_member(ca, g2) and self._class_member(cb, g1)))
        if not hit:
            return False
        cond = rule.get("condition")
        if cond in ("different_robot", "same_robot"):
            arm_a, arm_b = self._arm_of(qa), self._arm_of(qb)
            if arm_a is None or arm_b is None:
                return False
            same = self._robot_of_arm(arm_a) == self._robot_of_arm(arm_b)
            if cond == "different_robot" and same:
                return False
            if cond == "same_robot" and not same:
                return False
        if rule.get("scope") == "same_table":
            arm = self._arm_of(qa) or self._arm_of(qb)
            table = qb if qb.startswith("table") else qa
            if arm is None or self.own_table(arm) != table:
                return False
        return True

    def _category(self, rule: dict, qa: str, qb: str) -> str:
        cat = rule.get("margin_category", "")
        if cat == "self":  # 按机器人拆桶
            arm = self._arm_of(qa)
            cat = "self_F" if arm and arm[0] == "F" else "self_U"
        return cat

    def judge(self, qa: str, qb: str) -> PairVerdict:
        """任意两个实体全局名 -> 裁决。表/地面用 table_F/table_U/ground。"""
        ca, cb = self.classify(qa), self.classify(qb)
        if self._adjacent_exempt(qa, qb, ca, cb):
            return PairVerdict(keep=False, verdict="adjacency_exempt")
        for rule in self._rules:
            if not self._rule_matches(rule, qa, qb, ca, cb):
                continue
            v = rule["verdict"]
            if v == "allow":
                return PairVerdict(keep=False, verdict="allow")
            cat = self._category(rule, qa, qb)
            if v == "soft":
                return PairVerdict(keep=True, category=cat or "self", d_min=0.0,
                                   verdict="soft")
            if v == "conditional_exempt":
                # near_table 豁免的 applies_scope=same_table：只对自桌 conditional，
                # 对面桌按普通 forbid 处理
                arm = self._arm_of(qa) or self._arm_of(qb)
                table = qb if qb.startswith("table") else qa
                same = arm is not None and self.own_table(arm) == table
                return PairVerdict(keep=True, category=cat or "table",
                                   d_min=self._table_dmin(qa, qb), conditional=same,
                                   verdict="conditional_exempt" if same else "forbid")
            # forbid：d_min 按分桶（cross/self/table）；table 桶先查 per-link 覆写
            key = "cross" if cat == "cross" else ("table" if cat == "table" else "self")
            dm = self._table_dmin(qa, qb) if key == "table" else self.d_min[key]
            return PairVerdict(keep=True, category=cat or key,
                               d_min=dm, verdict="forbid")
        return PairVerdict(keep=False, verdict="no_rule")


def semantics_with_safety_overrides(yaml_path: str,
                                    safety_cfg: dict) -> ContactSemantics:
    """语义引擎 + duo_env safety 节运行时覆写，单一出处（C14）。

    覆写序列与 duo_env.__init__ 历史行为逐位一致（顺序敏感：d_min 分档 →
    per-link → d_warn）。抽成公共函数的原因（C12_HANDOFF §1"建议从 env yaml
    单源读取"）：训练器 cost 通道要与 env 塑形带用同一组 d_warn/d_min，
    train_lagrangian 侧复用本函数即消费同一份真值，杜绝再写第二处硬编码。

    - d_min_override（A6-W6 hot-fix 旋钮）：per-category d_min 重校，
      建对前生效（v5 self/cross 0.013、v6 cross 0.038）。
    - d_min_per_link（v6，Round 154 ②治根）：结构性悬停 link 的对桌行独立
      刹停边界（link2 0.015），必须在建对前注入随 pair 表烘焙。
    - d_warn_override（v6，Round 152 ②联动）：塑形带 0.05→0.08，只动
      d_warn 不动 d_soft（活跃集/tube 语义不变）。
    """
    sem = ContactSemantics(yaml_path)
    for k, v in (safety_cfg.get("d_min_override") or {}).items():
        sem.d_min[k] = float(v)
    for k, v in (safety_cfg.get("d_min_per_link") or {}).items():
        sem.d_min_per_link[str(k)] = float(v)
    if safety_cfg.get("d_warn_override") is not None:
        sem.d_warn = float(safety_cfg["d_warn_override"])
    return sem
