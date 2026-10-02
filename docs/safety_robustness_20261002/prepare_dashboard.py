"""Publish only finalized empirical evidence, preserving earlier negative results."""
from datetime import datetime
import argparse
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

ROOT = Path('/home/liyufeng/safeduo')
OUT = ROOT/'artifacts/safety_robustness_20261002'
DASH = Path('/home/liyufeng/safeduo-dashboard')
DATA = Path('/home/liyufeng/safeduo-dashboard-data/scientific_eval.json')
parser=argparse.ArgumentParser()
parser.add_argument('--preview-json', type=Path)
args=parser.parse_args()
r = json.loads((OUT/'results.json').read_text())
d = json.loads(DATA.read_text())
s = r['protected_total']
short_held = [x for x in r['rows'] if x['flow']=='held_random' and x['hold_steps']==30]
iid = next(x for x in r['rows'] if x['flow']=='uniform_random' and x['method']=='system0')
held = next(x for x in short_held if x['method']=='system0')
raw = next(x for x in short_held if x['method']=='raw')
r['summary'] = (f"固定 a31b 权重，改进来自执行安全层；修正质心参考点、危险行排序、条件接触阻尼，以及目标限位后的方向翻转。"
    f"本轮保护窗口共 {s['violations']}/{s['episodes']} 违规；最小臂/手球距 {s['arm_arm_min_mm']:.2f} mm。"
    '各窗口分布不同且相关，不报告独立样本安全置信界。')
r['overview_note'] = f"随机保持：无保护 {raw['violations']}/{raw['episodes']} → System 0 {held['violations']}/{held['episodes']}"
r['overview_detail'] = f"0.5秒保持；全部保护复测 {s['violations']}/{s['episodes']}；最小余量 {s['arm_arm_min_mm']:.2f}mm"
r['coverage_note'] = (f"同为新种子9/10、32环境、5秒、amp0.015，System 0 平均关节范围："
    f"逐步IID {100*iid['mean_joint_range_fraction']:.2f}% → 随机保持 {100*held['mean_joint_range_fraction']:.2f}%。"
    '范围为实测跨度/关节soft-limit跨度的平均；六配对暴露是进入80mm的跨窗口并集。'
    '保持随机没有目标引导，但存在时间相关性；初态和动力学仍固定，不能称操作空间已充分覆盖。')
r['negative_note'] = ('保留负结果：仅修正质心参考点曾由0/32退化至4/32，未推广。'
    '补充危险行排序及条件接触阻尼后，同条件恢复0/32。'
    f"随机保持0.5秒的旧/新策略 {r['legacy_held']['violations']}/{r['legacy_held']['episodes']} "
    f"→ {r['new_held']['violations']}/{r['new_held']['episodes']}；新增失败条件 {len(r['new_held_failures'])}。")
if 'target_bounds_fix' in r:
    fix=r['target_bounds_fix']
    r['negative_note'] += (f"风险排序版长保持曾有1/32违规，后置限位会把开离指令裁成闭合。"
        f"限位纳入投影后同条件1/32→0/32，原失败窗口余量 −32.68→{fix['new_env3_min_self_F_mm']:.2f}mm。")
long=next(x for x in r['rows'] if x['method']=='system0' and x['hold_steps']==90)
if long['violations']:
    r['negative_note'] += (f"长保持1.5秒/10秒组仍有 {long['violations']}/{long['episodes']} 违规，"
        f"最小臂/手球距 {long['arm_arm_min_mm']:.2f}mm；该范围尚未解决，不宣称全部安全。")
d['safety_robustness'] = r
d['updated'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
d['summary'] = (f"质心几何与风险约束修正后，扩大随机保持测试：无保护 {raw['violations']}/{raw['episodes']} "
    f"→ System 0 {held['violations']}/{held['episodes']}。本轮保护复测 {s['violations']}/{s['episodes']}，仍需初态、动力学和物体任务验证。")
d['historical_headline'] = d.get('historical_headline',
    [x for x in d['headline'] if not x['label'].startswith('0.5秒随机保持')])
cards=[]
for x in short_held:
    cards.append(dict(label=f"0.5秒随机保持 · {x['label'].split(' · ')[1]}", episodes=x['episodes'],
        violations=x['violations'],rate=x['violations']/x['episodes'],cp95_upper=None,
        status='bad' if x['violations'] else 'warn',
        note=f"新种子9/10；同一命令 tape；平均实测关节范围 {100*x['mean_joint_range_fraction']:.2f}%"))
d['headline'] = cards
d['coverage_assessment'] = dict(title='覆盖判定：随机范围扩大，仍未穷尽操作空间',
    summary=r['coverage_note'], items=[
        '关节增量逐关节独立均匀采样；新增0.5秒及1.5秒保持，持续同向运动增加。IID与保持随机按分布分别统计。',
        f"短保持六臂对暴露 {held['pair_union_exposed']}/6；每条窗口同时覆盖六对的数量 {held['all_six_pairs_exposed_episodes']}/{held['episodes']}。没有暴露的配对不作为避碰证明。",
        '匹配验证相同初态、checkpoint与逐窗口命令SHA；基座/桌面、初始姿态和动力学参数仍未随机化。',
        '六臂对目标测试重新覆盖4种EE目标间距×3种目标方向×两种幅度，但目标空间设计不等于实际球间距和速度均匀覆盖。'])
d['limitations'] = [
    '本轮只验证非豁免球层违规，不代表真实碰撞、接触力、物体夹持或四臂协作任务成功。',
    f"本轮最小臂/手球距 {s['arm_arm_min_mm']:.3f}mm，仍需感知误差、初始姿态、外力、质量/摩擦及真实执行延迟扰动验证。",
    '纯随机保持显著增加持续运动，但不是姿态空间或危险状态的均匀采样；相关窗口总数不提供独立样本置信证明。',
    '兜底仍使用有限32条风险约束，复杂多接触时可能遗漏危险行。actor输入保持原32条距离排序观测，未重新训练。',
    '质心修正单独引入4/32违规的负结果及上一轮新增2个桌面失败均保留，组合措施不能据此宣称普适安全。',
    '平均关节范围、运动路径和执行量比例不能替代任务完成率，也不能证明全操作空间覆盖。',
    '随机输入覆盖26个臂关节，未随机控制手指闭合；实际物体抓持和四臂协作仍需独立任务评测。',
]
prefix='https://github.com/asimfish/safeduo-dashboard/blob/main/docs/safety_robustness_20261002/'
d['links'] = [x for x in d['links'] if '/docs/safety_robustness_20261002/' not in x['url']]
d['links'] = [{'label':'最新几何与随机覆盖报告','url':prefix+'REPORT.md'},
              {'label':'速度参考点及遗漏约束诊断','url':prefix+'DIAGNOSIS.md'},
              {'label':'完全相同随机命令逐窗口 CSV','url':prefix+'paired_random.csv'},
              {'label':'新协议、源码哈希和完整结果','url':prefix+'results.json'}]+d['links']
(args.preview_json or DATA).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
dest=DASH/'docs/safety_robustness_20261002';dest.mkdir(parents=True,exist_ok=True)
for name in ['REPORT.md','DIAGNOSIS.md','PLAN.md','results.json','paired_random.csv','build_report.py',
             'run_validation.py','run_final_regression.py','prepare_dashboard.py','launch_hardware.json','geometry_test_red.log',
             'contact_test_red.log','target_limit_test_red.log','delivery_tests.log','make_plots.py',
             'check_browser.py','browser_check.log','consistency_check.json','VALIDATION.md','panel_results.png',
             'stage2_results.json','stage2_REPORT.md',
             'kinematics_and_margin.png','kinematics_and_margin.svg','random_coverage.png','random_coverage.svg',
             'random_pair_safety.png','random_pair_safety.svg']:
    shutil.copy2(OUT/name,dest/name)
for name in [x['args']['out'].split('/')[-1] for x in r['manifests']+r['diagnostic_manifests']]:
    target=dest/'protocols';target.mkdir(exist_ok=True)
    shutil.copy2(OUT/name/'protocol.json',target/f'{name}.json')
for manifest in r.get('reused_raw_controls',[]):
    name=manifest['args']['out'].split('/')[-1]
    shutil.copy2(OUT/name/'protocol.json',dest/'protocols'/f'raw_controls_{name}.json')
for src in ['src/safeduo/safety/geometry.py','src/safeduo/safety/backstop.py','src/safeduo/safety/sphere_distance.py',
            'src/safeduo/eval/research_battery.py','src/safeduo/configs/duo_env_a31_com_guard.yaml',
            'src/safeduo/configs/duo_env_a31_priority_guard.yaml','src/safeduo/configs/duo_env_a31_bounded_guard.yaml',
            'src/safeduo/envs/duo_env.py','tests/test_target_limit_projection.py','tests/test_safety_row_priority.py',
            'tests/test_isaac_geometry_reference.py','tests/test_research_battery.py','tests/test_r33_lookahead_band.py']:
    shutil.copy2(ROOT/src,dest/Path(src).name)
for name in ['live_browser_check.log','DELIVERY.json']:
    if (OUT/name).exists():
        shutil.copy2(OUT/name,dest/name)
for exported in dest.iterdir():
    if exported.suffix in {'.svg','.log'}:
        # Normalize generated whitespace without changing plots or log values.
        exported.write_text('\n'.join(line.rstrip() for line in exported.read_text().splitlines())+'\n')
print('Prepared finalized dashboard evidence',dest)
