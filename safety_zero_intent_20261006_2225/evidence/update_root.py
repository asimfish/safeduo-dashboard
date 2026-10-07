"""Change only this study card and the preceding owned study card."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re
from candidate_decision import decide
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_zero_worktree_20261006')
def strip_owned(s):
    for name in ['zeroIntentEvidence','trackingReserveEvidence']:
        s=re.sub(r'<section\b[^>]*\bid="'+name+r'"[^>]*>.*?</section>\n?','',s,flags=re.S)
    return s
def main():
    p=WORK/'index.html';before=p.read_text();assert p.read_bytes()==(HERE/'root_before_integrated.html').read_bytes()
    assert 'id="zeroIntentEvidence"' not in before
    matches=list(re.finditer(r'<section\b[^>]*\bid="trackingReserveEvidence"[^>]*>.*?</section>',before,re.S));assert len(matches)==1
    m=matches[0];previous=m.group().replace('最新完整实验：','上一轮完整实验：')
    r=json.loads((HERE/'holdout_results.json').read_text());v=json.loads((HERE/'visual_verification.json').read_text());z=json.loads((HERE/'ZERO_INTENT_AUDIT.json').read_text());t={x['mode']:x for x in r['totals']};zero={x['mode']:x for x in z['summary']};decision=decide(r)
    assert r['completed_method_windows']==576 and r['invalid_method_windows']==0
    new=f'<section class="sci-note" id="zeroIntentEvidence"><h2>最新完整实验：零输入目标保持与六步延迟</h2><p>192个新初态×三条件，恢复后576个有效窗口全部完成；首尝448完成/128因远程USD初始化失败未开始，原失败和上层误记PASS均保留。严格失败：原参考 {t["joint_reference"]["violations"]}/192；原收紧 {t["tight_reference"]["violations"]}/192；允许保持目标 {t["zero_inclusive"]["violations"]}/192。{decision["text"]}。</p><p>零输入前缀失败：基线 {zero["joint_reference"]["zero_prefix_strict_windows"]}/192，候选 {zero["zero_inclusive"]["zero_prefix_strict_windows"]}/192。{v["images"]}张真实九视角绑定自身state；原始失败、运动代价、覆盖及GPT‑6 Astra xhigh独立复算可查。初始GPU门禁失败、基线暂停/同进程恢复、只补未开始的候选及相机CUDA1失败后恢复原GPU0的事后操作修订均披露，物理安全尚未验证。</p><p><a href="docs/{HERE.name}/">打开完整结果、零输入诊断与原始图像 →</a></p></section>\n'
    after=before[:m.start()]+new+previous+before[m.end():]
    assert strip_owned(after)==strip_owned(before),'unowned root differs'
    p.write_text(after)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    with (HERE/'root_change_verification.json').open('x') as f:json.dump(dict(status='PASS_UNOWNED_ROOT_BYTE_EQUALITY',before_sha256=sha(before.encode()),after_sha256=sha(p.read_bytes()),unowned_sha256=sha(strip_owned(before).encode()),owned_cards=['zeroIntentEvidence','trackingReserveEvidence'],unowned_content_unchanged=True,utc=datetime.now(timezone.utc).isoformat()),f,indent=2)
    print('ROOT_OWNED_CARDS_PASS',flush=True)
if __name__=='__main__':main()
