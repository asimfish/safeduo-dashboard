"""Update only the two owned cards while preserving every inherited root byte."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re
HERE=Path(__file__).resolve().parent
WORK=Path('/home/liyufeng/safeduo-dashboard-feasible-guard-20261005')
OWN=['fifoPredictorEvidence','feasibleGuardEvidence']
def sha(b):return hashlib.sha256(b).hexdigest()
def section(s,tag):
    hits=list(re.finditer(r'<section\b[^>]*\bid="'+re.escape(tag)+r'"[^>]*>.*?</section>',s,re.S))
    assert len(hits)==1,(tag,len(hits));return hits[0]
def unowned(s):
    for tag in OWN:
        m=section(s,tag);s=s[:m.start()]+s[m.end():]
    return s
def main():
    path=WORK/'index.html';before=path.read_bytes();text=before.decode();r=json.loads((HERE/'holdout_results.json').read_text());v=json.loads((HERE/'visual_verification.json').read_text())
    assert r['completed_method_windows']==768 and r['invalid_method_windows']==0
    t={a['mode']:a for a in r['totals']};paired=[p for p in r['paired'] if p['a']=='joint_reference' and p['b']=='motion_admission']
    rescue=sum(p['rescued'] for p in paired);new=sum(p['new_failures'] for p in paired)
    card=f'<section class="sci-note" id="fifoPredictorEvidence"><h2>最新完整实验：FIFO 运动预测与随机对照</h2><p>192个新初态×四方法，768窗口完成，无效0。严格违规：原参考 {t["joint_reference"]["violations"]}/192；速度 {t["velocity_admission"]["violations"]}/192；PD {t["pd_admission"]["violations"]}/192；联合 {t["motion_admission"]["violations"]}/192。联合救回{rescue}个，新增失败{new}个。物理安全尚未验证。</p><p>实际九视角图像{v["images"]}张；全部案例、首次失败/FIFO、实际覆盖、预测漏报及GPT‑6 Astra xhigh独立复算可查。</p><p><a href="docs/{HERE.name}/">打开最新完整结果、逐案例曲线与图像 →</a></p></section>'
    m=section(text,OWN[0]);after=text[:m.start()]+card+text[m.end():]
    m=section(after,OWN[1]);prior=m.group();assert '最新完整实验：联合重求解与强度对照' in prior
    prior=prior.replace('最新完整实验：联合重求解与强度对照','上一轮完整实验：联合重求解与强度对照').replace('打开本轮完整结果、逐案例曲线与图像','查看上一轮完整结果、逐案例曲线与图像')
    after=after[:m.start()]+prior+after[m.end():]
    assert unowned(after)==unowned(text),'unowned root changed'
    (HERE/'ROOT_BEFORE_FINAL.html').write_bytes(before);path.write_bytes(after.encode())
    receipt=dict(status='PASS_EXACT_UNOWNED_ROOT_PRESERVATION',before_sha256=sha(before),after_sha256=sha(after.encode()),
        unowned_sha256=sha(unowned(text).encode()),owned_cards=OWN,foreign_dynamics_card_exact=section(text,'dynamicsPassiveEvidence').group()==section(after,'dynamicsPassiveEvidence').group(),
        utc=datetime.now(timezone.utc).isoformat())
    (HERE/'root_change_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['status'],flush=True)
if __name__=='__main__':main()
