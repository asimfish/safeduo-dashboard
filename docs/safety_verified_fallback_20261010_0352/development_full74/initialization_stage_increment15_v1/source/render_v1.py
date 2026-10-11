"""A compact bilingual stage comparison; no new image or safety inference."""
import html

STYLE='''
#nativeAttemptIncrement15{margin-top:3rem;border-top:3px solid #27798b;padding-top:1.3rem;overflow-wrap:anywhere}
.r15-controls{border:1px solid #ccd6da;padding:1rem;margin:1rem 0}.r15-controls label{display:inline-block;margin-right:1rem}
#nativeAttemptIncrement15 .r15-en{display:none}
#r15LanguageEN:checked~.r15-content .r15-zh{display:none}#r15LanguageEN:checked~.r15-content .r15-en{display:block}
#r15Summary:checked~.r15-content .r15-detail{display:none}
.r15-cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.7rem;margin:1.2rem 0}
.r15-cards>div{padding:1rem;background:#f1f5f6;border:1px solid #cbd7db}.r15-cards strong{display:block;font-size:1.25rem;margin-top:.35rem}
.r15-table-wrap{max-width:100%;overflow-x:auto}#r15StageTable{width:100%;min-width:740px;border-collapse:collapse}
#r15StageTable th,#r15StageTable td{text-align:left;padding:.8rem;vertical-align:top;border-bottom:1px solid #d7e1e5}
#r15EvidenceAppendix{margin:1.4rem 0;padding:1rem;background:#f6f8f9}#r15EvidenceAppendix code{overflow-wrap:anywhere;word-break:break-word}
@media(max-width:640px){.r15-cards{grid-template-columns:repeat(2,minmax(0,1fr))}.r15-cards>div:last-child{grid-column:1/-1}}
'''

def render(f,prefix):
    esc=html.escape
    utc=esc(f['snapshot_utc']);local=esc(f['snapshot_Asia_Shanghai'])
    result=f'''<section id="nativeAttemptIncrement15">
<h2>报告15：初始化阻点仍在，完整挑战尚未完成 / Report15: initialization remains blocked; the full challenge is incomplete</h2>
<input id="r15LanguageZH" type="radio" name="r15Language" checked aria-label="Report15 Chinese"><input id="r15LanguageEN" type="radio" name="r15Language" aria-label="Report15 English">
<input id="r15Detailed" type="radio" name="r15Detail" checked aria-label="Report15 detailed"><input id="r15Summary" type="radio" name="r15Detail" aria-label="Report15 summary">
<div class="r15-controls"><label for="r15LanguageZH">中文</label><label for="r15LanguageEN">English</label><label for="r15Detailed">详细 / Detailed</label><label for="r15Summary">摘要 / Summary</label></div>
<div class="r15-content">
<p class="r15-zh">固定科学快照：<time>{utc}</time>；北京时间 {local}。这里比较已关闭的 V10、V11、V12 重试，不表示实时运行状态。</p>
<p class="r15-en">Fixed scientific snapshot: <time>{utc}</time>; Asia/Shanghai {local}. This compares closed V10, V11 and V12 retries, rather than live process status.</p>
<div class="r15-zh"><h3>摘要</h3><p>本轮重试在环境初始化返回前终止，没有生成相机图片、状态回执或已提交数值行。它没有进入相机验收阶段，因此不能据此判定全部32组相机失败。此前 V10 的32组初始相机覆盖验收仍有效，但首步32条记录均触发数值拒绝，完整12步挑战与最终相机仍未完成。</p></div>
<div class="r15-en"><h3>Abstract</h3><p>The latest retry terminated before environment initialization returned, leaving no camera images, state receipts or committed numerical rows. It never reached camera assessment; this does not establish failure of all32 camera groups. V10's independently checked32 initial camera groups remain valid, while all32 first-step records were numerically denied. The full12-step challenge and final camera checks remain incomplete.</p></div>
<div id="r15AcceptanceIndicators" class="r15-cards">
<div>模型预测 / Model predictions<strong>40,960 · MODEL_ONLY</strong></div><div>V10初始覆盖 / V10 initial coverage<strong>32 / 32</strong></div>
<div>完整挑战验收 / Full challenge accepted<strong>0 / 32</strong></div><div>安全银行 / Safety bank<strong>0</strong></div><div>任务与留出 / Tasks and holdout<strong>未运行 / 未开启<br>NOT_RUN / UNOPENED</strong></div></div>
<div class="r15-detail"><div class="r15-zh"><h3>结果：记录到哪一步，就验收到哪一步</h3><p>V10 已验收的初始相机覆盖仍有效。V11 留下更多部分图像，V12 尚未走出初始化；这两次相关重试均未增加已验收的完整挑战前缀。</p></div>
<div class="r15-en"><h3>Results: assess only the stage actually recorded</h3><p>V10's accepted initial camera coverage remains valid. V11 retained partial images, while V12 did not complete initialization. Neither related retry adds an accepted full challenge prefix.</p></div>
<div class="r15-table-wrap"><table id="r15StageTable"><thead><tr><th>相关重试 / Related retry</th><th>已提交行 / Committed rows</th><th>原图与状态回执 / Images and state receipts</th><th>相机证据 / Camera evidence</th><th>完整挑战 / Full challenge</th></tr></thead><tbody>
<tr data-retry="V10"><th>V10</th><td>0, 1</td><td>1312原图；32初始状态回执<br>1312 originals;32 initial state receipts</td><td>32组 INITIAL 独立通过；480张已选原图<br>32 initial groups independently checked;480 selected originals</td><td>首步32/32记录数值拒绝；2..12与最终相机缺失<br>Recorded micro1 denials32/32; rows2..12 and final cameras missing</td></tr>
<tr data-retry="V11"><th>V11</th><td>0 only / 仅初始行</td><td>1112原图；0状态回执<br>1112 originals;0 state receipts</td><td>失败回执独立消费；完整组与全RAW未验收<br>Independent failed-receipt consumption; complete groups and full RAW unassessed</td><td>1..12缺失；完整挑战未通过<br>Rows1..12 missing; full challenge unaccepted</td></tr>
<tr data-retry="V12"><th>V12</th><td>无 / None</td><td>0原图；0状态回执<br>0 originals;0 state receipts</td><td>初始化尚未返回，相机阶段未进入<br>Initialization did not return; camera stage not reached</td><td>0..12全部缺失；物理推进未知<br>All rows0..12 missing; physical advance unknown</td></tr>
</tbody></table></div>
<div class="r15-zh"><h3>方法与实验设置</h3><p>V10、V11、V12 是同一组32候选的相关重试，不增加独立同分布样本。模型评估仍按原8192与补充32768分别保留分母，合计40960仅为模型预测。SOURCE与启动接口检查说明工程约束被消费，不等于原生测量完成，更不等于物理安全验收。</p>
<h3>失败分析与限制</h3><p>V12 最后记录到环境初始化前的 reset/render/app.update 路径，没有环境初始化返回标记。具体原因仍未知；一次等待不能证明 C++ 死锁。1200秒工作边界尾部一次仅剩约8毫秒预算的 NVML 查询超时，也不能单独证明驱动故障。</p><p>刷新后的父级登记在2.724秒内通过原30秒启动门禁，这是工程门禁修复的证据；宽视图优化能否提升原生速度尚未测量。V10 接触读数的独立 SDK 刷新时刻、真实驱动峰值、微步内峰值与全局误差界仍未知。任务未运行，正式留出未开启，整体要求尚未达成。</p></div>
<div class="r15-en"><h3>Methods and experimental setting</h3><p>V10, V11 and V12 reuse the same32 candidates and add no IID samples. The original8192 and supplemental32768 model cohorts keep separate denominators:40960 model predictions in total. SOURCE and startup-interface checks concern engineering boundaries; they establish neither completed native measurement nor physical safety.</p>
<h3>Failure analysis and limitations</h3><p>V12 last recorded the reset/render/app.update path before environment initialization returned. The cause remains unknown; a single wait is not evidence of a C++ deadlock. An NVML query with only about8ms remaining near the1200-second work deadline is not, by itself, proof of a driver fault.</p><p>The renewed parent registration passed the original30-second startup gate at an age of2.724seconds. This supports the operational gate repair; native speed gains from wide-view changes remain unmeasured. V10's independent contact SDK refresh epoch, actual drive peak, within-step peaks and global error bound remain unknown. Tasks remain unrun, formal holdout unopened, and the full requirements unmet.</p></div>
</div>
<details id="r15EvidenceAppendix"><summary>证据与复现 / Evidence and reproduction</summary>
<p>原有480张已选初始原图、两张科学图及全部历史证据保留原字节。本增量只投影已关闭回执，未重算几何或重新解码图片。Existing 480 selected originals, two scientific figures and historical evidence retain their original bytes; this increment projects closed receipts without geometry recomputation or image decoding.</p>
<p>V11/V12 independent consumers: actual worker/outer/tool = 2/2/2, CLOSED_FAILED_NATIVE_RECEIPTS_ONLY. This is a closed failure observation, not RAW or camera qualification. Native exits -9/raw9 and outer/tool1 remain preserved.</p>
<p>原30秒 freshness gate 与1200秒原生工作边界保留；本地报告构建没有 native/SDK/Git/网站动作。Historical Report13 session71948 terminal remains UNKNOWN_PERMANENT.</p>
<ul>'''
    for name in ['V11_parent','V11_reader_adoption','V12_parent','V12_init_refresh','V12_reader_semantic','V12_reader_handoff','V12_reader_final','public14_parent_adoption']:
        q=f['source_refs'][name]
        result+=f'<li><a href="{prefix}/evidence/{name}.json">{esc(name)}</a> · <code>{q["sha256"]}</code></li>'
    result+=f'''</ul><p><a href="{prefix}/FACTS_INCREMENT15_V1.json">紧凑事实 / Compact facts</a> · <a href="{prefix}/evidence/V12_REFRESH_COMPACT_V1.json">刷新门禁的限域投影 / Scoped refresh projection</a> · <a href="{prefix}/source/SOURCE_GRAPH_V1.json">可重算源码 / Reproducible source</a> · <a href="{prefix}/archive_report14/index.html">保留报告14快照 / Preserved Report14 snapshot</a></p>
<p>报告作者不自评科学PASS。Distinct report science peer: PENDING; reviewer selected by parent, not named by the user. No publication authorization is generated here.</p>
</details></div></section>'''
    return result
