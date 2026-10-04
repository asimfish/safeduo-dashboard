"""Add risk evidence UI; reuse the existing measured coverage renderer."""
from pathlib import Path
import re

path=Path('/home/liyufeng/safeduo-dashboard/index.html');s=path.read_text()
assert 'function renderRiskStrata' not in s
section='''  <div id="scientific-risk" style="display:none">
    <h3>六组臂对风险分层：稳定初态后的安全 / 不安全对照</h3>
    <div class="sci-note" id="scientific-risk-note"></div>
    <p class="small" id="scientific-risk-protocol"></p>
    <div class="table-wrap"><table id="scientific-risk-table"><thead><tr><th>初态分层 / 方法</th><th>完成 / 计划</th><th>全部16秒违规</th><th>超过5mm</th><th>cross / self_F / self_U / 桌面</th><th>零输入前缀违规</th><th>随机目标到达前违规</th></tr></thead><tbody></tbody></table></div>
    <p class="small" id="scientific-risk-bank"></p>
    <p class="small">初态和外部指令逐位一致。保护器也可能在零输入阶段产生避险动作，实际q前缀不必相同；下表单列首条随机目标到达前66步的观测差值。</p>
    <div class="table-wrap"><table id="scientific-risk-inputs"><thead><tr><th>种子</th><th>初态 / 指令一致</th><th>66步q前缀一致</th><th>最大q差(rad)</th><th>raw单独 / 保护单独 / 双方失败</th></tr></thead><tbody></tbody></table></div>
    <h3>逐臂对配额与实际风险暴露</h3>
    <div class="table-wrap"><table id="scientific-risk-pair-results"><thead><tr><th>风险初态臂对 / 方法</th><th>完成 / 24</th><th>3帧80mm：全窗 / 随机目标到达后</th><th>官方违规</th><th>超过5mm</th></tr></thead><tbody></tbody></table></div>
    <h3>初态与运动后的实测覆盖</h3>
    <label class="small">选择分层：<select id="scientific-risk-coverage-select" style="max-width:100%"></select></label>
    <p class="small" id="scientific-risk-coverage-note"></p>
    <div id="scientific-risk-heatmap"></div>
    <div class="table-wrap"><table id="scientific-risk-ee"><thead><tr><th>臂</th><th>初态 / 实测10cm格</th><th>实测x / y / z范围（米）</th></tr></thead><tbody></tbody></table></div>
    <div class="table-wrap"><table id="scientific-risk-pairs"><thead><tr><th>六组臂对</th><th>3帧80mm / 总窗口</th><th>5mm内 / 重叠</th><th>末端接近 / 远离 / 切向帧</th></tr></thead><tbody></tbody></table></div>
    <h3>相同初态和随机指令的全部分层距离曲线</h3>
    <label class="small">选择种子与臂对：<select id="scientific-risk-select" style="max-width:100%"></select></label>
    <div id="scientific-risk-trace" style="margin-top:8px"></div>
    <p class="small" id="scientific-risk-caption"></p>
    <h3>旧初态零输入诊断：无随机指令也会违规</h3>
    <p class="small">独立192窗口；外部输入全零且实际目标恒定。前版amp=0参数被CLI拒绝，未运行的窗口不计入测试。零时钟刷新未改变几何读数。桌面豁免撤回与物理初态漂移的详细记录见报告。</p>
    <div class="table-wrap"><table id="scientific-risk-zero"><thead><tr><th>旧初态种子</th><th>零输入违规</th><th>初态豁免桌面重叠</th><th>负余量豁免撤回</th><th>最大关节漂移</th></tr></thead><tbody></tbody></table></div>
    <div class="sci-note" id="scientific-risk-verdict"></div>
  </div>
'''
anchor='  <div id="scientific-random" style="display:none">'
assert s.count(anchor)==1;s=s.replace(anchor,section+anchor)
start=s.index('const renderRandomCoverage = index => {');end=s.index('const renderRandomSpace =',start)
block=s[start:end].replace('const renderRandomCoverage = index => {','const renderRandomCoverage = (index, sourceKey="safety_random_space", container="scientific-random") => {')
block=block.replace('SCIENTIFIC.safety_random_space.coverage_cases','SCIENTIFIC[sourceKey].coverage_cases')
block=re.sub(r'\$\("#scientific-random([^"\n]*)"\)',lambda m:'$(`#${container}'+m.group(1)+'`)',block)
s=s[:start]+block+s[end:]
render='''function renderRiskStrata(d) {
  const r=d.safety_risk_strata;
  $("#scientific-risk").style.display=r?.rows?.length ? "block" : "none";
  if (!r?.rows?.length) return;
  $("#scientific-risk-note").textContent=r.summary;
  $("#scientific-risk-protocol").textContent=r.protocol_note;
  $("#scientific-risk-bank").textContent=r.bank_note;
  $("#scientific-risk-verdict").textContent=r.verdict;
  $("#scientific-risk-table tbody").innerHTML=r.aggregate_rows.map(x=>`<tr><td><b>${esc(x.label)}</b><br>${esc(x.method_label)}</td><td class="num">${x.windows}/${x.planned_windows}</td><td class="num ${x.violations ? 'bad' : 'ok'}">${x.violations}/${x.windows}</td><td class="num">${x.deep}</td><td class="num">${x.class_violations.join(" / ")}</td><td class="num">${x.zero_prefix_violations}</td><td class="num">${x.before_first_random_delivery}</td></tr>`).join("");
  $("#scientific-risk-inputs tbody").innerHTML=r.paired_inputs.filter(x=>x.pre_first_random_delivery_q_exact != null).map(x=>`<tr><td>${esc(x.raw)}</td><td>是 / 是</td><td>${x.pre_first_random_delivery_q_exact ? "是" : "否"}</td><td class="num">${f(x.pre_first_random_delivery_q_max_diff_rad,6)}</td><td class="num">${x.raw_only_failure} / ${x.protected_only_failure} / ${x.both_failure}</td></tr>`).join("");
  $("#scientific-risk-pair-results tbody").innerHTML=r.pair_rows.map(x=>`<tr><td><b>${esc(x.label)}</b><br>${esc(x.method_label)}</td><td class="num">${x.windows}/24</td><td class="num">${x.target_pair_exposure.exposed_80mm} / ${x.target_pair_pressure_exposure.exposed_80mm}（/${x.windows}）</td><td class="num ${x.violations ? 'bad' : 'ok'}">${x.violations}/${x.windows}</td><td class="num">${x.deep}</td></tr>`).join("");
  $("#scientific-risk-zero tbody").innerHTML=r.zero_diagnostic.rows.map(x=>`<tr><td>${esc(x.id)}</td><td class="num bad">${x.violations}/${x.windows}</td><td class="num">${x.initial_exempt_table_overlap}</td><td class="num">${x.negative_table_exemption_withdrawal}</td><td class="num">${f(x.max_q_drift_rad,3)} rad</td></tr>`).join("");
  $("#scientific-risk-coverage-select").innerHTML=r.coverage_cases.map((x,i)=>`<option value="${i}">${esc(x.label)}</option>`).join("");
  $("#scientific-risk-coverage-select").onchange=e=>renderRandomCoverage(e.target.value,"safety_risk_strata","scientific-risk");
  $("#scientific-risk-coverage-select").value=String(r.default_coverage);
  renderRandomCoverage(r.default_coverage,"safety_risk_strata","scientific-risk");
  $("#scientific-risk-select").innerHTML=r.traces.map((x,i)=>`<option value="${i}">${esc(x.label)}</option>`).join("");
  $("#scientific-risk-select").onchange=e=>renderRobustnessTrace(e.target.value,"safety_risk_strata","scientific-risk");
  $("#scientific-risk-select").value=String(r.default_trace);
  renderRobustnessTrace(r.default_trace,"safety_risk_strata","scientific-risk");
  $("#scientific-random h3").textContent="上一轮：广域纯随机与全部容量失败";
  $("#scientific-protocol").textContent=r.protocol_note;
  $("#ov-scientific").innerHTML=`<b>${esc(r.summary)}</b><br><span class="small">完整窗口、六臂对暴露和失败曲线已分别报告。</span><div style="margin-top:4px"><a href="#scientific">风险分层配对结果 →</a></div>`;
}
'''
anchor='function renderRobustness(d) {';assert s.count(anchor)==1;s=s.replace(anchor,render+anchor)
anchor='  renderRandomSpace(d);';assert s.count(anchor)==1;s=s.replace(anchor,anchor+'\n  renderRiskStrata(d);')
path.write_text(s)
