from pathlib import Path

path=Path('/home/liyufeng/safeduo-dashboard/index.html')
text=path.read_text()
assert 'id="scientific-mechanism"' not in text
html='''  <div id="scientific-mechanism" style="display:none">
    <h3>目标积压机制：独立留出安全 / 不安全效果</h3>
    <div class="sci-note" id="scientific-mechanism-note"></div>
    <p class="small" id="scientific-mechanism-protocol"></p>
    <p class="small" id="scientific-mechanism-rule"></p>
    <div class="table-wrap"><table id="scientific-mechanism-table"><thead><tr><th>方法</th><th>完成 / 192</th><th>全16秒违规</th><th>超过5mm</th><th>cross / F / U / 桌面</th><th>压力期四臂同时动</th><th>输出 / 命令L2</th><th>目标积压P95</th><th>平均关节跨度 / 路径</th></tr></thead><tbody></tbody></table></div>
    <p class="small" id="scientific-mechanism-bank"></p>
    <h4>配对失败变化（同初态、同外部指令）</h4>
    <div class="table-wrap"><table id="scientific-mechanism-inputs"><thead><tr><th>种子 / 左 / 右</th><th>左独有 / 右独有失败</th><th>双方失败 / 双方通过</th><th>66步实际q前缀一致</th><th>最大q差rad</th></tr></thead><tbody></tbody></table></div>
    <h4>六组臂对配额与实际随机压力</h4>
    <div class="table-wrap"><table id="scientific-mechanism-pair-results"><thead><tr><th>臂对 / 方法</th><th>完成 / 24</th><th>≥3帧80mm：全窗 / 随机目标到达后</th><th>违规</th><th>超过5mm</th></tr></thead><tbody></tbody></table></div>
    <h4>全部帧的实测操作范围</h4>
    <label class="small">选择分层与方法：<select id="scientific-mechanism-coverage-select" style="max-width:100%"></select></label>
    <p class="small" id="scientific-mechanism-coverage-note"></p>
    <div id="scientific-mechanism-heatmap"></div>
    <div class="table-wrap"><table id="scientific-mechanism-ee"><thead><tr><th>臂</th><th>初态 / 实测10cm格</th><th>实测x / y / z范围（米）</th></tr></thead><tbody></tbody></table></div>
    <div class="table-wrap"><table id="scientific-mechanism-pairs"><thead><tr><th>六组臂对</th><th>≥3帧80mm / 总窗口</th><th>5mm内 / 重叠</th><th>末端接近 / 远离 / 切向帧</th></tr></thead><tbody></tbody></table></div>
    <h4>四方法配对曲线：改善、仍失败与新增失败</h4>
    <label class="small">选择种子与臂对：<select id="scientific-mechanism-select" style="max-width:100%"></select></label>
    <div id="scientific-mechanism-trace" style="margin-top:8px"></div>
    <p class="small" id="scientific-mechanism-caption"></p>
    <p class="small" id="scientific-mechanism-diagnostic"></p>
    <div class="sci-note" id="scientific-mechanism-verdict"></div>
  </div>
'''
text=text.replace('  <div id="scientific-risk"',html+'  <div id="scientific-risk"',1)
js='''const renderMechanism = d => {
  const r=d.safety_mechanism;
  $("#scientific-mechanism").style.display=r?.rows?.length ? "block" : "none";
  if (!r?.rows?.length) return;
  $("#scientific-mechanism-note").textContent=r.summary;
  $("#scientific-mechanism-protocol").textContent=r.protocol_note;
  $("#scientific-mechanism-rule").textContent=r.mechanism_note;
  $("#scientific-mechanism-bank").textContent=r.bank_note;
  $("#scientific-mechanism-diagnostic").textContent=r.diagnostic_note;
  $("#scientific-mechanism-verdict").textContent=r.verdict;
  $("#scientific-mechanism-table tbody").innerHTML=r.aggregate_rows.map(x=>`<tr><td><b>${esc(x.method_label)}</b></td><td class="num">${x.windows}/192</td><td class="num ${x.violations ? 'bad' : 'ok'}">${x.violations}/${x.windows}</td><td class="num">${x.deep}</td><td class="num">${x.class_violations.join(" / ")}</td><td class="num">${sciPct(x.four_arms_moving_fraction)}</td><td class="num">${f(x.exec_command_l2_ratio,3)}</td><td class="num">${f(x.pre_target_debt_abs_p95_rad,4)}rad</td><td class="num">${sciPct(x.mean_within_window_joint_range)} / ${f(x.mean_joint_path_rad,2)}rad</td></tr>`).join("");
  const names={raw:"raw",baseline:"原System 0",box_only:"增量限制",envelope_050:"积压限制"};
  $("#scientific-mechanism-inputs tbody").innerHTML=r.paired_inputs.filter(x=>x.both_failure != null).map(x=>`<tr><td>${x.seed}<br>${esc(names[x.left_mode])} / ${esc(names[x.right_mode])}</td><td class="num">${x.left_only_failure} / ${x.right_only_failure}</td><td class="num">${x.both_failure} / ${x.neither_failure}</td><td>${x.actual_q_prefix66_exact ? "是" : "否"}</td><td class="num">${f(x.actual_q_prefix66_max_diff_rad,6)}</td></tr>`).join("");
  $("#scientific-mechanism-pair-results tbody").innerHTML=r.pair_rows.map(x=>`<tr><td><b>${esc(x.label)}</b><br>${esc(x.method_label)}</td><td class="num">${x.windows}/24</td><td class="num">${x.target_pair_exposure.exposed_80mm} / ${x.target_pair_pressure_exposure.exposed_80mm}</td><td class="num ${x.violations ? 'bad' : 'ok'}">${x.violations}/${x.windows}</td><td class="num">${x.deep}</td></tr>`).join("");
  $("#scientific-mechanism-coverage-select").innerHTML=r.coverage_cases.map((x,i)=>`<option value="${i}">${esc(x.label)}</option>`).join("");
  $("#scientific-mechanism-coverage-select").onchange=e=>renderRandomCoverage(e.target.value,"safety_mechanism","scientific-mechanism");
  $("#scientific-mechanism-coverage-select").value=String(r.default_coverage);
  renderRandomCoverage(r.default_coverage,"safety_mechanism","scientific-mechanism");
  $("#scientific-mechanism-select").innerHTML=r.traces.map((x,i)=>`<option value="${i}">${esc(x.label)}</option>`).join("");
  $("#scientific-mechanism-select").onchange=e=>renderRobustnessTrace(e.target.value,"safety_mechanism","scientific-mechanism");
  $("#scientific-mechanism-select").value=String(r.default_trace);
  renderRobustnessTrace(r.default_trace,"safety_mechanism","scientific-mechanism");
  $("#scientific-risk h3").textContent="上一轮：六组臂对风险分层与稳定初态";
  $("#ov-scientific").innerHTML=`<b>${esc(r.summary)}</b><br><span class="small">${esc(r.verdict)}</span><div style="margin-top:4px"><a href="#scientific">最新机制实验 →</a></div>`;
};
'''
text=text.replace('function renderRiskStrata(d)',js+'function renderRiskStrata(d)',1)
text=text.replace('  renderRiskStrata(d);','  renderRiskStrata(d);\n  renderMechanism(d);',1)
path.write_text(text)
print('added latest mechanism section; existing tables and videos preserved')
