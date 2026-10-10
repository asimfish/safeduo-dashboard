'use strict';
const data = JSON.parse(document.getElementById('chartData').textContent);
const joint = document.getElementById('joint');
const svgNS = 'http://www.w3.org/2000/svg';
const node = (tag, attrs, text) => {
  const el = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  if (text !== undefined) el.textContent = text;
  return el;
};
const draw = (id, counts, low, high, nativeValues, unit, name) => {
  const svg = document.getElementById(id);
  svg.replaceChildren();
  svg.setAttribute('aria-label', `${name} ${unit}：8192 个生成样本，32 个原生初始读数`);
  svg.dataset.counts = JSON.stringify(counts);
  svg.dataset.low = String(low); svg.dataset.high = String(high);
  const left = 47, right = 540, top = 24, base = 211, width = right - left;
  const max = Math.ceil(Math.max(...counts) / 100) * 100;
  for (let i = 0; i <= 4; i++) {
    const y = base - (base - top) * i / 4;
    svg.append(node('line', {x1:left, y1:y, x2:right, y2:y, stroke:'#dbe5eb'}));
    svg.append(node('text', {x:left - 8, y:y + 4, 'text-anchor':'end'}, String(max * i / 4)));
  }
  counts.forEach((count, i) => {
    const height = count / max * (base - top);
    const bar = node('rect', {class:'histogram-bin', x:left + i * width / 32 + .8, y:base - height,
      width:width / 32 - 1.6, height, fill:'#287ba4', 'data-count':count});
    bar.append(node('title', {}, `${(low + (high-low)*i/32).toPrecision(5)}–${(low + (high-low)*(i+1)/32).toPrecision(5)} ${unit}：${count}`));
    svg.append(bar);
  });
  nativeValues.forEach((value, env) => {
    const x = left + (value - low) / (high - low) * width;
    const mark = node('line', {class:'native-mark', x1:x, x2:x, y1:218, y2:228, stroke:'#b96b14', 'stroke-width':1.5, 'data-value':value});
    mark.append(node('title', {}, `env ${env} point0：${value} ${unit}`)); svg.append(mark);
  });
  for (let i = 0; i <= 4; i++) {
    const value = low + (high - low) * i / 4;
    svg.append(node('text', {x:left + width * i / 4, y:250, 'text-anchor': i === 0 ? 'start' : i === 4 ? 'end' : 'middle'}, Number(value.toPrecision(4)).toString()));
  }
  svg.append(node('text', {x:right, y:275, 'text-anchor':'end'}, unit));
};
const renderJoint = () => {
  const index = Number(joint.value), j = data.joints[index];
  document.getElementById('jointDescription').textContent = `${j.native_name} · ${j.controlled26 ? '控制关节，位置取 hard/soft 交集' : '手部关节，位置取完整 hard 区间'} · 第 ${index + 1} / 74 个关节`;
  draw('positionChart', j.position_histogram, j.position_lower, j.position_upper,
    data.native.q.map(row => row[index]), 'rad', j.native_name);
  draw('velocityChart', j.velocity_histogram, -j.native_vmax, j.native_vmax,
    data.native.qd.map(row => row[index]), 'rad/s', j.native_name);
  document.getElementById('previousJoint').disabled = index === 0;
  document.getElementById('nextJoint').disabled = index === 73;
  document.getElementById('interactive').dataset.joint = String(index);
};
joint.addEventListener('change', renderJoint);
document.getElementById('previousJoint').addEventListener('click', () => { joint.value = String(Number(joint.value) - 1); renderJoint(); });
document.getElementById('nextJoint').addEventListener('click', () => { joint.value = String(Number(joint.value) + 1); renderJoint(); });
renderJoint();
document.getElementById('interactive').hidden = false;
document.getElementById('staticDistribution').open = false;
document.documentElement.dataset.ready = 'true';
