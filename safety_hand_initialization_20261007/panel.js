'use strict';
(async()=>{
 const response=await fetch('payload.json.gz');if(!response.ok)throw Error('data '+response.status);
 const stream=response.body.pipeThrough(new DecompressionStream('gzip'));
 const D=JSON.parse(await new Response(stream).text());window.safeduoEvidence=D;
 const $=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
 const label={initial:'初始',pickup:'取物',placement:'放置'},wl={startup:'启动0–3s',closed_calibration:'0.45闭合',open_after_calibration:'首次回开',closed_task_endpoint:'原任务闭合',open_after_task_endpoint:'第二次回开'};
 const pass=b=>`<span class="${b?'pass':'fail'}">${b?'通过':'失败'}</span>`;
 const allWin=(c,k)=>c.hands.every(h=>h.windows[k].qualified);
 $('verdict').innerHTML=D.summary.paragraphs.map(t=>'<p>'+t+'</p>').join('');
 for(const run of ['old_default','neutral_default'])for(const c of D.results[run].cases){
   const tr=document.createElement('tr');tr.innerHTML=`<td>${D.results[run].constructor_thumb_rad}</td><td>${c.post_write_thumb_rad}</td><td>${label[c.label]}</td><td>${pass(c.startup_qualified)}</td><td>${pass(allWin(c,'closed_calibration'))}</td><td>${pass(allWin(c,'closed_task_endpoint'))}</td><td>${pass(allWin(c,'open_after_calibration')&&allWin(c,'open_after_task_endpoint'))}</td><td>${pass(c.whole_path_contacts_qualified)}</td>`;$('cases').append(tr);
 }
 const names=['默认0，后写0','默认0，后写0.35','默认0.35，后写0','默认0.35，后写0.35'],colors=['#bc4829','#168251','#7a48a4','#2379bd'];
 $('legend').innerHTML=names.map((n,i)=>`<span style="color:${colors[i]}">● ${n}</span>`).join('');
 function svg(tag,attrs,text){const v=document.createElementNS(ns,tag);for(const [k,x]of Object.entries(attrs))v.setAttribute(k,x);if(text!==undefined)v.textContent=text;$('chart').append(v);return v;}
 function chart(){
  const e=2*Number($('pose').value),arm=$('arm').value,metric=$('metric').value,end=$('window').value==='startup'?3:18;
  const lines=['old_default','neutral_default'].flatMap(run=>[e,e+1].map(env=>D.series[run][env][arm]));
  const yfun=metric==='contact'?(v=>Math.log10(1+v)):(v=>v);let max=0;
  lines.forEach(l=>l.time.forEach((t,i)=>{if(t<=end)max=Math.max(max,yfun(l[metric][i]));}));max=Math.max(max,metric==='contact'?.1:.02)*1.07;
  $('chart').replaceChildren();const X=t=>65+900*t/end,Y=v=>315-270*yfun(v)/max;
  svg('path',{d:'M65 30V315H965',fill:'none',stroke:'#56697a'});
  for(let i=0;i<=6;i++){const t=i*end/6;svg('text',{x:X(t),y:343,'text-anchor':'middle',fill:'#354d60'},t.toFixed(1));}
  for(let i=0;i<=4;i++){const n=max*i/4,val=metric==='contact'?10**n-1:n;svg('text',{x:57,y:320-270*i/4,'text-anchor':'end',fill:'#354d60'},val<10?val.toFixed(2):val.toFixed(0));}
  if(metric==='contact')svg('line',{x1:65,x2:965,y1:Y(.1),y2:Y(.1),stroke:'#a73326','stroke-dasharray':'5 4'});
  lines.forEach((l,k)=>{const points=l.time.map((t,i)=>t<=end?X(t)+','+Y(l[metric][i]):null).filter(Boolean).join(' ');svg('polyline',{points,fill:'none',stroke:colors[k],'stroke-width':2,opacity:.88});});
  $('chart-note').textContent=metric==='contact'?'纵轴使用log10(1+N)，刻度标回原始N；红虚线为0.1 N门槛。重叠的曲线可能完全一致。':'纵轴为rad。与开放参考的误差在闭合时自然增大；追踪误差才反映是否达到当前闭合目标。';
 }
 function gallery(){
  const e=2*Number($('pose').value)+Number($('post').value),run=$('run').value,step=Number($('frame').value),c=D.results[run].cases[e];
  $('gallery').replaceChildren();for(const view of ['top','front','side']){
   const im=D.images.find(i=>i.run===run&&i.env===e&&i.step===step&&i.view===view);if(!im)throw Error('missing image');
   const f=document.createElement('figure'),a=document.createElement('a'),img=document.createElement('img'),cap=document.createElement('figcaption');
   a.href=im.url;a.target='_blank';a.rel='noopener';img.src=im.url;img.loading='lazy';img.width=1280;img.height=720;img.alt=`${label[c.label]} ${view} t=${im.state_time_s.toFixed(6)}s`;
   a.append(img);cap.textContent=`${view} · t=${im.state_time_s.toFixed(6)}s · ${run} env${e}`;f.append(a,cap);$('gallery').append(f);
  }
  $('hands').replaceChildren();$('peaks').replaceChildren();for(const h of c.hands){
   for(const [k,w]of Object.entries(h.windows)){const tr=document.createElement('tr');tr.innerHTML=`<td>${h.arm}</td><td>${wl[k]}</td><td>${w.max_error_rad.toFixed(6)}</td><td>${w.target_error_rad.toFixed(6)}</td><td>${w.pair_normal_max_n.toFixed(3)}</td><td>${pass(w.qualified)}</td>`;$('hands').append(tr);}
   const p=document.createElement('p'),v=h.whole_path_peak;p.textContent=`${h.arm}: ${v.normal_n.toFixed(3)} N @ ${v.time_s.toFixed(6)}s；${v.sensor} ↔ ${v.partner}`;$('peaks').append(p);
  }
 }
 for(const link of D.links){const li=document.createElement('li'),a=document.createElement('a');a.href=link.url;a.textContent=link.label;li.append(a);$('links').append(li);}
 for(const id of ['pose','arm','metric','window'])$(id).addEventListener('change',chart);
 for(const id of ['pose','run','post','frame'])$(id).addEventListener('change',gallery);
 chart();gallery();
})().catch(e=>{document.getElementById('error').textContent='证据加载失败：'+e.message;throw e;});
