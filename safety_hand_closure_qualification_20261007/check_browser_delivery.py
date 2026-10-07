import argparse,json,hashlib,re
from pathlib import Path
from playwright.sync_api import sync_playwright
pa=argparse.ArgumentParser();pa.add_argument('--base',required=True);pa.add_argument('--repo',type=Path,required=True);pa.add_argument('--out',type=Path,required=True);pa.add_argument('--archive-local',type=Path);a=pa.parse_args();a.out.mkdir(exist_ok=False);doc='/docs/safety_hand_closure_qualification_20261007/';records=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True)
 for width in [1440,390]:
  page=browser.new_page(viewport={'width':width,'height':950});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  if a.archive_local:
   def serve(route):
    url=route.request.url;rel=url.split('/safety_hand_closure_qualification_20261007/',1)[1];file=a.archive_local/rel;body=file.read_bytes();headers={'Access-Control-Allow-Origin':'*','Accept-Ranges':'bytes'};status=200;mime='application/octet-stream'
    if file.suffix=='.png':mime='image/png'
    elif file.suffix=='.webm':mime='video/webm'
    elif file.suffix=='.mp4':mime='video/mp4'
    rr=route.request.headers.get('range')
    if rr:
     match=re.fullmatch(r'bytes=(\d+)-(\d*)',rr);assert match,rr;start=int(match[1]);end=min(int(match[2]) if match[2] else len(body)-1,len(body)-1);headers['Content-Range']=f'bytes {start}-{end}/{len(body)}';body=body[start:end+1];status=206
    route.fulfill(status=status,body=body,content_type=mime,headers=headers)
   page.route('https://raw.githubusercontent.com/asimfish/safeduo-dashboard/**/safety_hand_closure_qualification_20261007/**',serve)
  page.goto(a.base+doc,wait_until='networkidle',timeout=60000);page.wait_for_function('window.safeduoReady===true',timeout=60000);assert page.locator('#error').inner_text()=='';data=page.evaluate('window.safeduoEvidence')
  curve_gate=page.evaluate('''()=>{let n=0,phases=[];for(const run of data.phase_keys){document.querySelector('#run').value=run;fillProfiles();const refs=[...new Set(data[run].registration.cases.map(c=>c.reference_time_s))];for(let i=0;i<data[run].registration.profiles.length;i++){document.querySelector('#profile').value=i;for(const ref of refs){document.querySelector('#reference').value=ref;for(const arm of ['U_L','U_R']){document.querySelector('#arm').value=arm;draw();const lines=[...document.querySelectorAll('#chart polyline')];if(lines.length!==6||lines.some(p=>/NaN|Infinity/.test(p.getAttribute('points'))||p.getAttribute('points').length<10))throw Error('bad curve '+run+'/'+i+'/'+ref+'/'+arm);const expected=data[run].rows.filter(x=>x.profile_index===i&&x.arm.startsWith('U')).length;if(document.querySelectorAll('#pathRows tr').length!==expected)throw Error('bad path rows');n++}}}phases.push(run)}let cells=0;for(const ff of ['0','0.475','1']){document.querySelector('#finger').value=ff;matrix();for(const button of document.querySelectorAll('#matrix button')){button.click();if(document.querySelector('#run').value!=='development'||document.querySelector('#profile').value!==button.dataset.profile)throw Error('bad matrix');cells++}}return {curves:n,phases,cells}}''')
  for run in data['phase_keys']:page.locator('#run').select_option(run);page.locator('#arm').select_option('U_R')
  paired_choices=0
  for ref in ['2','8','20']:
   page.locator('#pairedReference').select_option(ref)
   for arm in ['U_L','U_R']:page.locator('#pairedArm').select_option(arm);assert page.locator('#pairedChart polyline').count()==2;paired_choices+=1
  for arm in ['U_L','U_R']:page.locator('#recoveryArm').select_option(arm);assert page.locator('#recoveryChart polyline').count()==2
  if a.archive_local:gallery_choices=list(range(len(data['gallery'])))
  else:
   gallery_choices=set()
   for run in sorted({g['run'] for g in data['gallery']}):
    candidates=[i for i,g in enumerate(data['gallery']) if g['run']==run];gallery_choices.update([candidates[0],candidates[len(candidates)//2],candidates[-1]])
   for env in range(12):
    candidates=[i for i,g in enumerate(data['gallery']) if g['run']=='validation_fresh_v3' and g['env']==env];gallery_choices.add(candidates[-1])
   for env in range(8):
    candidates=[i for i,g in enumerate(data['gallery']) if g['run']=='paired_v3' and g['env']==env];gallery_choices.add(candidates[-1])
   gallery_choices=sorted(gallery_choices)
  images=0
  for i in gallery_choices:
   g=data['gallery'][i];page.locator('#frame').select_option(str(i));page.locator('#gallery').scroll_into_view_if_needed();page.wait_for_function('Array.from(document.querySelectorAll("#gallery img")).every(i=>i.complete&&i.naturalWidth===1280&&i.naturalHeight===720)',timeout=60000);imgs=page.locator('#gallery img');assert imgs.count()==len(g['images']);assert all(abs(r-1280/720)<.001 for r in imgs.evaluate_all('(xs)=>xs.map(i=>i.getBoundingClientRect().width/i.getBoundingClientRect().height)'));images+=len(g['images'])
  video_results=[]
  for i in range(page.locator('video').count()):
   video=page.locator('video').nth(i);video.scroll_into_view_if_needed();video.evaluate('(v)=>{v.muted=true;v.currentTime=0;return v.play()}');page.wait_for_function('(i)=>{const v=document.querySelectorAll("video")[i];return v.readyState>=3&&v.currentTime>.2&&v.videoWidth===1280}',arg=i,timeout=60000);result=video.evaluate('(v)=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration,time:v.currentTime,frames:v.getVideoPlaybackQuality().totalVideoFrames,error:v.error&&v.error.message,source:v.currentSrc})');assert not result['error'] and result['frames']>0 and result['height']==720;video_results.append(result);video.evaluate('(v)=>v.pause()')
  assert len(video_results)==len(data['videos']);assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth');assert page.locator('#allProfiles tr').count()==sum(len(data[x]['registration']['profiles']) for x in data['phase_keys']);page.locator('#run').select_option('validation');page.locator('#summary').scroll_into_view_if_needed();page.screenshot(path=str(a.out/f'panel_{width}.png'),full_page=True)
  for file in ['index.html','panel.js','SUMMARY.json','MEDIA_ARCHIVE.json','PUBLIC_MANIFEST.json']:
   response=page.request.get(a.base+doc+file);assert response.ok and response.body()==(a.repo/doc.lstrip('/')/file).read_bytes()
  page.goto(a.base+'/index.html#scientific',wait_until='domcontentloaded',timeout=60000)
  for key in ['handClosureEvidence','handInitializationEvidence','handSupportCalibrationEvidence','releaseFeedbackEvidence','queueViabilityEvidence','zeroIntentEvidence','trackingReserveEvidence','objectBindingEvidence']:assert page.locator('#'+key).is_visible(),key
  assert not errors,errors;records.append(dict(width=width,curve_gate=curve_gate,paired_choices=paired_choices,recovery_choices=2,gallery_selection='all' if a.archive_local else 'all_stages_and_all_new_reference_environments',gallery_frames=len(gallery_choices),images=images,video_decodes=video_results,no_overflow=True,js_errors=errors,prior_cards_intact=True));print('BROWSER_PASS',width,curve_gate,images,flush=True);page.close()
 browser.close()
(a.out/'receipt.json').write_text(json.dumps(dict(status='PASS_BROWSER_JOURNEY',base=a.base,local_media_route=a.archive_local is not None,records=records),indent=2)+'\n')
