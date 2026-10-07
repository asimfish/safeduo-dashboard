import argparse,json,gzip,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
doc='/docs/safety_hand_closure_qualification_20261007/';records=[]
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True)
 for width in [1440,390]:
  page=b.new_page(viewport={'width':width,'height':950});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(a.base+doc,wait_until='networkidle',timeout=60000);page.wait_for_function('window.safeduoReady===true',timeout=60000)
  assert page.locator('#error').inner_text()==''
  d=page.evaluate('window.safeduoEvidence');curves=0
  for run in ['development','validation']:
   page.locator('#run').select_option(run)
   refs=sorted({c['reference_time_s'] for c in d[run]['registration']['cases']})
   for i in range(len(d[run]['registration']['profiles'])):
    page.locator('#profile').select_option(str(i));assert page.locator('#pathRows tr').count()==6
    for ref in refs:
     page.locator('#reference').select_option(str(int(ref)) if float(ref).is_integer() else str(ref))
     for arm in ['U_L','U_R']:
      page.locator('#arm').select_option(arm)
      assert page.locator('#chart polyline').count()==5
      assert page.evaluate('[...document.querySelectorAll("#chart polyline")].every(p=>p.getAttribute("points").length>10&&!/NaN|Infinity/.test(p.getAttribute("points")))')
      curves+=1
  paired_choices=0
  for ref in ['2','8','20']:
   page.locator('#pairedReference').select_option(ref)
   for arm in ['U_L','U_R']:
    page.locator('#pairedArm').select_option(arm);assert page.locator('#pairedChart polyline').count()==2;paired_choices+=1
  cells=0
  for f in ['0','0.475','1']:
   page.locator('#finger').select_option(f)
   assert page.locator('#matrix button').count()==25
   for el in page.locator('#matrix button').all():
    el.click();assert page.locator('#run').input_value()=='development';assert page.locator('#profile').input_value()==el.get_attribute('data-profile');cells+=1
  images=0
  for i,g in enumerate(d['gallery']):
   page.locator('#frame').select_option(str(i));page.locator('#gallery').scroll_into_view_if_needed()
   page.wait_for_function('Array.from(document.querySelectorAll("#gallery img")).every(i=>i.complete&&i.naturalWidth===1280&&i.naturalHeight===720)',timeout=60000)
   imgs=page.locator('#gallery img');assert imgs.count()==len(g['images'])
   assert all(abs(r-1280/720)<.001 for r in imgs.evaluate_all('(xs)=>xs.map(i=>i.getBoundingClientRect().width/i.getBoundingClientRect().height)'))
   images+=len(g['images'])
  video_results=[]
  for i in range(page.locator('video').count()):
   v=page.locator('video').nth(i);v.scroll_into_view_if_needed();v.evaluate('(v)=>{v.muted=true;v.currentTime=0;return v.play()}')
   page.wait_for_function('(i)=>{const v=document.querySelectorAll("video")[i];return v.readyState>=3&&v.currentTime>.2&&v.videoWidth===1280}',arg=i,timeout=60000)
   result=v.evaluate('(v)=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration,time:v.currentTime,frames:v.getVideoPlaybackQuality().totalVideoFrames,error:v.error&&v.error.message})');assert not result['error'] and result['frames']>0;video_results.append(result);v.evaluate('(v)=>v.pause()')
  assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
  assert page.locator('#allProfiles tr').count()==sum(len(d[x]['registration']['profiles']) for x in ['development','validation'])
  page.locator('#summary').scroll_into_view_if_needed();page.screenshot(path=str(a.out/f'panel_{width}.png'),full_page=True)
  for file in ['index.html','panel.js','SUMMARY.json','MEDIA_ARCHIVE.json','PUBLIC_MANIFEST.json']:
   rr=page.request.get(a.base+doc+file);assert rr.ok and rr.body()==(a.repo/doc.lstrip('/')/file).read_bytes()
  page.goto(a.base+'/index.html#scientific',wait_until='networkidle',timeout=60000)
  for key in ['handClosureEvidence','handInitializationEvidence','handSupportCalibrationEvidence','releaseFeedbackEvidence','queueViabilityEvidence','zeroIntentEvidence','trackingReserveEvidence','objectBindingEvidence']:
   assert page.locator('#'+key).is_visible(),key
  assert not errors,errors
  records.append(dict(width=width,curves=curves,paired_choices=paired_choices,matrix_cells=cells,images=images,video_decodes=video_results,no_overflow=True,js_errors=errors,prior_cards_intact=True))
  print('BROWSER_PASS',width,curves,cells,images,flush=True);page.close()
 b.close()
(a.out/'receipt.json').write_text(json.dumps(dict(status='PASS_BROWSER_JOURNEY',base=a.base,records=records),indent=2)+'\n')
