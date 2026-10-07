import argparse,json,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
a.out.mkdir(exist_ok=False);receipts=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True)
 for width in (1440,390):
  page=browser.new_page(viewport={'width':width,'height':950},accept_downloads=True);errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(a.base+'/docs/safety_hand_support_calibration_20261007/',wait_until='networkidle',timeout=60000)
  page.wait_for_function("document.querySelector('#case').options.length===6")
  assert page.locator('#error').inner_text()==''
  assert page.locator('#verdict p').count()>=2
  plots=0;galleries=0
  for case in range(6):
   page.locator('#case').select_option(str(case));assert page.locator('#hands tr').count()==4
   for arm in ('F_L','F_R','U_L','U_R'):
    page.locator('#arm').select_option(arm)
    for metric in ('error','tracking','contact'):
     page.locator('#metric').select_option(metric);assert page.locator('#chart polyline').count()==2;plots+=1
   for condition in ('baseline_v2','self_off_v2'):
    page.locator('#condition').select_option(condition)
    for step in ('299','659','1199'):
     page.locator('#frame').select_option(step)
     page.locator('#gallery').scroll_into_view_if_needed()
     page.wait_for_function("document.querySelectorAll('#gallery img').length===3 && [...document.querySelectorAll('#gallery img')].every(i=>i.complete&&i.naturalWidth===1280&&i.naturalHeight===720)",timeout=60000)
     galleries+=1
  opening_galleries=0
  assert page.locator('#opening-hands tr').count()==8
  for case in range(4):
   page.locator('#opening-case').select_option(str(case))
   for method in ('original_zero','u_thumb_neutral'):
    page.locator('#opening-method').select_option(method)
    for step in ('299','659','1199'):
     page.locator('#opening-step').select_option(step)
     page.locator('#opening-gallery').scroll_into_view_if_needed()
     page.wait_for_function("document.querySelectorAll('#opening-gallery img').length===3 && [...document.querySelectorAll('#opening-gallery img')].every(i=>i.complete&&i.naturalWidth===1280&&i.naturalHeight===720)",timeout=60000)
     opening_galleries+=1
  for file in ('BASELINE_V2_RESULTS.json','SELF_OFF_V2_RESULTS.json','OPEN_POSE_RESULTS.json'):
   # Browser request proves the exposed result URL binds to the local sealed file.
   response=page.request.get(a.base+'/docs/safety_hand_support_calibration_20261007/'+file)
   assert response.ok and response.body()==(a.repo/'docs/safety_hand_support_calibration_20261007'/file).read_bytes()
  page.locator('#case').select_option('2');page.locator('#arm').select_option('U_R');page.locator('#metric').select_option('error');page.locator('#condition').select_option('baseline_v2');page.locator('#frame').select_option('1199');page.evaluate('window.scrollTo(0,0)')
  assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
  page.screenshot(path=str(a.out/f'panel_{width}.png'),full_page=True)
  page.goto(a.base+'/index.html#scientific',wait_until='networkidle',timeout=60000)
  for key in ('handSupportCalibrationEvidence','releaseFeedbackEvidence','trackingReserveEvidence','objectBindingEvidence'):assert page.locator('#'+key).is_visible()
  assert not errors,errors
  receipts.append(dict(width=width,plot_choices=plots,three_view_groups=galleries,opening_three_view_groups=opening_galleries,json_byte_checks=3,js_errors=errors,no_overflow=True,main_card_and_previous_cards_visible=True))
  print('BROWSER_COMPLETE',width,plots,galleries,flush=True);page.close()
 browser.close()
(a.out/'receipt.json').write_text(json.dumps(dict(status='PASS_NATIVE_CALIBRATION_USER_JOURNEY',base=a.base,viewports=receipts),indent=2)+'\n')
