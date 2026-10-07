import argparse,gzip,json
from pathlib import Path
from playwright.sync_api import sync_playwright
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(exist_ok=False);receipts=[];doc='/docs/safety_hand_initialization_20261007/'
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True)
    for width in [1440,390]:
        page=browser.new_page(viewport={'width':width,'height':950});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(a.base+doc,wait_until='networkidle',timeout=60000)
        page.wait_for_function('window.safeduoEvidence && document.querySelectorAll("#cases tr").length===12',timeout=60000)
        assert page.locator('#error').inner_text()==''
        plots=0;galleries=0
        for pose in ['0','1','2']:
            page.locator('#pose').select_option(pose)
            for arm in ['F_L','F_R','U_L','U_R']:
                page.locator('#arm').select_option(arm)
                for metric in ['contact','tracking','error']:
                    page.locator('#metric').select_option(metric)
                    for scope in ['all','startup']:
                        page.locator('#window').select_option(scope)
                        assert page.locator('#chart polyline').count()==4
                        assert page.evaluate('[...document.querySelectorAll("#chart polyline")].every(p=>p.getAttribute("points").length>10&&!/NaN|Infinity/.test(p.getAttribute("points")))')
                        plots+=1
            for run in ['old_default','neutral_default']:
                page.locator('#run').select_option(run)
                for post in ['0','1']:
                    page.locator('#post').select_option(post)
                    assert page.locator('#hands tr').count()==20
                    for step in ['0','659','1019','1379','1979']:
                        page.locator('#frame').select_option(step);page.locator('#gallery').scroll_into_view_if_needed()
                        page.wait_for_function('document.querySelectorAll("#gallery img").length===3&&[...document.querySelectorAll("#gallery img")].every(i=>i.complete&&i.naturalWidth===1280&&i.naturalHeight===720)',timeout=60000)
                        galleries+=1
        for name in ['SUMMARY.json','MEDIA_ARCHIVE.json','payload.json.gz']:
            response=page.request.get(a.base+doc+name);assert response.ok
            assert response.body()==(a.repo/doc.lstrip('/')/name).read_bytes(),name
        payload=gzip.decompress((a.repo/doc.lstrip('/')/'payload.json.gz').read_bytes())
        assert json.loads(payload)==page.evaluate('window.safeduoEvidence')
        assert page.locator('#links a').count()==14
        page.locator('#pose').select_option('1');page.locator('#arm').select_option('U_R');page.locator('#metric').select_option('contact');page.locator('#window').select_option('all')
        page.locator('#run').select_option('old_default');page.locator('#post').select_option('1');page.locator('#frame').select_option('1379')
        page.evaluate('[...document.images].forEach(i=>i.loading="eager")')
        page.wait_for_function('[...document.images].every(i=>i.complete&&i.naturalWidth===1280)',timeout=60000)
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        page.screenshot(path=str(a.out/f'panel_{width}.png'),full_page=True)
        page.goto(a.base+'/index.html#scientific',wait_until='networkidle',timeout=60000)
        for key in ['handInitializationEvidence','handSupportCalibrationEvidence','releaseFeedbackEvidence','zeroIntentEvidence','trackingReserveEvidence','objectBindingEvidence']:
            assert page.locator('#'+key).is_visible(),key
        assert not errors,errors
        receipts.append(dict(width=width,plot_choices=plots,three_view_groups=galleries,original_images_loaded=galleries*3,table_rows=12,hand_window_rows=20,byte_checks=3,payload_semantically_exact=True,js_errors=errors,no_overflow=True,prior_cards_intact=True))
        page.close();print('BROWSER_PASS',width,plots,galleries,flush=True)
    browser.close()
(a.out/'receipt.json').write_text(json.dumps(dict(status='PASS_NATIVE_INITIALIZATION_BROWSER_JOURNEY',base=a.base,viewports=receipts),indent=2)+'\n')
