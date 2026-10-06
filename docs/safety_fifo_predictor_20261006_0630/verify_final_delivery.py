"""Explicit full-UI evidence reuse plus a fresh native final-certificate smoke."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,re,subprocess
from playwright.sync_api import sync_playwright
from verify_remote import get
HERE=Path(__file__).resolve().parent
WORK=Path('/home/liyufeng/safeduo-dashboard-feasible-guard-20261005')
BASE='https://asimfish.github.io/safeduo-dashboard/'
def sha(b):return hashlib.sha256(b).hexdigest()
def load(n):return json.loads((HERE/n).read_text())
def card(text,tag):
    parts=re.findall(r'<section\b[^>]*\bid="'+tag+r'"[^>]*>.*?</section>',text,re.S);assert len(parts)==1;return parts[0]
def main():
    initial=load('remote_verification_initial.json');final=load('remote_verification.json');full=load('live_browser_initial.json')
    for r in [initial,final,full]:assert r['status'].startswith('PASS')
    assert final['commit']==final['pages_commit'] and final['commit']!=initial['commit']
    before={r['path']:r['sha256'] for r in initial['readback']};after={r['path']:r['sha256'] for r in final['readback']}
    assert before.keys()==after.keys();cert='docs/'+HERE.name+'/VERIFICATION.json'
    changes={p for p in before if before[p]!=after[p]};assert cert in changes and changes<={cert,'index.html'}
    original=(HERE/'PUBLIC_INITIAL_ROOT.html').read_text();current=(WORK/'index.html').read_text()
    for tag in ['fifoPredictorEvidence','feasibleGuardEvidence']:assert card(original,tag)==card(current,tag)
    git_changed=subprocess.check_output(['git','diff','--name-only',initial['commit'],final['commit']],cwd=WORK,text=True).splitlines()
    assert [p for p in git_changed if p.startswith('docs/'+HERE.name+'/')]==[cert]
    assert full['source_sha256']==sha((HERE/'check_panel.py').read_bytes()) and full['result_sha256']==sha((HERE/'holdout_results.json').read_bytes())
    scientific=get('https://raw.githubusercontent.com/asimfish/safeduo-dashboard/data/scientific_eval.json?fifo_final='+final['commit'])
    legacy=load('legacy_live_browser_initial.json');assert sha(scientific)==legacy['scientific_snapshot_sha256']
    url=BASE+'docs/'+HERE.name+'/?fifo_commit='+final['commit'];errors=[];served={}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu']);page=browser.new_page(viewport={'width':1440,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)))
        cdp=page.context.new_cdp_session(page);cdp.send('Network.enable',{'maxTotalBufferSize':128*1024*1024,'maxResourceBufferSize':64*1024*1024})
        def seen(e):
            if e['type']=='Document' and e['response']['url']==url:served['request_id']=e['requestId']
        cdp.on('Network.responseReceived',seen)
        response=page.goto(url,wait_until='domcontentloaded',timeout=180000);assert response.ok
        page.wait_for_function('window.SAFEDUO_READY===true',timeout=180000)
        b=cdp.send('Network.getResponseBody',{'requestId':served['request_id']});body=base64.b64decode(b['body']) if b['base64Encoded'] else b['body'].encode();assert sha(body)==full['page_sha256']==after['docs/'+HERE.name+'/index.html']
        actual=page.evaluate('window.SAFEDUO');result=load('holdout_results.json')
        assert actual['result']['cases']==result['cases'] and page.locator('#heatmap button').count()==192
        i=next((i for i,c in enumerate(result['cases']) if not c['methods']['joint_reference']['failed'] and c['methods']['motion_admission']['failed']),0)
        page.select_option('#case',str(i));page.select_option('#method','motion_admission')
        assert page.evaluate('current')==result['cases'][i]
        page.select_option('#cameraMode','motion_admission');count=page.locator('#cameraGroup option').count();page.select_option('#cameraGroup',str(count-1))
        page.locator('#gallery').scroll_into_view_if_needed();page.locator('#gallery img').evaluate_all('images=>images.forEach(im=>im.loading="eager")')
        page.wait_for_function('[...document.querySelectorAll("#gallery img")].every(im=>im.complete&&im.naturalWidth===1280&&im.naturalHeight===720)',timeout=180000)
        content=get(BASE+cert+'?fifo_commit='+final['commit']);assert sha(content)==after[cert]==sha((HERE/'VERIFICATION.json').read_bytes())
        verification=json.loads(content);assert verification['status']=='PASS' and verification['physical_safety_status']=='UNVALIDATED' and not verification['hardware_approved'] and not verification['production_promoted']
        page.evaluate('scrollTo(0,0)');page.screenshot(path=str(HERE/'panel_final.png'))
        page.set_viewport_size({'width':390,'height':844});assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
        assert not errors;browser.close()
    receipt=dict(status='PASS_FINAL_CERTIFICATE_DELIVERY',commit=final['commit'],pages_commit=final['pages_commit'],url=url,
       owned_changed_files=[cert],all_git_changed_files=git_changed,foreign_root_inheritance=before['index.html']!=after['index.html'],own_root_cards_unchanged=True,
       all_owned_published_bytes_rehashed=final['files'],asset_fixed_commit=load('ASSET_PUBLICATION.json')['commit'],
       reused_full_ui_receipt_sha256=sha((HERE/'live_browser_initial.json').read_bytes()),final_remote_receipt_sha256=sha((HERE/'remote_verification.json').read_bytes()),
       fresh_smoke='actual final native document exactSHA; all192cases data exact; one new-failure candidate selection; final9motionimages; actualpublicPASScertificate;390px/nooverflow/noJSerrors',
       reuse_scope='full768method-selection,5376curve-data,40combinedfilters,allgalleries links and nativeinitial9images initial suite; same owned HTML/payload/assets commit/source/results on final certificate deployment; no pretend second full suite',
       scientific_snapshot_sha256=sha(scientific),hardware_approved=False,production_promoted=False,source_sha256=sha(Path(__file__).read_bytes()),utc=datetime.now(timezone.utc).isoformat())
    with (HERE/'FINAL_DELIVERY_CLOSURE.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    with (HERE/'live_browser.json').open('x') as f:json.dump({**full,'final_commit':final['commit'],'explicit_evidence_reuse':receipt['reuse_scope'],'fresh_final_smoke_receipt':'FINAL_DELIVERY_CLOSURE.json'},f,indent=2);f.write('\n')
    print(receipt['status'],final['commit'],flush=True)
if __name__=='__main__':main()
