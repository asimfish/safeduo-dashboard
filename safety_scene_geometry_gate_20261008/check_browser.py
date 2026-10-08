import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import unquote,urlparse
from playwright.sync_api import sync_playwright
p=Path(__file__).resolve().parent;repo=Path('/home/liyufeng/safeduo-dashboard-benchmark-protocol-20261006');archive=Path('/home/liyufeng/safeduo-dashboard-object-media-20261006/safety_scene_geometry_gate_20261008')
pa=argparse.ArgumentParser();pa.add_argument('--live',action='store_true');a=pa.parse_args();out=p/('browser_public' if a.live else 'browser_local');out.mkdir(exist_ok=False);results=[]
base='https://asimfish.github.io/safeduo-dashboard/' if a.live else 'http://127.0.0.1:8973/';url=base+'docs/safety_scene_geometry_gate_20261008/'
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
    for width in [1440,390]:
        ctx=browser.new_context(viewport={'width':width,'height':1000});page=ctx.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        if not a.live:
            def route(rt):
                rel=unquote(urlparse(rt.request.url).path).split('/safety_scene_geometry_gate_20261008/',1)[-1];f=archive/rel
                if f.exists():rt.fulfill(path=str(f))
                else:rt.abort()
            page.route('https://raw.githubusercontent.com/**/safety_scene_geometry_gate_20261008/**',route)
        page.goto(url,wait_until='domcontentloaded',timeout=90000);page.wait_for_function("document.documentElement.dataset.loaded==='true'",timeout=120000)
        assert page.locator('#rows tbody tr').count()==576
        assert page.locator('#overview tbody tr').count()==3
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        for ref in page.locator('#curveRef option').all():
            page.select_option('#curveRef',ref.get_attribute('value'))
            for arm in ['U_L','U_R']:
                page.select_option('#curveArm',arm);assert '720帧' in page.locator('#curveInfo').inner_text()
        image_checks=[]
        for run in ['fresh_scene_batch','matched_scene']:
            page.select_option('#run',run)
            assert page.locator('#rows tbody tr').count()==(576 if run=='fresh_scene_batch' else 192)
            for env in page.locator('#env option').all():
                e=env.get_attribute('value');page.select_option('#env',e);page.select_option('#profile','11')
                for ph in ['359','659']:
                    page.select_option('#phase',ph);page.locator('#photos').scroll_into_view_if_needed()
                    for img in page.locator('#photos img').all():img.scroll_into_view_if_needed()
                    page.wait_for_function("[...document.querySelectorAll('#photos img')].length===4&&[...document.querySelectorAll('#photos img')].every(i=>i.complete&&i.naturalWidth===1280&&i.naturalHeight===720)",timeout=120000)
                    image_checks.append(dict(run=run,env=int(e),phase=int(ph),images=4))
        assert page.locator('#pairWorkspace tbody tr').count()==6
        assert page.locator('video').count()==6
        video_checks=[]
        for i,video in enumerate(page.locator('video').all()):
            video.scroll_into_view_if_needed();video.evaluate('(v)=>{v.muted=true;v.currentTime=0;return v.play()}');page.wait_for_function('(i)=>{let v=document.querySelectorAll("video")[i];return v.currentTime>.2&&v.videoWidth===1280&&v.videoHeight===720&&v.getVideoPlaybackQuality().totalVideoFrames>0}',arg=i,timeout=120000)
            video_checks.append(video.evaluate('(v)=>({src:v.currentSrc,width:v.videoWidth,height:v.videoHeight,time:v.currentTime,frames:v.getVideoPlaybackQuality().totalVideoFrames,error:v.error&&v.error.message})'));video.evaluate('(v)=>v.pause()')
        page.select_option('#run','fresh_scene_batch');page.select_option('#filter','failed');expected=page.evaluate('window.evidence.data.runs.fresh_scene_batch.rows.filter(r=>r.admitted&&!r.qualified).length');assert page.locator('#rows tbody tr').count()==expected
        page.select_option('#filter','all');page.evaluate('scrollTo(0,0)');page.screenshot(path=str(out/f'{width}_summary.png'),full_page=False)
        assert not errors,errors
        # New card plus old hand evidence card remain reachable on the main dashboard.
        page.goto(base,wait_until='domcontentloaded',timeout=90000);assert page.locator('#sceneGeometryEvidence').count()==1;assert page.locator('#handClosureEvidence').count()==1
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        results.append(dict(width=width,path_rows_fresh=576,path_rows_matched=192,curve_pairs_checked=8,image_checks=image_checks,images_checked=sum(x['images'] for x in image_checks),video_checks=video_checks,no_js_errors=True,no_horizontal_overflow=True,old_hand_card_preserved=True));ctx.close()
    browser.close()
(out/'receipt.json').write_text(json.dumps(dict(status='PASS_LIVE_BROWSER' if a.live else 'PASS_LOCAL_BROWSER',checked_utc=datetime.now(timezone.utc).isoformat(),url=url,checks=results),indent=2)+'\n');print(json.dumps({k:v for k,v in results[0].items() if k not in ['image_checks','video_checks']},indent=2))
