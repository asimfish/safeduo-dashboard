import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
receipts=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);context=browser.new_context()
    for width in (1440,390):
        page=context.new_page();page.set_viewport_size({'width':width,'height':1050});page.set_default_timeout(45000)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        response=page.goto(a.url,wait_until='networkidle');assert response.ok
        page.wait_for_function('window.safeduoReady===true');data=page.evaluate('D');assert len(data['results']['cases'])==24
        assert page.locator('#cases tr').count()==24 and page.locator('#totals tr').count()==3
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        choices=0;galleries=0
        for c in data['results']['cases']:
            key=str(c['block'])+'_'+str(c['env']);page.select_option('#caseSelect',key)
            for o in ('beam700','beam300'):
                page.select_option('#objectSelect',o)
                for m in ('lift_m','tilt_deg','hand_n','table_n'):
                    page.select_option('#metricSelect',m);assert '槽 '+str(c['env']) in page.locator('#caseDetail').inner_text();assert page.locator('#plot polyline').count()==2;choices+=1
            for step in page.locator('#imageStep option').evaluate_all('(xs)=>xs.map(x=>x.value)'):
                page.select_option('#imageStep',step)
                page.wait_for_function("[...document.querySelectorAll('#gallery img')].length===3&&[...document.querySelectorAll('#gallery img')].every(i=>i.complete&&i.naturalWidth===1280)")
                assert '槽 '+str(c['env']) in page.locator('#imageMeta').inner_text();galleries+=1
        assert choices==192 and galleries==120
        for i in range(3):
            v=page.locator('video').nth(i);v.scroll_into_view_if_needed()
            v.evaluate('(v)=>v.load()');page.wait_for_function('(i)=>document.querySelectorAll("video")[i].readyState>=2',arg=i)
            assert v.evaluate('(v)=>v.videoWidth')==1280 and abs(v.evaluate('(v)=>v.duration')-6.1)<.1
            assert v.evaluate('(v)=>v.play().then(()=>true)');page.wait_for_function('(i)=>document.querySelectorAll("video")[i].currentTime>.08',arg=i);v.evaluate('(v)=>v.pause()')
            v.evaluate('(v)=>{v.currentTime=3.8}');page.wait_for_function('(i)=>{const v=document.querySelectorAll("video")[i];return !v.seeking&&Math.abs(v.currentTime-3.8)<.1&&v.readyState>=2}',arg=i)
        for file in ('RESULTS.json','REGISTRATION.json'):
            with page.expect_download() as item:page.locator('a[href="'+file+'"][download]').click()
            assert item.value.failure() is None;item.value.save_as(a.out/(str(width)+'_'+file))
        assert errors==[] and not page.locator('#error').inner_text()
        page.screenshot(path=str(a.out/f'panel_{width}.png'),full_page=True)
        root=page.goto(a.url.split('/docs/')[0]+'/index.html#scientific',wait_until='networkidle');assert root.ok
        card=page.locator('#releaseFeedbackEvidence');assert card.is_visible() and '24' in card.inner_text()
        assert card.locator('a').get_attribute('href')=='docs/safety_release_feedback_20261006/'
        assert page.locator('#trackingReserveEvidence').is_visible() and page.locator('#objectBindingEvidence').is_visible()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');assert errors==[]
        receipts.append(dict(width=width,case_object_metric_choices=choices,all_case_gallery_groups=galleries,continuous_video_play_and_seek=3,downloaded_json=2,errors=errors,main_and_previous_cards_visible=True));page.close()
    browser.close()
(a.out/'receipt.json').write_text(json.dumps(dict(status='PASS_ACTUAL_BROWSER_ALL_CASES_GALLERIES_NATIVE_VIDEOS_AND_MAIN',url=a.url,viewports=receipts),indent=2)+'\n');print(json.dumps(receipts))
