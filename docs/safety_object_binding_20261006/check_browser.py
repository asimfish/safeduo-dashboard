"""Actual browser evidence: every case, metric, gallery and public download."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    receipts=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        for width in (1440,390):
            page=browser.new_page(viewport={'width':width,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            response=page.goto(a.url,wait_until='networkidle');assert response.ok
            page.wait_for_function('window.safeduoReady === true');d=page.evaluate('D')
            assert len(d['result']['cases'])==8 and page.locator('#cases tr').count()==8
            assert not page.locator('#error').inner_text();assert not errors
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            choices=0
            for e in range(8):
                page.select_option('#caseSelect',str(e))
                for obj in ('0','1'):
                    page.select_option('#objectSelect',obj)
                    for metric in ('height','contact','xy','tilt'):
                        page.select_option('#metricSelect',metric);assert f'槽位 {e}' in page.locator('#caseDetail').inner_text();choices+=1
            page.locator('#reviewVideo').scroll_into_view_if_needed()
            page.wait_for_function("document.querySelector('#reviewVideo').readyState>=2")
            video=page.locator('#reviewVideo');assert video.evaluate('(v)=>v.videoWidth')==1920
            assert video.evaluate('(v)=>v.play().then(()=>true)')
            page.wait_for_function("document.querySelector('#reviewVideo').currentTime>0.05")
            video.evaluate('(v)=>v.pause()')
            assert abs(video.evaluate('(v)=>v.duration')-88)<.1
            for e in range(8):
                page.select_option('#reviewCaseSelect',str(e))
                target=next(f['video_start_s'] for f in d['keyframe_video']['frames'] if f['env']==e)
                page.wait_for_function('(t)=>{const v=document.querySelector("#reviewVideo");return !v.seeking&&Math.abs(v.currentTime-t)<.1&&v.readyState>=2}',arg=target)
            groups=0
            for e in range(8):
                page.select_option('#imageCase',str(e))
                for step in page.locator('#imageStep option').evaluate_all('(es)=>es.map(e=>e.value)'):
                    page.select_option('#imageStep',step);page.wait_for_function("[...document.querySelectorAll('#gallery img')].length===3 && [...document.querySelectorAll('#gallery img')].every(i=>i.complete&&i.naturalWidth===1280)")
                    assert f'任务 {e}' in page.locator('#imageMeta').inner_text();groups+=1
            for file in ('native_state_timeseries.csv','development_contact.pdf','keyframe_review.mp4','keyframe_review.webm'):
                with page.expect_download() as download:page.locator(f'a[href="{file}"][download]').click()
                saved=download.value;assert saved.failure() is None;saved.save_as(a.out/(str(width)+'_'+file))
            assert not errors;page.select_option('#imageCase','7');page.select_option('#caseSelect','7');page.select_option('#metricSelect','contact')
            page.select_option('#imageStep','1571');page.select_option('#metricSelect','tilt')
            page.screenshot(path=str(a.out/f'panel_{width}.png'),full_page=True)
            root=page.goto(a.url.split('/docs/')[0]+'/index.html#scientific',wait_until='networkidle');assert root.ok
            card=page.locator('#objectBindingEvidence');assert card.is_visible();assert '8' in card.inner_text()
            assert card.locator('a').get_attribute('href')=='docs/safety_object_binding_20261006/'
            receipts.append(dict(width=width,case_object_metric_choices=choices,gallery_groups=groups,discrete_video_cases_seeked=8,video_playback_observed=True,video_duration_s=88,errors=errors,main_card_visible=True))
            page.close()
        browser.close()
    (a.out/'receipt.json').write_text(json.dumps(dict(status='PASS_ACTUAL_BROWSER_CASES_GALLERIES_DOWNLOADS_ROOT',url=a.url,viewports=receipts),indent=2)+'\n')
    print(json.dumps(receipts))
if __name__=='__main__':main()
