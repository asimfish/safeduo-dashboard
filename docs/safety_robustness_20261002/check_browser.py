"""Real-browser panel smoke test with frozen local data responses."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--live-url')
args=parser.parse_args()
DATA=Path('/home/liyufeng/safeduo-dashboard-data')
preview=OUT/'preview_scientific.json'
expected=json.loads(preview.read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    context=browser.new_context(viewport={'width':1280,'height':900},ignore_https_errors=True)
    errors=[]
    page=context.new_page()
    page.on('pageerror',lambda e: errors.append(str(e)))
    def local_data(route):
        name=Path(urlparse(route.request.url).path).name
        file=preview if name=='scientific_eval.json' else DATA/name
        if file.exists():
            route.fulfill(status=200,content_type='application/json',body=file.read_text())
        else:
            route.continue_()
    if not args.live_url:
        page.route('**/data/*.json*',local_data)
    page.goto(args.live_url or 'http://127.0.0.1:8892/#scientific',wait_until='domcontentloaded')
    page.wait_for_selector('#scientific-robustness-table tbody tr')
    actual=page.evaluate('SCIENTIFIC.safety_robustness')
    assert actual['candidate']==expected['safety_robustness']['candidate']
    assert actual['protected_total']==expected['safety_robustness']['protected_total']
    assert page.locator('#scientific-robustness-table tbody tr').count()==len(expected['safety_robustness']['rows'])
    assert page.locator('#scientific-robustness-regression tbody tr').count()==4
    assert page.locator('#scientific-headline .card').count()==3
    assert 'F_L' in page.locator('#scientific-robustness-trace svg title').text_content()
    for index,trace in enumerate(expected['safety_robustness']['traces']):
        page.select_option('#scientific-robustness-select',str(index))
        assert page.locator('#scientific-robustness-trace polyline').count()==len(trace['series'])
        assert 'NaN' not in page.locator('#scientific-robustness-trace').inner_html()
        assert 'Infinity' not in page.locator('#scientific-robustness-trace').inner_html()
        assert trace['caption']==page.locator('#scientific-robustness-caption').inner_text()
    page.select_option('#scientific-robustness-select','2')
    prefix='live_' if args.live_url else ''
    page.screenshot(path=str(OUT/f'{prefix}panel_desktop.png'),full_page=True)
    # Hide the sticky navigation only while exporting the whole results section.
    page.locator('header.top').evaluate("node => node.style.visibility = 'hidden'")
    page.locator('#scientific-robustness').screenshot(path=str(OUT/f'{prefix}panel_results.png'))
    page.locator('header.top').evaluate("node => node.style.visibility = ''")
    page.set_viewport_size({'width':390,'height':844})
    assert page.locator('#scientific-robustness').is_visible()
    overflow=page.evaluate('document.documentElement.scrollWidth>window.innerWidth+1')
    assert not overflow, 'page overflow at 390px'
    page.screenshot(path=str(OUT/f'{prefix}panel_mobile.png'),full_page=True)
    assert not errors, errors
    print(json.dumps({'status':'pass','browser':browser.version,'desktop':[1280,900],
        'mobile':[390,844],'tables':True,'selectors':True,'finite_svg':True,
        'candidate':actual['candidate'],'protected_total':actual['protected_total'],
        'live_url':args.live_url,'page_errors':errors}))
    browser.close()
