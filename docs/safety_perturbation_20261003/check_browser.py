"""Local frozen-preview and live browser checks for perturbation evidence."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parent
DATA=Path('/home/liyufeng/safeduo-dashboard-data')
parser=argparse.ArgumentParser();parser.add_argument('--live-url');args=parser.parse_args()
expected=json.loads((OUT/'preview_scientific.json').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1280,'height':900},ignore_https_errors=True)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    def local(route):
        name=Path(urlparse(route.request.url).path).name
        file=OUT/'preview_scientific.json' if name=='scientific_eval.json' else DATA/name
        if file.exists(): route.fulfill(status=200,content_type='application/json',body=file.read_text())
        else: route.continue_()
    if not args.live_url: page.route('**/data/*.json*',local)
    page.goto(args.live_url or 'http://127.0.0.1:8893/#scientific',wait_until='domcontentloaded')
    page.wait_for_selector('#scientific-perturbation-table tbody tr')
    actual=page.evaluate('SCIENTIFIC.safety_perturbation')
    assert actual['rows']==expected['safety_perturbation']['rows']
    assert page.locator('#scientific-perturbation-table tbody tr').count()==len(actual['rows'])
    assert page.locator('#scientific-headline .card').count()==3
    assert page.locator('#scientific-robustness-table tbody tr').count()==8
    assert '上一轮 512' in page.locator('#scientific-robustness h2').inner_text()
    assert '尚未解决' in page.locator('#scientific-perturbation-trace svg title').text_content()
    for section,key in [('scientific-perturbation','safety_perturbation'),('scientific-robustness','safety_robustness')]:
        for i,trace in enumerate(expected[key]['traces']):
            page.select_option(f'#{section}-select',str(i))
            assert page.locator(f'#{section}-trace polyline').count()==len(trace['series'])
            html=page.locator(f'#{section}-trace').inner_html()
            assert 'NaN' not in html and 'Infinity' not in html
            assert page.locator(f'#{section}-caption').inner_text()==trace['caption']
    page.select_option('#scientific-perturbation-select',str(actual['default_trace']))
    prefix='live_' if args.live_url else ''
    page.locator('header.top').evaluate("n=>n.style.visibility='hidden'")
    page.locator('#scientific-perturbation').screenshot(path=str(OUT/f'{prefix}panel_results.png'))
    page.locator('header.top').evaluate("n=>n.style.visibility=''")
    page.set_viewport_size({'width':390,'height':844})
    assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth+1')
    page.screenshot(path=str(OUT/f'{prefix}panel_mobile.png'),full_page=True)
    assert not errors,errors
    print(json.dumps(dict(status='pass',browser=browser.version,live_url=args.live_url,
                         rows=len(actual['rows']),history_preserved=True,mobile_no_page_overflow=True,
                         chart_selectors=True,finite_svg=True,page_errors=errors)))
    browser.close()
