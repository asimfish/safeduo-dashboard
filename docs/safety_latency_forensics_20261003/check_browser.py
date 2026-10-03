"""Render complete current evidence, historical tables, all charts and mobile layout."""
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
        if file.exists():route.fulfill(status=200,content_type='application/json',body=file.read_text())
        else:route.continue_()
    if not args.live_url:page.route('**/data/*.json*',local)
    page.goto(args.live_url or 'http://127.0.0.1:8893/#scientific',wait_until='domcontentloaded')
    page.wait_for_selector('#scientific-forensics-table tbody tr')
    actual=page.evaluate('SCIENTIFIC.safety_forensics')
    assert actual['rows']==expected['safety_forensics']['rows']
    assert actual['holdout_rows']==expected['safety_forensics']['holdout_rows']
    assert actual['candidate_status']=='experimental_not_promoted'
    assert actual['candidate_total']['violations']==7 and actual['candidate_total']['episodes']==192
    assert page.locator('#scientific-headline .card').count()==4
    assert page.locator('#scientific-forensics-table tbody tr').count()==4
    assert page.locator('#scientific-forensics-holdout tbody tr').count()==4
    assert page.locator('#scientific-perturbation-table tbody tr').count()==30
    assert page.locator('#scientific-robustness-table tbody tr').count()==8
    assert '冻结参考512' in page.locator('#scientific-robustness h2').inner_text()
    assert '配置勘误' in page.locator('#scientific-perturbation-note').inner_text()
    assert '仍未解决' in page.locator('#scientific-forensics-trace svg title').text_content()
    for section,key in [('scientific-forensics','safety_forensics'),
                        ('scientific-perturbation','safety_perturbation'),
                        ('scientific-robustness','safety_robustness')]:
        for i,trace in enumerate(expected[key]['traces']):
            page.select_option(f'#{section}-select',str(i))
            assert page.locator(f'#{section}-trace polyline').count()==len(trace['series'])
            html=page.locator(f'#{section}-trace').inner_html()
            assert 'NaN' not in html and 'Infinity' not in html
            assert page.locator(f'#{section}-caption').inner_text()==trace['caption']
    page.select_option('#scientific-forensics-select',str(actual['default_trace']))
    prefix='live_' if args.live_url else ''
    page.locator('header.top').evaluate("n=>n.style.visibility='hidden'")
    page.locator('#scientific-forensics').screenshot(path=str(OUT/f'{prefix}panel.png'))
    page.locator('header.top').evaluate("n=>n.style.visibility=''")
    page.set_viewport_size({'width':390,'height':844})
    assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth+1')
    assert not errors,errors
    print(json.dumps(dict(status='pass',browser=browser.version,live_url=args.live_url,
        factor_rows=4,holdout_rows=4,previous_rows=30,frozen_reference_rows=8,
        mobile_no_page_overflow=True,charts_finite=True,page_errors=errors)))
    browser.close()
