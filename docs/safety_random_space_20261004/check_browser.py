"""Verify random coverage, matched traces, and preservation of previous evidence."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parent
DATA=Path('/home/liyufeng/safeduo-dashboard-data')
parser=argparse.ArgumentParser();parser.add_argument('--live-url');parser.add_argument('--local-url',default='http://127.0.0.1:8893/#scientific');args=parser.parse_args()
expected=json.loads((OUT/'preview_scientific.json').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1280,'height':900},ignore_https_errors=True)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    if not args.live_url:
        def local(route):
            name=Path(urlparse(route.request.url).path).name
            file=OUT/'preview_scientific.json' if name=='scientific_eval.json' else DATA/name
            if file.exists():route.fulfill(status=200,content_type='application/json',body=file.read_text())
            else:route.continue_()
        page.route('**/data/*.json*',local)
    page.goto(args.live_url or args.local_url,wait_until='domcontentloaded')
    page.wait_for_selector('#scientific-random-table tbody tr')
    actual=page.evaluate('SCIENTIFIC');assert actual==expected
    r=actual['safety_random_space']
    assert page.locator('#scientific-random-table tbody tr').count()==11
    assert '576个窗口无效' in page.locator('#scientific-random-verdict').inner_text()
    assert '后续策略成败' in page.locator('#scientific-random-bank').inner_text()
    for i,c in enumerate(r['coverage_cases']):
        page.select_option('#scientific-random-coverage-select',str(i))
        assert page.locator('#scientific-random-heatmap rect').count()==260
        assert page.locator('#scientific-random-pairs tbody tr').count()==6
        assert page.locator('#scientific-random-ee tbody tr').count()==4
        assert page.locator('#scientific-random-coverage-note').inner_text()==c['note']
        html=page.locator('#scientific-random-heatmap').inner_html()
        assert 'NaN' not in html and 'Infinity' not in html
    for selector,n in [('scientific-bounded-table',4),('scientific-bounded-stops',18),
                        ('scientific-queue-controls',9),('scientific-queue-stops',12),
                        ('scientific-forensics-table',4),('scientific-forensics-holdout',4),
                        ('scientific-perturbation-table',30),('scientific-robustness-table',8)]:
        assert page.locator(f'#{selector} tbody tr').count()==n,selector
    trace_count=0
    for section,key in [('scientific-random','safety_random_space'),('scientific-bounded','safety_bounded_stop'),
                        ('scientific-queue','safety_queue_braking'),('scientific-forensics','safety_forensics'),
                        ('scientific-perturbation','safety_perturbation'),('scientific-robustness','safety_robustness')]:
        for i,trace in enumerate(expected[key]['traces']):
            page.select_option(f'#{section}-select',str(i));trace_count+=1
            assert page.locator(f'#{section}-trace polyline').count()==len(trace['series'])
            html=page.locator(f'#{section}-trace').inner_html();assert 'NaN' not in html and 'Infinity' not in html
            assert page.locator(f'#{section}-caption').inner_text()==trace['caption']
    page.select_option('#scientific-random-coverage-select',str(r['default_coverage']))
    if r['traces']:page.select_option('#scientific-random-select',str(r['default_trace']))
    page.locator('header.top').evaluate("n=>n.style.visibility='hidden'")
    page.locator('#scientific-random').screenshot(path=str(OUT/('live_panel.png' if args.live_url else 'panel.png')))
    page.set_viewport_size({'width':390,'height':844})
    assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth+1')
    assert not errors,errors
    print(json.dumps({'status':'pass','browser':browser.version,'live_url':args.live_url,
                      'aggregate_rows':11,'coverage_cases':len(r['coverage_cases']),'traces_checked':trace_count,
                      'legacy_preserved':True,'mobile_no_page_overflow':True,'page_errors':errors}))
    browser.close()
