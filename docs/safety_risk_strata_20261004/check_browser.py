"""Verify new risk UI and all legacy coverage/traces in a real browser."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
DATA=Path('/home/liyufeng/safeduo-dashboard-data')
parser=argparse.ArgumentParser();parser.add_argument('--live-url');parser.add_argument('--local-url');args=parser.parse_args()
expected=json.loads((HERE/'preview_scientific.json').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport=dict(width=1280,height=900),ignore_https_errors=True)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    if not args.live_url:
        def local(route):
            name=Path(urlparse(route.request.url).path).name
            file=HERE/'preview_scientific.json' if name=='scientific_eval.json' else DATA/name
            if file.exists():route.fulfill(status=200,content_type='application/json',body=file.read_text())
            else:route.continue_()
        page.route('**/data/*.json*',local)
    page.goto(args.live_url or args.local_url,wait_until='domcontentloaded')
    page.wait_for_selector('#scientific-risk-table tbody tr')
    actual=page.evaluate('SCIENTIFIC');assert actual==expected
    risk=actual['safety_risk_strata'];old=actual['safety_random_space']
    assert page.locator('#scientific-risk-table tbody tr').count()==len(risk['aggregate_rows'])
    assert page.locator('#scientific-risk-pair-results tbody tr').count()==len(risk['pair_rows'])
    assert page.locator('#scientific-risk-zero tbody tr').count()==3
    assert page.locator('#scientific-risk-inputs tbody tr').count()==len(risk['paired_inputs'])
    assert '不删除' in page.locator('#scientific-risk-protocol').inner_text()
    assert page.locator('#scientific-risk-bank').inner_text()==risk['bank_note']
    assert page.locator('#scientific-random-table tbody tr').count()==11
    coverage_count=0
    for section,data in [('scientific-risk',risk),('scientific-random',old)]:
        for i,c in enumerate(data['coverage_cases']):
            coverage_count+=1;page.select_option('#'+section+'-coverage-select',str(i))
            assert page.locator('#'+section+'-heatmap rect').count()==260
            assert page.locator('#'+section+'-ee tbody tr').count()==4
            assert page.locator('#'+section+'-pairs tbody tr').count()==6
            assert page.locator('#'+section+'-coverage-note').inner_text()==c['note']
            html=page.locator('#'+section+'-heatmap').inner_html();assert 'NaN' not in html and 'Infinity' not in html
    for selector,n in [('scientific-bounded-table',4),('scientific-bounded-stops',18),('scientific-queue-controls',9),('scientific-queue-stops',12),
                        ('scientific-forensics-table',4),('scientific-forensics-holdout',4),('scientific-perturbation-table',30),('scientific-robustness-table',8)]:
        assert page.locator('#'+selector+' tbody tr').count()==n,selector
    traces=0
    for section,key in [('scientific-risk','safety_risk_strata'),('scientific-random','safety_random_space'),('scientific-bounded','safety_bounded_stop'),
                        ('scientific-queue','safety_queue_braking'),('scientific-forensics','safety_forensics'),('scientific-perturbation','safety_perturbation'),('scientific-robustness','safety_robustness')]:
        for i,t in enumerate(actual[key]['traces']):
            traces+=1;page.select_option('#'+section+'-select',str(i))
            assert page.locator('#'+section+'-trace polyline').count()==len(t['series'])
            assert page.locator('#'+section+'-caption').inner_text()==t['caption']
            html=page.locator('#'+section+'-trace').inner_html();assert 'NaN' not in html and 'Infinity' not in html
    page.select_option('#scientific-risk-coverage-select',str(risk['default_coverage']))
    if risk['traces']:page.select_option('#scientific-risk-select',str(risk['default_trace']))
    page.locator('header.top').evaluate("n=>n.style.visibility='hidden'")
    page.locator('#scientific-risk').screenshot(path=str(HERE/('live_panel.png' if args.live_url else 'panel.png')))
    page.set_viewport_size(dict(width=390,height=844))
    assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth+1')
    assert not errors,errors
    print(json.dumps(dict(status='pass',browser=browser.version,live_url=args.live_url,coverage_cases_checked=coverage_count,traces_checked=traces,
                          legacy_preserved=True,mobile_no_page_overflow=True,page_errors=errors)))
    browser.close()
