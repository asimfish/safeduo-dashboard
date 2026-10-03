"""Check the live research journey and every preserved scientific section."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parent
DATA=Path('/home/liyufeng/safeduo-dashboard-data')
parser=argparse.ArgumentParser()
parser.add_argument('--live-url')
args=parser.parse_args()
expected=json.loads((OUT/'preview_scientific.json').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1280,'height':900},ignore_https_errors=True)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    if not args.live_url:
        def local(route):
            name=Path(urlparse(route.request.url).path).name
            file=OUT/'preview_scientific.json' if name=='scientific_eval.json' else DATA/name
            if file.exists():route.fulfill(status=200,content_type='application/json',body=file.read_text())
            else:route.continue_()
        page.route('**/data/*.json*',local)
    page.goto(args.live_url or 'http://127.0.0.1:8893/#scientific',wait_until='domcontentloaded')
    page.wait_for_selector('#scientific-bounded-table tbody tr')
    actual=page.evaluate('SCIENTIFIC')
    assert actual==expected,'rendered data differs from prepared evidence'
    r=actual['safety_bounded_stop']
    assert r['candidate_status']=='experimental_not_promoted'
    assert page.locator('#scientific-bounded-table tbody tr').count()==4
    assert page.locator('#scientific-bounded-stops tbody tr').count()==18
    assert '事后' in page.locator('#scientific-bounded-note').inner_text()
    assert '没有前瞻检测器' in page.locator('#scientific-bounded-protocol').inner_text()
    assert '不构成System 0前瞻安全性能' in page.locator('#scientific-bounded-verdict').inner_text()
    assert page.locator('#scientific-bounded-table tbody tr').nth(1).inner_text().find('31')>=0
    for selector,n in [('scientific-queue-controls',9),('scientific-queue-stops',12),
                        ('scientific-forensics-table',4),('scientific-forensics-holdout',4),
                        ('scientific-perturbation-table',30),('scientific-robustness-table',8)]:
        assert page.locator(f'#{selector} tbody tr').count()==n,selector
    for section,key in [('scientific-bounded','safety_bounded_stop'),('scientific-queue','safety_queue_braking'),
                        ('scientific-forensics','safety_forensics'),('scientific-perturbation','safety_perturbation'),
                        ('scientific-robustness','safety_robustness')]:
        for i,trace in enumerate(expected[key]['traces']):
            page.select_option(f'#{section}-select',str(i))
            assert page.locator(f'#{section}-trace polyline').count()==len(trace['series'])
            html=page.locator(f'#{section}-trace').inner_html()
            assert 'NaN' not in html and 'Infinity' not in html
            assert page.locator(f'#{section}-caption').inner_text()==trace['caption']
    page.select_option('#scientific-bounded-select',str(r['default_trace']))
    page.locator('header.top').evaluate("n=>n.style.visibility='hidden'")
    page.locator('#scientific-bounded').screenshot(path=str(OUT/('live_panel.png' if args.live_url else 'panel.png')))
    page.locator('header.top').evaluate("n=>n.style.visibility=''")
    page.set_viewport_size({'width':390,'height':844})
    assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth+1')
    assert not errors,errors
    print(json.dumps(dict(status='pass',browser=browser.version,live_url=args.live_url,
                          rows=4,stops=18,traces=len(r['traces']),historic_data_preserved=True,
                          mobile_no_page_overflow=True,page_errors=errors)))
    browser.close()
