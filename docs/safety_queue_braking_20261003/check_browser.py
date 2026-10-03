"""Verify current delay/hold evidence and preserved historical data/charts."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
DATA = Path('/home/liyufeng/safeduo-dashboard-data')
parser = argparse.ArgumentParser()
parser.add_argument('--live-url')
args = parser.parse_args()
expected = json.loads((OUT / 'preview_scientific.json').read_text())
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 900}, ignore_https_errors=True)
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    def local(route):
        name = Path(urlparse(route.request.url).path).name
        file = OUT / 'preview_scientific.json' if name == 'scientific_eval.json' else DATA / name
        if file.exists():
            route.fulfill(status=200, content_type='application/json', body=file.read_text())
        else:
            route.continue_()
    if not args.live_url:
        page.route('**/data/*.json*', local)
    page.goto(args.live_url or 'http://127.0.0.1:8893/#scientific', wait_until='domcontentloaded')
    page.wait_for_selector('#scientific-queue-controls tbody tr')
    actual = page.evaluate('SCIENTIFIC.safety_queue_braking')
    assert actual == expected['safety_queue_braking']
    assert actual['candidate_status'] == 'experimental_not_promoted'
    for key in ['safety_forensics', 'safety_perturbation', 'safety_robustness', 'headline', 'historical_headline']:
        assert page.evaluate(f'SCIENTIFIC.{key}') == expected[key], key
    assert page.locator('#scientific-queue-controls tbody tr').count() == 9
    count = sum(len(x['trajectories']) for x in actual['stop_diagnostics'])
    assert page.locator('#scientific-queue-stops tbody tr').count() == count
    assert '事后' in page.locator('#scientific-queue-diagnostic-note').inner_text()
    assert '不记作策略改进' in page.locator('#scientific-queue-verdict').inner_text()
    assert all(actual['cold_100_replay_exact_fields'].values())
    assert page.locator('#scientific-headline .card').count() == 4
    assert page.locator('#scientific-forensics-table tbody tr').count() == 4
    assert page.locator('#scientific-forensics-holdout tbody tr').count() == 4
    assert page.locator('#scientific-perturbation-table tbody tr').count() == 30
    assert page.locator('#scientific-robustness-table tbody tr').count() == 8
    for section, key in [('scientific-queue', 'safety_queue_braking'),
                         ('scientific-forensics', 'safety_forensics'),
                         ('scientific-perturbation', 'safety_perturbation'),
                         ('scientific-robustness', 'safety_robustness')]:
        for i, trace in enumerate(expected[key]['traces']):
            page.select_option(f'#{section}-select', str(i))
            assert page.locator(f'#{section}-trace polyline').count() == len(trace['series'])
            html = page.locator(f'#{section}-trace').inner_html()
            assert 'NaN' not in html and 'Infinity' not in html
            assert page.locator(f'#{section}-caption').inner_text() == trace['caption']
    page.select_option('#scientific-queue-select', str(actual['default_trace']))
    page.locator('header.top').evaluate("n=>n.style.visibility='hidden'")
    page.locator('#scientific-queue').screenshot(path=str(OUT / ('live_panel.png' if args.live_url else 'panel.png')))
    page.locator('header.top').evaluate("n=>n.style.visibility=''")
    page.set_viewport_size({'width': 390, 'height': 844})
    assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth+1')
    assert not errors, errors
    print(json.dumps(dict(status='pass', browser=browser.version, live_url=args.live_url,
                          delay_rows=9, diagnostic_rows=count, historic_tables_preserved=True,
                          all_charts_finite=True, mobile_no_page_overflow=True, page_errors=errors)))
    browser.close()
