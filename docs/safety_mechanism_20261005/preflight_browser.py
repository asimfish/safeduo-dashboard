"""Preflight only: check current candidate UI while the fixed experiment runs."""
import json
from pathlib import Path
import socket
import subprocess
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
DASH=Path('/home/liyufeng/safeduo-dashboard')
DATA=Path('/home/liyufeng/safeduo-dashboard-data')
expected=json.loads((HERE/'preview_scientific.json').read_text())
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
server=subprocess.Popen(['/usr/bin/python3','-m','http.server',str(port),'--bind','127.0.0.1'],cwd=DASH,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport=dict(width=1280,height=900))
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        def route(r):
            name=Path(urlparse(r.request.url).path).name
            file=HERE/'preview_scientific.json' if name=='scientific_eval.json' else DATA/name
            if file.exists():r.fulfill(status=200,content_type='application/json',body=file.read_text())
            else:r.continue_()
        page.route('**/data/*.json*',route)
        page.goto(f'http://127.0.0.1:{port}/#scientific',wait_until='domcontentloaded')
        page.wait_for_selector('#scientific-mechanism-table tbody tr')
        assert page.evaluate('SCIENTIFIC')==expected
        new=expected['safety_mechanism']
        assert page.locator('#scientific-mechanism-table tbody tr').count()==len(new['aggregate_rows'])
        for i,c in enumerate(new['coverage_cases']):
            page.select_option('#scientific-mechanism-coverage-select',str(i))
            assert page.locator('#scientific-mechanism-heatmap rect').count()==260
            assert page.locator('#scientific-mechanism-coverage-note').inner_text()==c['note']
        for i,c in enumerate(new['traces']):
            page.select_option('#scientific-mechanism-select',str(i))
            assert page.locator('#scientific-mechanism-trace polyline').count()==len(c['series'])
            assert page.locator('#scientific-mechanism-caption').inner_text()==c['caption']
        page.locator('header.top').evaluate("n=>n.style.visibility='hidden'")
        page.locator('#scientific-mechanism-table').screenshot(path=str(HERE/'preflight_table.png'))
        page.set_viewport_size(dict(width=390,height=844))
        assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth+1')
        assert not errors,errors
        print(json.dumps(dict(status='pass',scope='partial-data UI preflight; not experiment completion',
                              coverage=len(new['coverage_cases']),traces=len(new['traces']),mobile_no_overflow=True,errors=errors)))
        browser.close()
finally:
    server.terminate();server.wait(timeout=10)
