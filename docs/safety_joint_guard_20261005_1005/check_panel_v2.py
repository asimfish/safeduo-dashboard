"""Every case/class, outcome filter, coverage heatmap, image and raw native link."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--live',action='store_true');args=parser.parse_args()
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport=dict(width=1380,height=920));page.on('pageerror',lambda e:errors.append(str(e)))
    expected_page=Path('/home/liyufeng/safeduo-dashboard-joint-guard-20261005/docs')/HERE.name/'index.html'
    served={}
    def capture_document(route):
        actual=route.fetch()
        assert actual.ok
        body=actual.body()
        served['sha256']=hashlib.sha256(body).hexdigest()
        assert served['sha256']==hashlib.sha256(expected_page.read_bytes()).hexdigest(), 'served page bytes differ from reviewed artifact'
        route.fulfill(response=actual,body=body)
    page.route(args.url,capture_document)
    page.goto(args.url,wait_until='networkidle');D=page.evaluate('D')
    page_sha=served['sha256']
    assert D['result']==json.loads((HERE/'results.json').read_text()), 'served result differs from closed analysis'
    assert len(D['cases'])==len(D['result']['paired_four_patterns'])*64
    assert page.locator('#metrics tbody tr').count()==4
    assert page.locator('#diagnostics tbody tr').count()==len([r for r in D['result']['rows'] if r['status']=='complete'])
    assert page.locator('#visualDifferences tbody tr').count()==2
    assert page.locator('#coverageMetrics tbody tr').count()==4
    assert '运动范围下降' in page.locator('#motionTradeoff').inner_text()
    drawings=0
    for i,case in enumerate(D['cases']):
        page.select_option('#case',str(i))
        for cls in range(4):
            page.select_option('#class',label=['cross','F自碰','U自碰','桌面'][cls])
            assert page.locator('#curve polyline').count()==4
            assert page.locator('#caseMetrics tbody tr').count()==4
            assert f"env{case['env']}" in page.locator('#caseNote').inner_text()
            html=page.locator('#curve').inner_html();assert 'NaN' not in html and 'Infinity' not in html
            drawings+=1
    actuation_drawings=0
    for i,case in enumerate(D['cases']):
        page.select_option('#case',str(i))
        for metric in range(3):
            page.select_option('#actuationSelect',label=['发出目标积压（rad）','实测关节速度（rad/s）','实际目标增量（rad/步）'][metric])
            assert page.locator('#actuationCurve polyline').count()==4
            assert f"env{case['env']}" in page.locator('#actuationNote').inner_text()
            html=page.locator('#actuationCurve').inner_html();assert 'NaN' not in html and 'Infinity' not in html
            actuation_drawings+=1
    filters=0
    for mode in ['admission_guard','envelope_guard','joint_guard']:
        page.select_option('#compare',mode)
        for s in ['all','-1','0','1','2','3','4','5']:
            for o in ['all','rescued','regressed','both','safe']:
                page.select_option('#stratum',s);page.select_option('#outcome',o)
                expected=0
                for c in D['cases']:
                    b=c['methods']['baseline_guard']['failed'];v=c['methods'][mode]['failed']
                    if (s=='all' or c['stratum']==int(s)) and (o=='all' or o=='rescued' and b and not v or o=='regressed' and not b and v or o=='both' and b and v or o=='safe' and not b and not v):expected+=1
                assert page.locator('#case option').count()==expected,(mode,s,o)
                filters+=1
    page.select_option('#compare','joint_guard');page.select_option('#stratum','all');page.select_option('#outcome','all')
    for i in range(4):
        page.select_option('#coverageSelect',str(i));assert page.locator('#heatmap rect').count()==260
        assert page.locator('#coverageNote').inner_text()
    images=0
    for i,g in enumerate(D['galleries']):
        page.select_option('#gallerySelect',str(i));page.locator('#gallery').scroll_into_view_if_needed()
        page.locator('#gallery img').evaluate_all("xs=>xs.forEach(x=>x.loading='eager')")
        page.wait_for_function("[...document.querySelectorAll('#gallery img')].every(x=>x.complete&&x.naturalWidth===1280&&x.naturalHeight===720)",timeout=45000)
        assert page.locator('#gallery img').count()==9
        for k,key in enumerate(['state','before','after']):
            link=page.locator('#galleryNote a').nth(k).get_attribute('href')
            response=page.request.get(urljoin(args.url,link));assert response.ok
            expected=g['state_sha256'] if key=='state' else g[key+'_sha256']
            assert hashlib.sha256(response.body()).hexdigest()==expected
        images+=9
    for link in page.locator('#docs a').all():
        response=page.request.get(urljoin(args.url,link.get_attribute('href')));assert response.ok
    page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(HERE/('live_panel.png' if args.live else 'panel.png')))
    page.set_viewport_size(dict(width=390,height=844));assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth+1')
    assert not errors,errors
    receipt=dict(status='PASS',live=args.live,cases=len(D['cases']),class_drawings=drawings,method_polylines=drawings*4,
        actuation_drawings=actuation_drawings,actuation_polylines=actuation_drawings*4,filters=filters,coverage_cases=4,images=images,state_and_native_http_digests=len(D['galleries'])*3,
        mobile_no_overflow=True,page_errors=errors,browser=browser.version,url=args.url,
        page_sha256=page_sha,result_sha256=hashlib.sha256((HERE/'results.json').read_bytes()).hexdigest(),
        utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        document_capture='real URL fetched through Playwright API and exact byte response forwarded to browser; no inspector body-cache reliance')
    (HERE/('live_browser.json' if args.live else 'browser.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));browser.close()
