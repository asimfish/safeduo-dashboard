"""Verify new risk UI and all legacy coverage/traces in a real browser."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

HERE=Path('/home/liyufeng/safeduo/artifacts/safety_risk_strata_20261004')
OUTPUT=Path(__file__).resolve().parent
DATA=Path('/home/liyufeng/safeduo-dashboard-data')
parser=argparse.ArgumentParser();parser.add_argument('--live-url');parser.add_argument('--local-url');parser.add_argument('--snapshot');args=parser.parse_args()
snapshot=Path(args.snapshot) if args.snapshot else OUTPUT/'legacy_scientific_snapshot_current.json'
expected=json.loads(snapshot.read_text())
# Foreign data branch is read only. Compare live UI to the freshly frozen snapshot.
# Ownership of changes is proved by root compare-and-swap and Git diff, not assumed data immutability.
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport=dict(width=1280,height=900),ignore_https_errors=True)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    if not args.live_url:
        def local(route):
            name=Path(urlparse(route.request.url).path).name
            file=snapshot if name=='scientific_eval.json' else DATA/name
            if file.exists():route.fulfill(status=200,content_type='application/json',body=file.read_text())
            else:route.continue_()
        page.route('**/data/*.json*',local)
    page.goto(args.live_url or args.local_url,wait_until='domcontentloaded')
    page.locator('a[data-tab="scientific"]').click()
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
    assert page.locator('#scientific-mechanism-table tbody tr').count()==len(actual['safety_mechanism']['aggregate_rows'])
    assert page.locator('#scientific-mechanism-pair-results tbody tr').count()==len(actual['safety_mechanism']['pair_rows'])
    assert page.locator('#queueEnvelopeEvidence').count()==1
    admission=actual['safety_predictive_admission']
    for suffix,key in [('table','aggregate_rows'),('pair-results','pair_rows'),('safe','aggregate_rows'),('joint','operation_joint_rows'),('speed-table','speed_comparison_rows')]:
        assert page.locator('#scientific-admission-'+suffix+' tbody tr').count()==len(admission[key]),suffix
    assert page.locator('#scientific-admission-inputs tbody tr').count()==sum(x.get('both_failure') is not None for x in admission['paired_inputs'])
    assert page.locator('#scientific-admission-zero-table tbody tr').count()==len(admission['zero_command_diagnostic']['aggregate_rows'])
    coverage_count=0
    for section,data in [('scientific-risk',risk),('scientific-random',old),('scientific-mechanism',actual['safety_mechanism']),('scientific-admission',admission)]:
        for i,c in enumerate(data['coverage_cases']):
            coverage_count+=1;page.select_option('#'+section+'-coverage-select',str(i))
            assert page.locator('#'+section+'-heatmap rect').count()==260
            assert page.locator('#'+section+'-ee tbody tr').count()==4
            assert page.locator('#'+section+'-pairs tbody tr').count()==6
            assert page.locator('#'+section+'-coverage-note').inner_text()==c['note']
            html=page.locator('#'+section+'-heatmap').inner_html();assert 'NaN' not in html and 'Infinity' not in html
    assert page.locator('#jointGuardEvidence').count()==1
    assert page.locator('#feasibleGuardEvidence').count()==1
    assert page.locator('#scientific-refinement-table tbody tr').count()==len(actual['safety_refinement']['rows'])
    for selector,n in [('scientific-bounded-table',4),('scientific-bounded-stops',18),('scientific-queue-controls',9),('scientific-queue-stops',12),
                        ('scientific-forensics-table',4),('scientific-forensics-holdout',4),('scientific-perturbation-table',30),('scientific-robustness-table',8)]:
        assert page.locator('#'+selector+' tbody tr').count()==n,selector
    traces=0
    for section,key in [('scientific-admission','safety_predictive_admission'),('scientific-mechanism','safety_mechanism'),('scientific-risk','safety_risk_strata'),('scientific-random','safety_random_space'),('scientific-bounded','safety_bounded_stop'),
                        ('scientific-queue','safety_queue_braking'),('scientific-forensics','safety_forensics'),('scientific-perturbation','safety_perturbation'),('scientific-robustness','safety_robustness')]:
        for i,t in enumerate(actual[key]['traces']):
            traces+=1;page.select_option('#'+section+'-select',str(i))
            assert page.locator('#'+section+'-trace polyline').count()==len(t['series'])
            assert page.locator('#'+section+'-caption').inner_text()==t['caption']
            html=page.locator('#'+section+'-trace').inner_html();assert 'NaN' not in html and 'Infinity' not in html
    for i,t in enumerate(actual['safety_refinement']['traces']):
        page.select_option('#scientific-trace-select',str(i));traces+=1
        assert page.locator('#scientific-trace polyline').count()==2
        assert 'NaN' not in page.locator('#scientific-trace').inner_html()
    page.select_option('#scientific-risk-coverage-select',str(risk['default_coverage']))
    if risk['traces']:page.select_option('#scientific-risk-select',str(risk['default_trace']))
    page.locator('header.top').evaluate("n=>n.style.visibility='hidden'")
    page.locator('#scientific-risk').screenshot(path=str(OUTPUT/('legacy_live_panel.png' if args.live_url else 'legacy_panel.png')))
    page.set_viewport_size(dict(width=390,height=844))
    assert not page.evaluate('document.documentElement.scrollWidth>window.innerWidth+1')
    assert not errors,errors
    receipt=dict(status='pass',browser=browser.version,live_url=args.live_url,coverage_cases_checked=coverage_count,traces_checked=traces,
        scientific_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),foreign_data_branch_read_only=True,
        legacy_preserved=True,mobile_no_page_overflow=True,page_errors=errors)
    (OUTPUT/('legacy_live_browser.json' if args.live_url else 'legacy_browser.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    browser.close()
