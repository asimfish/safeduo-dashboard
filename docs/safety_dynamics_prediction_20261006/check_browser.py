"""Actual Chromium journeys; running and final evidence remain distinguishable."""
from pathlib import Path
import argparse,json,urllib.request
from playwright.sync_api import sync_playwright
def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--phase',choices=['running','complete'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);errors=[];checks=[]
    with sync_playwright() as api:
        browser=api.chromium.launch(headless=True)
        for width in (1440,390):
            page=browser.new_page(viewport=dict(width=width,height=950))
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(a.url,wait_until='networkidle');page.wait_for_function("document.querySelector('#calibration').children.length===8")
            assert page.locator('#calibration tr').count()==8
            assert page.locator('#complete').inner_text().endswith(' / 384')
            if a.phase=='complete':
                page.wait_for_function("document.querySelector('#case').options.length===192")
                assert page.locator('#physical tr').count()==2 and page.locator('#models tr').count()==8
                page.wait_for_function("document.querySelector('#caseRows').children.length===2")
                page.select_option('#case',index=191);page.wait_for_function("current && current.id===document.querySelector('#case').value")
                for f in ('bad','miss','all'):
                    page.select_option('#filter',f)
                    page.wait_for_function("!document.querySelector('#case').options.length || (current && current.id===document.querySelector('#case').value)")
                assert page.locator('#case option').count()==192
            else:
                assert page.locator('#newResults').is_hidden() and page.locator('#pending').is_visible()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1')
            page.screenshot(path=str(a.out/f'{a.phase}_{width}.png'),full_page=True)
            checks.append(dict(width=width,historical_rows=8,document_overflow=False,phase=a.phase))
            page.close()
        browser.close()
    if a.phase=='complete':
        with urllib.request.urlopen(a.url+'data.json') as r:data=json.load(r)
        assert len(data['cases'])==192 and data['completed_windows']==384
        for meta in data['cases']:
            with urllib.request.urlopen(a.url+'cases/'+meta['id']+'.json') as r:case=json.load(r)
            for c in case['controllers']:
                assert len(c['margins'])==960 and all(len(row)==4 for row in c['margins'])
                values=[v for row in c['margins'] for v in row]
                assert min(values)==c['min_margin_m'] and any(v<0 for v in values)==c['bad']
        checks.append(dict(full_case_readbacks=192,controllers=384,actual_margin_signs_exact=True))
    assert not errors,errors
    result=dict(status='PASS',checks=checks,javascript_errors=errors,url=a.url)
    (a.out/'browser.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
