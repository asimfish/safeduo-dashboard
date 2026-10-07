"""Native browser reads all cases/methods, filters, coverage and camera bindings."""
from pathlib import Path
import argparse,gzip,hashlib,json
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
PUBLIC=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_zero_worktree_20261006/docs')/HERE.name


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--tag',required=True);args=parser.parse_args()
    result=json.loads((HERE/'holdout_results.json').read_text());expected=json.loads(gzip.decompress((PUBLIC/'payload.json.gz').read_bytes()))
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu'])
        page=browser.new_page(viewport=dict(width=1440,height=1000));page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.url,wait_until='domcontentloaded',timeout=120000);page.wait_for_function('window.SAFEDUO_READY===true',timeout=120000)
        actual=page.evaluate('window.SAFEDUO');assert actual==dict(payload=expected,result=result)
        assert page.locator('#totals tr').count()==4 and page.locator('#heatmap button').count()==192
        totals=page.locator('#totals tr').evaluate_all('rows=>rows.slice(1).map(r=>[...r.querySelectorAll("td")].slice(1,5).map(c=>Number(c.textContent)))')
        assert totals==[[r['violations'],r['deep'],r['strict_env_steps'],r['deep_env_steps']] for r in result['totals']]
        classes=page.locator('#classResults tr').evaluate_all('rows=>rows.slice(1).map(r=>[...r.querySelectorAll("td")].slice(2).map(c=>Number(c.textContent)))')
        assert classes==[[r['strict_windows'],r['deep_windows'],r['strict_env_steps'],r['deep_env_steps']] for r in expected['classSummary']]
        checked=0
        for i,c in enumerate(result['cases']):
            page.select_option('#case',str(i))
            for mode,s in c['methods'].items():
                page.select_option('#method',mode);assert page.evaluate('window.SAFEDUO_SELECTION')==dict(case=i,method=mode,curves=s['curves'],actuation=s['actuation'],indices=s['indices'])
                assert page.locator('#curves polyline').count()==7
                assert 'NaN' not in page.locator('#curves').inner_html();checked+=1
        def category(c):
            a,b=c['methods']['joint_reference']['failed'],c['methods']['zero_inclusive']['failed']
            return ('both' if b else 'rescued') if a else ('new' if b else 'neither')
        filters={}
        for st in ['all',*[str(i) for i in range(-1,6)]]:
            page.select_option('#stratum',st)
            for out in ['all','rescued','new','both','neither']:
                page.select_option('#outcome',out)
                count=sum((st=='all' or c['stratum']==int(st)) and (out=='all' or category(c)==out) for c in result['cases'])
                assert page.locator('#heatmap button').count()==count;filters[st+'/'+out]=count
        page.select_option('#stratum','all');page.select_option('#outcome','all')
        coverage=0
        for mode in expected['modes']:
            page.select_option('#coverageMode',mode)
            c=next(r['measured'] for r in result['coverage'] if r['mode']==mode)
            for stage in ['initial_joint','visited_joint']:
                page.select_option('#coverageStage',stage)
                titles=page.locator('#coverage rect title').all_text_contents()
                assert titles==[f'关节{j+1} 区间{b+1}：{c[stage]["counts"][j][b]}' for j in range(26) for b in range(10)]
                coverage+=len(titles)
        bindings=0
        for i,g in enumerate(expected['visual']['galleries']):
            page.select_option('#cameraGroup',str(i));base=expected['assetBase']+'visual/'+g['mode']+'/'
            assert page.locator('#gallery img').evaluate_all('xs=>xs.map(x=>x.src)')==[base+im['path'] for im in g['images']]
            assert page.locator('#stateLinks a').evaluate_all('xs=>xs.map(x=>x.href)')==[base+g[k] for k in ['state','before','after']]
            bindings+=len(g['images'])
        for i in sorted({0,len(expected['visual']['galleries'])-1}):
            page.select_option('#cameraGroup',str(i));page.locator('#gallery').scroll_into_view_if_needed()
            page.locator('#gallery img').evaluate_all('xs=>xs.forEach(x=>x.loading="eager")')
            page.wait_for_function('Array.from(document.querySelectorAll("#gallery img")).every(x=>x.complete&&x.naturalWidth===1280&&x.naturalHeight===720)',timeout=120000)
        assert page.locator('#failures tr').count()==1+expected['failure']['cases']
        assert page.locator('#fixedFeasibility tr').count()==28
        assert page.locator('#zeroIntent tr').count()==4
        zero=page.locator('#zeroIntent tr').evaluate_all('rows=>rows.slice(1).map(r=>[...r.querySelectorAll("td")].slice(1,7).map(c=>Number(c.textContent)))')
        assert zero==[[r['zero_prefix_strict_windows'],r['zero_prefix_strict_env_steps'],r['bounds_excluding_zero_env_steps'],r['nonzero_effective_target_env_steps'],r['reference_prelimit_injection_env_steps'],r['original_projector_nonzero_return_env_steps']] for r in expected['zeroSummary']]
        assert str(expected['invariant']['candidate_env_steps']) in page.locator('#invariantNote').inner_text()
        assert page.locator('#executionNote').inner_text()==expected['executionText']
        fixed=page.locator('#fixedFeasibility tr').evaluate_all('rows=>rows.slice(1).map(r=>[...r.querySelectorAll("td")].slice(1).map(c=>Number(c.textContent)))')
        assert fixed==[[r['pre_step'],r['cases'],r['F_infeasible'],r['U_infeasible']] for r in expected['fixedSummary']]
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(HERE/(args.tag+'_desktop.png')),full_page=False)
        page.set_viewport_size(dict(width=390,height=844));page.evaluate('window.scrollTo(0,0)');assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
        assert not errors,errors
        page.screenshot(path=str(HERE/(args.tag+'_mobile.png')),full_page=False)
        receipt=dict(status='PASS_ALL_NATIVE_PANEL_CHECKS',url=args.url,browser=browser.version,cases_methods=checked,curvedata_arrays=checked*7,filters=filters,coverage_cells=coverage,image_url_bindings=bindings,all_payload_and_results_exact=True,mobile_no_overflow=True,errors=errors)
        with (HERE/(args.tag+'_browser.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
        print('NATIVE_BROWSER_PASS',checked,coverage,bindings,flush=True);browser.close()


if __name__=='__main__':main()
