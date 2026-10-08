"""Actual Chromium journeys against immutable experiment arrays and original images."""
from pathlib import Path
import argparse,json,datetime,re
from playwright.sync_api import sync_playwright
H=Path(__file__).resolve().parent
A=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_native_assets_20261007')/H.name
MODES=['joint_reference','zero_inclusive','box_admission','adaptive_joint']
CLASSES=['跨臂','Franka自碰','UR自碰','台面'];ARMS=['F_L','F_R','U_L','U_R']
def j(p):return json.loads(Path(p).read_text())
def main(url,live):
    result=j(H/'ACCEPTANCE_RESULT.json');galleries=j(A/'galleries.json');cases=j(A/'case_table.json');checks=[];errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1440,'height':1000})
        if not live:
            def route(r):
                marker='/'+H.name+'/'
                relative=r.request.url.split(marker,1)[1].split('?',1)[0]
                p=A/relative
                assert p.is_file() and A in p.resolve().parents
                r.fulfill(path=str(p),headers={'Access-Control-Allow-Origin':'*'})
            context.route('https://raw.githubusercontent.com/asimfish/safeduo-dashboard/**/'+H.name+'/**',route)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url,wait_until='domcontentloaded',timeout=60000)
        page.wait_for_function("document.querySelector('#data-status').textContent.includes('512 条案例')",timeout=60000)
        assert page.locator('#adoption-status').inner_text()=='拒绝采用'
        assert page.locator('#method-rows tr').count()==4
        for i,row in enumerate(result['aggregated']):
            cells=page.locator('#method-rows tr').nth(i).locator('td')
            assert cells.nth(2).inner_text()==str(row['strict_windows'])
            assert cells.nth(3).inner_text()==str(row['deep_windows'])
        def select(mode,block,env):
            page.select_option('#mode',mode);page.select_option('#block',str(block));page.select_option('#env',str(env))
            job=f'b{block}_{mode}'
            page.wait_for_function("([job,env])=>document.querySelector('#curve-status').textContent.includes('已读取 '+job+' / env '+env+'。')",arg=[job,env],timeout=60000)
            assert page.locator('#case-rows tr').count()==64
            assert page.locator('#margin-plot path[data-class]').evaluate_all("els=>els.map(e=>e.dataset.class)")==CLASSES
            assert page.locator('#force-plot path[data-arm]').evaluate_all("els=>els.map(e=>e.dataset.arm)")==ARMS
            raw=j(A/'curves'/f'{job}.json')
            for plot,readout,tk,vk,names,unit in [('margin','margin','times','margin_by_env',CLASSES,'m'),('force','force','native_hand_micro_times','native_hand_max_by_env',ARMS,'N')]:
                for key,index in [('Home',0),('End',len(raw[tk])-1)]:
                    page.locator(f'#{plot}-plot svg').focus();page.keyboard.press(key)
                    expected=page.evaluate("([t,names,v,unit])=>'窗口 t='+String(t)+' s · '+names.map((n,i)=>n+' '+String(v[i])+' '+unit).join(' · ')",[raw[tk][index],names,raw[vk][index][env],unit])
                    assert page.locator(f'#{readout}-readout').inner_text()==expected
            assert '控制步 960 个；物理微步 1920 个' in page.locator('#curve-detail').inner_text()
            row=next(c for c in cases if c['mode']==mode and c['block']==block and c['env']==env)
            if row['first_failure_step'] is None and not row['strict']:assert page.locator('#first-failure').inner_text()=='未违规'
            return job
        def gallery(group):
            select(group['mode'],group['block'],group['env_id'])
            own=sorted([g for g in galleries if g['job_id']==group['job_id'] and g['env_id']==group['env_id']],key=lambda g:(g['step'],g['capture_kind']))
            page.select_option('#capture',str(own.index(group)))
            assert page.locator('#gallery-images img').count()==21
            actual=page.locator('#gallery-images img').evaluate_all("els=>els.map(e=>{e.loading='eager';return e.src;})")
            assert [u.split('/'+H.name+'/',1)[1] for u in actual]==[im['path'] for im in group['images']]
            page.wait_for_function("()=>[...document.querySelectorAll('#gallery-images img')].length===21&&[...document.querySelectorAll('#gallery-images img')].every(i=>i.complete&&i.naturalWidth===1280&&i.naturalHeight===720)",timeout=90000)
            assert page.locator('#state-source').get_attribute('href').endswith(group['state_url'])
            assert page.locator('#gallery-images').get_attribute('data-representative')=='false'
            checks.append({'kind':'original21_gallery','job':group['job_id'],'env':group['env_id'],'step':group['step'],'trigger':group['capture_kind']})
        for mode in MODES:
            for block in [0,1]:
                group=next(g for g in galleries if g['mode']==mode and g['block']==block and g['env_id']==0 and g['step']==959 and g['capture_kind']=='scheduled')
                gallery(group)
        for kind in ['first_failure','first_native_contact']:
            group=next(g for g in galleries if g['mode']=='adaptive_joint' and g['capture_kind']==kind)
            gallery(group)
        # A missing case must remain missing; representative choice affects only its gallery.
        missing=next(c for c in cases if c['mode']=='adaptive_joint' and not any(g['mode']==c['mode'] and g['block']==c['block'] and g['env_id']==c['env'] for g in galleries))
        select(missing['mode'],missing['block'],missing['env'])
        assert page.locator('#gallery-empty').is_visible() and '未采图' in page.locator('#gallery-empty').inner_text()
        before=page.locator('#margin-plot').inner_html();force=page.locator('#force-plot').inner_html();readout=page.locator('#curve-status').inner_text()
        page.select_option('#representative-env','0')
        assert page.locator('#env').input_value()==str(missing['env'])
        assert page.locator('#gallery-images').get_attribute('data-representative')=='true'
        assert page.locator('#margin-plot').inner_html()==before and page.locator('#force-plot').inner_html()==force
        assert page.locator('#curve-status').inner_text()==readout
        checks.append({'kind':'uncaptured_case_and_explicit_representative_preserved','env':missing['env']})
        select('adaptive_joint',0,0)
        page.screenshot(path=str(H/('live_panel.png' if live else 'local_panel.png')),full_page=False)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'), 'mobile horizontal overflow'
        page.screenshot(path=str(H/('live_mobile.png' if live else 'local_mobile.png')),full_page=False)
        assert not errors,errors
        context.close()
        static=browser.new_context(java_script_enabled=False,viewport={'width':1440,'height':1000})
        p=static.new_page();p.goto(url,wait_until='domcontentloaded',timeout=60000)
        assert p.locator('#adoption-status').inner_text()=='拒绝采用' and p.locator('#method-rows tr').count()==4
        assert p.locator('#default-gallery img').count()==21
        for i,row in enumerate(result['aggregated']):assert p.locator('#method-rows tr').nth(i).locator('td').nth(2).inner_text()==str(row['strict_windows'])
        static.close();browser.close()
    receipt=dict(status='PASS_ACTUAL_CHROMIUM_NATIVE_EVIDENCE_JOURNEYS',url=url,live=live,cases=512,method_block_switches=8,full_curve_lengths=[960,1920],raw_first_last_sample_exact=True,original21_galleries=checks,desktop_mobile_noJS=True,js_errors=errors,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (H/('live_browser.json' if live else 'local_browser.json')).open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--live',action='store_true');a=parser.parse_args();main(a.url,a.live)
