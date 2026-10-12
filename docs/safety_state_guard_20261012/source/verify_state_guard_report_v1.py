"""Terminal report delivery checks; separate from physical/general acceptance."""
from pathlib import Path
import ast,csv,hashlib,json,re,sys

D=Path(__file__).resolve().parent
W=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_loaded_duration_20261012')
P=W/'docs/safety_state_guard_20261012'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def artifacts():
    manifest=read(P/'PUBLIC_MANIFEST_V1.json');rows=manifest['files']
    for row in rows:
        f=P/row['path'];assert f.is_file() and f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
        if f.suffix=='.py':ast.parse(f.read_text())
        if f.suffix=='.json':json.loads(f.read_text())
        if f.suffix in ('.html','.json','.py','.csv','.svg'):
            content=f.read_text()
            assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|Bearer\s+[A-Za-z0-9_-]{16,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)',content),row['path']
    result=read(P/'STATE_GUARD_RESULTS_V1.json');assert result['new_controlled_native_microsteps']==7920
    assert [(r['version'],r['physical_pass'],r['original_q_AND_qd_residual_pass']) for r in result['cases']]==[(109,True,False),(110,True,False),(112,False,True),(113,True,True)]
    assert result['brake']['measured_emergency_brake_case_pass']
    assert not any(r['all4_actual_movement_admission_pass'] or r['genuine_full_task_pass'] for r in result['cases'])
    assert not result['fullSystem0_accepted'] and result['input_random_finger_dimensions']==0
    assert result['native_long_random_30_120_600s_completed']==result['new_learned_training_jobs']==result['formal_holdouts']==0
    with (P/'PER_NATIVE_STEP_V1.csv').open() as f:assert sum(1 for _ in csv.DictReader(f))==7920
    media=read(P/'MEDIA_CUSTODY_V1.json');assert len(media['native_videos'])==16
    for row in media['native_videos']:
        assert sha(P/row['public_path'])==row['sha256'] and row['max_timestamp_error_s']<=1.1e-6
    for href in re.findall(r'href="([^"]+)"',(P/'index.html').read_text()):
        target=(P/href.split('#')[0]).resolve()
        assert target.is_file() or (target.is_dir() and (target/'index.html').is_file()),href
    insertion=read(D/'DASHBOARD_ROOT_INSERTION_V1.json')
    before=(D/'DASHBOARD_INDEX_BEFORE_STATE_GUARD_V1.html').read_bytes()
    assert (W/'index.html').read_bytes().replace(insertion['inserted_card'].encode(),b'',1)==before
    receipt={'artifact_files_verified':len(rows),'bytes':manifest['total_bytes'],'native_csv_rows':7920,
        'native_videos':16,'old_root_content_byte_preserved':True,'all_links_resolved':True,
        'physical_results_and_nonacceptance_claims_match_audits':True,'secret_pattern_scan_pass':True}
    (D/'REPORT_ARTIFACT_GATE_V1.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('REPORT_ARTIFACT_PASS',len(rows),'files',flush=True)


def browser():
    from playwright.sync_api import sync_playwright
    contexts=[]
    with sync_playwright() as pw:
        b=pw.chromium.launch(executable_path='/home/liyufeng/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox'])
        for width,js in ((1280,True),(390,True),(390,False)):
            ctx=b.new_context(viewport={'width':width,'height':900},java_script_enabled=js)
            page=ctx.new_page();page.goto(P.as_uri()+'/index.html',wait_until='load')
            assert page.locator('h1').inner_text()=='实际状态防护与原生制动'
            assert page.locator('video').count()==16
            assert page.locator('.alert').inner_text().find('任意长时间、大范围运动仍未验收')>=0
            assert page.locator('.metric b').all_text_contents()[0]=='7920'
            if js:
                page.wait_for_function('Array.from(document.querySelectorAll("video")).every(v=>v.readyState>=1)',timeout=30000)
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
                for i in range(16):
                    v=page.locator('video').nth(i);assert v.evaluate('(v)=>v.videoWidth===1280 && v.videoHeight===720')
                    v.evaluate('(v)=>{v.currentTime=0;v.play()}')
                    page.wait_for_function('(i)=>document.querySelectorAll("video")[i].currentTime>0.05',arg=i,timeout=10000)
                    v.evaluate('(v)=>v.pause()')
                page.locator('#slow').click();assert page.locator('video').first.evaluate('(v)=>v.playbackRate')==.2
                page.locator('#normal').click();assert page.locator('video').first.evaluate('(v)=>v.playbackRate')==1
                v=page.locator('video').last;v.evaluate('(v)=>{v.currentTime=0;v.pause()}');v.focus();page.keyboard.press('Space')
                page.wait_for_function('document.querySelectorAll("video")[15].currentTime>0.05',timeout=10000)
                v.evaluate('(v)=>v.pause()')
            else:
                box=page.locator('main').bounding_box();assert box['x']>=0 and box['x']+box['width']<=width+1
            dest=D/f'state_guard_report_{width}_js{int(js)}.png';page.screenshot(path=str(dest),full_page=True)
            page.goto((W/'index.html').as_uri(),wait_until='load');link=page.locator('#actualStateBrakeEvidence a')
            assert link.get_attribute('href')=='docs/safety_state_guard_20261012/'
            contexts.append({'width':width,'javascript':js,'videos':16,'all_native_videos_played':js,'native_Space_playback_tested':js,'root_link_verified':True,'screenshot':str(dest)})
            ctx.close()
        b.close()
    (D/'REPORT_BROWSER_GATE_V1.json').write_text(json.dumps({'contexts':contexts,'delivery_pass':True,'fullSystem0_accepted':False},indent=2)+'\n')
    print('REPORT_BROWSER_PASS3_CONTEXTS16_VIDEOS_AND_NATIVE_SPACE',flush=True)


if __name__=='__main__':
    if '--artifacts' in sys.argv:artifacts()
    elif '--browser' in sys.argv:browser()
    else:raise SystemExit('select --artifacts or --browser; browser environment needs no NumPy')
