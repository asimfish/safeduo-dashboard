"""Delivery gate for a failed short native trial, never a broad-safety pass."""
from pathlib import Path
import ast,csv,hashlib,json,re,sys

E=Path(__file__).resolve().parent
W=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_loaded_duration_20261012')
P=W/'docs/safety_full50_motion_20261013'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def artifacts():
    manifest=read(P/'PUBLIC_MANIFEST_V1.json')
    for row in manifest['files']:
        f=P/row['path'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
        if f.suffix=='.py':ast.parse(f.read_text())
        if f.suffix=='.json':read(f)
        if f.suffix in ['.html','.json','.py','.csv','.svg','.urdf']:
            assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|Bearer\s+[A-Za-z0-9_-]{16,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)',f.read_text()),row['path']
    result=read(P/'FULL50_RESULTS_V1.json');assert result['physics_events']==832 and len(result['motor_rows'])==50
    assert result['all_original_physical_gates_pass'] and not result['original_q_AND_qd_residuals_pass']
    assert not result['full_preregistered_window_complete'] and not result['finite_preregistered_physical_and_model_pass']
    assert result['ordinary_finger_motor_target_dimensions_updated']==24 and result['ordinary_controls_after_new_random_FIFO_delivery']==16
    assert not result['original_all4_arm_movement_admission_pass'] and not result['genuine_full_task_pass'] and not result['fullSystem0_accepted']
    causal=read(P/'CONTACT_CAUSAL_RESULT_V1.json')
    assert causal['precontact_physical_events_compared']==824 and causal['precontact_every_compared_physical_field_byte_equal']
    assert causal['filtered_contact_peak_normal_N']==0 and causal['native_event824_in_filtered_world']['maximum_qd_residual_rad_s']<1e-4
    assert causal['original_model_failure']['event']==1040 and not causal['counts_towards_random_safety_pass_rate']
    with (P/'ACTUAL50_MOTOR_COVERAGE_V1.csv').open() as f:assert sum(1 for _ in csv.DictReader(f))==50
    with (P/'PER_NATIVE_STEP_V1.csv').open() as f:assert sum(1 for _ in csv.DictReader(f))==832
    for row in read(P/'MEDIA_CUSTODY_V1.json')['views']:
        assert sha(P/row['public_path'])==row['sha256'] and row['max_timestamp_error_s']<=1.1e-6
    for href in re.findall(r'href="([^"]+)"',(P/'index.html').read_text()):
        path=(P/href.split('#')[0]).resolve();assert path.is_file() or (path.is_dir() and (path/'index.html').is_file()),href
    insertion=read(E/'DASHBOARD_FULL50_INSERTION_V1.json')
    assert (W/'index.html').read_bytes().replace(insertion['card'].encode(),b'',1)==(E/'DASHBOARD_INDEX_BEFORE_FULL50_V1.html').read_bytes()
    (E/'FULL50_REPORT_ARTIFACT_GATE_V1.json').write_text(json.dumps({'files':len(manifest['files']),'native_rows':832,'motor_rows':50,'old_root_bytes_preserved':True,'claims_match_failed_original_model_oracle':True,'all_links_and_SHA_valid':True,'fullSystem0_accepted':False},indent=2)+'\n')
    print('FULL50_REPORT_ARTIFACT_PASS',len(manifest['files']),'files',flush=True)


def browser():
    from playwright.sync_api import sync_playwright
    rows=[]
    with sync_playwright() as p:
        b=p.chromium.launch(executable_path='/home/liyufeng/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',args=['--no-sandbox'])
        for width,js in ((1280,True),(390,True),(390,False)):
            ctx=b.new_context(viewport={'width':width,'height':900},java_script_enabled=js);page=ctx.new_page();page.goto((P/'index.html').as_uri(),wait_until='load')
            assert page.locator('h1').inner_text()=='全50电机随机动作诊断'
            assert '未通过' in page.locator('.alert').inner_text() and '长时间、大范围任意运动仍未验收' in page.locator('.alert').inner_text()
            assert '116不计入任何安全或任务成功率' in page.locator('#contactCausal').inner_text()
            assert page.locator('tbody tr').count()==50 and page.locator('video').count()==4
            assert page.locator('.metric b').all_text_contents()==['50','24 / 24','0.433秒','0']
            if js:
                page.wait_for_function('Array.from(document.querySelectorAll("video")).every(v=>v.readyState>=1)')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                for i in range(4):
                    v=page.locator('video').nth(i);assert v.evaluate('(v)=>v.videoWidth===1280&&v.videoHeight===720')
                    v.evaluate('(v)=>v.play()');page.wait_for_function('(i)=>document.querySelectorAll("video")[i].currentTime>.05',arg=i);v.evaluate('(v)=>v.pause()')
                page.locator('#slow').click();assert page.locator('video').first.evaluate('(v)=>v.playbackRate')==.2
                page.locator('#normal').click();assert page.locator('video').first.evaluate('(v)=>v.playbackRate')==1
                v=page.locator('video').last;v.evaluate('(v)=>{v.pause();v.currentTime=0}');v.focus();page.keyboard.press('Space');page.wait_for_function('document.querySelectorAll("video")[3].currentTime>.05');v.evaluate('(v)=>v.pause()')
            else:
                box=page.locator('main').bounding_box();assert box['x']>=0 and box['x']+box['width']<=width+1
            page.screenshot(path=str(E/f'full50_report_{width}_js{int(js)}.png'),full_page=True)
            page.goto((W/'index.html').as_uri());assert page.locator('#full50MotionEvidence a').get_attribute('href')=='docs/safety_full50_motion_20261013/'
            rows.append({'width':width,'javascript':js,'native_videos':4,'motor_rows':50,'video_playback_and_native_Space':js,'root_link_verified':True});ctx.close()
        b.close()
    (E/'FULL50_REPORT_BROWSER_GATE_V1.json').write_text(json.dumps({'contexts':rows,'delivery_pass':True,'fullSystem0_accepted':False},indent=2)+'\n');print('FULL50_BROWSER_PASS3_CONTEXTS4_VIDEOS50_MOTOR_ROWS',flush=True)


if __name__=='__main__':
    if '--artifacts' in sys.argv:artifacts()
    elif '--browser' in sys.argv:browser()
    else:raise SystemExit('select --artifacts or --browser')
