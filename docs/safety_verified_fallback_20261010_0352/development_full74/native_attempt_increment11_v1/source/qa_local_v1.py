"""Future actual REPORT11 local browser gate. No live/native reads or publication.

The partial overlay resolves declared new files first and inherits baseline W
files by manifest SHA. Browser software policy reuses the closed REPORT10 gate.
Run only after a real closed-terminal build; SOURCE readiness is not browser PASS.
"""
import argparse
import hashlib
import http.server
import json
import mimetypes
import socket
import threading
from pathlib import Path
from urllib.parse import unquote,urlparse

R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
W=Path('/home/liyufeng/safeduo-dashboard-verified-fallback-20261010-0352')
REL='docs/safety_verified_fallback_20261010_0352/development_full74/'
SHELL='/home/liyufeng/safeduo/velocity_browser_venv_20261008_1519/browsers/chromium_headless_shell-1187/chrome-linux/headless_shell'

def need(value,message):
    if not value:raise ValueError(message)

def digest(raw):return hashlib.sha256(raw).hexdigest()

def small_json(path,sha):
    with Path(path).open('rb') as f:raw=f.read(4*1024**2+1)
    need(len(raw)<=4*1024**2 and digest(raw)==sha,'bound small JSON')
    return json.loads(raw)

def main():
    from playwright.sync_api import sync_playwright
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=('local','public'),default='local');a=p.parse_args()
    plan=small_json(a.plan,a.plan_sha256);need(plan['schema']=='astra.report.increment11.overlay_plan.v1','actual REPORT11 candidate plan')
    manifest=small_json(plan['manifest']['path'],plan['manifest']['sha256']);facts=small_json(plan['source_facts']['path'],plan['source_facts']['sha256'])
    need(facts['packet_ref'] and facts['bank_count']==0 and facts['safety_acceptance'] is False,'actual bound terminal facts, no promotion')
    out=Path(a.out);need(out.is_absolute() and out.resolve().is_relative_to(R/'astra_report_increment11_v1') and not out.exists(),'exclusive owned browser output')
    files=dict(manifest['files'],**{'panel_manifest.json':plan['manifest']['sha256']});sources=plan['file_sources']
    need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in files),'declared relative assets')
    out.mkdir(parents=True,exist_ok=False);errors=[];modes=[];served=[];server=None;thread=None;browser=None;pw=None;closed={}
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            try:
                path=unquote(urlparse(self.path).path);need(path.startswith('/'+REL),'declared panel route')
                leaf=path[len('/'+REL):] or 'index.html';need(leaf in files,'declared manifest asset')
                src=Path(sources[leaf]['path']) if leaf in sources else W/REL/leaf
                need(not src.is_symlink() and src.stat().st_size<=100*1024**2,'bounded regular asset')
                raw=src.read_bytes();need(digest(raw)==files[leaf],'actual served bytes '+leaf)
                self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(leaf)[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
                served.append(dict(path=leaf,sha256=digest(raw)))
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as e:errors.append(str(e));self.send_error(404)
    try:
        base='https://asimfish.github.io/safeduo-dashboard/'+REL;port=None
        if a.mode=='local':
            server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
            thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05});thread.start();port=server.server_address[1]
            base='http://127.0.0.1:'+str(port)+'/'+REL
        pw=sync_playwright().start();browser=pw.chromium.launch(executable_path=SHELL,headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-sandbox'])
        for width in (1440,390):
            for js in (True,False):
                ctx=browser.new_context(viewport=dict(width=width,height=1000 if width==1440 else 844),java_script_enabled=js,reduced_motion='reduce')
                try:
                    page=ctx.new_page();bad=[];page.on('pageerror',lambda e:bad.append(str(e)));page.on('response',lambda r:bad.append([r.url,r.status]) if r.status>=400 else None)
                    def route(r):
                        u=urlparse(r.request.url)
                        if (a.mode=='local' and u.scheme=='http' and u.hostname=='127.0.0.1' and u.port==port) or (a.mode=='public' and u.scheme=='https' and u.hostname=='asimfish.github.io'):r.continue_()
                        else:bad.append(['EXTERNAL',r.request.url]);r.abort()
                    page.route('**/*',route);response=page.goto(base,wait_until='domcontentloaded',timeout=20000)
                    need(response.status==200 and digest(response.body())==files['index.html'],'actual HTML SHA')
                    section=page.locator('#nativeAttemptIncrement11');need(section.count()==1,'one REPORT11 section')
                    for lang in ('ZH','EN'):
                        page.locator('#r11Language'+lang).check()
                        for detail in ('Detailed','Summary'):
                            page.locator('#r11'+detail).check();need(section.locator('.r11-result').is_visible(),'summary outcome always visible')
                            need(section.locator('.r11-detail').is_visible()==(detail=='Detailed'),'native detail control works without JS')
                            hidden='.r11-en' if lang=='ZH' else '.r11-zh'
                            need(section.locator(hidden).first.is_hidden(),'bilingual control hides other language')
                    page.locator('#r11LanguageZH').check();page.locator('#r11Detailed').check()
                    need(section.locator('#native11RequirementsTable tbody tr').count()==8,'all requirement layers')
                    need(section.locator('#native11StrongCameraTable tbody tr').count()==5,'all strong camera thresholds')
                    need('332.812' in section.inner_text() and 'NONADDITIVE' in section.inner_text() and 'UNKNOWN' in section.inner_text(),'honest diagnostic boundaries')
                    summary=page.request.get(base+'summary.json',timeout=15000);need(summary.status==200 and digest(summary.body())==files['summary.json'],'actual served summary SHA')
                    j=summary.json();need(j['asof_utc']==facts['snapshot_utc'] and j['asof_AsiaShanghai']==facts['snapshot_Asia_Shanghai'],'summary/body/facts same frozen UTC/local instant')
                    need(page.locator('#reportAsOf time').get_attribute('datetime')==facts['snapshot_utc'],'body cutoff binding')
                    fig_ids=('figure1','figure2','figure3','calibrationFigure','mechanismV6Figure','model05CalibrationFigure','model05ScreenFigure','model06SupplementFigure')
                    for ident in fig_ids:
                        fig=page.locator('#'+ident);need(fig.count()==1,'old figure retained');image=fig.locator('img');image.scroll_into_view_if_needed(timeout=6000)
                        image.evaluate('(e)=>e.decode()');need(image.evaluate('(e)=>e.naturalWidth>0&&getComputedStyle(e).objectFit!=="cover"'),'loaded uncropped old science figure')
                    raw=page.locator('#camera img');need(raw.count()==15,'all15 original native RGB retained')
                    for image in raw.all():image.scroll_into_view_if_needed(timeout=6000);image.evaluate('(e)=>e.decode()')
                    anchor_ids=page.locator('a[href^="#"]').evaluate_all('(a)=>a.map(x=>x.hash.slice(1))')
                    need(all(page.locator('[id="'+i+'"]').count()==1 for i in anchor_ids),'all internal anchors resolve once')
                    overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth+1');need(not overflow,'desktop/mobile page has no horizontal overflow')
                    section.scroll_into_view_if_needed(timeout=6000);shot=out/('report11_'+str(width)+'_'+str(js)+'.png');page.screenshot(path=str(shot),full_page=False)
                    need(not bad,'browser response/JS/network errors')
                    modes.append(dict(width=width,JS=js,EN_ZH_summary_detailed=True,figures=8,original_RGB=15,screenshot=dict(path=str(shot),sha256=digest(shot.read_bytes()))))
                finally:ctx.close()
    except Exception as e:errors.append(type(e).__name__+': '+str(e));raise
    finally:
        if browser:browser.close();closed['browser']=not browser.is_connected()
        if pw:pw.stop();closed['playwright']=True
        if server:
            server.shutdown();server.server_close();thread.join(5)
            with socket.socket() as sk:sk.settimeout(.2);closed['HTTP_server']=not thread.is_alive() and sk.connect_ex(('127.0.0.1',port))!=0
        (out/'BROWSER_REPORT_V1.json').write_text(json.dumps(dict(status='PASS_REAL_LOCAL_BROWSER' if len(modes)==4 and not errors and all(closed.values()) else 'FAILED_PRESERVED',modes=modes,errors=errors,closed=closed,served=served,frozen_snapshot_utc=facts['snapshot_utc']),ensure_ascii=False,indent=2)+'\n')
    need(len(modes)==4 and not errors and all(closed.values()),'actual complete browser closure')

if __name__=='__main__':main()
