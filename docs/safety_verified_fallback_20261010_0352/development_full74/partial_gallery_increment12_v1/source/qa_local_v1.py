"""Real local four-mode REPORT12 gate; unchanged software Chromium policy."""
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
def need(v,message):
    if not v:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def small_json(path,sha):
    p=Path(path);need(not p.is_symlink(),'regular bound JSON')
    with p.open('rb') as f:raw=f.read(4*1024**2+1)
    need(len(raw)<=4*1024**2 and digest(raw)==sha,'bound small JSON');return json.loads(raw)

def main():
    from playwright.sync_api import sync_playwright
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('plan','plan-sha256','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=small_json(a.plan,a.plan_sha256)
    need(plan['schema']=='astra.report.increment12.overlay_plan.v1','actual report12 overlay plan')
    manifest=small_json(plan['manifest']['path'],plan['manifest']['sha256']);facts=small_json(plan['source_facts']['path'],plan['source_facts']['sha256'])
    need(facts['packet_ref'] and facts['bank_count']==0 and facts['safety_acceptance'] is False and facts['gallery']['complete_groups']==0,'real closed frozen facts, partial-only gallery')
    out=Path(a.out);need(out.is_absolute() and not out.exists() and out.resolve().is_relative_to(R/'astra_report_increment12_v1'),'exclusive own browser output')
    files=dict(manifest['files'],**{'panel_manifest.json':plan['manifest']['sha256']});sources=plan['file_sources']
    need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in files),'safe declared assets')
    out.mkdir(parents=True,exist_ok=False);errors=[];modes=[];served=[];browser=None;pw=None;server=None;thread=None;closed={}
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            try:
                path=unquote(urlparse(self.path).path);need(path.startswith('/'+REL),'scoped panel route')
                leaf=path[len('/'+REL):] or 'index.html';need(leaf in files,'declared manifest asset')
                src=Path(sources[leaf]['path']) if leaf in sources else W/REL/leaf
                need(not src.is_symlink() and src.is_file() and src.stat().st_size<=100*1024**2,'bounded regular asset')
                raw=src.read_bytes();need(digest(raw)==files[leaf],'served exact bytes '+leaf)
                self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(leaf)[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
                served.append(dict(path=leaf,sha256=digest(raw)))
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as e:errors.append(str(e));self.send_error(404)
    try:
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
                        if u.scheme=='http' and u.hostname=='127.0.0.1' and u.port==port:r.continue_()
                        else:bad.append(['EXTERNAL',r.request.url]);r.abort()
                    page.route('**/*',route);response=page.goto(base,wait_until='domcontentloaded',timeout=20000)
                    need(response.status==200 and digest(response.body())==files['index.html'],'actual HTML SHA')
                    section=page.locator('#nativeAttemptIncrement12');need(section.count()==1,'one report12 section')
                    for lang in ('ZH','EN'):
                        page.locator('#r12Language'+lang).check()
                        for detail in ('Detailed','Summary'):
                            page.locator('#r12'+detail).check();need(section.locator('.r12-result').is_visible(),'summary outcome remains visible')
                            need(section.locator('.r12-detail').is_visible()==(detail=='Detailed'),'native noJS summary/detail control')
                            need(section.locator('.r12-en' if lang=='ZH' else '.r12-zh').first.is_hidden(),'native bilingual control')
                    page.locator('#r12LanguageZH').check();page.locator('#r12Detailed').check()
                    need(section.locator('.r12-gallery img').count()==4,'four exact partial views')
                    gallery=[]
                    for e in facts['gallery']['entries']:
                        stem=e['stem'];fig=section.locator('#r12_'+stem);image=fig.locator('img')
                        image.scroll_into_view_if_needed(timeout=6000);image.evaluate('(e)=>e.decode()')
                        need(image.evaluate('(e)=>e.naturalWidth===1280&&e.naturalHeight===720&&getComputedStyle(e).objectFit!=="cover"'),'original1280 uncropped display')
                        need('PARTIAL_INITIAL_ONLY' in fig.inner_text() and 'NOT_QUALIFIED' in fig.inner_text(),'visible partial-only caption')
                        name=plan['source_facts']['path'].split('/overlay/',1)[1].rsplit('/',1)[0]+'/assets/'+stem+'.png'
                        actual=page.request.get(base+name,timeout=15000)
                        need(actual.status==200 and digest(actual.body())==e['original_image']['sha256']==files[name],'actual served original PNG SHA')
                        gallery.append(dict(stem=stem,sha256=digest(actual.body()),width=1280,height=720))
                    need('43323' in section.inner_text() and '601' in section.inner_text() and 'NEW_PROPOSED_NOT_ADOPTED' in section.inner_text(),'typed taxonomy and held-state boundary visible')
                    need(section.locator('#native12EvidenceTable tbody tr').count()==5,'all evidence scope rows')
                    response=page.request.get(base+'summary.json',timeout=15000);need(response.status==200 and digest(response.body())==files['summary.json'],'actual summary SHA')
                    summary=response.json();need(summary['asof_utc']==facts['snapshot_utc'] and summary['asof_AsiaShanghai']==facts['snapshot_Asia_Shanghai'],'summary/body/facts same UTC/local instant')
                    need(page.locator('#reportAsOf time').get_attribute('datetime')==facts['snapshot_utc'],'body cutoff exact')
                    figures=('figure1','figure2','figure3','calibrationFigure','mechanismV6Figure','model05CalibrationFigure','model05ScreenFigure','model06SupplementFigure')
                    for ident in figures:need(page.locator('#'+ident).count()==1,'old science figure retained')
                    need(page.locator('#camera img').count()==15,'original15RGB retained separately')
                    anchors=page.locator('a[href^="#"]').evaluate_all('(a)=>a.map(x=>x.hash.slice(1))')
                    need(all(page.locator('[id="'+i+'"]').count()==1 for i in anchors),'all internal anchors resolve once')
                    need(not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'no page horizontal overflow')
                    section.scroll_into_view_if_needed(timeout=6000);shot=out/('report12_'+str(width)+'_'+str(js)+'.png');page.screenshot(path=str(shot),full_page=False)
                    page.locator('#r12_U_R_attempt_04').scroll_into_view_if_needed(timeout=6000);urshot=out/('UR04_'+str(width)+'_'+str(js)+'.png');page.screenshot(path=str(urshot),full_page=False)
                    need(not bad,'no browser response/JS/network errors')
                    modes.append(dict(width=width,JS=js,EN_ZH_summary_detailed=True,figures=8,original_RGB=15,new_partial_RGB=4,gallery_original_SHA=gallery,anchors=True,page_overflow=False,screenshot=dict(path=str(shot),sha256=digest(shot.read_bytes())),UR04_screenshot=dict(path=str(urshot),sha256=digest(urshot.read_bytes()))))
                finally:ctx.close()
    except Exception as e:errors.append(type(e).__name__+': '+str(e));raise
    finally:
        if browser:browser.close();closed['browser']=not browser.is_connected()
        if pw:pw.stop();closed['playwright']=True
        if server:
            server.shutdown();server.server_close();thread.join(5)
            with socket.socket() as sk:sk.settimeout(.2);closed['HTTP_server']=not thread.is_alive() and sk.connect_ex(('127.0.0.1',port))!=0
        (out/'BROWSER_REPORT_V1.json').write_text(json.dumps(dict(status='PASS_REAL_LOCAL_BROWSER' if len(modes)==4 and not errors and all(closed.values()) else 'FAILED_PRESERVED',modes=modes,errors=errors,closed=closed,served=served,frozen_snapshot_utc=facts['snapshot_utc'],publication_performed=False),ensure_ascii=False,indent=2)+'\n')
    need(len(modes)==4 and not errors and all(closed.values()),'actual complete local browser closure')

if __name__=='__main__':main()
