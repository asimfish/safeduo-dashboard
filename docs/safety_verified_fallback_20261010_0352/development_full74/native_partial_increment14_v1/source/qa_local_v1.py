"""Fresh bounded software browser, desktop/mobile JS/noJS, bilingual controls."""
import argparse,hashlib,http.server,json,mimetypes,threading
from pathlib import Path
from urllib.parse import unquote,urlparse
import common_v1 as c
SHELL='/home/liyufeng/safeduo/velocity_browser_venv_20261008_1519/browsers/chromium_headless_shell-1187/chrome-linux/headless_shell'
def main():
    from playwright.sync_api import sync_playwright
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ['plan','plan-sha256','out']:p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=c.read({'path':a.plan,'sha256':a.plan_sha256});facts=c.read(plan['source_facts']);manifest=c.read(plan['manifest']);out=Path(a.out)
    c.need(plan['schema']=='astra.report.increment14.overlay_plan.v1' and not out.exists() and out.resolve().is_relative_to(c.O),'actual14 plan and exclusive browser root');out.mkdir(parents=True,exist_ok=False)
    sources=plan['file_sources'];files=dict(manifest['files'],panel_manifest__unused='');files.pop('panel_manifest__unused');files['panel_manifest.json']=plan['manifest']['sha256']
    errors=[];served={};modes=[];pw=browser=server=thread=None
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            try:
                path=unquote(urlparse(self.path).path);c.need(path.startswith('/'+c.REL),'scoped local route');leaf=path[len('/'+c.REL):] or 'index.html';c.need(leaf in files,'declared asset '+leaf)
                src=Path(sources[leaf]['path']) if leaf in sources else c.W/c.REL/leaf
                raw=c.buffer({'path':str(src),'sha256':files[leaf]},100*1024**2)
                self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(leaf)[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);served[leaf]=files[leaf]
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as e:errors.append(str(e));self.send_error(404)
    closed={}
    try:
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05});thread.start();port=server.server_address[1];base='http://127.0.0.1:'+str(port)+'/'+c.REL
        pw=sync_playwright().start();browser=pw.chromium.launch(executable_path=SHELL,headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-sandbox'])
        for width in [1440,390]:
            for js in [True,False]:
                ctx=browser.new_context(viewport={'width':width,'height':1000 if width==1440 else 844},java_script_enabled=js,reduced_motion='reduce')
                try:
                    page=ctx.new_page();bad=[];page.on('pageerror',lambda e:bad.append(str(e)));page.on('response',lambda r:bad.append([r.url,r.status]) if r.status>=400 else None)
                    def route(r):
                        u=urlparse(r.request.url)
                        if u.scheme=='http' and u.hostname=='127.0.0.1' and u.port==port:r.continue_()
                        else:bad.append(['EXTERNAL',r.request.url]);r.abort()
                    page.route('**/*',route);resp=page.goto(base,wait_until='domcontentloaded',timeout=20000);c.need(resp.status==200 and hashlib.sha256(resp.body()).hexdigest()==files['index.html'],'served exact candidate HTML')
                    section=page.locator('#nativeAttemptIncrement14');c.need(section.count()==1,'single Report14 section');checks=[]
                    for lang in ['ZH','EN']:
                        page.locator('#r14Language'+lang).check()
                        for detail in ['Detailed','Summary']:
                            page.locator('#r14'+detail).check();c.need(section.locator('.r14-'+lang.lower()).first.is_visible(),'language control without JS');c.need(section.locator('.r14-detail').is_visible()==(detail=='Detailed'),'detail control without JS')
                            c.need(page.locator('#r14AcceptanceIndicators').inner_text().find('32 / 32')>=0,'initial32 card');c.need(page.locator('#r14AcceptanceIndicators').inner_text().find('0 / 32')>=0,'fullchallenge zero card')
                            over=page.evaluate('({w:innerWidth,d:document.documentElement.scrollWidth})');c.need(over['d']<=over['w']+1,'whole document horizontal overflow')
                            section.locator('h2').first.scroll_into_view_if_needed();shot=out/f'report14_{width}_{js}_{lang}_{detail}.png';page.screenshot(path=str(shot),full_page=False)
                            checks.append({'language':lang,'detail':detail,'overflow':over,'screenshot':c.ref(shot)})
                    page.locator('#r14LanguageZH').check();page.locator('#r14Detailed').check()
                    newimages=section.locator('img');c.need(newimages.count()==6,'two scientific plots plus four original previews')
                    for i in range(newimages.count()):
                        im=newimages.nth(i);im.scroll_into_view_if_needed();im.evaluate('(i)=>i.decode()');c.need(im.evaluate('(i)=>i.naturalWidth>0 && i.naturalHeight>0'),'actual image decoded')
                    page.locator('#r14ScopeTable').screenshot(path=str(out/f'scopes_{width}_{js}.png'))
                    page.locator('#r14All32').screenshot(path=str(out/f'figure32_{width}_{js}.png'))
                    page.locator('#r14EvidenceAppendix > summary').click();c.need(page.locator('#r14EvidenceAppendix').get_attribute('open') is not None,'evidence disclosure works')
                    c.need(page.locator('#r14EvidenceAppendix').inner_text().find('UNKNOWN')>=0,'unknown limits in appendix')
                    page.goto(base+c.PREFIX+'/gallery/index.html',wait_until='domcontentloaded');c.need(page.locator('ol li a').count()==32,'all32 gallery links')
                    for lane in [17,27]:
                        page.goto(base+c.PREFIX+f'/gallery/env_{lane:03d}.html',wait_until='domcontentloaded');ims=page.locator('img');c.need(ims.count()==15,'full15 original group page')
                        for i in range(15):
                            im=ims.nth(i);im.scroll_into_view_if_needed();im.evaluate('(i)=>i.decode()');c.need(im.evaluate('(i)=>i.naturalWidth===1280 && i.naturalHeight===720'),'real full-size original rendered')
                        over=page.evaluate('({w:innerWidth,d:document.documentElement.scrollWidth})');c.need(over['d']<=over['w']+1,'gallery horizontal overflow')
                    c.need(not bad,'page/network errors '+str(bad));modes.append(dict(width=width,javascript=js,EN_ZH_detailed_summary=checks,new_images_decoded=6,gallery_groups_rendered=[17,27],original_gallery_images_per_group=15,errors=bad))
                    print(json.dumps({'browser_mode_closed':{'width':width,'javascript':js}}),flush=True)
                finally:ctx.close()
    except Exception as e:
        errors.append(type(e).__name__+': '+str(e));raise
    finally:
        if browser:browser.close();closed['browser_close_returned']=True
        if pw:pw.stop();closed['playwright_stop_returned']=True
        if server:server.shutdown();server.server_close();closed['server_closed']=True
        if thread:thread.join(timeout=5);closed['server_thread_absent']=not thread.is_alive()
        c.write(out/'BROWSER_REPORT_V1.json',dict(schema='astra.report14.local.browser.v1',status='PASS_LOCAL_PRESENTATION_ONLY' if len(modes)==4 and not errors else 'FAILED_LOCAL_PRESENTATION',plan={'path':a.plan,'sha256':a.plan_sha256},HTML_SHA=files['index.html'],frozen_snapshot_utc=facts['snapshot_utc'],modes=modes,errors=errors,served_exact_SHA=served,resource_policy='REUSED_SOFTWARE_CHROMIUM_AS_INFINITY_DISABLE_GPU_CORE_WALL360_THREADS1',actual_GPU_or_native_launch=False,new_RAW_math_or_mask_audit=False,all480_byte_copies_verified_by_build_not_all480_browser_decoded=True,closure=closed))
    c.need(len(modes)==4 and not errors,'four actual modes closed')
if __name__=='__main__':main()
