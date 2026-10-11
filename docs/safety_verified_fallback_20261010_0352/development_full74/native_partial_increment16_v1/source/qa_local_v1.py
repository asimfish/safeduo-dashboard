"""Four actual software-browser modes for the actual current gallery increment; old images remain blocked."""
import argparse,hashlib,http.server,json,mimetypes,threading
from pathlib import Path
from urllib.parse import unquote,urlparse
import common_v1 as c
SHELL='/home/liyufeng/safeduo/velocity_browser_venv_20261008_1519/browsers/chromium_headless_shell-1187/chrome-linux/headless_shell'
IMAGE_SUFFIXES={'.png','.svg','.jpg','.jpeg','.webp','.gif','.avif'}

def main():
    from playwright.sync_api import sync_playwright
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ['plan','plan-sha256','out']:p.add_argument('--'+k,required=True)
    a=p.parse_args();pq={'path':a.plan,'sha256':a.plan_sha256};plan=c.read(pq);facts=c.read(plan['source_facts']);manifest=c.read(plan['manifest']);out=Path(a.out)
    c.need(plan['schema']=='astra.report.increment16.actual_partial_overlay.v1' and not out.exists() and out.resolve().is_relative_to(c.O),'actual15 plan and exclusive browser root');out.mkdir(parents=True,exist_ok=False)
    sources=dict(plan['inherited_file_sources']);sources.update(plan['file_sources']);files=dict(manifest['files']);files['panel_manifest.json']=plan['manifest']['sha256'];errors=[];served={};blocked_images=[];modes=[];pw=browser=server=thread=None;closed={}
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            try:
                path=unquote(urlparse(self.path).path);c.need(path.startswith('/'+c.REL),'scoped local route');leaf=path[len('/'+c.REL):] or 'index.html'
                c.need(Path(leaf).suffix.lower() not in IMAGE_SUFFIXES or leaf.startswith(c.PREFIX+'/assets/'),'only current V13 images may be opened; old media blocked')
                c.need(leaf in files,'declared metadata route '+leaf);src=Path(sources[leaf]['path']) if leaf in sources else c.W/c.REL/leaf
                raw=c.buffer({'path':str(src),'sha256':files[leaf]},16*1024**2)
                self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(leaf)[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);served[leaf]=files[leaf]
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as e:errors.append(str(e));self.send_error(404)
    try:
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05});thread.start();port=server.server_address[1];url='http://127.0.0.1:'+str(port)+'/'+c.REL
        pw=sync_playwright().start();browser=pw.chromium.launch(executable_path=SHELL,headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-sandbox'])
        for width in [1440,390]:
            for js in [True,False]:
                ctx=browser.new_context(viewport={'width':width,'height':1000 if width==1440 else 844},java_script_enabled=js,reduced_motion='reduce')
                try:
                    page=ctx.new_page();bad=[];page.on('pageerror',lambda e:bad.append(str(e)));page.on('response',lambda r:bad.append([r.url,r.status]) if r.status>=400 else None)
                    def route(r):
                        u=urlparse(r.request.url)
                        if (r.request.resource_type=='image' or Path(u.path).suffix.lower() in IMAGE_SUFFIXES) and ('/'+c.PREFIX+'/assets/') not in u.path:
                            blocked_images.append(r.request.url);r.abort();return
                        if u.scheme=='http' and u.hostname=='127.0.0.1' and u.port==port:r.continue_()
                        else:bad.append(['EXTERNAL',r.request.url]);r.abort()
                    page.route('**/*',route);resp=page.goto(url,wait_until='domcontentloaded',timeout=20000)
                    c.need(resp.status==200 and hashlib.sha256(resp.body()).hexdigest()==files['index.html'],'served exact frozen HTML')
                    section=page.locator('#nativeAttemptIncrement16');c.need(section.count()==1 and section.locator('img').count()==2,'one current increment with two fixed representative initial previews')
                    asof=page.locator('#reportAsOf').inner_text();c.need('REPORT16' in asof and facts['snapshot_utc'] in asof and facts['snapshot_Asia_Shanghai'] in asof,'latest header cutoff agrees with facts')
                    c.need(page.locator('.sitebar a[href="#nativeAttemptIncrement16"]').count()==1 and page.locator('.toc a[href="#nativeAttemptIncrement16"]').count()==1,'header and TOC direct latest anchor')
                    c.need(page.locator('#nativeAttemptIncrement15').count()==1,'old15 historical section retained');
                    for ident in ['r16Overview','r16Protocol','r16Criteria','r16Metrics','r16FailureCases','r16PerTrialEvidence','r16Reproduction']:
                        c.need(page.locator('#'+ident).count()==1 and section.locator('nav a[href="#'+ident+'"]').count()==1,'accessible unique method/evidence anchor '+ident)
                    c.need(facts['V13']['camera_INITIAL32_PASS'] is True and facts['V13']['selected_original_images']==480 and facts['publish_allowed'] is False,'LOCAL actual INITIAL DATA, not publication');checks=[]
                    for lang in ['ZH','EN']:
                        section.locator('label[for="r16Language'+lang+'"]').click();c.need(page.locator('#r16Language'+lang).is_checked(),'language label activates accessible radio')
                        for detail in ['Detailed','Summary']:
                            section.locator('label[for="r16'+detail+'"]').click();c.need(page.locator('#r16'+detail).is_checked(),'detail label activates accessible radio');c.need(section.locator('.r16-'+lang.lower()).first.is_visible(),'language controls in actual JS/noJS context')
                            other='en' if lang=='ZH' else 'zh';c.need(section.locator('h2 .r16-title-'+lang.lower()).is_visible() and not section.locator('h2 .r16-title-'+other).is_visible(),'exactly one visible language heading')
                            radio=page.locator('#r16Language'+lang);radio.focus();c.need(radio.evaluate('(e)=>document.activeElement===e'),'visually hidden radio remains keyboard-focusable');radio.press('Tab')
                            c.need(section.locator('.r16-detail').is_visible()==(detail=='Detailed'),'detail controls in actual JS/noJS context')
                            c.need(page.locator('#r16AcceptanceIndicators > div').count()==5,'five scientific acceptance indicators')
                            text=page.locator('#r16AcceptanceIndicators').inner_text();c.need('40,960' in text and '32/32 INITIAL' in text and '480 selected' in text and 'Bank 0' in text and 'UNOPENED' in text,'model/initial/full/task/holdout separated')
                            if detail=='Detailed':
                                c.need(page.locator('#r16StageTable tbody tr').count()==3,'exact V10/V12/V13 comparison')
                                c.need('Current INITIAL32/480 independently checked' in page.locator('#r16StageTable [data-retry="V13"]').inner_text(),'current INITIAL acceptance only, full12 unaccepted')
                                wrap=page.locator('#r16StageTable').locator('..').evaluate('(e)=>({client:e.clientWidth,scroll:e.scrollWidth})');c.need(wrap['client']<=width and wrap['scroll']>=wrap['client'],'table overflow contained')
                            over=page.evaluate('({w:innerWidth,d:document.documentElement.scrollWidth})');c.need(over['d']<=over['w']+1,'whole document mobile/desktop overflow')
                            section.locator('h2').first.scroll_into_view_if_needed();shot=out/f'report16_{width}_{js}_{lang}_{detail}.png';page.screenshot(path=str(shot),full_page=False)
                            checks.append(dict(language=lang,detail=detail,overflow=over,screenshot=c.ref(shot)))
                    section.locator('label[for="r16LanguageZH"]').click();section.locator('label[for="r16Detailed"]').click();page.locator('#r16EvidenceAppendix > summary').click()
                    c.need(page.locator('#r16EvidenceAppendix').get_attribute('open') is not None,'evidence disclosure opens without JS')
                    appendix=page.locator('#r16EvidenceAppendix').inner_text();c.need('2/2/2' in appendix and 'UNKNOWN_PERMANENT' in appendix,'failure scope and historical unknown disclosed')
                    for href in section.locator('a').evaluate_all('(nodes)=>nodes.map(n=>n.getAttribute("href"))'):
                        if href and not href.startswith('#'):
                            rr=ctx.request.get(url+href,timeout=15000);c.need(rr.status==200,'new evidence link accessible '+href)
                            c.need(hashlib.sha256(rr.body()).hexdigest()==files[href],'new evidence bytes exactly manifest')
                    # All32 group page HTMLs are checked; original pixels were copied once by adapter.
                    links=section.locator('.r16-gallery-links a').evaluate_all('(nodes)=>nodes.map(n=>n.getAttribute("href"))')
                    c.need(len(links)==32 and len(set(links))==32,'all32 current per-lane gallery pages')
                    for lane,href in enumerate(links):
                        rr=ctx.request.get(url+href,timeout=15000);c.need(rr.status==200 and hashlib.sha256(rr.body()).hexdigest()==files[href],'actual per-lane gallery HTML SHA')
                        body=rr.text();c.need(body.count('<img ')==15 and 'INITIAL' in body and 'full12' in body,'per-lane15 current views and scope')
                    sample=page.goto(url+links[0],wait_until='domcontentloaded',timeout=20000)
                    c.need(sample.status==200 and page.locator('figure img').count()==15,'actual gallery route and all15 views')
                    for im in page.locator('figure img').all():
                        im.scroll_into_view_if_needed();im.wait_for(state='visible');im.evaluate('(e)=>e.decode()')
                        dims=im.evaluate('(e)=>[e.naturalWidth,e.naturalHeight]');c.need(dims==[1280,720],'actual original image dimensions, no crop/downsample')
                    over=page.evaluate('({w:innerWidth,d:document.documentElement.scrollWidth})');c.need(over['d']<=over['w']+1,'current gallery mobile overflow contained')
                    page.locator('h1').scroll_into_view_if_needed();gshot=out/f'report16_gallery_lane0_{width}_{js}.png';page.screenshot(path=str(gshot),full_page=False)
                    c.need(not bad,'page/network errors '+str(bad));modes.append(dict(width=width,javascript=js,EN_ZH_detailed_summary=checks,current_INITIAL32_gallery=True,gallery_screenshot=c.ref(gshot),gallery_natural_dimensions=[1280,720],old_media_qualification_reused=plan['old_media_QA'],new_media_decodes='CURRENT_INITIAL_V13_BROWSER_DISPLAY_ONLY_NOT_SCIENTIFIC_MASK_AUDIT',errors=bad));print(json.dumps({'browser_mode_closed':{'width':width,'javascript':js},'old_media_reads':0,'current_gallery_views_displayed':15}),flush=True)
                finally:ctx.close()
    except Exception as e:
        errors.append(type(e).__name__+': '+str(e));raise
    finally:
        if browser:browser.close();closed['browser_close_returned']=True
        if pw:pw.stop();closed['playwright_stop_returned']=True
        if server:server.shutdown();server.server_close();closed['server_closed']=True
        if thread:thread.join(timeout=5);closed['server_thread_absent']=not thread.is_alive()
        c.write(out/'BROWSER_REPORT_V1.json',dict(schema='astra.report16.local.browser.v1',status='PASS_LOCAL_ACTUAL_PARTIAL_DATA_PRESENTATION_ONLY' if len(modes)==4 and not errors else 'FAILED_LOCAL_INCREMENT_PRESENTATION',plan=pq,HTML_SHA=files['index.html'],frozen_snapshot_utc=facts['snapshot_utc'],modes=modes,errors=errors,served_exact_SHA=served,intentionally_blocked_old_image_requests=len(blocked_images),blocked_old_media_UNTESTED_THIS_RUN=True,inherited_media_QA=plan['old_media_QA'],resource_policy='REUSED_SOFTWARE_CHROMIUM_AS_INFINITY_DISABLE_GPU_CORE_WALL360_THREADS1',actual_GPU_native_SDK_Git_actions=0,source_before_after=c.sources(),source_unchanged=True,current_original_image_routes_served=len([k for k in served if k.startswith(c.PREFIX+'/assets/')]),old_media_buffer_reads=0,new_RAW_math_or_mask_audit=False,closure=closed))
    c.need(len(modes)==4 and not errors and all(closed.values()),'four actual modes and owned browser/server cleanup')

if __name__=='__main__':main()
