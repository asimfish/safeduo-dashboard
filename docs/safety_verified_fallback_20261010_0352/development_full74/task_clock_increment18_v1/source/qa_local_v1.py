"""Real LOCAL16 current text/navigation/controls; deliberately no image decoding."""
import argparse,datetime,hashlib,http.server,json,mimetypes,threading
from pathlib import Path
from urllib.parse import unquote,urlparse
import common_v1 as c
IMAGES={'.png','.svg','.jpg','.jpeg','.webp','.gif','.avif'}
PRIMARY=tuple('r18'+x for x in ('Overview','Methods','Criteria','Metrics','Failures','Evidence','Reproduction'))
def main():
    from playwright.sync_api import sync_playwright
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('candidate','candidate-sha256','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();before=c.sources();cq={'path':a.candidate,'sha256':a.candidate_sha256};h,plan,facts,manifest=c.authenticate(cq);out=Path(a.out)
    c.need(not out.exists() and out.resolve().is_relative_to(c.O),'exclusive LOCAL QA products');out.mkdir(parents=True,exist_ok=False)
    sources=dict(plan['inherited_file_sources']);sources.update(plan['file_sources']);files=dict(manifest['files']);files['panel_manifest.json']=plan['manifest']['sha256']
    served={};blocked=[];errors=[];modes=[];pw=browser=server=thread=None;closed={}
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            try:
                leaf=c.route_leaf(unquote(urlparse(self.path).path));c.need(leaf in files,'declared route')
                c.need(Path(leaf).suffix.lower() not in IMAGES,'no old/new PNG or science-plot reads in this text delta')
                src=Path(sources[leaf]['path']) if leaf in sources else c.W/c.REL/leaf
                raw=c.buffer({'path':str(src),'sha256':files[leaf]},16*1024**2)
                self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(leaf)[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);served[leaf]=files[leaf]
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as e:errors.append(str(e));self.send_error(404)
    try:
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05});thread.start();port=server.server_address[1];url='http://127.0.0.1:'+str(port)+'/'+c.REL
        pw=sync_playwright().start();browser=pw.chromium.launch(executable_path=c.SHELL,headless=True,args=c.LAUNCH_ARGS)
        for width in (1440,390):
            for js in (True,False):
                ctx=browser.new_context(viewport={'width':width,'height':1000 if width==1440 else 844},java_script_enabled=js,reduced_motion='reduce')
                try:
                    page=ctx.new_page();bad=[];page.on('pageerror',lambda e:bad.append(str(e)));page.on('response',lambda r:bad.append([r.url,r.status]) if r.status>=400 else None)
                    def route(r):
                        u=urlparse(r.request.url)
                        if r.request.resource_type=='image' or Path(u.path).suffix.lower() in IMAGES:blocked.append(r.request.url);r.abort();return
                        if u.scheme=='http' and u.hostname=='127.0.0.1' and u.port==port:r.continue_()
                        else:bad.append(['EXTERNAL',r.request.url]);r.abort()
                    page.route('**/*',route);resp=page.goto(url,wait_until='domcontentloaded',timeout=20000)
                    c.need(resp.status==200 and hashlib.sha256(resp.body()).hexdigest()==h['HTML']['sha256'],'actual frozen HTML bytes')
                    s=page.locator('#nativeAttemptIncrement18');c.need(s.count()==1 and s.locator('img').count()==0 and page.locator('#reportMain>section').first.get_attribute('id')=='nativeAttemptIncrement18','one current latest-first metadata increment')
                    human=lambda x:datetime.datetime.fromisoformat(x).astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
                    asof=page.locator('#reportAsOf').inner_text();c.need(human(facts['engineering_snapshot_utc']) in asof and human(facts['scientific_snapshot_utc']) in asof and 'Shanghai UTC+8' in asof,'separate human-readable dates to seconds; exact provenance remains in facts')
                    c.need(page.locator('.sitebar a[href="#r18Overview"]').count()==1,'top overview reaches18')
                    c.need(page.locator('.r18-nav-top a').get_attribute('href')=='#r18Overview','latest overview remains directly accessible')
                    nav=page.locator('.r18-desktop-primary' if width==1440 else '.r18-mobile-primary');c.need(nav.locator('a').evaluate_all('(ns)=>ns.map(n=>n.getAttribute("href"))')==['#'+x for x in PRIMARY[1:]],'six secondary methods/results/evidence anchors')
                    menu=page.locator('#r18MethodsNav');c.need(menu.get_attribute('open') is None,'mobile method navigation defaults closed')
                    if width==390:
                        c.need(menu.is_visible() and not page.locator('.r18-desktop-primary').is_visible() and not nav.locator('a').first.is_visible(),'mobile uses closed native details, not a long desktop TOC')
                        mb=page.locator('.r18-toc').bounding_box();c.need(mb and mb['height']<190,'mobile navigation takes less than190px before latest report')
                        menu.locator('>summary').focus();menu.locator('>summary').press('Enter');c.need(menu.get_attribute('open') is not None and nav.locator('a').first.is_visible(),'keyboard native disclosure works with or without JS');nav.locator('a[href="#r18Evidence"]').click();c.need(page.evaluate('location.hash')=='#r18Evidence','collapsed method navigation reaches evidence');menu.locator('>summary').press('Enter') if menu.locator('>summary').evaluate('(e)=>document.activeElement===e') else menu.locator('>summary').click();c.need(menu.get_attribute('open') is None,'mobile navigation closes before reading')
                    else:c.need(nav.locator('a').first.is_visible() and not menu.is_visible(),'desktop retains direct secondary navigation')
                    for ident in PRIMARY:c.need(page.locator('#'+ident).count()==1,'unique current anchor')
                    c.need(page.locator('#r18HistoryNav').get_attribute('open') is None and page.locator('#nativeAttemptIncrement17').count()==1 and page.locator('#nativeAttemptIncrement16').count()==1,'collapsed native history and old report sections retained')
                    combos=[]
                    for lang in ('ZH','EN'):
                        s.locator('label[for="r18'+lang+'"]').click();c.need(page.locator('#r18'+lang).is_checked(),'accessible native language toggle')
                        for detail in ('Detailed','Summary'):
                            s.locator('label[for="r18'+detail+'"]').click();c.need(page.locator('#r18'+detail).is_checked(),'native detail toggle')
                            c.need(s.locator('h2 .r18-title-'+lang.lower()).is_visible() and not s.locator('h2 .r18-title-'+('en' if lang=='ZH' else 'zh')).is_visible(),'one language title')
                            hero=page.locator('header.hero');active='.r18-chrome-'+lang.lower();inactive='.r18-chrome-'+('en' if lang=='ZH' else 'zh');c.need(hero.locator('h1 '+active).is_visible() and not hero.locator('h1 '+inactive).is_visible() and hero.locator('.subtitle '+active).is_visible() and not hero.locator('.subtitle '+inactive).is_visible(),'hero title/subtitle honor selected language without JavaScript')
                            ht=hero.locator('h1').inner_text();c.need(('Robot safety screening' in ht and not any('\u4e00'<=x<='\u9fff' for x in ht)) if lang=='EN' else ('机器人安全筛查' in ht and 'Robot safety' not in ht),'visible hero contains only selected-language title')
                            current_date_text=s.inner_text();c.need(facts['engineering_snapshot_utc'] not in current_date_text and facts['scientific_snapshot_utc'] not in current_date_text and human(facts['engineering_snapshot_utc']) in current_date_text,'visible section dates to seconds, no microsecond ISO paragraph')
                            radio=page.locator('#r18'+lang);box=radio.bounding_box();c.need(box and box['width']<=1 and box['height']<=1,'bare radio hidden');radio.focus();c.need(radio.evaluate('(e)=>document.activeElement===e'),'keyboard focus retained');radio.press('Tab')
                            c.need(s.locator('.r18-extra').is_visible()==(detail=='Detailed'),'summary/detailed content actually switches')
                            c.need(s.locator('.r18-cards>div').count()==5,'five acceptance dimensions')
                            cards=s.locator('.r18-cards').inner_text();c.need(all(x in cards for x in ('40,960','MODEL_ONLY','Bank0','NOT_RUN','UNOPENED')),'model/native/safety/tasks/holdout kept separate')
                            text=s.inner_text();c.need('MISSING' in text and (('20项' in text and '5项' in text) if lang=='ZH' else ('Twenty' in text and 'five' in text)),'clock checks and next schema/provider blocker immediately visible')
                            if detail=='Detailed':
                                c.need(s.locator('#r18ScopeTable tbody tr').count()==3,'checks/models/native denominators have separate rows')
                                c.need(all(x in text for x in ('10,168,233,984','20,401,094,656','10,232,860,672','5,921,033,027,584','UNKNOWN')),'exact historical capacity and unknown scope')
                            page.locator('.sitebar a[href="#r18Overview"]').click();c.need(page.evaluate('location.hash')=='#r18Overview','latest overview actual anchor')
                            over=page.evaluate('({w:innerWidth,d:document.documentElement.scrollWidth})');c.need(over['d']<=over['w']+1,'desktop/mobile document overflow')
                            s.locator('h2').scroll_into_view_if_needed();shot=out/f'report18_{width}_{js}_{lang}_{detail}.png';page.screenshot(path=str(shot),full_page=False);combo=dict(language=lang,detail=detail,overflow=over,screenshot=c.ref(shot))
                            if detail=='Summary':
                                hero.scroll_into_view_if_needed();hs=out/f'report18_{width}_{js}_{lang}_{detail}_hero.png';page.screenshot(path=str(hs),full_page=False);combo['hero_screenshot']=c.ref(hs)
                            combos.append(combo)
                    history=page.locator('#r18HistoryNav');history.locator('>summary').click();c.need(history.get_attribute('open') is not None,'history opens without JS')
                    history.locator('a[href="#nativeAttemptIncrement17"]').first.click();c.need(page.evaluate('location.hash')=='#nativeAttemptIncrement17','Report17 snapshot remains directly readable')
                    old=page.locator('#r17HistoryNav');old.locator('>summary').click();c.need(old.get_attribute('open') is not None,'old nested native history retains access');old.locator('>summary').click();history.locator('>summary').click()
                    c.need(page.locator('#nativeAttemptIncrement16 .r16-gallery-links a').count()==32,'all32 inherited V13 gallery links retained; images not reread')
                    s.locator('#r18Evidence>summary').click();c.need('UNKNOWN_PERMANENT' in s.locator('#r18Evidence').inner_text() and 'NOT_ATTESTED' in s.locator('#r18Evidence').inner_text(),'truthful terminal and backend provenance')
                    raw=s.locator('a[href$="archive_report17/index.html"]');c.need(raw.get_attribute('download') is not None,'raw historical HTML is explicitly a download, not broken archived browsing')
                    for href in s.locator('a').evaluate_all('(ns)=>ns.map(n=>n.getAttribute("href"))'):
                        if href and not href.startswith('#'):
                            rr=ctx.request.get(url+href,timeout=15000);c.need(rr.status==200 and hashlib.sha256(rr.body()).hexdigest()==files[href],'exact evidence/source/download bytes')
                    c.need(not bad,'no page/network failures '+str(bad));modes.append(dict(width=width,javascript=js,combinations=combos,native_history_JS_off_accessible=True,old32_gallery_links_preserved=True,old_gallery_image_dimensions_inherited_from_closed_Report17_NOT_redecoded=True,errors=bad));print(json.dumps({'browser_mode_closed':{'width':width,'javascript':js},'language_detail_combinations':4}),flush=True)
                finally:ctx.close()
    except Exception as e:errors.append(type(e).__name__+': '+str(e));raise
    finally:
        if browser:browser.close();closed['browser_close_returned']=True
        if pw:pw.stop();closed['playwright_stop_returned']=True
        if server:server.shutdown();server.server_close();closed['server_closed']=True
        if thread:thread.join(timeout=5);closed['server_thread_absent']=not thread.is_alive()
        c.write(out/'BROWSER_REPORT_V1.json',dict(schema='astra.report18.LOCAL.actual_browser.v1',status='PASS_LOCAL_PRESENTATION_4MODES_16COMBINATIONS' if len(modes)==4 and not errors else 'FAILED_LOCAL_PRESENTATION',candidate=cq,source_graph=c.ref(c.D/'SOURCE_GRAPH_V1.json'),HTML=h['HTML'],engineering_snapshot_utc=facts['engineering_snapshot_utc'],scientific_snapshot_utc=facts['scientific_snapshot_utc'],modes=modes,mode_count=len(modes),language_detail_combinations=sum(len(x['combinations']) for x in modes),errors=errors,served_exact_SHA=served,intentionally_blocked_historical_image_requests=len(blocked),RAW_PNG_pixel_camera_or_model_audits=0,scientific_image_payload_reads=0,source_before=before,source_after=c.sources(),source_unchanged=c.sources()==before,resource_policy='ORIGINAL_SOFTWARE_CHROMIUM_AS_INFINITY_CORE360_THREADS1_CUDA_EMPTY_DISABLE_GPU',native_GPU_SDK_Git_calls=0,publish_performed=False,closure=closed))
    c.need(len(modes)==4 and not errors and all(closed.values()) and c.sources()==before,'actual LOCAL16 cleanup and source unchanged')
if __name__=='__main__':main()
