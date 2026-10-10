"""Report09 frozen local bilingual journeys; exact metadata and real browser closure."""
import argparse,datetime,hashlib,http.server,json,mimetypes,os,socket,threading,time
from pathlib import Path
from urllib.parse import urlparse,unquote
from playwright.sync_api import sync_playwright
R=Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
W=Path('/home/liyufeng/safeduo-dashboard-verified-fallback-20261010-0352')
REL='docs/safety_verified_fallback_20261010_0352/development_full74/'
PUBLIC='https://asimfish.github.io/safeduo-dashboard/'+REL
SHELL='/home/liyufeng/safeduo/velocity_browser_venv_20261008_1519/browsers/chromium_headless_shell-1187/chrome-linux/headless_shell'
def need(ok,message):
 if not ok:raise ValueError(message)
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--overlay',required=True);p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=['local','public'],required=True);a=p.parse_args()
 overlay=Path(a.overlay);out=Path(a.out);need(out.is_absolute() and out.resolve().is_relative_to(R) and not out.exists(),'exclusive NAS QA product root');out.mkdir()
 raw=(overlay/'panel_manifest.json').read_bytes();need(len(raw)<=2*1024**2 and digest(raw)==a.manifest_sha256,'bound report manifest')
 pr=Path(a.plan).read_bytes();need(len(pr)<=4*1024**2 and digest(pr)==a.plan_sha256,'bound partial overlay plan');plan=json.loads(pr);sources=plan['file_sources'];manifest=json.loads(raw);files=dict(manifest['files']);files['panel_manifest.json']=a.manifest_sha256
 need(files['index.html']==digest((overlay/'index.html').read_bytes()),'bound main HTML')
 need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in files),'relative manifest leaves')
 summaryraw=Path(sources['summary.json']['path']).read_bytes();need(len(summaryraw)<=4*1024**2 and digest(summaryraw)==files['summary.json'],'bound top-level summary bytes')
 factsraw=Path(plan['source_facts']['path']).read_bytes();need(len(factsraw)<=4*1024**2 and digest(factsraw)==plan['source_facts']['sha256'],'bound typed facts bytes')
 summary=json.loads(summaryraw);facts=json.loads(factsraw);epoch=facts['snapshot_utc'];local=datetime.datetime.fromisoformat(epoch).astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
 need(summary['asof_utc']==epoch and summary['asof_AsiaShanghai']==local==facts['snapshot_Asia_Shanghai'] and datetime.datetime.fromisoformat(epoch)==datetime.datetime.fromisoformat(local),'summary UTC/local and facts exact same frozen instant')
 errors=[];modes=[];served=[];screens={};server=None;thread=None;browser=None;pw=None;browser_closed=False;pw_closed=False;server_closed=a.mode=='public'
 class Handler(http.server.BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_GET(self):
   try:
    u=unquote(urlparse(self.path).path);need(u.startswith('/'+REL),'panel route only');n=u[len('/'+REL):] or 'index.html';need(n in files,'declared payload only')
    src=Path(sources[n]['path']) if n in sources else W/REL/n
    need(not src.is_symlink() and src.stat().st_size<=100*1024**2,'bounded regular declared file');b=src.read_bytes();need(digest(b)==files[n],'exact served payload '+n)
    self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(n)[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);served.append(dict(path=n,sha256=digest(b)))
   except (BrokenPipeError,ConnectionResetError):pass
   except BaseException as e:errors.append('HTTP '+str(e));self.send_error(404)
 try:
  base=PUBLIC
  if a.mode=='local':
   server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;port=server.server_address[1];thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05});thread.start();base='http://127.0.0.1:'+str(port)+'/'+REL
  pw=sync_playwright().start();browser=pw.chromium.launch(executable_path=SHELL,headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-sandbox'])
  for width in [1440,390]:
   for js in [True,False]:
    ctx=browser.new_context(viewport=dict(width=width,height=1000 if width==1440 else 844),java_script_enabled=js,reduced_motion='reduce')
    try:
     page=ctx.new_page();bad=[];page_errors=[];page.on('pageerror',lambda e:page_errors.append(str(e)));page.on('response',lambda r:bad.append([r.url,r.status]) if r.status>=400 else None)
     def route(r):
      u=urlparse(r.request.url);ok=(u.hostname=='127.0.0.1' and u.port==port and u.scheme=='http') if a.mode=='local' else (u.hostname=='asimfish.github.io' and u.scheme=='https')
      if ok:r.continue_()
      else:bad.append(['EXTERNAL',r.request.url]);r.abort()
     page.route('**/*',route);resp=page.goto(base+'?report='+files['index.html'][:12],wait_until='domcontentloaded',timeout=15000);need(resp.status==200 and digest(resp.body())==files['index.html'],'exact main HTTP response')
     text=page.locator('body').inner_text();need('完整安全验收尚未通过' in text and '0/128' in text,'primary negative finding retained');need('Asia/Shanghai' in text and 'UTC' in text,'date boundary both zones')
     need('76' in text and '19' in text and '18' in text and '128' in text,'new closed experiments discoverable')
     need(page.locator('h1').count()==1 and page.locator('main').count()==1 and page.locator('.toc').count()==1,'single main report with TOC')
     for ident,count in [('fixed128SeedTableV3',4),('fixed128QuotaTableV3',16),('v6All16Table',16),('v6ReasonCountsTable',7)]:need(page.locator('#'+ident+' tbody tr').count()==count,'complete lookup '+ident)
     scientific=[]
     for ident in ['figure1','figure2','figure3','calibrationFigure','mechanismV6Figure','model05CalibrationFigure','model05ScreenFigure','model06SupplementFigure']:
      fig=page.locator('#'+ident);need(fig.count()==1,'scientific figure anchor '+ident);img=fig.locator('img');need(img.count()==1,'scientific image '+ident);src=img.get_attribute('src');need(src in files and img.get_attribute('alt'),'declared scientific figure')
      fig.scroll_into_view_if_needed(timeout=6000)
      for poll in range(200):
       if img.evaluate('(e)=>e.complete&&e.naturalWidth>0&&e.naturalHeight>0'):break
       time.sleep(.05)
      else:raise ValueError('scientific image not loaded '+src)
      need(img.evaluate('(e)=>getComputedStyle(e).objectFit')!='cover','scientific figure uncropped')
      scientific.append(dict(id=ident,path=src,sha256=files[src]))
     need('17:53:40' in text and '8192' in text and '8018' in text and '174' in text and 'EMPTY' in text and 'HELD' in text and 'proposal45' in text,'historical cutoff and complete MODEL_ONLY shortfall discoverable');need(page.locator('#model05CalibrationTable tbody tr').count()==2 and page.locator('#model05CellsTable tbody tr').count()==16,'complete formalcal and16cells tables')
     need(all(word in text for word in ['32768','31986','782','40960','DERIVED_PROSPECTIVE32_NOT_ADOPTED','SOURCE+INPUT']),'report06 exact ModelOnly boundaries discoverable')
     for ident,count in [('supplement06CellsTable',16),('supplement06CandidatesTable',32),('supplement06DenominatorsTable',3)]:need(page.locator('#'+ident+' tbody tr').count()==count,'complete report06 table '+ident)
     need(page.locator('#reportAsOf time').count()==1 and page.locator('#reportAsOf time').get_attribute('datetime')==epoch,'body datetime equals summary and facts UTC')
     r=page.request.get(base+'summary.json?bound='+files['summary.json'][:16],timeout=15000);need(r.status==200 and digest(r.body())==files['summary.json'],'actual browser-served summary exact SHA')
     live_summary=r.json();need(live_summary['asof_utc']==epoch and live_summary['asof_AsiaShanghai']==local,'each actual mode same served UTC/local instant')
     latest=page.locator('#nativeAttemptIncrement07');need(latest.count()==1,'single report07 section');latesttext=latest.inner_text()
     need(all(word in latesttext for word in ['21:30:27.935059','NOT_EVAL','NOT_STARTED','9400','11000','21:29:22','BANK0','UNKNOWN','8192','32768','782']),'exact07 historical truth and unpromoted model scope')
     need(page.locator('#native07AttemptTable tbody tr').count()==2 and page.locator('#native07EngineeringTable tbody tr').count()==3,'nativefailed versus SOURCE_CPU engineering rows')
     need('不是实时' in page.locator('#reportAsOf').inner_text() and '2026-10-11 05:30:27.935059' in page.locator('#reportAsOf').inner_text(),'historicalUTC/AsiaShanghai boundary')
     need('native0' in page.locator('#native07Executive').inner_text() and 'outer1' in page.locator('#native07Executive').inner_text() and 'tool1' in page.locator('#native07Executive').inner_text(),'native0 does not promote failed measurement')
     newest=page.locator('#nativeAttemptIncrement08');need(newest.count()==1,'single report08 failed metadata section');newtext=newest.inner_text()
     need(all(word in newtext for word in ['22:03:37.385121','22:04:35','−11','raw139','1086.5235147180501','1200','MISSING','24','UNKNOWN','BANK0','8192','32768','782','NOT_RUN','UNOPENED']),'actual08 failure and missing evidence truth')
     need(page.locator('#native08AttemptTable tbody tr').count()==2 and page.locator('#native08GapsTable tbody tr').count()==4 and page.locator('#native08All32Table tbody tr').count()==32,'same32 failure gaps and complete metadata denominator')
     need('06:03:37.385121' in page.locator('#reportAsOf').inner_text() and '不是实时' in page.locator('#reportAsOf').inner_text(),'new08 correct frozen local snapshot')
     need('−11' in page.locator('#native08Executive').inner_text() and 'outer1' in page.locator('#native08Executive').inner_text() and 'tool1' in page.locator('#native08Executive').inner_text(),'actual failed native remains failed')
     increment=page.locator('#nativeAttemptIncrement09');need(increment.count()==1,'single09 frozen metadata section');t09=increment.inner_text()
     need(all(word in t09 for word in ['23:22:37.346130','07:22:37.346130','13641','919','920','orphan','allocated13','micro0','NOT_EVAL','UNKNOWN','PENDING','profile_runtime_v5','69fc','fb0','f577','8192','32768','40960','BANK0','NOT_RUN','UNOPENED']),'exact09 evidence and absent acceptance')
     need(increment.locator('[lang=en]').count()>=5 and '部分证据文件核验完成' in t09,'Chinese and English accessible frozen limits')
     for ident,count in [('native09InventoryTable',4),('native09CPUAndRuntimeTable',3),('native09ShapesTable',6)]:need(page.locator('#'+ident+' tbody tr').count()==count,'complete09 metadata table '+ident)
     need('07:22:37.346130' in page.locator('#reportAsOf').inner_text() and '不是实时' in page.locator('#reportAsOf').inner_text(),'09 correct exact UTC-derived local top-level timestamp')
     need(live_summary['increment09']['snapshot_utc']==epoch and live_summary['increment09']['snapshot_Asia_Shanghai']==local and live_summary['increment09']['allocated_rows']==13 and live_summary['increment09']['actual_committed_rows']==1 and live_summary['increment09']['CPU_API_actual_exit']==1 and live_summary['increment09']['CPU_API_outer_wait'] is None and not live_summary['increment09']['CPU_API_native_started'],'each mode serves exact09 summary with failedCPU and null outer')
     need(live_summary['increment09']['bank_count']==0 and live_summary['increment09']['camera_qualification']=='UNKNOWN_NOT_AUDITED' and live_summary['increment09']['CUDA_Kit_loaded_identity']=='PENDING_ACTUAL_NATIVE' and not live_summary['increment09']['crash_cause_repair_or_speedup_demonstrated'],'pending native identity never promoted')
     projection='native_attempt_increment09_v1/PROCESS_LOCAL_BINDING_SELECTED_V1.json';r=page.request.get(base+projection+'?bound='+files[projection][:16],timeout=15000);need(r.status==200 and digest(r.body())==files[projection],'actual selected binding projection SHA')
     binding=r.json();need(binding['query_stdout_and_memory_telemetry_copied'] is False and 'stdout' not in binding and 'query_argv' not in binding and binding['native_loaded_library_readback']=='PENDING_ACTUAL_NATIVE','no copied query telemetry or native identity fiction')
     gallery=page.locator('#pilotSelectedViews');need(gallery.locator('img').count()==15,'all15 original images')
     loaded=[]
     for i in range(15):
      fig=gallery.locator('figure').nth(i);img=fig.locator('img');src=img.get_attribute('src');need(src in files and fig.get_attribute('data-sha256')==files[src],'exact original figure identity')
      img.scroll_into_view_if_needed(timeout=6000)
      for poll in range(200):
       if img.evaluate('(e)=>e.complete&&e.naturalWidth===1280&&e.naturalHeight===720'):break
       time.sleep(.05)
      else:raise ValueError('original image not loaded '+src)
      need(img.get_attribute('alt') and 'INITIAL_ONLY' in fig.inner_text() and 'proposal92' in fig.inner_text(),'explicit image scope and alt')
      style=img.evaluate('(e)=>({fit:getComputedStyle(e).objectFit,w:e.getBoundingClientRect().width,h:e.getBoundingClientRect().height})');need(style['fit']!='cover' and abs(style['w']/style['h']-1280/720)<.02,'uncropped original aspect')
      loaded.append(dict(path=src,sha256=files[src]))
     for group in ['F_L','F_R','U_L','U_R','wide']:need(page.locator('#pilotViews_'+group+' img').count()==3,'five native image groups')
     links=page.locator('main a,.toc a').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("href"))')
     for href in links:
      if not href:continue
      u=urlparse(href)
      if u.scheme or href.startswith('../'):continue
      if href.startswith('#'):need(page.locator(href).count()==1,'unique internal anchor '+href)
      else:need(unquote(u.path) in files,'declared download '+href)
     overflow=page.evaluate('({w:innerWidth,scroll:document.documentElement.scrollWidth})');need(overflow['scroll']<=width+1,'no document horizontal overflow')
     tag=str(width)+'_'+('js' if js else 'nojs');page.locator('header.hero').scroll_into_view_if_needed();top=out/(tag+'_top.png');page.screenshot(path=str(top),full_page=False,animations='disabled');screens[str(top)]=digest(top.read_bytes())
     for ident in ['mechanismV6Figure','model05CalibrationFigure','model05ScreenFigure','model06SupplementFigure']:
      fp=out/(tag+'_'+ident+'.png');page.locator('#'+ident).screenshot(path=str(fp),animations='disabled',timeout=10000);screens[str(fp)]=digest(fp.read_bytes())
     sp=out/(tag+'_native07.png');page.locator('#nativeAttemptIncrement07').screenshot(path=str(sp),animations='disabled',timeout=10000);screens[str(sp)]=digest(sp.read_bytes())
     newestshot=out/(tag+'_native08.png');newest.screenshot(path=str(newestshot),animations='disabled',timeout=10000);screens[str(newestshot)]=digest(newestshot.read_bytes())
     incshot=out/(tag+'_native09.png');increment.screenshot(path=str(incshot),animations='disabled',timeout=10000);screens[str(incshot)]=digest(incshot.read_bytes())
     gp=out/(tag+'_all15.png');gallery.screenshot(path=str(gp),animations='disabled',timeout=10000);screens[str(gp)]=digest(gp.read_bytes())
     need(not bad and not page_errors,'browser/network errors');modes.append(dict(width=width,javascript=js,original_images=loaded,scientific_figures=scientific,mechanism_pair_rows=16,criteria_rows=7,seed_rows=4,quota_rows=16,overflow=overflow,links_checked=len(links),page_errors=page_errors,failed_responses=bad));print('REPORT_MODE_CLOSED',tag,flush=True)
    finally:ctx.close()
 except BaseException as e:errors.append(type(e).__name__+': '+str(e));raise
 finally:
  if browser:
   try:browser.close();browser_closed=not browser.is_connected()
   except BaseException as e:errors.append('close browser '+str(e))
  if pw:
   try:pw.stop();pw_closed=True
   except BaseException as e:errors.append('close playwright '+str(e))
  if server:
   server.shutdown();server.server_close();thread.join(5)
   with socket.socket() as s:s.settimeout(.2);server_closed=s.connect_ex(('127.0.0.1',port))!=0 and not thread.is_alive()
  report=dict(summary_body_facts_datetime_consistent=True,summary_sha256=files['summary.json'],frozen_snapshot_utc=epoch,frozen_snapshot_local=local,status='PASS_REAL_BROWSER' if len(modes)==4 and not errors and browser_closed and pw_closed and server_closed else 'FAILED_PRESERVED',mode=a.mode,manifest_sha256=a.manifest_sha256,main_sha256=files['index.html'],modes=modes,errors=errors,screenshots=screens,served=served,browser_closed=browser_closed,playwright_closed=pw_closed,preview_closed=server_closed)
  (out/'BROWSER_REPORT_V1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 need(report['status']=='PASS_REAL_BROWSER','complete actual browser report gate')
if __name__=='__main__':main()
