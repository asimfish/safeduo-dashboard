"""One real mobile/noJS check for the corrected existing evidence appendix."""
import argparse,hashlib,http.server,json,mimetypes,socket,threading
from pathlib import Path
from urllib.parse import urlparse,unquote
from playwright.sync_api import sync_playwright
import qa_local_v1 as oldqa
def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('plan','plan-sha256','out'):ap.add_argument('--'+key,required=True)
    a=ap.parse_args();p=oldqa.small_json(a.plan,a.plan_sha256);m=oldqa.small_json(p['manifest']['path'],p['manifest']['sha256']);facts=oldqa.small_json(p['source_facts']['path'],p['source_facts']['sha256'])
    out=Path(a.out);oldqa.need(not out.exists() and out.is_absolute() and out.resolve().is_relative_to(oldqa.R/'astra_report_increment13_v1'),'exclusive phone check');out.mkdir(parents=True,exist_ok=False)
    files=dict(m['files'],**{'panel_manifest.json':p['manifest']['sha256']});errors=[];closed={};browser=None;pw=None;server=None;thread=None;result={}
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            try:
                path=unquote(urlparse(self.path).path);oldqa.need(path.startswith('/'+oldqa.REL),'scoped panel route');name=path[len('/'+oldqa.REL):] or 'index.html';oldqa.need(name in files,'declared asset')
                src=Path(p['file_sources'][name]['path']) if name in p['file_sources'] else oldqa.W/oldqa.REL/name
                oldqa.need(not src.is_symlink() and src.stat().st_size<=100*1024**2,'bounded regular asset');raw=src.read_bytes();oldqa.need(hashlib.sha256(raw).hexdigest()==files[name],'served exactSHA')
                self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(name)[0] or 'application/octet-stream');self.end_headers();self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as error:errors.append(str(error));self.send_error(404)
    try:
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05});thread.start();port=server.server_address[1];base='http://127.0.0.1:'+str(port)+'/'+oldqa.REL
        pw=sync_playwright().start();browser=pw.chromium.launch(executable_path=oldqa.SHELL,headless=True,args=['--disable-gpu','--disable-dev-shm-usage','--no-sandbox']);ctx=browser.new_context(viewport=dict(width=390,height=844),java_script_enabled=False,reduced_motion='reduce')
        try:
            page=ctx.new_page();bad=[];page.on('pageerror',lambda e:bad.append(str(e)))
            def route(r):
                u=urlparse(r.request.url)
                if u.scheme=='http' and u.hostname=='127.0.0.1' and u.port==port:r.continue_()
                else:bad.append(['external',r.request.url]);r.abort()
            page.route('**/*',route);response=page.goto(base,wait_until='domcontentloaded',timeout=20000);oldqa.need(response.status==200 and hashlib.sha256(response.body()).hexdigest()==p['changed_files']['index.html'],'actual corrected HTML SHA')
            section=page.locator('#nativeAttemptIncrement13');section.locator('#r13EvidenceAppendix>summary').click();section.locator('#r13ExpertEvidenceTable').scroll_into_view_if_needed(timeout=6000)
            for lang in ['ZH','EN']:
                page.locator('#r13Language'+lang).check();text=section.locator('#r13ExpertEvidenceTable').inner_text();oldqa.need('1024' in text and '128' in text and 'forearm/wrist2' in text and 'wide128' not in text,'exact corrected wide/critical thresholds in '+lang)
            literal=json.loads(section.locator('#r13ExactEvidence').inner_text());oldqa.need(literal['V9']['registered_campaign_owned_fresh_absent']==91 and literal['V9']['registered_identity_breakdown']==dict(native=89,CPU_check=1,outer=1) and 'registered_native_owned_fresh_absent' not in literal['V9'],'campaign identity breakdown, no native91 claim')
            oldqa.need(not page.evaluate('document.documentElement.scrollWidth>innerWidth+1'),'phone page no overflow');page.locator('#r13LanguageZH').check();section.locator('#r13ExpertEvidenceTable').scroll_into_view_if_needed(timeout=6000);shot=out/'report13_repair_camera_390_noJS.png';page.screenshot(path=str(shot),full_page=False)
            oldqa.need(not bad and not errors,'one actual phone check clean');result=dict(width=390,JS=False,EN_ZH_corrected_appendix=True,page_overflow=False,screenshot=dict(path=str(shot),sha256=hashlib.sha256(shot.read_bytes()).hexdigest()),new_phone_runs=1,new_fourmode_runs=0,old_four_modes_reused_by_exact_inert_delta=True)
        finally:ctx.close()
    except Exception as error:errors.append(type(error).__name__+': '+str(error));raise
    finally:
        if browser:browser.close();closed['browser']=not browser.is_connected()
        if pw:pw.stop();closed['playwright']=True
        if server:
            server.shutdown();server.server_close();thread.join(5)
            with socket.socket() as sk:sk.settimeout(.2);closed['HTTP_server']=not thread.is_alive() and sk.connect_ex(('127.0.0.1',port))!=0
        (out/'MOBILE_WORDING_REPAIR_QA_V1.json').write_text(json.dumps(dict(status='PASS_REAL_SINGLE_MOBILE_WORDING_CHECK' if result and not errors and all(closed.values()) else 'FAILED_PRESERVED',result=result,closed=closed,errors=errors,plan=dict(path=a.plan,sha256=a.plan_sha256),frozen_snapshot_utc=facts['snapshot_utc'],camera_qualification=False,publish_performed=False),ensure_ascii=False,indent=2)+'\n')
    oldqa.need(result and not errors and all(closed.values()),'phone true closure')
if __name__=='__main__':main()
