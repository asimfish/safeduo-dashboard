"""Actual browser checks of every paired case, filters, images and downloads."""
from pathlib import Path
import argparse,asyncio,gzip,hashlib,json
from playwright.async_api import async_playwright
HERE=Path(__file__).resolve().parent
PUBLIC=Path('/home/liyufeng/safeduo-dashboard-feasible-guard-20261005/docs')/HERE.name
async def main(url,tag):
    result=json.loads((HERE/'holdout_results.json').read_text());visual=json.loads((HERE/'visual_verification.json').read_text())
    expected=json.loads(gzip.decompress((PUBLIC/'payload.json.gz').read_bytes()))
    independent=json.loads((HERE/'ASTRA_FINAL_SCORE.json').read_text())
    assert expected['nativeDuration']=={m:dict(strict=independent['counts'][m]['strict']['env_steps'],deep=independent['counts'][m]['deep']['env_steps']) for m in expected['modes']}
    expected['result']['cases']=result['cases']
    errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu'])
        page=await browser.new_page(viewport=dict(width=1440,height=1000))
        page.on('pageerror',lambda error:errors.append(str(error)))
        await page.goto(url,wait_until='domcontentloaded',timeout=180000)
        await page.wait_for_function('window.SAFEDUO_READY===true',timeout=180000)
        actual=await page.evaluate('window.SAFEDUO')
        assert actual==expected,'native browser payload differs from bound compact build'
        assert await page.locator('#heatmap button').count()==192
        text=await page.locator('#totals').inner_text()
        for t in result['totals']:
            assert str(t['violations']) in text and str(t['deep']) in text
        duration=await page.locator('#duration tr').evaluate_all('rows=>rows.slice(1).map(row=>[...row.querySelectorAll("td")].slice(1).map(c=>Number(c.textContent)))')
        assert duration==[[expected['nativeDuration'][m]['strict'],expected['nativeDuration'][m]['deep']] for m in expected['modes']]
        await page.evaluate('''() => { const draw=chart; window.__charts={}; chart=function(id,x,series,legend) {window.__charts[id]={x,series,legend};return draw(id,x,series,legend)} }''')
        checked=0;plotted=0
        for index,c in enumerate(result['cases']):
            await page.select_option('#case',str(index))
            assert f'env {c["env"]}' in await page.locator('#caseheading').inner_text()
            failures=[f for f in expected['failureDetails'] if f['seed']==c['seed'] and f['env']==c['env']]
            assert await page.locator('#failureMechanism tr').count()==1+len(failures)
            for mode,method in c['methods'].items():
                await page.select_option('#method',mode)
                bound=await page.evaluate('({heading:document.querySelector("#caseheading").textContent,case:current,method:document.querySelector("#method").value})')
                assert bound['case']==c and bound['method']==mode
                drawn=await page.evaluate('window.__charts')
                time=[(t+1)*expected['dt'] for t in method['indices']]
                assert drawn['margin']['x']==time
                assert drawn['margin']['series']==[[v[k] for v in method['curves']] for k in range(4)]
                for k,name in enumerate(['debt','velocity','increment']):
                    assert drawn[name]['x']==time and drawn[name]['series']==[[v[k] for v in method['actuation']]]
                painted=await page.evaluate('''() => ['margin','debt','velocity','increment'].map(id => {const c=document.getElementById(id);const b=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=3;i<b.length;i+=4)if(b[i])n++;return n})''')
                assert all(n>100 for n in painted),'empty native curve canvas'
                checked+=1;plotted+=7
        filter_counts={};cross_filter_counts={}
        for st in ['all',*map(str,range(-1,6))]:
            await page.select_option('#stratum',st)
            for out in ['all','rescued','new','both','neither']:
                await page.select_option('#outcome',out)
                def cat(c):
                    a=c['methods']['joint_reference']['failed'];b=c['methods']['motion_admission']['failed']
                    return ('both' if b else 'rescued') if a else ('new' if b else 'neither')
                count=sum((st=='all' or c['stratum']==int(st)) and (out=='all' or cat(c)==out) for c in result['cases'])
                assert await page.locator('#heatmap button').count()==count
                cross_filter_counts[st+'/'+out]=count
        await page.select_option('#stratum','all')
        for outcome in ('all','rescued','new','both','neither'):
            await page.select_option('#outcome',outcome)
            def category(c):
                r=c['methods']['joint_reference']['failed'];m=c['methods']['motion_admission']['failed']
                return ('both' if m else 'rescued') if r else ('new' if m else 'neither')
            count=sum(outcome=='all' or category(c)==outcome for c in result['cases'])
            assert await page.locator('#heatmap button').count()==count
            filter_counts[outcome]=count
        await page.select_option('#outcome','all')
        strata_counts={}
        for s in range(-1,6):
            await page.select_option('#stratum',str(s))
            count=sum(c['stratum']==s for c in result['cases'])
            assert await page.locator('#heatmap button').count()==count
            strata_counts[str(s)]=count
        await page.select_option('#stratum','all')
        await page.select_option('#case','0')
        coverage_checked=0
        for mode in expected['modes']:
            await page.select_option('#coverageMode',mode)
            m=next(c['measured'] for c in result['coverage'] if c['mode']==mode)
            for id,key in [('initialHeat','initial_joint'),('visitedHeat','visited_joint')]:
                rects=await page.locator('#'+id+' rect').evaluate_all('xs=>xs.map(x=>({joint:Number(x.dataset.joint),bin:Number(x.dataset.bin),count:Number(x.dataset.count)}))')
                assert rects==[dict(joint=j,bin=b,count=m[key]['counts'][j][b]) for j in range(26) for b in range(10)]
                coverage_checked+=len(rects)
            assert await page.locator('#eeRanges tr').count()==5
        image_urls=[]
        for mode in ('joint_reference','motion_admission'):
            await page.select_option('#cameraMode',mode)
            galleries=[g for g in visual['galleries'] if g['mode']==mode]
            assert await page.locator('#cameraGroup option').count()==len(galleries)
            for index,g in enumerate(galleries):
                await page.select_option('#cameraGroup',str(index))
                assert await page.locator('#gallery img').count()==9
                urls=await page.locator('#gallery img').evaluate_all('(images)=>images.map(im=>({src:im.src,alt:im.alt}))')
                assert [u['src'] for u in urls]==[expected['assetBase']+'visual/'+mode+'/'+im['path'] for im in g['images']]
                image_urls.extend(u['src'] for u in urls)
        assert len(image_urls)==visual['images']
        await page.select_option('#cameraMode','joint_reference');await page.select_option('#cameraGroup','0')
        await page.locator('#gallery').scroll_into_view_if_needed()
        await page.locator('#gallery img').evaluate_all('(images)=>images.forEach(im=>im.loading="eager")')
        await page.wait_for_function('[...document.querySelectorAll("#gallery img")].every(im=>im.complete&&im.naturalWidth===1280&&im.naturalHeight===720)',timeout=180000)
        links=await page.locator('#docs a').evaluate_all('(links)=>links.map(a=>({text:a.textContent,href:a.getAttribute("href")}))')
        assert links==[dict(text=d['title'],href=d['path']) for d in expected['docs']]
        await page.locator('h1').scroll_into_view_if_needed();await page.screenshot(path=str(HERE/f'panel_{tag}.png'),full_page=False)
        await page.set_viewport_size(dict(width=390,height=844));await page.reload(wait_until='domcontentloaded')
        await page.wait_for_function('window.SAFEDUO_READY===true',timeout=180000)
        overflow=await page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
        assert not overflow,'mobile page horizontal overflow'
        await page.screenshot(path=str(HERE/f'panel_{tag}_mobile.png'),full_page=False)
        assert not errors,errors
        await browser.close()
    receipt=dict(status='PASS_ALL_CASES_FILTERS_GALLERIES_AND_MOBILE',url=url,cases=192,method_cases_checked=checked,
         actual_curves_checked=plotted,coverage_heatmap_cells_checked=coverage_checked,combined_filters=cross_filter_counts,outcome_filters=filter_counts,stratum_filters=strata_counts,gallery_groups=visual['groups'],all_image_links_bound=len(image_urls),
         native_nine_initial_images_loaded=True,mobile_no_overflow=True,page_errors=errors,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),result_sha256=hashlib.sha256((HERE/'holdout_results.json').read_bytes()).hexdigest(),
         page_sha256=hashlib.sha256((PUBLIC/'index.html').read_bytes()).hexdigest(),
         payload_sha256=hashlib.sha256((PUBLIC/'payload.json.gz').read_bytes()).hexdigest(),
         scope='allcase selection/data/filters/galleries matched localrawderived results; allpublicfile bytes verified separately')
    with (HERE/(('live_browser_'+tag+'.json') if tag!='local' else 'local_browser.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(receipt['status'],checked,visual['images'],flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--url',required=True);a.add_argument('--tag',default='local');args=a.parse_args();asyncio.run(main(args.url,args.tag))
