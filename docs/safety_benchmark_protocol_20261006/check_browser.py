"""Actual Chromium journeys for all325 pair choices and research downloads."""
from pathlib import Path
import argparse,csv,io,json,urllib.request
from playwright.sync_api import sync_playwright
def get(url):
    with urllib.request.urlopen(url,timeout=30) as r:return r.read()
def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);errors=[];checks=[]
    with sync_playwright() as api:
        browser=api.chromium.launch(headless=True)
        for width in (1440,390):
            page=browser.new_page(viewport=dict(width=width,height=950));page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(a.url,wait_until='networkidle');page.wait_for_function("document.body.dataset.ready==='true'")
            assert page.locator('#outcomes tr').count()==4 and page.locator('#coverage tr').count()==4
            assert page.locator('#pair option').count()==325 and page.locator('#mode option').count()==4
            for category,n in [('random_motion',6),('real_task',8),('robustness',6),('all',20)]:
                page.select_option('#category',category);assert page.locator('#matrix tr').count()==n
            swept=page.evaluate("""() => {
                let visited=0;for(const g of D.groups){document.querySelector('#mode').value=g.mode;
                for(let i=0;i<D.pairs.length;i++){document.querySelector('#pair').value=String(i);pairView();
                const actual=document.querySelector('#pairSummary').textContent;
                const expected='已访问格：初态 '+g.initial_counts[i].filter(v=>v>0).length+' /100；全程 '+g.full_counts[i].filter(v=>v>0).length+' /100；非负前缀 '+g.prefix_counts[i].filter(v=>v>0).length+' /100。未访问格尚无可达性归类。';
                if(actual!==expected)throw Error('pair '+g.mode+'/'+i);visited++;}}
                return visited;}""")
            assert swept==1300
            page.select_option('#mode','joint_reference');page.select_option('#pair','305')
            assert page.evaluate("Array.from(document.images).every(im=>im.complete && im.naturalWidth>0)")
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
            assert 'NOT_READY_TO_REGISTER' in page.locator('#readiness').inner_text()
            assert page.locator('#error').inner_text()==''
            page.screenshot(path=str(a.out/f'panel_{width}.png'),full_page=True)
            page.locator('.heatmaps').screenshot(path=str(a.out/f'pairs_{width}.png'))
            checks.append(dict(width=width,matrix_categories=[6,8,6,20],pair_journeys=1300,images_loaded=4,document_overflow=False))
            # Main-page entry must be present in the actual scientific tab.
            main=browser.new_page(viewport=dict(width=width,height=950));main.goto(a.url.split('/docs/')[0]+'/#scientific',wait_until='domcontentloaded')
            main.locator('[data-tab="scientific"]').click();card=main.locator('#benchmarkProtocolEvidence')
            assert card.is_visible();assert card.locator('a').get_attribute('href')=='docs/safety_benchmark_protocol_20261006/'
            main.close();page.close()
        browser.close()
    pairs=list(csv.DictReader(io.StringIO(get(a.url+'all_joint_pairs.csv').decode())))
    windows=list(csv.DictReader(io.StringIO(get(a.url+'all_window_motion.csv').decode())))
    assert len(pairs)==1300 and len(windows)==768
    prior=list(csv.DictReader(io.StringIO(get(a.url+'prior_task_cases.csv').decode())))
    assert len(prior)==192 and sum(r['task_pass']=='True' for r in prior)==19
    for name in ('events_and_mobility','coverage_grid','motion_ranges','prior_task_failures'):
        assert get(a.url+'figures/'+name+'.pdf').startswith(b'%PDF')
    assert not errors,errors
    result=dict(status='PASS',url=a.url,checks=checks,pair_csv_rows=1300,window_csv_rows=768,prior_task_rows=192,pdf_downloads=4,javascript_errors=errors)
    (a.out/'browser.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
