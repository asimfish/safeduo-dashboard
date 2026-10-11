"""Only readable dates, language-aware hero and native compact mobile navigation."""
import datetime,html,re
def human(value):
    return datetime.datetime.fromisoformat(value).astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')+' (Shanghai UTC+8)'
def bi(zh,en,tag='span'):
    return f'<{tag} class="r18-chrome-zh" lang="zh-CN">{zh}</{tag}><{tag} class="r18-chrome-en" lang="en">{en}</{tag}>'
def extra_css():
    return '''
/* Report18 release2 presentation chrome only; historical sections untouched. */
.r18-chrome-en{display:none}body:has(#r18EN:checked) .r18-chrome-zh{display:none}body:has(#r18EN:checked) .r18-chrome-en{display:block}
.hero .r18-chrome-zh,.hero .r18-chrome-en{max-width:100%;overflow-wrap:normal}
.hero h1{font-size:clamp(1.8rem,3.25vw,3rem);line-height:1.2}
.hero .asof{max-width:58rem;margin-inline:auto;line-height:1.6;overflow-wrap:normal}
.r18-clock{white-space:nowrap}.r18-desktop-primary{display:grid;gap:.65rem;margin:1rem 0}.r18-nav-top{margin:.7rem 0}
#r18MethodsNav{display:none}#r18MethodsNav summary,#r18HistoryNav summary{cursor:pointer}
.r18-mobile-primary{display:grid;gap:.55rem;padding:.65rem 0}#r18HistoryNav{margin-top:.75rem}
@media(max-width:900px){
 .r18-toc{position:static!important;display:block;margin-bottom:.85rem;padding:0 0 .5rem;border-bottom:none}
 #r18MethodsNav,#r18HistoryNav{padding:0 .5rem;margin:.35rem 0}
 #r18MethodsNav>summary,#r18HistoryNav>summary{font-size:13px;line-height:1.5;padding:.4rem 0}
 .r18-desktop-primary{display:none}#r18MethodsNav{display:block;margin-top:.6rem}
 .r18-toc>strong{display:none}.r18-nav-top{margin:.35rem 0}
 #r18MethodsNav>summary,#r18HistoryNav>summary{padding:.35rem 0}
 #r18HistoryNav{margin-top:.35rem}.hero h1{font-size:1.85rem}.hero .subtitle{font-size:1rem}
 .hero .asof{font-size:.82rem;line-height:1.5}
}
'''
def refine(old,facts):
    page,n=re.subn(r'(<style id="report18ModesV1">)(.*?)(</style>)',lambda m:m[1]+m[2]+extra_css()+m[3],old,count=1,flags=re.S)
    assert n==1
    eng=human(facts['engineering_snapshot_utc']);science=human(facts['scientific_snapshot_utc'])
    hero='<header class="hero"><div class="eyebrow">'+bi('实验进展与验收边界','Experiment progress and acceptance limits')+'</div>'
    hero+='<h1>'+bi('机器人安全筛查<br>时钟防护与任务准入','Robot safety screening<br>Clock guards and task readiness')+'</h1>'
    hero+='<p class="subtitle">'+bi('工程检查已闭合；真实任务、完整原生准入与安全验收仍待完成。','Engineering checks are closed; real tasks, complete native admission and safety acceptance remain incomplete.')+'</p>'
    hero+='<p class="asof" id="reportAsOf">'+bi(f'报告18 · 工程进展 {eng}<br>实验快照 {science} · 非实时',f'Report18 · Engineering progress {eng}<br>Experiment snapshot {science} · Not live')+'</p></header>'
    page,n=re.subn(r'<header class="hero">.*?</header>',lambda m:hero,page,count=1,flags=re.S);assert n==1
    oldtoc_start=page.index('<nav class="toc r18-toc"');oldtoc_end=page.index('<main id="reportMain">',oldtoc_start)
    toc=page[oldtoc_start:oldtoc_end]
    nested_start=toc.index('<nav class="toc r17-toc"');nested_end=toc.rindex('</nav></details></nav>')+len('</nav>')
    oldnav=toc[nested_start:nested_end]
    method=[]
    for k,z,e in [('Methods','方法','Methods'),('Criteria','判据','Criteria'),('Metrics','指标','Metrics'),('Failures','失败与限制','Failures and limits'),('Evidence','证据','Evidence'),('Reproduction','复现','Reproduction')]:
        method.append(f'<a href="#r18{k}">{z} / {e}</a>')
    links=''.join(method)
    nav='<nav class="toc r18-toc" aria-label="Report18 and historical reports"><strong>报告18 / Report18</strong>'
    nav+='<div class="r18-nav-top"><a href="#r18Overview">最新概览 / Latest overview</a></div>'
    nav+='<div class="r18-desktop-primary">'+links+'</div>'
    nav+='<details id="r18MethodsNav"><summary>方法与证据 / Methods and evidence</summary><div class="r18-mobile-primary">'+links+'</div></details>'
    nav+='<details id="r18HistoryNav"><summary>历史报告 / Historical reports</summary><a href="#nativeAttemptIncrement17">报告17冻结科学快照 / Frozen Report17 science snapshot</a>'+oldnav+'</details></nav>'
    page=page[:oldtoc_start]+nav+page[oldtoc_end:]
    marker='<section id="nativeAttemptIncrement18"';a=page.index(marker);b=page.index('<section id="nativeAttemptIncrement17"',a);s=page[a:b]
    for value in (facts['engineering_snapshot_utc'],facts['scientific_snapshot_utc'],facts['capacity']['observed_utc']):
        assert s.count(html.escape(value))==2
        s=s.replace(html.escape(value),html.escape(human(value)))
    page=page[:a]+s+page[b:]
    page,n=re.subn(r'<title>.*?</title>','<title>Report18 · 时钟防护与任务准入 / Clock guards and task readiness</title>',page,count=1,flags=re.S);assert n==1
    return page
def projection(old,new,facts):
    marker='<section id="nativeAttemptIncrement17"';assert new[new.index(marker):]==old[old.index(marker):]
    start='<section id="nativeAttemptIncrement18"';a=old[old.index(start):old.index(marker)]
    b=new[new.index(start):new.index(marker)]
    for value in (facts['engineering_snapshot_utc'],facts['scientific_snapshot_utc'],facts['capacity']['observed_utc']):
        a=a.replace(html.escape(value),html.escape(human(value)))
    assert a==b
    nav='<nav class="toc r17-toc"';oa=old[old.index(nav):old.index('<main id="reportMain">')]
    na=new[new.index(nav):new.index('<main id="reportMain">')]
    assert oa==na
    return dict(allowed_presentation_regions=['document_title','hero_title_subtitle_readable_dates','scoped_chrome_CSS','Report18_navigation','six_visible_Report18_date_strings'],
      historical_Report17_body_and_tail_byte_exact=True,historical_navigation_byte_exact=True,
      Report18_experiment_and_SOURCE_claims_equal_after_only_date_format_normalization=True,
      displayed_time_zone='Shanghai UTC+8',display_precision_seconds=True,provenance_cutoffs_microseconds_unchanged=True)

