"""Latest report first; original history text and every historical anchor preserved."""
import re
from html.parser import HTMLParser
import common_v1 as c

SECTIONS=(('r17Overview','概览 / Overview'),('r17Methods','方法 / Methods'),
          ('r17Criteria','判据 / Criteria'),('r17Metrics','指标 / Metrics'),
          ('r17Failures','失败分析 / Failures'),('r17Evidence','证据 / Evidence'),
          ('r17Reproduction','复现 / Reproduction'))

class Anchors(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.hrefs=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.hrefs.append(a['href'])

def compose(old,latest,style,facts):
    main='<main id="reportMain">'
    c.need(old.count(main)==1,'one original main container')
    shell,history=old.split(main,1)
    match=re.search(r'<nav class="toc"[^>]*>.*?</nav>',shell,re.S)
    c.need(match is not None,'one original sidebar navigation')
    old_nav=match.group(0);links=re.findall(r'<a\b[^>]*href="#[^"]+"[^>]*>.*?</a>',old_nav,re.S)
    parsed=Anchors();parsed.feed(old_nav)
    c.need(len(links)==len(parsed.hrefs) and len(links)>20,'complete original historical TOC, no silently dropped links')
    reports=[];studies=[]
    for link in links:
        (reports if re.search(r'href="#(?:nativeAttemptIncrement|modelSupplementIncrement)',link) else studies).append(link)
    primary=''.join('<a href="#'+ident+'">'+label+'</a>' for ident,label in SECTIONS)
    nav=('<nav class="toc r17-toc" aria-label="报告17与历史报告目录">'
         '<strong>报告17 / Report17</strong><div class="r17-primary-nav">'+primary+'</div>'
         '<details id="r17HistoryNav"><summary>历史报告 / Historical reports</summary>'
         '<details id="r17HistoryIncrementNav"><summary>报告06–16 / Reports06–16</summary>'+''.join(reports)+'</details>'
         '<details id="r17HistoryStudyNav"><summary>早期实验、方法与下载 / Earlier studies, methods and downloads</summary>'+''.join(studies)+'</details>'
         '</details></nav>')
    shell=shell[:match.start()]+nav+shell[match.end():]
    bar=re.search(r'<nav class="sitebar"[^>]*>.*?</nav>',shell,re.S)
    c.need(bar is not None,'original site navigation')
    newbar=('<nav class="sitebar" aria-label="站点导航">'
            '<a href="#r17Overview">最新报告17 / Latest Report17</a>'
            '<a href="#r17Reproduction">证据与复现 / Evidence and reproduction</a>'
            '<a href="#r17HistoryNav">历史导航 / History</a>'
            '<a href="../">实验总览</a><a href="../paired_full128/">历史配对完整面板</a></nav>')
    shell=shell[:bar.start()]+newbar+shell[bar.end():]
    shell=shell.replace('<body>','<body class="report17-ia-draft">',1)
    asof='<p class="asof" id="reportAsOf">REPORT17 LOCAL DRAFT · '+facts['snapshot_utc']+' UTC / Asia/Shanghai '+facts['snapshot_Asia_Shanghai']+' · 固定输入，非实时 / Fixed inputs, not live</p>'
    shell,count=re.subn(r'<p class="asof" id="reportAsOf">.*?</p>',lambda _:asof,shell,count=1,flags=re.S)
    c.need(count==1,'actual draft input cutoff visible in header')
    shell=shell.replace('</head>','<style id="report17DraftModesV1">'+style+'</style></head>',1)
    intro=('<div id="r17HistoricalReports" class="r17-history-intro"><h2>历史报告与原始实验 / Historical reports and original studies</h2>'
           '<p>以下保留各版原文与原锚点；结论按各自截止时点和实验分母解读。'
           'The original text and anchors below retain each historical cutoff and experimental denominator.</p></div>')
    page=shell+main+latest+intro+history
    c.need(page[page.index('<section id="executiveResults"'):]==history,'entire historical main payload byte exact after relocation')
    old_parse=Anchors();old_parse.feed(history);new_parse=Anchors();new_parse.feed(page)
    c.need(len(new_parse.ids)==len(set(new_parse.ids)),'all page IDs unique after relocation')
    c.need(set(old_parse.ids).issubset(set(new_parse.ids)),'all original historical anchors retained')
    for href in parsed.hrefs:
        c.need(href in new_parse.hrefs and href[1:] in new_parse.ids,'each historical TOC link still resolves: '+href)
    for ident,_ in SECTIONS:c.need(ident in new_parse.ids,'seven live current section anchors')
    c.need(page.index('<section id="nativeAttemptIncrement17"')<page.index('<section id="executiveResults"')<page.index('<section id="nativeAttemptIncrement16"'),'latest report before old executive/V6 and all earlier sections')
    c.need(' open' not in nav and nav.count('<details')==3,'native initially-collapsed history groups; no JS dependency')
    return page,dict(latest_report_first=True,seven_primary_sections=[ident for ident,_ in SECTIONS],old_toc_links_preserved=len(links),history_native_details_groups=3,history_initially_collapsed=True,history_JS_dependency=False,entire_original_main_payload_byte_exact=True,old_historical_anchor_count=len(old_parse.ids),all_original_anchors_retained=True,all_fragment_links_resolve=True,all_page_ids_unique=True,old_executive_V6_after_latest=True,media_payload_reads=0,actual_browser_modes_run=0)
