"""Prospective report05 builder. Runs only after all actual input chains close."""
import argparse
import csv
import datetime
import html
from html.parser import HTMLParser
import io
import json
from pathlib import Path

from binding_v1 import BASE,H,HERE,OUT,R,STATES,buffer,calibration,closed_wait,digest,get,metadata_rules,need,read,ready_inputs,ref,sha,stage_chain,tool0,validate_row,write

PREFIX='model_point0_increment05_v1'

def replace_id(page, tag, ident, replacement):
    """Replace exactly one element, including nested elements of the same tag."""
    lines=page.splitlines(keepends=True);offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    class Locate(HTMLParser):
        def __init__(self):super().__init__();self.depth=0;self.start=None;self.ranges=[]
        def at(self):row,col=self.getpos();return offsets[row-1]+col
        def handle_starttag(self,t,a):
            if self.depth:
                if t==tag:self.depth+=1
            elif t==tag and dict(a).get('id')==ident:self.start=self.at();self.depth=1
        def handle_endtag(self,t):
            if self.depth and t==tag:
                self.depth-=1
                if not self.depth:self.ranges.append((self.start,page.index('>',self.at())+1))
    parser=Locate();parser.feed(page);need(len(parser.ranges)==1 and not parser.depth,'exact closed baseline element '+ident)
    first,last=parser.ranges[0]
    return page[:first]+replacement+page[last:]

def table(ident,caption,heads,rows):
    return '<div class="table-window" tabindex="0"><table id="'+ident+'"><caption>'+html.escape(caption)+'</caption><thead><tr>'+''.join('<th scope="col">'+html.escape(x)+'</th>' for x in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def copy_small(q,path):
    raw=buffer(q,16*1024**2);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw)
    need(sha(path)==q['sha256'],'exact small evidence copy')

def peer_chain(q, expected, field):
    peer=read(q)
    need(peer.get('reviewer') in ('Desc','James') and not peer['blocking_findings'] and (peer.get('status') in ('PASS','COND_PASS','PASS_DATA_ONLY') or peer.get('verdict') in ('PASS_DATA_ONLY','PASS_MODEL_REPLAY_DATA_ONLY')),'closed distinct DATA review')
    need(get(peer,field)==expected,'peer binds exact result')
    closed_wait(peer['actual_worker_wait'])
    tool0(read(peer['actual_tool_wait']),'parent_tool')
    need(peer.get('actual_worker_exit')==0 and peer.get('actual_tool_exit')==0,'peer own actual terminal exits')
    return peer

def stream_rows(report,report_ref,out):
    need(report['schema']=='astra.full8192.model_stream.v2' and report['status']=='COMPLETE_MODEL_ONLY_PENDING_ACTUAL_CPU_WAIT_AND_REVIEW' and report['denominator']==report['completed']==8192 and report['not_evaluated']==0 and not report.get('error'),'actual complete8192; partial never complete')
    need(report['native_tested']==report['camera_tested']==report['eligible']==report['bank_count']==0 and report['global_error_bound']=='UNKNOWN','MODEL_ONLY cannot promote native/camera/bank')
    rules=metadata_rules();root=Path(report_ref['path']).parent;compact=[];ledger=[]
    target=out/PREFIX/'data/ALL8192_MODEL_FAILURE_TABLE.csv';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',newline='') as stream:
        columns=['global_id','seed','proposal_id','family','model_status','finite_distance_count','negative_distance_count','minimum_model_gap_m','known_or_reserved','native_status','camera_status','eligible','block_rows_sha256','block_model_gaps_sha256']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();detail_path=out/PREFIX/'data/ALL8192_MODEL_DISPOSITIONS.jsonl';detail=detail_path.open('x')
        for first in range(0,8192,32):
            block=root/'blocks'/f'block_{first:04d}';cp=block/'COMMIT.json';cq=dict(path=str(cp),sha256=report['artifacts'][str(cp)])
            commit=read(cq);need(commit['first_global_id']==first and commit['count']==32 and commit['scope']=='MODEL_ONLY_NOT_NATIVE','exact committed32 block')
            expected={str(block/name) for name in ('q74.npy','qd74.npy','model_raw9021.npy','ROWS.json')}
            need(set(commit['files'])==expected and all(report['artifacts'][p]==s for p,s in commit['files'].items()),'actual artifact ledger agrees without re-decoding arrays')
            rows_ref=dict(path=str(block/'ROWS.json'),sha256=commit['files'][str(block/'ROWS.json')]);rows=read(rows_ref);need(len(rows)==32,'actual32 row packet')
            gap_ref=dict(path=str(block/'model_raw9021.npy'),sha256=commit['files'][str(block/'model_raw9021.npy')])
            ledger.append(dict(first_global_id=first,count=32,commit=cq,rows=rows_ref,model_gaps=gap_ref,q74=dict(path=str(block/'q74.npy'),sha256=commit['files'][str(block/'q74.npy')]),qd74=dict(path=str(block/'qd74.npy'),sha256=commit['files'][str(block/'qd74.npy')]),scope='MODEL_ONLY; producer closed SHA ledger; large array bytes not recopied or independently reevaluated by report builder'))
            for lane,row in enumerate(rows):
                validate_row(row,first+lane,rules);detail.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                compact.append({k:v for k,v in row.items() if k!='negative_rows'})
                writer.writerow({**{k:row[k] for k in columns if k in row},'negative_distance_count':len(row['negative_rows']),'block_rows_sha256':rows_ref['sha256'],'block_model_gaps_sha256':gap_ref['sha256']})
    detail.close()
    accounting=rules.exhaustive_accounting(compact);need(accounting==report['accounting'],'all16 cells independently counted from8192 committed metadata rows')
    gaps=write(out/PREFIX/'data/ALL256_MODEL_GAPS_SHA_MANIFEST.json',dict(report=report_ref,blocks=ledger,model_array_shape=[8192,9021],full_float64_model_gap_bytes=591265792,full_gaps_retained_on_NAS=True,full_arrays_copied_to_public_git=False,raw_native_removed=False,block_negative_row_IDs_retained_on_NAS=True,public_failure_table='ALL8192_MODEL_FAILURE_TABLE.csv',archive_decodes=0))
    return compact,accounting,rules.prospective_first32(compact),ref(target),gaps

def v7_evidence(inputs,base_plan):
    peer=peer_chain(inputs['V7_DATA_peer'],inputs['V7_RAW_handoff'],['exact_author_handoff'])
    hand=read(inputs['V7_RAW_handoff']);raw=read(hand['RAW_report'])
    need(peer['exact_author_report']==hand['RAW_report'] and peer['related_rows']==416 and peer['new_IID_proposals']==0 and raw['status']=='CLOSED_VALID_MECHANISM_DIAGNOSTIC_DATA' and not raw['errors'],'same16 V7 diagnostic DATA chain')
    need(raw['bank_count']==0 and raw['safety_acceptance'] is False and raw['camera']=='NOT_TESTED_DIAGNOSTIC_ONLY','numeric DATA not safety/camera')
    flags=raw['data']['denial13x32x7'];order=raw['data']['reason_order'];need(len(flags)==13 and all(len(x)==32 for x in flags),'all13x32 temporal flags')
    csv_ref=base_plan['file_sources']['mechanism_v6_addendum_v1/data/ALL416_RELATED_NUMERIC_FLAGS_V6.csv']
    prior=list(csv.DictReader(io.StringIO(buffer(csv_ref).decode())))
    need(len(prior)==416 and all(len(row)==7 and all(type(v) is bool for v in row) for micro in flags for row in micro),'actual416×7 boolean predicates')
    old=[];current=[]
    for record in prior:
        micro,lane=int(record['micro']),int(record['lane'])
        old.extend(record[name]=='True' for name in order);current.extend(flags[micro][lane])
    need(bytes(old)==bytes(current),'V6/V7 same416 predicate vectors; no new IID credit')
    need(all(any(any(flags[micro][lane]) for micro in range(13)) for lane in range(32)),'all32 numerically denied in recorded window')
    return dict(status='PASS_DATA_ONLY',same16_repeat=True,AB_windows=32,states_per_window=13,physics_steps=12,related_temporal_rows=416,new_IID_proposals=0,all32_numeric_denied=True,V6_V7_416x7_predicate_vectors_identical=True,predicate_comparison_basis='bound old V6 CSV and V7 independently audited report JSON; no raw array reread',numeric_scope_model='NOT_READY_IN_ORIGINAL_NUMERIC_SCOPE',separate_final_frame_binding=inputs['frame_binding'],camera='NOT_TESTED',bank_count=0,Drive='UNKNOWN',wrench_frame='UNKNOWN',global_error_bound='UNKNOWN',formal_holdout=False)

def cell_availability(rows):
    cells=[]
    for seed in range(4):
        for family in range(4):
            options=[r for r in rows if r['seed']==seed and r['family']==family and r['model_status']=='MODEL_PREDICTED_INITIAL_CLEAR']
            all_ids=[r['proposal_id'] for r in options];reserved=[r['proposal_id'] for r in options if r['known_or_reserved']];unseen=[r['proposal_id'] for r in options if not r['known_or_reserved']]
            cells.append(dict(seed=seed,family=family,total_clear_IDs=all_ids,reserved_clear_IDs=reserved,unseen_clear_IDs=unseen,earliest2_diagnostic_IDs=unseen[:2]))
    return cells

def figures(cal,accounting,availability,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='closed-report05'
    labels=['Calibration64','Development verification64'];groups=[cal['replay']['calibration'],cal['replay']['development_verification']]
    source=write(out/PREFIX/'data/FIGURE_SOURCES_V1.json',dict(calibration=cal['replay'],point0_position_max_abs_error_m=cal['point0_position_max_abs_error_m'],diagnostic_tolerance_m=cal['diagnostic_tolerance_m'],model_only_cells=accounting['cells'],all16_availability=availability,scope='Observed fixed DEVELOPMENT replay and complete MODEL_ONLY accounting; no new native samples; no global error bound'))
    fig,ax=plt.subplots(figsize=(9,4.7));x=[0,1]
    for delta,key,color,label in [(-.16,'max_abs_error_m','#205c83','Observed maximum'),(.16,'RMS_error_m','#a65331','Observed RMS')]:
        bars=ax.bar([v+delta for v in x],[g[key]*1e6 for g in groups],width=.3,label=label,color=color);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    ax.set_xticks(x,labels);ax.set_ylabel('Observed raw9021 discrepancy (µm)');ax.set_ylim(0,6);ax.legend(fontsize=8);ax.set_title('Formal fixed DEVELOPMENT replay:128 existing native point0 states')
    fig.text(.02,.03,'Original diagnostic tolerance50 µm (outside plot range). Both sign disagreements0. 64+64 are not holdout.\nObserved max4.771 µm is not a global or physical error bound.',fontsize=8)
    fig.tight_layout(rect=(0,.11,1,1));dest=out/PREFIX/'figures/calibration64_64';dest.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('svg','png'):fig.savefig(dest.with_suffix('.'+ext),dpi=150,metadata={'Creator':'Source-bound report05'} if ext=='svg' else None)
    plt.close(fig)
    fig,(ax,bx)=plt.subplots(2,1,figsize=(11,8),gridspec_kw={'height_ratios':[2,1.2]});cells=accounting['cells'];bottom=[0]*16
    for status,color,label in zip(STATES,('#8c939a','#a65331','#205c83'),('UNKNOWN0','NEGATIVE8018','PREDICTED INITIAL CLEAR174')):
        values=[c['counts'][status] for c in cells];ax.bar(range(16),values,bottom=bottom,color=color,label=label);bottom=[a+b for a,b in zip(bottom,values)]
    ax.set_xticks(range(16),[f"s{c['seed']}/f{c['family']}" for c in cells]);ax.set_ylim(0,550);ax.set_ylabel('Original cell denominator512');ax.set_title('All8192 retained; prediction labels only');ax.legend(fontsize=8,loc='lower center')
    available=[len(a['unseen_clear_IDs']) for a in availability];bars=bx.bar(range(16),available,color=['#a65331' if n<2 else '#205c83' for n in available]);bx.bar_label(bars,padding=2,fontsize=8);bx.axhline(2,linestyle='--',color='#8c3929',label='Original quota2/cell; no borrowing');bx.set_xticks(range(16),[f"s{c['seed']}/f{c['family']}" for c in cells]);bx.set_ylabel('Unreserved predicted CLEAR');bx.set_ylim(0,27);bx.legend(fontsize=8)
    fig.suptitle('Schedule EMPTY / HELD:seed3/family1 has1, requires2',fontsize=14);fig.text(.02,.015,'Reserved172 stays in denominator8192; only future candidate filter. One CLEAR(proposal92) is reserved. Native/camera NOT_TESTED; bank0.',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.95));dest=out/PREFIX/'figures/model8192_cells'
    for ext in ('svg','png'):fig.savefig(dest.with_suffix('.'+ext),dpi=150,metadata={'Creator':'Source-bound report05'} if ext=='svg' else None)
    plt.close(fig)
    # Only newly generated SVG serialization whitespace; original figures untouched.
    for p in (out/PREFIX/'figures').glob('*.svg'):p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    return source

def export_recompute(inputs,out):
    root=out/PREFIX/'recompute';root.mkdir()
    runner=read(inputs['screen_runner_receipt']);graph=read(inputs['model_runner_source_graph']);frame=read(inputs['frame_binding']);exports={}
    sources=[('fk_v2.py',next(dict(path=p,sha256=s) for p,s in graph['files'].items() if p.endswith('/fk_v2.py'))),('prospective_rules_v1.py',next(dict(path=p,sha256=s) for p,s in graph['files'].items() if p.endswith('/prospective_rules_v1.py'))),('science.py',next(dict(path=p,sha256=s) for p,s in graph['files'].items() if p.endswith('/science.py'))),('evidence_io.py',next(dict(path=p,sha256=s) for p,s in graph['files'].items() if p.endswith('/evidence_io.py'))),('model.json',inputs['model']),('geometry.json',frame['geometry']),('parameters.npz',frame['parameter']),('body_identity.npz',frame['body_identity']),('PROPOSALS.json',runner['cohort']),*[(key+'.npy',q) for key,q in runner['input_files'].items()],('ORIGINAL_RUNNER_SOURCE_GRAPH.json',inputs['model_runner_source_graph'])]
    for name,q in sources:copy_small(q,root/name);exports[name]=dict(original=q,sha256=q['sha256'],bytes=(root/name).stat().st_size)
    for name in ('recompute_model_only.py','recompute_io_v2.py'):
        dest='io_v2.py' if name=='recompute_io_v2.py' else name;copy_small(ref(HERE/name),root/dest);exports[dest]=dict(original=ref(HERE/name),sha256=sha(root/dest),bytes=(root/dest).stat().st_size)
    copy_small(ref(HERE/'RECOMPUTE_README.txt'),root/'README.txt')
    return write(root/'RECOMPUTE_BUNDLE_V1.json',dict(status='MODEL_ONLY_CPU_RECOMPUTATION_SOURCE_AND_ORIGINAL_INPUTS',files=exports,source_reviewed_math_byteexact=['fk_v2.py','science.py','evidence_io.py','prospective_rules_v1.py'],command='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python -B recompute_model_only.py --out NEW_DIRECTORY',large_gaps_retained_NAS=True,native_launches=0,bank_count=0,global_error_bound='UNKNOWN'))

def build(inputs,out,inputs_ref):
    ready_inputs(inputs)
    base=read(inputs['base_publication']);need(base['commit']==BASE and base['status']=='PASS_PUBLISHED_CANDIDATE04_WITH_PRESENTATION_ONLY_WRAP','exact genuinely published04 base')
    base_plan=read(base['plan']);base_manifest=read(base['manifest']);page=buffer(base['HTML']).decode();summary=read(base_plan['file_sources']['summary.json'])
    frame=read(inputs['frame_binding']);model=read(inputs['model']);frame_peer=read(inputs['final_frame_DATA_peer'])
    need(frame['schema']=='astra.full8192.native_frame_binding.v2' and frame['synthetic'] is False and frame['model']==inputs['model'] and frame['frame_export_review']==inputs['final_frame_DATA_peer'],'actual model/frame/FINAL DATA binding')
    need(model['schema']=='astra.full74.fk.frame_model.v2' and model['coordinates']=='ALL74_INDEPENDENT_NO_MIMIC_PROJECTION' and len(model['joint_names74'])==74,'typed74 independent model')
    need(frame_peer['status'] in ('PASS','COND_PASS') and not frame_peer['blocking_findings'] and frame_peer['model']==inputs['model'] and frame_peer['independently_verified']['all4736_signed_motion_directions'] is True and frame_peer['independently_verified']['all256_world_root_corroborations'] is True,'separate signed4736/roots256 FINAL model DATA scope')
    point=read(inputs['point0_reviews']);need(point['status']=='PASS_DATA_ONLY_ALL4_NEW_FRAME_BOUND_REVIEWS_CLOSED' and point['frame_binding']==inputs['frame_binding'] and len(point['point_review_refs'])==4,'four actual new frame-bound point0 reviews')
    finalized=read(inputs['point0_finalize']);need(inputs['point0_finalize']['sha256']=='fa60dfa116f6722a36ee89b6a3ac7b8d6caff7d7c69e2d1701a67f6d909c2ca5','parent actual formal FINALIZE closure preserved')
    cal=calibration(read(inputs['formal_calibration']),inputs['frame_binding'],inputs['calibration_input'])
    stage_chain(inputs['calibration_terminal'],inputs['formal_calibration']);peer_chain(inputs['formal_calibration_peer'],inputs['formal_calibration'],['model_replay'])
    stage_chain(inputs['model8192_terminal'],inputs['model8192']);peer_chain(inputs['model8192_DATA_peer'],inputs['model8192'],inputs['model8192_peer_result_field'])
    sp=peer_chain(inputs['model8192_DATA_peer'],inputs['model8192'],['screen_report']);need(sp['prospective_schedule']==inputs['schedule'],'same genuinely closed peer schedule');closed_wait(inputs['point0_finalize_wait']);tool0(finalized['actual_tool'],'parent_tool');need(finalized['calibration_input']==inputs['calibration_input'] and finalized['frame_binding']==inputs['frame_binding'],'true formal FINALIZE input chain')
    need(not out.exists() and out.resolve().is_relative_to(OUT),'exclusive new report05 output; no W/Git/native writes');out.mkdir(parents=True)
    overlay=out/'overlay';overlay.mkdir()
    rows,counts,expected_schedule,failure_table,gaps=stream_rows(read(inputs['model8192']),inputs['model8192'],overlay)
    schedule=read(inputs['schedule']);need(all(schedule[k]==expected_schedule[k] for k in ('status','candidates','shortfall','bank_count','eligible','requires_new_parent_adoption')),'exact original deterministic2/cell/no borrowing/shortfall schedule');availability=cell_availability(rows);need(schedule['all16_availability']==availability,'independent all16 unseen/reserved IDs agree with distinct peer')
    v7=v7_evidence(inputs,base_plan);plot_source=figures(cal,counts,availability,overlay);recompute_bundle=export_recompute(inputs,overlay)
    proofs={name:inputs[name] for name in ('model','frame_binding','final_frame_DATA_peer','point0_reviews','point0_finalize','calibration_input','formal_calibration','formal_calibration_peer','model8192','model8192_DATA_peer','schedule','V7_RAW_handoff','V7_DATA_peer','preliminary_DIAG','preliminary_DIAG_closed','preliminary_DIAG_peer')}
    for name,q in proofs.items():copy_small(q,overlay/PREFIX/'evidence'/(name+'.json'))
    for name,q in [('index.html',base['HTML']),('panel_manifest.json',base['manifest']),('summary.json',base_plan['file_sources']['summary.json'])]:copy_small(q,overlay/PREFIX/'archive_candidate04'/name)
    for name in ('binding_v1.py','build_report_v1.py','prepare_v1.py'):
        copy_small(ref(HERE/name),overlay/PREFIX/'source'/name)
    copy_small(inputs_ref,overlay/PREFIX/'source/EXACT_REPORT_INPUTS_V1.json')
    facts=dict(schema='astra.report.increment05.actual_facts.v1',snapshot_utc=inputs['snapshot_utc'],base_publication=inputs['base_publication'],model=inputs['model'],frame_binding=inputs['frame_binding'],FINAL_DATA=inputs['final_frame_DATA_peer'],signed_direction_observations=4736,root_corroborations=256,existing_native_point0_states=128,new_native_point0_states=0,formal_calibration=cal,earlier_diagnostic_is_not_formal_calibration=True,model8192=counts,schedule=schedule,V7_numeric=v7,large_model_gaps_NAS_manifest=gaps,complete8192_failure_table=failure_table,figure_sources=plot_source,recomputation_bundle=recompute_bundle,reserved_filter_total=172,model_total_counts={k:sum(c['counts'][k] for c in counts['cells']) for k in STATES},unreserved_prediction_clear=173,all16_availability=availability,no_new_physics_or_camera=True,all_source_inputs=inputs_ref,bank_count=0,safety_acceptance=False,task_acceptance=False,Drive='UNKNOWN',wrench_frame='UNKNOWN',global_error_bound='UNKNOWN',formal_holdout=False)
    write(overlay/PREFIX/'FACTS_INCREMENT05_V1.json',facts)
    utc=datetime.datetime.fromisoformat(inputs['snapshot_utc']);asof=utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    page=replace_id(page,'p','reportAsOf','<p class="asof" id="reportAsOf">报告快照截至 <time datetime="'+html.escape(inputs['snapshot_utc'])+'">'+asof+'</time>：新增正式64+64模型回放、完整8192 MODEL_ONLY与确定性schedule缺额、EMPTY／HELD；V7独立numeric RAW／DATA已闭合。旧04截止17:53:40正文与数据可下载，原5图及15张INITIAL_ONLY图保持各自时点。</p>')
    repeat='<div id="v7ActualCollection"><h3>V7独立numeric RAW／DATA已闭合；同16提案重复运行</h3><p>保留32 AB×13状态／12物理步、416相关时序观测；416×7数值拒绝向量与V6报告逐项相同，全部32窗口数值拒绝。不是新IID试验，15组B仍为不可比较的A重复，仅102可构造。numeric RAW原scope的frame NOT_READY历史标签保留；后续完整signed4736／roots256模型是另一条已闭合FINAL DATA证据链，不能逆改原receipt。相机NOT_TESTED、bank0，Drive／wrench／globalbound UNKNOWN。<a href="'+PREFIX+'/evidence/V7_DATA_peer.json">[V7不同作者DATA]</a></p></div>'
    page=replace_id(page,'div','v7ActualCollection',repeat)
    calrows=[[name,64,f"{cal['replay'][key]['max_abs_error_m']:.12g}",f"{cal['replay'][key]['RMS_error_m']:.12g}",cal['replay'][key]['sign_disagreements']] for name,key in [('calibration','calibration'),('development verification','development_verification')]]
    cellrows=[[c['seed'],c['family'],512,*[c['counts'][s] for s in STATES],c['reserved'],len(a['unseen_clear_IDs']),2,'缺1 / HELD' if len(a['unseen_clear_IDs'])<2 else '仅诊断满足2'] for c,a in zip(counts['cells'],availability)]
    screen='<section id="latestExperiments"><h2>正式64+64回放与完整8192 MODEL_ONLY：要求仍未达成</h2><p class="negative"><strong>缺口：seed3/family1只有proposal45一个未reserved预测CLEAR，quota要求2。原32项schedule EMPTY／HELD，不借配额；174个预测CLEAR不是安全状态，bank0。</strong></p><p>完整74独立关节模型由实际两phase的signed4736／roots256核对及不同作者FINAL DATA闭合支持；固定64+64仍来自旧128原生point0，非新128试验、非holdout。先前8904诊断与本次正式runner回放分开保留。</p>'+table('model05CalibrationTable','固定development64+64，误差单位m',['固定半组','状态数','观测最大误差m','RMS误差m','sign disagreements'],calrows)+'<figure id="model05CalibrationFigure"><img src="'+PREFIX+'/figures/calibration64_64.svg" alt="正式development64+64观测回放误差；非全局误差界"><figcaption>图6：原5e-5 m阈值仅为模型回放诊断门槛，观测误差不是物理或全局上界。</figcaption></figure><p>完整8192逐项记账：NEGATIVE8018、预测CLEAR174、UNKNOWN0；原16 cells各512全部保留。172个reserved只从后续候选中排除，不删除8192分母；其中预测CLEAR的proposal92已reserved，未reserved预测CLEAR173个。MODEL_PREDICTED_INITIAL_CLEAR仅为预测初态标签；native／camera均NOT_TESTED，不能计为合格状态。</p>'+table('model05CellsTable','8192 MODEL_ONLY全部16 cells',['seed','family','分母','UNKNOWN','NEGATIVE','预测INITIAL CLEAR','reserved总数','未reserved CLEAR','要求','quota诊断'],cellrows)+'<figure id="model05ScreenFigure"><img src="'+PREFIX+'/figures/model8192_cells.svg" alt="完整8192模型分类，16个原seed family cells各512"><figcaption>图7：仅MODEL_ONLY，未增加真实camera、physics或policy bank。</figcaption></figure><p>确定性schedule状态：<strong>'+html.escape(schedule['status'])+'</strong>；候选数 '+str(len(schedule['candidates']))+'，shortfall cells '+str(len(schedule['shortfall']))+'。原每cell最早2个未用预测CLEAR、不可借用配额；存在缺额时整份schedule为空／HELD，不允许借用，不是原生准入。</p><p>完整8192逐项failure table、256 blocks/gaps SHA与可重算源代码下载；591MB模型gap原件保留NAS，旧native RAW不删除。<a href="'+PREFIX+'/data/ALL8192_MODEL_FAILURE_TABLE.csv" download>完整8192逐项表</a> · <a href="'+PREFIX+'/data/ALL256_MODEL_GAPS_SHA_MANIFEST.json" download>全部gaps SHA/路径索引</a> · <a href="'+PREFIX+'/FACTS_INCREMENT05_V1.json" download>本次事实与输入</a> · <a href="'+PREFIX+'/source/EXACT_REPORT_INPUTS_V1.json" download>可重算输入refs</a> · <a href="'+PREFIX+'/recompute/RECOMPUTE_BUNDLE_V1.json" download>完整可重算代码／模型／提案输入SHA索引</a> · <a href="'+PREFIX+'/recompute/recompute_model_only.py" download>CPU逐block重算入口</a> · <a href="'+PREFIX+'/data/ALL8192_MODEL_DISPOSITIONS.jsonl" download>8192逐项9021负row IDs与flags</a> · <a href="'+PREFIX+'/archive_candidate04/index.html">历史04</a></p><p>完整安全与任务仍未验收；bank0；Drive、wrench、CONTACT、全局误差界UNKNOWN，正式holdout未打开。未来真实12micro强相机与独立raw核验仍必需。模型仅检查初态几何，尚不覆盖74硬限、74速度、26软限、82 scalar、动态隐式驱动／接触／终端或全局界。</p></section>'
    page=replace_id(page,'section','latestExperiments',screen)
    need('<style id="publicationMobileWrapV1">main{overflow-wrap:anywhere}</style>' in page,'reuse closed04 wrap; no new style')
    with (overlay/'index.html').open('x') as f:f.write(page)
    summary['historical_candidate04_summary']=PREFIX+'/archive_candidate04/summary.json';summary['increment05']=facts;summary['asof_utc']=inputs['snapshot_utc'];summary['mechanism_v7_numeric_DATA_current']=v7
    write(overlay/'summary.json',summary)
    source_map={p.relative_to(overlay).as_posix():ref(p) for p in overlay.rglob('*') if p.is_file()}
    updated=json.loads(json.dumps(base_manifest));updated['files'].update({n:q['sha256'] for n,q in source_map.items()});updated['snapshot_updated_utc']=inputs['snapshot_utc'];updated['increment05']=dict(facts=PREFIX+'/FACTS_INCREMENT05_V1.json',scope='MODEL_ONLY and same16 numeric repeat; bank0',model=inputs['model'],frame_binding=inputs['frame_binding'],formal_holdout=False,safety_acceptance=False)
    write(overlay/'panel_manifest.json',updated);source_map['panel_manifest.json']=ref(overlay/'panel_manifest.json')
    changed={n:q['sha256'] for n,q in source_map.items()};inherited={n:s for n,s in base_manifest['files'].items() if n not in changed}
    need(all(updated['files'][n]==s for n,s in base_manifest['files'].items() if n not in ('index.html','summary.json')),'all original04 evidence/5fig/15RGB preserved by exact SHA metadata')
    plan=dict(schema='astra.report.increment05.overlay_plan.v1',base_commit=BASE,base_manifest=base['manifest'],before_sha256={n:base['manifest']['sha256'] if n=='panel_manifest.json' else base_manifest['files'][n] for n in changed if n=='panel_manifest.json' or n in base_manifest['files']},overlay=str(overlay),overlay_kind='PARTIAL_METADATA_DELTA_RESOLVE_FILE_SOURCES',file_sources=source_map,changed_files=changed,file_bytes={n:Path(q['path']).stat().st_size for n,q in source_map.items()},inherited_files=inherited,manifest=ref(overlay/'panel_manifest.json'),old64_file_sources_preserved_as_metadata=base['plan'],legacy_figures=5,additional_scientific_figures=2,original_RGB=15,archive_decodes=0,large_model_arrays_in_public_git=False,source_inputs=inputs_ref,scientific_peer='PENDING_NEW05_DESC_PEER',browser_publication='NOT_RUN')
    write(out/'REPORT_OVERLAY_PLAN_V1.json',plan)
    return write(out/'CANDIDATE05_BUILD_HANDOFF_V1.json',dict(status='BUILT_REAL_CLOSED_DATA_PENDING_CPU_WAIT_DESC_BROWSER_PUBLIC',inputs=inputs_ref,plan=ref(out/'REPORT_OVERLAY_PLAN_V1.json'),manifest=plan['manifest'],HTML=ref(overlay/'index.html'),facts=ref(overlay/PREFIX/'FACTS_INCREMENT05_V1.json'),source_graph=ref(HERE/'SOURCE_GRAPH_V1.json'),old04_immutable=True,native_GPU_Git_W_actions=0,bank_count=0,safety_acceptance=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--inputs',required=True);parser.add_argument('--inputs-sha256',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();q=dict(path=args.inputs,sha256=args.inputs_sha256)
    print(json.dumps(build(read(q),Path(args.out),q)),flush=True)

if __name__=='__main__':main()
