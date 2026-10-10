"""Report05 source/input entry; preparation does not run model/native/Git/browser."""
import argparse
import ast
import json
from pathlib import Path
from binding_v1 import HERE,OUT,buffer,need,read,ready_inputs,ref,sha,write

def verify_sources():
    graph=read(ref(HERE/'SOURCE_GRAPH_V1.json'))
    need(graph['schema']=='astra.report.increment05.source_graph.v1','own source graph')
    for path,digest in graph['files'].items():need(sha(path)==digest,'own source exact before execution')
    for path,digest in graph['reused_external_sources'].items():need(sha(path)==digest,'narrow exact reused source bytes')
    return graph

def check_preparation():
    graph=verify_sources();template=read(ref(HERE/'INPUTS_TEMPLATE_V1.json'))
    need(template['status']=='TEMPLATE_ONLY' and template['template_only'] is True,'preparation not outcome')
    need(template['future_result_counts'] is None and template['future_figure_values'] is None,'no future counts/figures fabricated')
    for name in ('formal_calibration_peer','model8192','model8192_DATA_peer','schedule'):
        need(template[name]=={'path':None,'sha256':None},'actual pending exact ref '+name)
    for path in graph['files']:
        if path.endswith('.py'):ast.parse(Path(path).read_bytes(),filename=path)
    return dict(status='PASS_SOURCE_TEMPLATE_ONLY',source_graph=ref(HERE/'SOURCE_GRAPH_V1.json'),inputs_template=ref(HERE/'INPUTS_TEMPLATE_V1.json'),actual8192_results_read=0,scientific_figures_generated=0,native_GPU_Git_W_actions=0)

def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--mode',choices=('check-source','check-inputs','build'),required=True)
    parser.add_argument('--inputs');parser.add_argument('--inputs-sha256');parser.add_argument('--out')
    args=parser.parse_args();verify_sources()
    if args.mode=='check-source':
        need(not any((args.inputs,args.inputs_sha256,args.out)),'source check cannot read future outputs')
        print(json.dumps(check_preparation()),flush=True);return 0
    need(args.inputs and args.inputs_sha256,'parent exact actual result input bindings')
    q=dict(path=args.inputs,sha256=args.inputs_sha256);inputs=ready_inputs(read(q))
    if args.mode=='check-inputs':
        need(args.out is None,'input gate cannot build silently')
        print(json.dumps(dict(status='EXACT_REFS_PRESENT_PENDING_BUILD_SEMANTIC_GATE',inputs=q,native_GPU_Git_W_actions=0)),flush=True);return 0
    need(args.out,'exclusive new report05 output')
    from build_report_v1 import build
    result=build(inputs,Path(args.out),q);verify_sources();print(json.dumps(result),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
