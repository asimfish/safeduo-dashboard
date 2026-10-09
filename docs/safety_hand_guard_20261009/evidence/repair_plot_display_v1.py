"""Run only the own builder's figure block; all native/score inputs stay identical."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import textwrap

import numpy as np

import build_report_v1 as b

P=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifestpath=P/'PUBLIC_PAYLOAD_MANIFEST.json'
    manifest=json.loads(manifestpath.read_text())
    assert not (P/'PLOT_DISPLAY_REPAIR_V1.json').exists()
    shutil.copyfile(manifestpath,P/'PUBLIC_PAYLOAD_MANIFEST_BEFORE_PLOT_DISPLAY_V1.json')
    data={};bindings={}
    for method in b.METHODS:
        path=b.FIX/f'PAIRED_METRICS_batch0_{method}_V1.npz';bindings[str(path)]=sha(path)
        with np.load(path) as z:data[method]={k:z[k] for k in ['global_input_id','gap_m','force_N','hard_violation_rad','velocity_exceedance_rad_s']}
    source=(P/'build_report_v1.py').read_text()
    block=textwrap.dedent(source[source.index('    plt.rcParams.update'):source.index('    def table')])
    namespace={**vars(b),'metrics':data,'result':b.read(b.FIX/'FIXEDHAND_TRIPLET64_RESULT_V1.json'),'hold':b.read(b.H/'HOLD_FALLBACK64_RESULT_V1.json'),'margin':b.read(b.H/'HAND_MARGIN64_RESULT_V1.json'),'physics_dt':.008333}
    exec(compile(block,'own_builder_plot_block','exec'),namespace)
    for path,expected in bindings.items():assert sha(Path(path))==expected
    for name,expected in manifest['files'].items():
        if not name.endswith(('.png','.svg','.pdf')):assert sha(b.W/name)==expected,'Nonplot was unexpectedly modified '+name
    receipt=dict(status='PASS_DISPLAY_ONLY_FIGURE_REPAIR_NATIVE_METRIC_BYTES_UNCHANGED',changes=['Common outcome legend outside plotted bars','Nonnegative normal-force/hard/speed metrics use nonnegative axes with actual-data upper limits'],unchanged_native_metric_SHA256=bindings,native_or_score_reexecuted=False,source_sha256=sha(P/'build_report_v1.py'))
    (P/'PLOT_DISPLAY_REPAIR_V1.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for name in ['build_report_v1.py','repair_plot_display_v1.py','PLOT_DISPLAY_REPAIR_V1.json']:
        shutil.copyfile(P/name,b.D/'evidence'/name)
    manifest['files']={str(p.relative_to(b.W)):sha(p) for p in sorted(b.D.rglob('*')) if p.is_file()};manifest['files']['index.html']=sha(b.W/'index.html')
    manifest.update(total_files=len(manifest['files']),total_bytes=sum((b.W/n).stat().st_size for n in manifest['files']))
    manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(['git','add','--',*manifest['files']],cwd=b.W,check=True)
    print(receipt['status'],flush=True)


if __name__=='__main__':main()
