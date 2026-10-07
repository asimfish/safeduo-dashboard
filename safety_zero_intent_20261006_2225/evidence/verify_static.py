"""Terminal source/immutability gate; scientific outcome is reported separately."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,re,shutil,subprocess,tempfile

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_zero_worktree_20261006')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plans=[json.loads(p.read_text()) for p in sorted((HERE/'plans').glob('holdout_*_plan.json'))]
    hashes={}
    for p in plans:
        for rel,s in p['source_sha256'].items():
            file=Path(p['cwd'])/rel;assert sha(file)==s,file;hashes[str(file)]=s
        assert sha(p['checkpoint_path'])==p['checkpoint_sha256']
        for file,s in p['research_source_sha256'].items():assert sha(file)==s,file
    intake=json.loads((HERE/'INTAKE.json').read_text());assert sha(Path(intake['prior'])/'SEALED.json')==intake['prior_seal_sha256']
    py=[]
    for p in sorted(HERE.rglob('*.py')):
        if '__pycache__' not in p.parts:ast.parse(p.read_text());py.append(str(p.relative_to(HERE)))
    node=shutil.which('node');assert node,'native JavaScript parser required'
    scripts=re.findall(r'<script\b[^>]*>(.*?)</script>',(HERE/'panel_template.html').read_text(),re.S);assert scripts
    for text in scripts:
        with tempfile.NamedTemporaryFile('w',suffix='.js') as f:
            f.write(text);f.flush();r=subprocess.run([node,'--check',f.name],capture_output=True,text=True);assert r.returncode==0,r.stderr
    public=WORK/'docs'/HERE.name
    assert public.is_dir() and len(list(public.iterdir()))>=8,'complete public bundle must exist before static delivery gate'
    patterns=[r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',r'\bsk-[A-Za-z0-9_-]{24,}\b']
    scanned=0
    asset_root=Path('/mnt/nas/data/lyf/double_hand/safeduo_dashboard_zero_assets_20261006')/HERE.name
    assert asset_root.is_dir(), 'pinned evidence bundle required'
    for p in list(public.rglob('*'))+list(asset_root.rglob('*')):
        if p.is_file() and p.suffix in ['.py','.json','.md','.html','.log']:
            text=p.read_text();assert not any(re.search(pattern,text) for pattern in patterns),p;scanned+=1
    diff=subprocess.run(['git','-c','core.filemode=false','-C',str(WORK),'diff','--check','--','index.html','docs/'+HERE.name],capture_output=True,text=True)
    assert diff.returncode==0,diff.stdout+diff.stderr
    result=dict(status='PASS_SOURCE_STATIC_IMMUTABILITY_GATE',production_sources=len(hashes),checkpoint_sha256=plans[0]['checkpoint_sha256'],research_hashes_per_plan=len(plans[0]['research_source_sha256']),prior_seal_unchanged=True,python_ast_files=py,native_javascript_scripts=len(scripts),public_text_files_scanned=scanned,git_owned_diff_check_pass=True,physical_safety_certified=False,utc=datetime.now(timezone.utc).isoformat())
    with (HERE/'static_verification.json').open('x') as f:json.dump(result,f,indent=2)
    print('STATIC_GATE_PASS',len(hashes),len(py),scanned,flush=True)
if __name__=='__main__':main()
