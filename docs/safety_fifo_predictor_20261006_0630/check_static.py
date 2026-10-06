"""Static checks for owned delivery only; retain frozen whitespace exceptions."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,re,subprocess,tempfile
HERE=Path(__file__).resolve().parent
WORK=Path('/home/liyufeng/safeduo-dashboard-feasible-guard-20261005')
PUBLIC=WORK/'docs'/HERE.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parsed=0
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),str(p));parsed+=1
    code='\n'.join(re.findall(r'<script[^>]*>(.*?)</script>',(PUBLIC/'index.html').read_text(),re.S))
    with tempfile.NamedTemporaryFile('w',suffix='.js') as f:
        f.write(code);f.flush();subprocess.run(['node','--check',f.name],check=True,capture_output=True)
    paths=[WORK/'index.html',*sorted(p for p in PUBLIC.rglob('*') if p.is_file())]
    text=[];secret=re.compile(r'(?:sk-[A-Za-z0-9_-]{24,}|tk-[a-z0-9]{20,}-[a-z0-9]{8,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    for p in paths:
        if p.suffix not in ('.py','.html','.md','.json','.log'):continue
        assert not secret.search(p.read_text()),'credential-like bytes in an owned delivery file: '+p.name
        text.append(p)
    checks=[];frozen_warnings=[]
    for staged in (False,True):
        argv=['git','diff','--check']+(['--cached'] if staged else [])+['--','index.html','docs/'+HERE.name]
        out=subprocess.run(argv,cwd=WORK,text=True,capture_output=True)
        if out.returncode:
            names=set(re.findall(r'^(.*?):\d+:',out.stdout,re.M))
            assert names and all(n.startswith('docs/'+HERE.name+'/') for n in names)
            for n in names:
                relative=n.split('docs/'+HERE.name+'/',1)[1];src=HERE/relative;pub=WORK/n
                assert src.is_file() and sha(src)==sha(pub),'mutable/new delivery whitespace error '+n
                frozen_warnings.append(dict(path=n,source_sha256=sha(src),scope='preserved source-bound original artifact; not reformatted for delivery'))
        checks.append(dict(staged=staged,exit_code=out.returncode,diagnostic_sha256=hashlib.sha256(out.stdout.encode()).hexdigest()))
    r=dict(status='PASS_OWNED_SYNTAX_SECRETS_AND_REVIEWED_DIFF',utc=datetime.now(timezone.utc).isoformat(),python_sources_parsed=parsed,
      inline_javascript_checked=True,owned_text_files_checked=len(text),credential_like_bytes_found=False,diff_checks=checks,frozen_whitespace_warnings=frozen_warnings,
      source_sha256=sha(Path(__file__)),page_sha256=sha(PUBLIC/'index.html'),scope='static and owned-stage integrity only; original immutable whitespace warnings explicitly retained')
    (HERE/'static_verification.json').write_text(json.dumps(r,indent=2)+'\n');print(r['status'],parsed,len(text),len(frozen_warnings),flush=True)
if __name__=='__main__':main()
