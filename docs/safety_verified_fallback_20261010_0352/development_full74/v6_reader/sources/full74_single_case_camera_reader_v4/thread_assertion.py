"""Identify reached thread checks from reviewed source; do not invent raw values."""
import ast,hashlib
from pathlib import Path
from evidence_io import HERE,require


def dump(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def verify_reached_assertion(e,anchor,request):
    contract=e.js(HERE/'THREAD_ASSERTION_CONTRACT_V2.json')
    entry=Path(anchor['entry']);helper=Path(anchor['thread_helper'])
    tree=ast.parse(e.raw(entry,request['source_files'][str(entry)],1024**2))
    shared=ast.parse(e.raw(helper,request['source_files'][str(helper)],1024**2))
    check=next(n for n in shared.body if isinstance(n,ast.FunctionDef) and n.name=='assert_threads')
    require(dump(check)==contract['assert_threads_AST_sha256'],'exact reviewed strict thread assertion')
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    cls=next(n for n in ast.walk(main) if isinstance(n,ast.ClassDef) and n.name=='SingleThreadAppLauncher')
    config=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_config_resolution')
    require(dump(config)==contract['launcher_config_AST_sha256'],'actual app config threads1 before startup')
    block=next(n for n in main.body if isinstance(n,ast.Try) and any(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and
        ast.unparse(x.value.func)=='launcher.__init__' for x in n.body))
    calls={ast.unparse(n.value.func):i for i,n in enumerate(block.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)}
    require(calls['launcher.__init__']<calls['assert_threads']<calls['run_in_existing_parent_app'],'thread assertion unavoidable after app and before complete diagnostic')
    return dict(validated=True,complete_native_and_outer_actual0=True,source_entry=str(entry),entry_sha256=request['source_files'][str(entry)],
        helper=str(helper),helper_sha256=request['source_files'][str(helper)],assert_threads_AST_sha256=dump(check),
        launcher_config_AST_sha256=dump(config),raw_environment_record_persisted=False)
