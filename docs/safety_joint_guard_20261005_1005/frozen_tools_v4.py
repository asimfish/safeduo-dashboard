"""Compile exactly the bytes whose SHA is recorded as the execution identity."""
import argparse
import hashlib
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent

def load_source(filename):
    path=HERE/filename
    content=path.read_bytes()
    module=types.ModuleType('safeduo_frozen_'+path.stem)
    module.__file__=str(path)
    module.EXECUTED_SOURCE_SHA256=hashlib.sha256(content).hexdigest()
    exec(compile(content,str(path),'exec'),module.__dict__)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==module.EXECUTED_SOURCE_SHA256
    return module

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('target',choices=['analysis','visual']);a=parser.parse_args()
    load_source('analyze_v4.py' if a.target=='analysis' else 'verify_native_visual_v2.py').main()
