"""Bounded JSON streaming for exactly32 retained camera receipt references.

Only the two known camera aggregate files use this reader. Ordinary Evidence.js
stays at16MiB. At most one4MiB receipt is decoded at once; nested metadata is
hashed canonically and independently compared with each SHA-bound state JSON.
"""
import hashlib
import json
from pathlib import Path
import re
from evidence_io import parse_json,require

MAX_PACKET=32*1024**2
MAX_RECORD=4*1024**2
CHUNK=64*1024
META=('status','reasons','arms','wide_context','image_count','qualified_image_count')
RECEIPT_KEYS={'env_id','step','substep','capture_kind','status','reasons','state','sha256','image_count','qualified_image_count','arms','wide_context','scope'}
TOKENS=re.compile(rb'["{}\[\]\\]')
SPACE=re.compile(rb'[ \t\r\n]*')
DELIM=re.compile(rb'[,}\] \t\r\n]')


def metadata_digest(value):
    fields={k:value.get(k) for k in META}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


class Stream:
    def __init__(self,stream,chunk=CHUNK):
        self.stream=stream;self.chunk=chunk;self.buf=b'';self.pos=0;self.eof=False;self.digest=hashlib.sha256();self.bytes=0;self.peak_buffer_bytes=0

    def grow(self):
        if self.eof:return False
        data=self.stream.read(self.chunk)
        if not data:self.eof=True;return False
        self.digest.update(data);self.bytes+=len(data);require(self.bytes<=MAX_PACKET,'camera packet exceeds32MiB')
        self.buf+=data;self.peak_buffer_bytes=max(self.peak_buffer_bytes,len(self.buf));return True

    def space(self):
        while True:
            self.pos=SPACE.match(self.buf,self.pos).end()
            if self.pos<len(self.buf) or self.eof:return
            self.buf=b'';self.pos=0;self.grow()

    def peek(self):
        self.space();return self.buf[self.pos:self.pos+1]

    def take(self,token):
        require(self.peek()==token,'camera JSON delimiter '+repr(token));self.pos+=1

    def value(self,limit=MAX_RECORD):
        self.space();self.buf=self.buf[self.pos:];self.pos=0
        require(bool(self.buf),'truncated camera JSON value')
        first=self.buf[:1];compound=first in (b'{',b'[');quoted=first==b'"'
        scan=0;depth=0;inside=False
        while True:
            if compound or quoted:
                match=TOKENS.search(self.buf,scan)
                if match:
                    at=match.start();ch=self.buf[at:at+1]
                    require(at<limit,'camera JSON value exceeds4MiB bound')
                    if inside and ch==b'\\':
                        if at+1>=len(self.buf):
                            require(self.grow(),'truncated camera JSON escape');continue
                        scan=at+2;continue
                    scan=at+1
                    if ch==b'"':
                        inside=not inside
                        if quoted and not inside:return self.finish(scan,limit)
                    elif not inside:
                        if ch in (b'{',b'['):depth+=1;require(depth<=64,'camera JSON nesting limit64')
                        elif ch in (b'}',b']'):
                            depth-=1
                            if depth==0:return self.finish(scan,limit)
                    continue
            else:
                match=DELIM.search(self.buf,scan)
                if match:return self.finish(match.start(),limit)
                if self.eof:return self.finish(len(self.buf),limit)
                scan=len(self.buf)
            require(len(self.buf)<=limit,'camera JSON value exceeds4MiB bound')
            require(self.grow(),'truncated camera JSON compound/string')

    def finish(self,end,limit):
        require(0<end<=limit,'camera JSON value bound')
        value=parse_json(self.buf[:end]);self.pos=end;return value


def project(record,slot):
    require(isinstance(record,dict) and set(record)==RECEIPT_KEYS,'camera receipt exact field inventory')
    require(type(record['env_id']) is int and record['env_id']==slot and record['step']==-1 and
            record['substep'] is None and record['capture_kind']=='initial','exact ordered32 initial slots')
    require(record['status'] in ('qualified','failed'),'camera metadata status')
    p=Path(record['state'])
    require(not p.is_absolute() and '..' not in p.parts and str(p)==record['state'] and len(p.parts)==5 and
            p.parts[:2]==('qualified_views','initial') and p.parent.name==f'env_{slot:03d}' and p.name=='state.json','camera state reference slot/path')
    digest=record['sha256'];require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None,'camera state SHA')
    for key,maximum in (('image_count',96),('qualified_image_count',12)):
        require(type(record[key]) is int and 0<=record[key]<=maximum,'camera count bound '+key)
    return {k:record[k] for k in ('env_id','step','substep','capture_kind','status','state','sha256','image_count','qualified_image_count')}|{'_metadata_sha256':metadata_digest(record)}


def read_index(e,path,expected=None,*,kind,chunk=CHUNK):
    path=Path(path);require(kind in ('worker','shared'),'camera packet kind')
    expected_name='initial_camera_receipts.json' if kind=='worker' else 'receipts.json'
    require(path.name==expected_name,'specialized camera index filename only')
    digest=e.hash(path,expected,limit=MAX_PACKET)
    result={};receipts=None;keys=set()
    with path.open('rb') as stream:
        s=Stream(stream,chunk);s.take(b'{')
        while s.peek()!=b'}':
            key=s.value(4096);require(isinstance(key,str) and key not in keys,'duplicate/nonstring camera packet key');keys.add(key);s.take(b':')
            if key=='receipts':
                receipts=[];s.take(b'[')
                while s.peek()!=b']':
                    require(len(receipts)<32,'more than32 camera receipts')
                    receipts.append(project(s.value(),len(receipts)))
                    if s.peek()==b']':break
                    s.take(b',');require(s.peek()!=b']','trailing comma in camera receipts')
                s.take(b']');require(len(receipts)==32,'exactly32 camera receipts required')
            else:result[key]=s.value()
            if s.peek()==b'}':break
            s.take(b',');require(s.peek()!=b'}','trailing comma in camera packet')
        s.take(b'}');require(s.peek()==b'' and s.eof,'trailing camera JSON content')
        require(s.digest.hexdigest()==digest,'camera packet bytes changed during streaming')
    wanted={'qualified','receipts'} if kind=='worker' else {'receipts','native_invariant','native_before_all64','native_final_all64','scope'}
    require(keys==wanted and receipts is not None,'camera packet exact top-level field inventory')
    require(len({r['state'] for r in receipts})==32 and len({str(Path(r['state']).parent.parent) for r in receipts})==1,'one capture and32 distinct state paths')
    if kind=='worker':require(type(result['qualified']) is bool,'worker aggregate boolean')
    e.hash(path,digest,limit=MAX_PACKET)
    result['receipts']=receipts
    result['_stream']=dict(file_bytes=s.bytes,peak_buffer_bytes=s.peak_buffer_bytes,packet_limit_bytes=MAX_PACKET,
        receipt_limit_bytes=MAX_RECORD,decoded_receipts=32,retained_full_receipt_objects=0,sha256=digest)
    return result
