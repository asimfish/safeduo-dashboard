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

MAX_PACKET=16*1024**2
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
        self.digest.update(data);self.bytes+=len(data);require(self.bytes<=MAX_PACKET,'camera packet exceeds16MiB')
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

