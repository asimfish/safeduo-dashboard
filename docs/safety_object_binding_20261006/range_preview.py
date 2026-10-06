"""Loopback delivery preview with explicit HTTP byte-range video support."""
import http.server,os,re
class RangePreview(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
    def send_head(self):
        self.remaining=None
        path=self.translate_path(self.path)
        if os.path.isdir(path):return super().send_head()
        try:f=open(path,'rb')
        except OSError:return super().send_head()
        size=os.fstat(f.fileno()).st_size
        value=self.headers.get('Range');m=re.fullmatch(r'bytes=(\d*)-(\d*)',value or '')
        if not m:
            self.send_response(200);self.send_header('Content-type',self.guess_type(path));self.send_header('Content-Length',str(size));self.send_header('Accept-Ranges','bytes');self.end_headers();return f
        left,right=m.groups()
        if left:start=int(left);end=int(right) if right else size-1
        elif right:start=max(0,size-int(right));end=size-1
        else:start=size;end=size-1
        end=min(end,size-1)
        if start> end or start>=size:
            f.close();self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return None
        self.send_response(206);self.send_header('Content-type',self.guess_type(path));self.send_header('Accept-Ranges','bytes');self.send_header('Content-Range',f'bytes {start}-{end}/{size}');self.remaining=end-start+1;self.send_header('Content-Length',str(self.remaining));self.end_headers();f.seek(start);return f
    def copyfile(self,source,output):
        if self.remaining is None:return super().copyfile(source,output)
        while self.remaining:
            chunk=source.read(min(65536,self.remaining))
            if not chunk:break
            output.write(chunk);self.remaining-=len(chunk)
