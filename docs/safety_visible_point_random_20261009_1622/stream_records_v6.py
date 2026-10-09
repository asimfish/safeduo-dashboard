"""Bounded-memory exact typed records; streaming ZIP exports stay np.load compatible."""
from pathlib import Path
import json
import mmap
import zipfile
import numpy as np
class StreamRecords:
    def __init__(self, root, count):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.count=int(count);self.filled=0;self.arrays={};self.schema={}
    def append(self, record):
        if self.filled>=self.count:raise ValueError('registered record denominator exceeded')
        if self.arrays and set(record)!=set(self.arrays):raise ValueError('record schema changed')
        for key,value in record.items():
            if not key.replace('_','').isalnum():raise ValueError('unsafe field name')
            a=np.asarray(value)
            if a.dtype.hasobject:raise ValueError('object field forbidden')
            if key not in self.arrays:
                self.schema[key]=dict(dtype=a.dtype.str,shape=list(a.shape),bytes=int(a.nbytes*self.count))
                self.arrays[key]=np.lib.format.open_memmap(self.root/(key+'.npy'),mode='w+',dtype=a.dtype,shape=(self.count,)+a.shape)
            target=self.arrays[key]
            if target.dtype!=a.dtype or target.shape[1:]!=a.shape:raise ValueError('record dtype or shape changed')
            target[self.filled]=a
        self.filled+=1
        if self.filled%8==0 or self.filled==self.count:
            for a in self.arrays.values():
                a.flush();a._mmap.madvise(mmap.MADV_DONTNEED)
            (self.root/'progress.json').write_text(json.dumps(dict(filled=self.filled,registered=self.count,schema=self.schema),indent=2)+'\n')
    def close(self, output):
        if self.filled!=self.count:raise ValueError('incomplete recorded denominator')
        for a in self.arrays.values():a.flush()
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
            for key,a in self.arrays.items():
                with archive.open(key+'.npy','w',force_zip64=True) as stream:
                    np.lib.format.write_array(stream,a,allow_pickle=False)
                a._mmap.madvise(mmap.MADV_DONTNEED)
        for a in self.arrays.values():a._mmap.close()
        with np.load(output,allow_pickle=False) as z:
            if set(z.files)!=set(self.arrays):raise ValueError('export schema differs')
        (self.root/'closed.json').write_text(json.dumps(dict(status='complete',filled=self.filled,registered=self.count,output=str(output),schema=self.schema),indent=2)+'\n')
