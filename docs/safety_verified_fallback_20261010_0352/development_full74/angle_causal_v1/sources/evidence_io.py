"""Bounded, read-only evidence access. Inodes are deliberately not identities on CIFS."""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import stat
import sys
import zipfile

H = Path('/home/liyufeng/safeduo/artifacts/safety_verified_fallback_20261010_0352')
R = Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352')
HERE = H / 'astra/full74_camera_angle_diagnostic_v1'
OUTPUT = R / 'astra_full74_independent_reader_v1/camera_angle_diagnostic_v1'
SCREEN = H / 'astra/full74_screen'
NATIVE = R / 'astra_full74_screen_native_v2'
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
MAX_FILE = 128 * 1024**2


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def cpu_limits():
    require(sys.dont_write_bytecode, 'launch with python -B')
    for key in THREADS:
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def contained(path, root):
    p, r = Path(path).absolute(), Path(root).absolute()
    require(p.is_relative_to(r) and p.resolve() == p, 'path escapes or uses symlink: ' + str(p))
    return p


def metadata(s):
    require(stat.S_ISREG(s.st_mode), 'regular file required')
    return dict(dev=s.st_dev, size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def fingerprint(path, limit=None):
    p = Path(path)
    require(p.resolve() == p.absolute(), 'symlink input')
    with p.open('rb') as f:
        before = metadata(os.fstat(f.fileno()))
        require(limit is None or before['size'] <= limit, 'file exceeds bounded read size')
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1024**2), b''):
            digest.update(block)
        require(before == metadata(os.fstat(f.fileno())) == metadata(p.stat()), 'file changed during hashing')
    return dict(path=str(p), sha256=digest.hexdigest(), **before)


def sha(path):
    return fingerprint(path)['sha256']


def pairs_no_duplicates(pairs):
    d = {}
    for k, v in pairs:
        require(k not in d, 'duplicate JSON key: ' + k)
        d[k] = v
    return d


def parse_json(raw):
    def reject(x):
        raise ValueError('nonfinite JSON constant ' + x)
    return json.loads(raw, object_pairs_hook=pairs_no_duplicates, parse_constant=reject)


def write_new(path, value):
    p = contained(path, OUTPUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


class Evidence:
    def __init__(self):
        self.ledger = {}
        self.external_frozen = None

    def hash(self, path, expected=None, limit=None):
        p = Path(path).absolute()
        if not any(p.is_relative_to(r) for r in (H, R, Path('/home/liyufeng/safeduo'))):
            if self.external_frozen is None:
                anchors = self.js(HERE / 'anchors.json')
                binding = SCREEN / 'INPUT_BINDING_V1.json'
                self.external_frozen = dict(self.js(binding, anchors['files'][str(binding)])['files'])
                for name, digest in anchors['external_operational_sources'].items():
                    require(anchors['files'][name] == digest, 'operational source anchor disagreement')
                    if name in self.external_frozen:
                        require(self.external_frozen[name] == digest, 'operational source changes frozen509')
                    self.external_frozen[name] = digest
            require(str(p) in self.external_frozen and expected == self.external_frozen[str(p)],
                    'external read requires exact frozen509 or pinned operational path and SHA')
        record = fingerprint(p, limit)
        if expected is not None:
            require(record['sha256'] == expected, 'SHA mismatch: ' + str(p))
        if str(p) in self.ledger:
            require(self.ledger[str(p)] == record, 'evidence changed: ' + str(p))
        self.ledger[str(p)] = record
        return record['sha256']

    def raw(self, path, expected=None, limit=MAX_FILE):
        digest = self.hash(path, expected, limit)
        with Path(path).open('rb') as f:
            raw = f.read(limit + 1)
        require(len(raw) <= limit and hashlib.sha256(raw).hexdigest() == digest, 'read bytes changed or oversized')
        return raw

    def js(self, path, expected=None):
        return parse_json(self.raw(path, expected, 16*1024**2))

    def npy(self, path, shape=None, dtype=None):
        import numpy as np
        raw = self.raw(path)
        a = np.load(io.BytesIO(raw), allow_pickle=False)
        require(isinstance(a, np.ndarray) and not a.dtype.hasobject, 'plain NPY required')
        require(a.nbytes <= MAX_FILE, 'array too large')
        if shape is not None:
            require(a.shape == tuple(shape), 'array shape: ' + str(path))
        if dtype is not None:
            require(a.dtype == np.dtype(dtype), 'array dtype: ' + str(path))
        return a

    def npz(self, path, expected=None):
        import numpy as np
        raw = self.raw(path, expected)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            members = z.infolist()
            require(len(members) <= 256 and len({v.filename for v in members}) == len(members), 'NPZ duplicate/member bound')
            require(sum(v.file_size for v in members) <= MAX_FILE, 'NPZ expansion bound')
            require(all('/' not in v.filename and v.filename.endswith('.npy') for v in members), 'NPZ member name')
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            result = {k: z[k] for k in z.files}
        require(all(not v.dtype.hasobject for v in result.values()), 'object NPZ forbidden')
        return result

    def npy_first_row(self,path,shape,dtype):
        """Hash all bytes but read only initialized row0; never inspect tail rows."""
        import numpy as np
        self.hash(path,limit=MAX_FILE)
        with Path(path).open('rb') as f:
            version=np.lib.format.read_magic(f)
            require(version in ((1,0),(2,0)),'bounded NPY header version')
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            actual,fortran,dt=reader(f,max_header_size=10000)
            require(actual==tuple(shape) and not fortran and dt==np.dtype(dtype) and not dt.hasobject,'native stream shape/dtype/order')
            count=int(np.prod(shape[1:]));total=int(np.prod(shape))*dt.itemsize
            require(f.tell()+total==Path(path).stat().st_size and total<=MAX_FILE,'native stream exact file size')
            raw=f.read(count*dt.itemsize)
            require(len(raw)==count*dt.itemsize,'truncated native initial row')
        self.hash(path,limit=MAX_FILE)
        return np.frombuffer(raw,dtype=dt).reshape(shape[1:]).copy()

    def verify(self, files, root=None):
        require(isinstance(files, dict) and files, 'empty file binding')
        for p, digest in files.items():
            if root is not None:
                contained(p, root)
            self.hash(p, digest)

    def recheck(self):
        for p, old in list(self.ledger.items()):
            require(fingerprint(p) == old, 'input changed before closure: ' + p)


def equal_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order='C') == b.tobytes(order='C')


def exact(a, b, name):
    require(equal_bits(a, b), 'dtype/shape/byte mismatch: ' + name)
