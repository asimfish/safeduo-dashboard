"""Remove line-end whitespace while proving unchanged SVG element semantics."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

P = Path(__file__).resolve().parent
W = Path('/home/liyufeng/safeduo-dashboard-response-probe-20261009')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(node):
    return (node.tag, tuple(sorted((k, ' '.join(v.split())) for k, v in node.attrib.items())),
            ' '.join((node.text or '').split()), ' '.join((node.tail or '').split()),
            tuple(normalized(child) for child in node))


def main():
    mp = P / 'PUBLIC_PAYLOAD_MANIFEST.json'
    manifest = json.loads(mp.read_text())
    before = P / 'PUBLIC_PAYLOAD_BEFORE_SVG_NORMALIZATION_V1.json'
    assert not before.exists()
    before.write_bytes(mp.read_bytes())
    archive = P / 'SVG_BEFORE_NORMALIZATION_V1'
    archive.mkdir()
    records = []
    for name, expected in manifest['files'].items():
        path = W / name
        assert sha(path) == expected
        if path.suffix != '.svg':
            continue
        old = path.read_bytes()
        shutil.copyfile(path, archive / path.name)
        new = '\n'.join(line.rstrip() for line in old.decode().splitlines()) + '\n'
        assert normalized(ET.fromstring(old)) == normalized(ET.fromstring(new))
        path.write_text(new)
        records.append(dict(path=name, before_sha256=expected, after_sha256=sha(path),
                            normalized_XML_elements_attributes_text_identical=True))
    assert len(records) == 12
    out = W / manifest['slug']
    proof = dict(status='PASS_SVG_WHITESPACE_ONLY_XML_SEMANTICS_UNCHANGED',
                 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), records=records,
                 native_or_scoring_reexecuted=False, all8_native_archives_unchanged=True)
    (P / 'SVG_NORMALIZATION_RECEIPT_V1.json').write_text(json.dumps(proof, indent=2) + '\n')
    shutil.copyfile(P / 'SVG_NORMALIZATION_RECEIPT_V1.json', out / 'evidence/SVG_NORMALIZATION_RECEIPT_V1.json')
    shutil.copyfile(Path(__file__), out / 'evidence/normalize_svg_v1.py')
    page = out / 'index.html'
    text = page.read_text()
    marker = '<a href="evidence/BUNDLE_RECEIPT_V1.json">'
    assert text.count(marker) == 1
    text = text.replace(marker, '<p><a href="evidence/SVG_NORMALIZATION_RECEIPT_V1.json">SVG 格式修正的逐图语义证明</a> · <a href="evidence/normalize_svg_v1.py">修正脚本</a></p>' + marker, 1)
    page.write_text(text)
    files = [W / 'index.html'] + sorted(q for q in out.rglob('*') if q.is_file())
    manifest.update(files={str(q.relative_to(W)): sha(q) for q in files}, total_files=len(files),
                    total_bytes=sum(q.stat().st_size for q in files))
    mp.write_text(json.dumps(manifest, indent=2) + '\n')
    print(proof['status'], manifest['total_files'], flush=True)


if __name__ == '__main__':
    main()
