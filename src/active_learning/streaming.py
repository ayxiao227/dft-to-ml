"""Bounded JSON readers/writers, including legacy arrays and SQLite JSON metadata."""
import hashlib
import io
import json
import ijson
from .config import canonical


def json_parts(value):
    """Byte-identical to canonical(), without a full ledger/string copy."""
    if isinstance(value, dict):
        yield '{'
        for n, key in enumerate(sorted(value)):
            if n: yield ','
            yield from json_parts(key)
            yield ':'
            yield from json_parts(value[key])
        yield '}'
    elif isinstance(value, (list, tuple)):
        yield '['
        for n, item in enumerate(value):
            if n: yield ','
            yield from json_parts(item)
        yield ']'
    elif isinstance(value, str):
        yield '"'
        for start in range(0, len(value), 16384):
            yield json.dumps(value[start:start+16384], ensure_ascii=True)[1:-1]
        yield '"'
    else:
        yield canonical(value)


class MetaReader(io.RawIOBase):
    """Read a TEXT meta value as UTF-8 bytes without loading it into Python."""
    def __init__(self, db, key):
        row=db.execute('SELECT rowid FROM meta WHERE key=?',(key,)).fetchone()
        self.blob=db.blobopen('meta','value',row[0],readonly=True) if row else None
    def readable(self): return True
    def readinto(self, target):
        chunk=self.blob.read(len(target)) if self.blob else b''
        target[:len(chunk)]=chunk
        return len(chunk)
    def close(self):
        if self.blob:self.blob.close();self.blob=None
        super().close()


def meta_values(s, key, prefix):
    with io.BufferedReader(MetaReader(s.db, key)) as stream:
        yield from ijson.items(stream, prefix, use_float=True)


def prediction_records(path):
    """Legacy JSON arrays and v2 JSONL; one record at a time."""
    with path.open('rb') as stream:
        if path.suffix == '.jsonl':
            for line in stream:
                yield json.loads(line)
        else:
            yield from ijson.items(stream, 'item', use_float=True)


def rows_hash(rows):
    digest = hashlib.sha256()
    digest.update(b'[')
    for n, row in enumerate(rows):
        if n: digest.update(b',')
        for part in json_parts(row): digest.update(part.encode())
    digest.update(b']')
    return digest.hexdigest()
