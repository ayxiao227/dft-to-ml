"""Verified, durable report stages, separate from the experiment's immutable artifacts."""
import json
import sqlite3
import time
from pathlib import Path
from .artifacts import file_hash, publish_file, atomic_write
from .config import canonical, hashed
from .state import crash

SETTINGS = {'sql_cache_kib': 8192, 'ingest_batch': 1, 'plot_limit': 20000,
            'plot_seed': 271828, 'fingerprint_chunk': 1024, 'working_memory_target_gib': 2}


def log(message, start=None):
    elapsed='' if start is None else f' elapsed={time.monotonic()-start:.1f}s'
    print(f'Report: {message}{elapsed}', flush=True)


class StageCache:
    def __init__(self, root, identity, failpoint=None):
        self.root=Path(root)/'.report-work'/identity
        self.root.mkdir(parents=True, exist_ok=True)
        self.failpoint=failpoint
        try:
            self.db=sqlite3.connect(self.root/'progress.sqlite3')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS stages(key TEXT PRIMARY KEY, manifest TEXT NOT NULL)')
            if self.db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise sqlite3.DatabaseError('quick_check failed')
            self.db.commit()
        except sqlite3.DatabaseError as exc:
            if hasattr(self,'db'):self.db.close()
            raise ValueError(f'Corrupt report progress ledger retained at {self.root}; restore a verified copy or move this work directory aside and retry: {exc}') from exc

    def run(self, key, build):
        token=hashed(key); target=self.root/'objects'/token
        try:row=self.db.execute('SELECT manifest FROM stages WHERE key=?',(key,)).fetchone()
        except sqlite3.DatabaseError as exc:raise ValueError(f'Corrupt report progress ledger retained at {self.root}; move this work directory aside and retry: {exc}') from exc
        self.last_cached=bool(row)
        if row:
            try:
                doc=json.loads(row[0])
                if 'stage.json' not in doc['files']:raise ValueError('missing stage payload hash')
                for name,sha in doc['files'].items():
                    if file_hash(target/name)!=sha:raise ValueError('hash mismatch: '+name)
                if json.loads((target/'stage.json').read_text())!=doc['data']:raise ValueError('stage payload/ledger mismatch')
            except (OSError,ValueError,KeyError) as exc:
                raise ValueError(f'Corrupt reporting stage {key}; evidence retained at {target}. Restore verified files or move this report work directory aside and retry: {exc}') from exc
            log(f'{key}: cached (verified)')
            return doc['data'], {name:target/name for name in doc['files'] if name!='stage.json'}
        # An interrupted copy may leave a partial anonymous publication file.
        # Stop before producing another one; preserve it for explicit review.
        remnants=list(target.glob('.*')) if target.exists() else []
        if remnants:raise ValueError(f'Interrupted report publication evidence retained at {target}: {remnants}. Move these temporary files aside after checking that no writer is active, then retry.')
        start=time.monotonic();log(f'{key}: started')
        pending=self.root/'pending'/token;pending.mkdir(parents=True, exist_ok=True)
        data,files=build(pending)
        # The payload is itself a verified file, so ledger data cannot drift unnoticed.
        payload=canonical(data).encode();atomic_write(pending/'stage.json',payload)
        files={**files,'stage.json':pending/'stage.json'}
        crash('before_report_publication',self.failpoint)
        hashes={}
        for name,path in files.items():
            publish_file(target/name,path);hashes[name]=file_hash(target/name)
        crash('after_report_publication',self.failpoint)
        doc={'data':data,'files':hashes}
        self.db.execute('INSERT INTO stages VALUES (?,?)',(key,canonical(doc)));self.db.commit()
        crash('after_report_commit',self.failpoint)
        # Only owned, successfully published pending files are removed.
        for path in files.values():Path(path).unlink(missing_ok=True)
        log(f'{key}: committed {len(files)} files',start)
        return data,{name:target/name for name in files if name!='stage.json'}

    def close(self):self.db.close()


def progress(root):
    """Read committed stage names without a writer lock or heavyweight verification."""
    result=[]
    base=Path(root)/'.report-work'
    if not base.exists():return result
    for path in sorted(base.glob('*/progress.sqlite3')):
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=1) as db:
            db.execute('PRAGMA query_only=ON')
            result.append({'identity':path.parent.name,'committed_stages':[r[0] for r in db.execute('SELECT key FROM stages ORDER BY rowid')],
                           'verification':'committed snapshot only; cached files verified on resume'})
    return result
