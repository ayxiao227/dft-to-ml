"""Authoritative SQLite ledger with one process writer and immutable artifacts."""
from contextlib import contextmanager
import fcntl
import json
import sqlite3
from pathlib import Path
from .config import canonical, hashed, validate
from .artifacts import atomic_write, file_hash

SCHEMA=1
DDL='''
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE molecules(id TEXT PRIMARY KEY, record TEXT NOT NULL);
CREATE TABLE partitions(repeat INTEGER NOT NULL, id TEXT NOT NULL REFERENCES molecules(id),
 origin TEXT NOT NULL CHECK(origin IN ('initial','pool','validation','final')), group_key TEXT NOT NULL,
 PRIMARY KEY(repeat,id));
CREATE TABLE runs(repeat INTEGER, strategy TEXT, finalized INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(repeat,strategy));
CREATE TABLE rounds(repeat INTEGER, strategy TEXT, round INTEGER, status TEXT NOT NULL,
 metadata TEXT NOT NULL, PRIMARY KEY(repeat,strategy,round),
 FOREIGN KEY(repeat,strategy) REFERENCES runs(repeat,strategy));
CREATE TABLE selections(repeat INTEGER, strategy TEXT, round INTEGER, id TEXT REFERENCES molecules(id),
 detail TEXT NOT NULL, PRIMARY KEY(repeat,strategy,id),
 FOREIGN KEY(repeat,strategy,round) REFERENCES rounds(repeat,strategy,round));
CREATE TABLE attempts(repeat INTEGER, strategy TEXT, id TEXT REFERENCES molecules(id), attempt INTEGER,
 round INTEGER NOT NULL, calculation_id TEXT UNIQUE NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('reserved','prepared','pending','accepted','rejected','failed','excluded')),
 reason TEXT, input_record TEXT, log_hash TEXT, evidence TEXT, cost TEXT,
 PRIMARY KEY(repeat,strategy,id,attempt),
 FOREIGN KEY(repeat,strategy,round) REFERENCES rounds(repeat,strategy,round));
CREATE UNIQUE INDEX one_active_attempt ON attempts(repeat,strategy,id)
 WHERE status IN ('reserved','prepared','pending');
CREATE TABLE labels(repeat INTEGER, strategy TEXT, id TEXT, attempt INTEGER, value REAL NOT NULL,
 provenance TEXT NOT NULL, PRIMARY KEY(repeat,strategy,id),
 FOREIGN KEY(repeat,strategy,id,attempt) REFERENCES attempts(repeat,strategy,id,attempt));
CREATE TABLE artifacts(path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, kind TEXT NOT NULL);
CREATE TABLE checkpoints(repeat INTEGER, strategy TEXT, round INTEGER, path TEXT NOT NULL REFERENCES artifacts(path),
 predictions TEXT NOT NULL REFERENCES artifacts(path), metadata TEXT NOT NULL,
 PRIMARY KEY(repeat,strategy,round), FOREIGN KEY(repeat,strategy) REFERENCES runs(repeat,strategy));
CREATE TABLE metrics(repeat INTEGER, strategy TEXT, round INTEGER, evaluator TEXT, partition TEXT,
 data TEXT NOT NULL, PRIMARY KEY(repeat,strategy,round,evaluator,partition));
CREATE TABLE events(sequence INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, data TEXT NOT NULL);
'''
TABLES=('molecules','partitions','runs','rounds','selections','attempts','labels','artifacts','checkpoints','metrics','events')


def crash(point, requested=None):
    """Explicit local test hook. Exceptions model process loss at commit boundaries."""
    if point==requested:raise RuntimeError('Injected crash: '+point)

class Store:
    def __init__(self,root,create=False):
        self.root=Path(root).resolve();self.create=create
        self.db=None;self.lock=None

    def __enter__(self):
        if not self.root.is_dir():raise ValueError('Experiment directory does not exist')
        self.lock=(self.root/'writer.lock').open('a+b')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close();raise RuntimeError('Experiment has an active writer; lock is never deleted by age')
        try:
            path=self.root/'state.sqlite3'
            if not self.create and not path.exists():raise ValueError('Experiment database missing')
            self.db=sqlite3.connect(path,timeout=0,isolation_level=None)
            self.db.row_factory=sqlite3.Row
            self.db.execute('PRAGMA foreign_keys=ON');self.db.execute('PRAGMA synchronous=FULL')
            if self.create:
                self.db.executescript(DDL)
                self.db.execute('INSERT INTO meta VALUES (?,?)',('schema',canonical(SCHEMA)))
            else:self.verify()
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def __exit__(self,*args):
        if self.db:self.db.close();self.db=None
        if self.lock:fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close();self.lock=None

    def rows(self,sql,parameters=()):return [dict(r) for r in self.db.execute(sql,parameters)]
    def meta(self,key,default=None):
        r=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        return json.loads(r[0]) if r else default
    def set_meta(self,key,value):self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,canonical(value)))

    def _digest(self):
        data={table:self.rows(f'SELECT * FROM {table} ORDER BY '+','.join(str(i+1) for i in range(len(self.db.execute(f'SELECT * FROM {table} LIMIT 0').description)))) for table in TABLES}
        data['meta']=self.rows("SELECT * FROM meta WHERE key != 'integrity' ORDER BY key")
        return hashed(data)

    def verify(self):
        if self.meta('schema')!=SCHEMA:raise ValueError('Unsupported state schema')
        if self.meta('integrity')!=self._digest():raise ValueError('Database integrity/config/partition drift')
        config=validate(self.meta('config'))
        if hashed(config)!=self.meta('config_hash'):raise ValueError('Config hash changed')
        from .datasets import verify_sources
        verify_sources(config)
        for a in self.rows('SELECT * FROM artifacts'):
            path=self.root/a['path']
            if not path.is_relative_to(self.root) or file_hash(path)!=a['sha256']:
                raise ValueError('Artifact hash/path changed: '+a['path'])
        import numpy as np, sklearn
        from rdkit import rdBase
        if self.meta('environment')!={'numpy':np.__version__,'sklearn':sklearn.__version__,'rdkit':rdBase.rdkitVersion}:
            raise ValueError('Pinned environment changed; new experiment required')

    @contextmanager
    def transaction(self,failpoint=None):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.set_meta('integrity',self._digest())
            crash('before_database_commit',failpoint)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK');raise
        crash('after_database_commit',failpoint)

    def artifact(self,relative,content,kind,failpoint=None):
        path=self.root/relative
        if not path.resolve().is_relative_to(self.root):raise ValueError('Unsafe artifact path')
        sha=__import__('hashlib').sha256(content).hexdigest()
        old=self.db.execute('SELECT sha256 FROM artifacts WHERE path=?',(relative,)).fetchone()
        if old:
            if old[0]!=sha or file_hash(path)!=sha:raise ValueError('Immutable artifact conflict')
            return sha
        # Orphan after crash can be reused only with matching exact intended bytes.
        if path.exists() and file_hash(path)!=sha:
            raise ValueError('Unreferenced artifact conflicts; retain it for review: '+relative)
        atomic_write(path,content)
        crash('after_artifact_rename',failpoint)
        self.db.execute('INSERT INTO artifacts VALUES (?,?,?)',(relative,sha,kind))
        return sha

    def artifact_file(self,relative,kind):
        """Register an already staged durable file without materializing its bytes."""
        path=self.root/relative
        if not path.resolve().is_relative_to(self.root):raise ValueError('Unsafe artifact path')
        with path.open('rb') as stream:__import__('os').fsync(stream.fileno())
        sha=file_hash(path)
        self.db.execute('INSERT INTO artifacts VALUES (?,?,?)',(relative,sha,kind))
        return sha

    def event(self,kind,data):self.db.execute('INSERT INTO events(kind,data) VALUES (?,?)',(kind,canonical(data)))
    def ensure_mutable(self):
        if self.meta('finalization') is not None:raise ValueError('Experiment frozen/finalized; acquisition and retraining forbidden')

    def reservation_ids(self,key):
        repeat,strategy,round_number=json.loads(key)
        rows=self.rows('SELECT id,status FROM attempts WHERE repeat=? AND strategy=? AND round=? ORDER BY id',(repeat,strategy,round_number))
        if not rows or any(r['status']!='reserved' for r in rows):raise ValueError('Reservation absent or already delivered')
        return [r['id'] for r in rows]

    def labels(self,repeat,strategy):
        return {r['id']:r['value'] for r in self.rows('SELECT id,value FROM labels WHERE repeat=? AND strategy=? ORDER BY id',(repeat,strategy))}

    def complete_round(self,repeat,strategy,round_number):
        pending=self.db.execute("SELECT COUNT(*) FROM attempts WHERE repeat=? AND strategy=? AND round=? AND status IN ('reserved','prepared','pending')",(repeat,strategy,round_number)).fetchone()[0]
        if not pending:self.db.execute("UPDATE rounds SET status='terminal' WHERE repeat=? AND strategy=? AND round=?",(repeat,strategy,round_number))

    def deliver(self,repeat,strategy,round_number,deliveries,failpoint=None,before_commit=None):
        """Atomic batch ingestion; each delivery includes exact attempt/hash evidence."""
        self.ensure_mutable()
        with self.transaction(failpoint):
            for d in deliveries:
                key=(repeat,strategy,d['id'],d['attempt'])
                r=self.db.execute('SELECT * FROM attempts WHERE repeat=? AND strategy=? AND id=? AND attempt=?',key).fetchone()
                if not r or r['round']!=round_number:raise ValueError('Unregistered attempt/round')
                if r['log_hash'] is not None or r['status'] in ('accepted','rejected','failed','excluded'):
                    if r['log_hash']==d.get('log_hash') and r['status']==d['status'] and r['evidence']==canonical(d['evidence']):continue
                    raise ValueError('Conflicting replay quarantined; existing result retained')
                if d['status'] not in ('accepted','rejected','failed','excluded'):raise ValueError('Only terminal deliveries')
                if d['status']=='accepted':
                    value=float(d['value'])
                    import math
                    if not math.isfinite(value) or value<0:raise ValueError('Nonfinite/negative label')
                    self.db.execute('INSERT INTO labels VALUES (?,?,?,?,?,?)',(*key,value,canonical(d['evidence'])))
                self.db.execute('UPDATE attempts SET status=?,reason=?,log_hash=?,evidence=? WHERE repeat=? AND strategy=? AND id=? AND attempt=?',
                    (d['status'],d.get('reason'),d.get('log_hash'),canonical(d['evidence']),*key))
            if before_commit:before_commit()
            crash('after_label_delivery',failpoint)
            self.complete_round(repeat,strategy,round_number)
            self.event('delivery',{'repeat':repeat,'strategy':strategy,'round':round_number,'ids':[d['id'] for d in deliveries]})
