"""Disk-backed report inputs/outputs; no experiment-wide prediction frame."""
import csv
import json
import shutil
import sqlite3
from pathlib import Path
import pandas as pd
from .streaming import prediction_records


class Outputs(dict):
    def __init__(self,root):super().__init__();self.root=Path(root)
    def path(self,name):return self.root/name
    def __setitem__(self,name,value):
        path=self.path(name)
        if isinstance(value,pd.DataFrame):value.to_csv(path,index=False)
        elif isinstance(value,Path):
            if value!=path:shutil.copyfile(value,path)
        else:path.write_bytes(value)
        super().__setitem__(name,path)

    def rows(self,name,rows):
        """Union CSV columns in encounter order with a bounded disk spool."""
        fields={};spool=self.root/(name+'.rows.jsonl')
        with spool.open('w') as stream:
            for row in rows:
                fields.update(dict.fromkeys(row))
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        with self.path(name).open('w',newline='') as out,spool.open() as stream:
            writer=csv.DictWriter(out,fieldnames=list(fields),lineterminator='\n')
            if fields:writer.writeheader()
            for line in stream:writer.writerow(json.loads(line))
        spool.unlink();super().__setitem__(name,self.path(name))


class Predictions:
    def __init__(self,s,work):
        self.db=sqlite3.connect(work/'predictions.sqlite3')
        self.db.execute('PRAGMA cache_size=-4096');self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('CREATE TABLE predictions(repeat INTEGER,strategy TEXT,evaluator TEXT,partition TEXT,round INTEGER,data TEXT)')
        paths=[s.root/cp['predictions'] for cp in s.db.execute('SELECT predictions FROM checkpoints ORDER BY repeat,strategy,round')]
        state=s.meta('finalization',{})
        if state.get('status')=='completed':
            if state.get('format_version')==2:
                from .finalization import verified_results
                docs=verified_results(s)
                paths.extend(s.root/d['predictions']['path'] for d in docs.values())
            else:paths.append(s.root/'final/predictions.json')
        count=0
        with (work/'predictions.csv').open('w',newline='') as stream:
            writer=None
            for path in paths:
                for row in prediction_records(path):
                    if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row)
                    self.db.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',
                        (*[row[k] for k in ('repeat','strategy','evaluator','partition','round')],json.dumps(row,allow_nan=False)))
                    count+=1
        if not count:raise ValueError('No saved predictions to report')
        self.db.execute('CREATE INDEX membership ON predictions(partition,strategy,evaluator,repeat,round)')
        self.db.execute('CREATE TABLE latest AS SELECT repeat,strategy,evaluator,partition,MAX(round) AS round FROM predictions GROUP BY repeat,strategy,evaluator,partition')
        self.db.execute('CREATE INDEX latest_keys ON latest(partition,strategy,evaluator,repeat,round)')
        self.db.commit()

    def partitions(self):return [r[0] for r in self.db.execute('SELECT DISTINCT partition FROM predictions ORDER BY partition')]
    def groups(self,evaluator,by_repeat=False):
        # Exactly one latest diagnostic group is resident at a time. SQL spills sorts to disk.
        columns='partition,strategy'+(',repeat' if by_repeat else '')
        for keys in self.db.execute('SELECT DISTINCT '+columns+' FROM latest WHERE evaluator=? ORDER BY '+columns,(evaluator,)):
            where=' AND '.join('p.'+c+'=?' for c in columns.split(','))
            query='SELECT p.data FROM predictions p JOIN latest l USING(repeat,strategy,evaluator,partition,round) WHERE p.evaluator=? AND '+where+' ORDER BY p.rowid'
            frame=pd.DataFrame(json.loads(r[0]) for r in self.db.execute(query,(evaluator,*keys)))
            yield tuple(keys),frame
            del frame
    def close(self):self.db.close()
