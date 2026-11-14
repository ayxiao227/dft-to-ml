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
    """Stream all rows to CSV chunks; index only latest diagnostic rows on disk."""
    def __init__(self,s,work,cache):
        from .report_cache import log, SETTINGS
        from .artifacts import file_hash
        import time
        start=time.monotonic()
        paths=[s.root/cp['predictions'] for cp in s.db.execute('SELECT predictions FROM checkpoints ORDER BY repeat,strategy,round')]
        state=s.meta('finalization',{})
        if state.get('status')=='completed':
            if state.get('format_version')==2:
                from .finalization import verified_results
                paths.extend(s.root/d['predictions']['path'] for d in verified_results(s).values())
            else:paths.append(s.root/'final/predictions.json')
        latest={}
        for cp in s.db.execute('SELECT repeat,strategy,MAX(round) FROM checkpoints GROUP BY repeat,strategy'):
            latest[(cp[0],cp[1])]=cp[2]
        chunks=[];count=0;new_seconds=0.;new_artifacts=0
        for index,path in enumerate(paths):
            key='ingest/'+str(index)+'/'+file_hash(path)
            def build(directory):
                db=sqlite3.connect(directory/'latest.sqlite3')
                try:
                    db.execute('DROP TABLE IF EXISTS predictions')
                    db.execute('CREATE TABLE predictions(repeat INTEGER,strategy TEXT,evaluator TEXT,partition TEXT,round INTEGER,ident TEXT,y REAL,pred REAL,residual REAL,ae REAL,u REAL,data TEXT)')
                    n=0;fields=None
                    with (directory/'rows.csv').open('w',newline='') as out:
                        for row in prediction_records(path):
                            if fields is None:
                                fields=list(row);writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
                            writer.writerow(row);n+=1
                            if row['round']==latest[(row['repeat'],row['strategy'])]:
                                db.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                                    (*[row[k] for k in ('repeat','strategy','evaluator','partition','round','source_record_id','observed_gap_ev','predicted_gap_ev','residual_ev','absolute_error_ev','uncertainty_ev')],json.dumps(row,allow_nan=False)))
                            if n%100000==0:log(f'ingestion artifact {index+1}/{len(paths)}: {n:,} rows (uncommitted)',start)
                    db.commit()
                finally:db.close()
                return {'count':n,'fields':fields},{'rows.csv':directory/'rows.csv','latest.sqlite3':directory/'latest.sqlite3'}
            tick=time.monotonic()
            data,files=cache.run(key,build);chunks.append((data,files));count+=data['count']
            if not cache.last_cached:new_seconds+=time.monotonic()-tick;new_artifacts+=1
            eta=f'{new_seconds/new_artifacts*(len(paths)-index-1):.1f}s if remaining artifacts have comparable sizes' if new_artifacts else 'unavailable (cached work only)'
            log(f'ingestion {index+1}/{len(paths)} artifacts; {count:,} committed rows; '+('cached' if cache.last_cached else 'new')+'; approximate ETA '+eta,start)
        if not count:raise ValueError('No saved predictions to report')
        def combine(directory):
            path=directory/'predictions.sqlite3';path.unlink(missing_ok=True)
            db=sqlite3.connect(path)
            db.execute(f"PRAGMA cache_size=-{SETTINGS['sql_cache_kib']}");db.execute('PRAGMA temp_store=FILE')
            try:
                db.execute('CREATE TABLE predictions(repeat INTEGER,strategy TEXT,evaluator TEXT,partition TEXT,round INTEGER,ident TEXT,y REAL,pred REAL,residual REAL,ae REAL,u REAL,data TEXT)')
                with (directory/'predictions.csv').open('w',newline='') as out:
                    fields=next(data['fields'] for data,_ in chunks if data['fields']);writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
                    for data,files in chunks:
                        if not data['fields']:continue
                        if set(data['fields'])!=set(fields):raise ValueError('Prediction CSV schema changed')
                        with files['rows.csv'].open(newline='') as inp:
                            # csv parser preserves quoted embedded newlines and numeric text.
                            reader=csv.DictReader(inp);writer.writerows(reader)
                        db.execute('ATTACH DATABASE ? AS chunk',(str(files['latest.sqlite3']),))
                        db.execute('INSERT INTO predictions SELECT * FROM chunk.predictions');db.commit()
                        db.execute('DETACH DATABASE chunk')
                db.execute('CREATE INDEX membership ON predictions(evaluator,partition,strategy,repeat)')
                db.execute('CREATE INDEX uncertainty ON predictions(evaluator,partition,strategy,repeat,u,ident)')
                db.execute('CREATE INDEX worst ON predictions(evaluator,partition,strategy,ae DESC,ident)')
                db.commit()
            finally:db.close()
            return {'count':count},{'predictions.sqlite3':path,'predictions.csv':directory/'predictions.csv'}
        _,files=cache.run('prediction-index',combine)
        self.csv=files['predictions.csv']
        self.db=sqlite3.connect(files['predictions.sqlite3'].resolve().as_uri()+'?mode=ro',uri=True)
        self.db.execute(f"PRAGMA cache_size=-{SETTINGS['sql_cache_kib']}");self.db.execute('PRAGMA temp_store=FILE')

    def partitions(self):return [r[0] for r in self.db.execute('SELECT DISTINCT partition FROM predictions ORDER BY partition')]
    def keys(self,evaluator,by_repeat=False):
        columns='partition,strategy'+(',repeat' if by_repeat else '')
        return [tuple(r) for r in self.db.execute('SELECT DISTINCT '+columns+' FROM predictions WHERE evaluator=? ORDER BY '+columns,(evaluator,))]
    def query(self,evaluator,keys,columns='data',order='rowid',limit=None):
        names=('partition','strategy','repeat')[:len(keys)]
        where=' AND '.join(k+'=?' for k in names)
        sql='SELECT '+columns+' FROM predictions WHERE evaluator=? AND '+where+' ORDER BY '+order
        if limit is not None:sql+=' LIMIT '+str(int(limit))
        return self.db.execute(sql,(evaluator,*keys))
    def count(self,evaluator,keys):return self.query(evaluator,keys,'COUNT(*)').fetchone()[0]
    def rows(self,evaluator,keys,order='rowid',limit=None):
        return (json.loads(r[0]) for r in self.query(evaluator,keys,order=order,limit=limit))
    def groups(self,evaluator,by_repeat=False):
        """Internal API: groups expose row iterators, never whole DataFrames."""
        for keys in self.keys(evaluator,by_repeat):yield keys,self.rows(evaluator,keys)
    def close(self):self.db.close()
