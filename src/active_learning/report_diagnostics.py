"""Streaming numeric diagnostics and bounded chemistry/fingerprint work."""
import json
import sqlite3
from functools import lru_cache
import numpy as np
from rdkit import DataStructs
from src.descriptors import input_features
from .config import canonical, hashed
from .report_cache import log


class Chemistry:
    """At most 4096 decoded summaries; distinct input structures are cached on disk."""
    def __init__(self,path):
        try:
            self.db=sqlite3.connect(path)
            self.db.execute('CREATE TABLE IF NOT EXISTS chemistry(smiles TEXT PRIMARY KEY,data TEXT,sha TEXT)')
            if self.db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise sqlite3.DatabaseError('quick_check failed')
        except sqlite3.DatabaseError as exc:
            if hasattr(self,'db'):self.db.close()
            raise ValueError(f'Corrupt chemistry cache retained at {path}; restore a verified copy or move this report work directory aside: {exc}') from exc
        self.pending=0
    @lru_cache(maxsize=4096)
    def get(self,smiles):
        row=self.db.execute('SELECT data,sha FROM chemistry WHERE smiles=?',(smiles,)).fetchone()
        if row:
            try:
                value=json.loads(row[0])
                if hashed([smiles,value])!=row[1]:raise ValueError('hash mismatch')
            except ValueError as exc:raise ValueError('Corrupt chemistry cache retained; move report work aside: '+smiles) from exc
            return value
        f,mol=input_features(smiles)
        tags=['all','cyclic' if mol.GetRingInfo().NumRings()>0 else 'acyclic',
              'heavy_atoms_'+str(mol.GetNumHeavyAtoms()),'elements_'+','.join(sorted({a.GetSymbol() for a in mol.GetAtoms()}))]
        tags.extend(name for name in ['n_nitroso','n_amine_noxide','n_carbonyl','n_conj_carbonyl_v2','n_azo'] if f[name]>0)
        value={'tags':tags,'features':{k:f[k] for k in ['HeavyAtomCount','n_nitroso','n_amine_noxide','n_carbonyl','n_conj_carbonyl_v2']}}
        self.db.execute('INSERT INTO chemistry VALUES (?,?,?)',(smiles,canonical(value),hashed([smiles,value])))
        self.pending+=1
        if self.pending%1024==0:self.db.commit()
        return value
    def close(self):
        self.db.commit();self.db.close();self.get.cache_clear()


def bitvectors(rows,chunk=1024):
    """Pack bounded row chunks and release expanded uint8 fingerprint rows."""
    result=[]
    for start in range(0,len(rows),chunk):
        for packed in np.packbits(rows[start:start+chunk],axis=1,bitorder='little'):
            result.append(DataStructs.CreateFromBinaryText(packed.tobytes()))
    return result


class FingerprintLookup:
    """Reuse packed references across batches/rounds; cap retained vectors at 4096."""
    def __init__(self,features,limit=4096):
        from collections import OrderedDict
        self.features=features;self.limit=limit;self.cache=OrderedDict()
    def get(self,ids,chunk):
        result=[]
        for start in range(0,len(ids),chunk):
            current=ids[start:start+chunk]
            missing=list(dict.fromkeys(i for i in current if i not in self.cache))
            fresh=dict(zip(missing,bitvectors(self.features.fingerprints(missing),chunk)))
            for ident in current:
                value=fresh[ident] if ident in fresh else self.cache[ident]
                result.append(value);self.cache[ident]=value;self.cache.move_to_end(ident)
            while len(self.cache)>self.limit:self.cache.popitem(last=False)
        return result


def read_fingerprints(s):
    # Prediction/feature files have already passed Store verification. Reporting
    # needs binary fingerprints only: do not fault in the unused model X array.
    from .features import CandidateFeatures
    m=s.meta('features');n=len(m['ids'])
    return CandidateFeatures(tuple(m['ids']),tuple(m['smiles']),(),np.empty((n,0)),
        np.load(s.root/'binary.npy',mmap_mode='r',allow_pickle=False),m['settings_hash'])


def batch_diversity(features, selected, reference, chunk=1024, lookup=None):
    """Exact ordered minimum Tanimoto distance. Earlier selections include failures.

    Reference membership is supplied by the ledger, never inferred from selections.
    Bulk similarity uses packed bit vectors; no candidate × reference × bits tensor.
    Each unordered within-batch pair is calculated once and reused for the later row.
    """
    lookup=lookup or FingerprintLookup(features)
    references=lookup.get(reference,chunk)
    candidates=lookup.get(selected,chunk)
    minimum=np.ones(len(selected))
    for i,fp in enumerate(candidates):
        for vectors,stop in [(references,len(references)),(candidates,i)]:
            for start in range(0,stop,chunk):
                refs=vectors[start:min(start+chunk,stop)]
                similarities=DataStructs.BulkTanimotoSimilarity(fp,refs)
                # Existing binary diagnostic defines empty/empty distance as 0;
                # RDKit defines its similarity as 0, so preserve that contract.
                best=1. if fp.GetNumOnBits()==0 and any(r.GetNumOnBits()==0 for r in refs) else max(similarities)
                minimum[i]=min(minimum[i],1-best)
    return minimum.tolist()


def ranking(predictions,keys):
    """Exact uncertainty order, retaining source-ID ties and all rows; SQL spills."""
    n=predictions.count('committee',keys)
    fractions=[.1,.25,.5,.75,1.]
    cutoffs=[max(1,int(np.ceil(n*f))) for f in fractions]
    # np.array_split convention: first n % bins bins have one extra member.
    bins=min(5,n);sizes=[n//bins+(i<n%bins) for i in range(bins)]
    boundaries=np.cumsum(sizes);b=0;total=0.;sums=[0.,0.];diagnostics=[];selective=[]
    for i,(u,ae) in enumerate(predictions.query('committee',keys,'u,ae',order='u,ident'),1):
        total+=ae;sums[0]+=u;sums[1]+=ae
        for f,cutoff in zip(fractions,cutoffs):
            if i==cutoff:selective.append({'partition':keys[0],'strategy':keys[1],'repeat':keys[2],'retained_fraction':f,'count':i,'MAE_eV':total/i})
        if i==boundaries[b]:
            diagnostics.append({'partition':keys[0],'strategy':keys[1],'repeat':keys[2],'bin':b,'count':sizes[b],'mean_uncertainty_ev':sums[0]/sizes[b],'MAE_eV':sums[1]/sizes[b]})
            b+=1;sums=[0.,0.]
    return diagnostics,selective


def rf_summary(predictions,keys,chemistry,limit,seed):
    """Exact subgroup totals and worst rows; only plotted points use reservoir sampling."""
    subgroups={};rng=np.random.default_rng(seed);sample=[];n=0
    for row in predictions.rows('rf',keys):
        n+=1
        for tag in chemistry.get(row['input_smiles'])['tags']:
            key=(row['repeat'],tag);count,total=subgroups.get(key,(0,0.))
            subgroups[key]=(count+1,total+row['absolute_error_ev'])
        if n<=limit:sample.append(row)
        else:
            index=int(rng.integers(n))
            if index<limit:sample[index]=row
        if n%25000==0:log(f'RF {keys}: {n:,} rows processed (uncommitted)')
    worst=list(predictions.rows('rf',keys,order='ae DESC,ident',limit=12))
    return {'subgroups':[[*key,*value] for key,value in sorted(subgroups.items())],
            'worst':worst,'sample':sample,'population':n,'seed':seed,'limit':limit}
