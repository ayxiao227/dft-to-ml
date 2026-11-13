"""Four exact batch strategies. Memory O(N + chunk_size * fingerprint_length)."""
import numpy as np
from .config import hashed, seed_for


def disagreement(member_predictions):
    p=np.asarray(member_predictions,dtype=float)
    if p.ndim!=2 or len(p)<2 or not np.isfinite(p).all():raise ValueError('Invalid committee predictions')
    return p.mean(axis=0),p.std(axis=0,ddof=1)


def distances(candidates, reference, chunk_size):
    out=np.empty(len(candidates),dtype=float)
    reference=np.asarray(reference,dtype=np.uint8)
    for start in range(0,len(candidates),chunk_size):
        chunk=np.asarray(candidates[start:start+chunk_size],dtype=np.uint8)
        intersection=np.sum(chunk & reference,axis=1,dtype=np.int32)
        union=np.sum(chunk | reference,axis=1,dtype=np.int32)
        out[start:start+len(chunk)]=1-np.divide(intersection,union,out=np.ones(len(chunk)),where=union!=0)
    return out


def maxmin(ids, fingerprints, reference, count, chunk_size=4096):
    if len(ids)!=len(set(ids)) or count>len(ids) or count<0:raise ValueError('Invalid diversity candidates/count')
    order=sorted(range(len(ids)),key=lambda i:ids[i]);ids=[ids[i] for i in order]
    fp=fingerprints.reordered(order) if hasattr(fingerprints,'reordered') else np.asarray(fingerprints)[order]
    minimum=np.ones(len(ids));available=np.ones(len(ids),dtype=bool)
    for ref in reference:minimum=np.minimum(minimum,distances(fp,ref,chunk_size))
    chosen=[]
    for _ in range(count):
        index=int(np.argmax(np.where(available,minimum,-np.inf)))
        chosen.append({'source_record_id':ids[index],'diversity_distance':float(minimum[index]),'diversity_rank':len(chosen)+1})
        available[index]=False
        minimum=np.minimum(minimum,distances(fp,fp[index],chunk_size))
    return chosen


def candidate_subset(ids, cap, seed):
    ids=sorted(set(ids))
    if cap is None or len(ids)<=cap:return ids
    return sorted(sorted(ids,key=lambda i:(hashed([seed,i]),i))[:cap])


def select(strategy, eligible, labeled, features, count, config, repeat, round_number, scorer=None):
    eligible=sorted(eligible)
    if len(set(eligible))!=len(eligible) or set(eligible)&set(labeled):raise ValueError('Duplicate/acquired eligible IDs')
    seed=config['repeat_seeds'][repeat];a=config['acquisition']
    candidates=candidate_subset(eligible,a['candidate_cap'],seed_for(seed,repeat,round_number,'candidate_cap'))
    count=min(count,len(candidates))
    if count<0:raise ValueError('Negative batch')
    scores={};warning=None
    if strategy in ('uncertainty','uncertainty_diversity'):
        if scorer is None:raise ValueError('Uncertainty strategy requires committee scorer')
        predicted=scorer(candidates)
        if len(predicted)!=len(candidates) or not np.isfinite(predicted).all():raise ValueError('Nonfinite acquisition disagreement')
        scores=dict(zip(candidates,map(float,predicted)))
        if len(set(scores.values()))<=1:warning='Degenerate equal disagreement; deterministic tie ordering retained'
    if strategy=='random':
        chosen=np.random.default_rng(seed_for(seed,repeat,round_number,'random_selection')).choice(candidates,size=count,replace=False).tolist()
        rows=[{'source_record_id':i,'diversity_distance':None,'diversity_rank':None} for i in chosen]
    elif strategy=='uncertainty':
        rows=[{'source_record_id':i,'diversity_distance':None,'diversity_rank':None} for i in sorted(candidates,key=lambda i:(-scores[i],i))[:count]]
    elif strategy in ('diversity','uncertainty_diversity'):
        shortlist=candidates
        if strategy=='uncertainty_diversity':
            length=min(len(candidates),max(count,min(20*count,a['shortlist_cap'])))
            shortlist=sorted(candidates,key=lambda i:(-scores[i],i))[:length]
        rows=maxmin(shortlist,features.fingerprints(shortlist),features.fingerprints(sorted(labeled)),count,a['chunk_size'])
    else:raise ValueError('Unknown strategy')
    for order,row in enumerate(rows):
        row["selection_order"] = order
        ident=row['source_record_id'];row['uncertainty_ev']=scores.get(ident)
        row['uncertainty_rank']=sorted(scores,key=lambda i:(-scores[i],i)).index(ident)+1 if ident in scores else None
    return rows,{'candidate_ids':candidates,'candidate_hash':hashed(candidates),'capped':a['candidate_cap'] is not None,'warning':warning,
                 'seeds':{'cap':seed_for(seed,repeat,round_number,'candidate_cap'),'random':seed_for(seed,repeat,round_number,'random_selection')}}
