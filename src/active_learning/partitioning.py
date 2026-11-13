"""Immutable four-way membership; group policy is a topology heuristic."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from src.splits import chemical_group
from .config import seed_for, hashed


def freeze(records, config, repeat):
    records=sorted(records,key=lambda r:r.source_record_id)
    ids=[r.source_record_id for r in records];n=len(ids)
    if len({r.input_structure_key for r in records})!=n:
        raise ValueError('Duplicate input structures before partitioning')
    optimized=[r.optimized_structure_key for r in records if r.optimized_structure_key]
    if len(set(optimized))!=len(optimized):raise ValueError('Duplicate optimized structures before partitioning')
    groups=[chemical_group(r.input_smiles) for r in records]
    s=config['split'];seed=config['repeat_seeds'][repeat]
    final_seed=seed_for(seed,repeat,0,'split_final');val_seed=seed_for(seed,repeat,0,'split_validation')
    if n<5:raise ValueError(f'Insufficient rows: {n}')
    if s['mode']=='group':
        if len(set(groups))<5:raise ValueError(f'Insufficient groups: {len(set(groups))}; rows: {n}; need >=5')
        dev,final=next(GroupShuffleSplit(n_splits=1,test_size=s['final_fraction'],random_state=final_seed).split(ids,groups=groups))
        dg=np.asarray(groups)[dev]
        if len(set(dg))<2:raise ValueError(f'Insufficient development groups: {len(set(dg))}')
        train,val=next(GroupShuffleSplit(n_splits=1,test_size=s['validation_fraction']/(1-s['final_fraction']),random_state=val_seed).split(dev,groups=dg))
        universe,validation=dev[train],dev[val]
    else:
        dev,final=train_test_split(np.arange(n),test_size=s['final_fraction'],random_state=final_seed)
        universe,validation=train_test_split(dev,test_size=s['validation_fraction']/(1-s['final_fraction']),random_state=val_seed)
    b=config['budgets']
    if len(universe)<b['max_labels']:
        raise ValueError(f'Development universe {len(universe)} < max_labels {b["max_labels"]}; groups={len(set(groups))}, rows={n}')
    universe=sorted(universe,key=lambda i:ids[i])
    initial=set(np.random.default_rng(seed_for(seed,repeat,0,'initial')).choice(universe,size=b['initial'],replace=False).tolist())
    roles={i:'pool' for i in universe};roles.update({i:'initial' for i in initial})
    roles.update({int(i):'validation' for i in validation});roles.update({int(i):'final' for i in final})
    rows=[{'source_record_id':ident,'input_structure_key':r.input_structure_key,'group_key':groups[i],'origin':roles[i]} for i,(ident,r) in enumerate(zip(ids,records))]
    sizes={role:sum(r['origin']==role for r in rows) for role in ['initial','pool','validation','final']}
    return {'rows':rows,'hash':hashed(rows),'sizes':sizes,'seeds':{'final':final_seed,'validation':val_seed,'initial':seed_for(seed,repeat,0,'initial')},
            'mode':s['mode'],'policy':s['policy'],'achieved_fractions':{k:v/n for k,v in sizes.items()}}
