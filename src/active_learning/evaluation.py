"""Defined metric edges and prediction records; no fitting in this module."""
import numpy as np
from scipy.stats import spearmanr


def metrics(observed,predicted,uncertainty=None):
    y=np.asarray(observed,dtype=float);p=np.asarray(predicted,dtype=float)
    if y.shape!=p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Invalid metric membership/values')
    residual=p-y;absolute=np.abs(residual)
    reason='fewer than two targets' if len(y)<2 else ('constant targets' if np.ptp(y)==0 else None)
    out={'n':len(y),'MAE_eV':float(absolute.mean()),'RMSE_eV':float(np.sqrt(np.mean(residual**2))),
         'R2':None if reason else float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)), 'R2_reason':reason,
         'median_AE_eV':float(np.quantile(absolute,.5,method='linear')),'P90_AE_eV':float(np.quantile(absolute,.9,method='linear'))}
    if uncertainty is not None:
        u=np.asarray(uncertainty,dtype=float)
        if u.shape!=y.shape or not np.isfinite(u).all():raise ValueError('Invalid uncertainty')
        why='constant uncertainty/error or insufficient rows' if len(y)<2 or np.ptp(u)==0 or np.ptp(absolute)==0 else None
        out['uncertainty_error_spearman']=None if why else float(spearmanr(u,absolute).statistic)
        out['spearman_reason']=why
    return out


def prediction_rows(s,repeat,strategy,round_number,checkpoint_hash,ids,observed,predictions,label_count,attempt_count,partition):
    records={r['id']:__import__('json').loads(r['record']) for r in s.rows('SELECT * FROM molecules')}
    rows=[]
    for evaluator,values in predictions.items():
        if evaluator=='uncertainty':continue
        for i,ident in enumerate(ids):
            y=observed[ident]['gap_ev'];p=values[i];unc=predictions['uncertainty'][i] if evaluator=='committee' else None
            rows.append({'repeat':repeat,'strategy':strategy,'round':round_number,'partition':partition,
                'label_count':label_count,'attempt_count':attempt_count,'source_record_id':ident,
                'input_structure_key':records[ident]['input_structure_key'],'input_smiles':records[ident]['input_smiles'],
                'checkpoint_hash':checkpoint_hash,'evaluator':evaluator,'observed_gap_ev':y,
                'predicted_gap_ev':p,'residual_ev':p-y,'absolute_error_ev':abs(p-y),'uncertainty_ev':unc,'units':'eV'})
    return rows
