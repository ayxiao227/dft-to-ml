"""Figures and tables exclusively from saved predictions, selections and audit state."""
import io
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from .config import canonical,hashed
from .costs import coverage
from src.descriptors import input_features

COLORS={'random':'#737373','diversity':'#0072B2','uncertainty':'#D55E00','uncertainty_diversity':'#009E73'}
REPORT_VERSION='saved_diagnostics_v1'


def _frame_bytes(frame):return frame.to_csv(index=False).encode()


def _savefig(fig,name,outputs):
    for extension in ['png','svg']:
        buffer=io.BytesIO();fig.savefig(buffer,format=extension,dpi=150,bbox_inches='tight',metadata={'Date':None} if extension=='svg' else None)
        outputs[name+'.'+extension]=buffer.getvalue()
    plt.close(fig)


def _draw_empty(axis,message):
    axis.text(.5,.5,message,ha='center',va='center',transform=axis.transAxes);axis.set_axis_off()


def report(s):
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'dft-to-ml-active-learning-v1'})
    config=s.meta('config');checkpoints=s.rows('SELECT * FROM checkpoints ORDER BY repeat,strategy,round')
    predictions=[]
    for cp in checkpoints:predictions.extend(json.loads((s.root/cp['predictions']).read_text()))
    if s.meta('finalization',{}).get('status')=='completed':predictions.extend(json.loads((s.root/'final/predictions.json').read_text()))
    if not predictions:raise ValueError('No saved predictions to report')
    p=pd.DataFrame(predictions);outputs={};m=[]
    for row in s.rows('SELECT * FROM metrics ORDER BY repeat,strategy,round,evaluator,partition'):
        cp=next(c for c in checkpoints if (c['repeat'],c['strategy'],c['round'])==(row['repeat'],row['strategy'],row['round']))
        data=json.loads(cp['metadata']);attempts=s.rows('SELECT * FROM attempts WHERE repeat=? AND strategy=? AND round<=?',(row['repeat'],row['strategy'],row['round']))
        cost=coverage(attempts)
        m.append({**{k:v for k,v in row.items() if k!='data'},**json.loads(row['data']),
                  'label_count':data['label_count'],'attempt_count':data['attempt_count'],
                  'allocated_cpu_hours':cost['allocated_cpu_hours']['measured_sum'] if cost['allocated_cpu_hours']['complete'] else None,
                  'cost_known_attempts':cost['allocated_cpu_hours']['known_attempts'],
                  'cost_total_attempts':cost['allocated_cpu_hours']['total_attempts'],
                  'evaluator_seconds':data['evaluator_seconds'],'committee_seconds':data['committee_seconds'],'prediction_seconds':data['prediction_seconds']})
    metrics=pd.DataFrame(m);outputs['predictions.csv']=_frame_bytes(p);outputs['metrics.csv']=_frame_bytes(metrics)
    attempts=pd.DataFrame(s.rows('SELECT * FROM attempts ORDER BY repeat,strategy,round,id,attempt'))
    # Final heldout label evidence is never exposed in a development report.
    if s.meta('finalization',{}).get('status')!='completed':
        attempts=attempts.drop(columns=['evidence','log_hash'],errors='ignore')
    outputs['attempts.csv']=_frame_bytes(attempts)
    selections=pd.DataFrame([{**r,**json.loads(r['detail'])} for r in s.rows('SELECT * FROM selections ORDER BY repeat,strategy,round,id')])
    outputs['selections.csv']=_frame_bytes(selections.drop(columns=['detail'],errors='ignore'))
    outputs['attrition.csv']=_frame_bytes(pd.DataFrame(json.loads((s.root/'curation.json').read_text())))
    outputs['rounds.csv']=_frame_bytes(pd.DataFrame([{**r,**json.loads(r['metadata'])} for r in s.rows('SELECT * FROM rounds ORDER BY repeat,strategy,round')]))
    outputs['cost_coverage.json']=canonical(coverage(s.rows('SELECT * FROM attempts'))).encode()
    summaries=[];paired=[];threshold=[]
    for axis in ['label_count','attempt_count','allocated_cpu_hours']:
        current=metrics.dropna(subset=[axis]).sort_values('round').drop_duplicates(['repeat','strategy','partition','evaluator',axis],keep='last')
        if not current.empty:
            grouped=current.groupby(['partition','evaluator','strategy',axis],dropna=False)['MAE_eV'].agg(['mean','std','count']).reset_index()
            grouped['axis']=axis;grouped=grouped.rename(columns={axis:'budget'});summaries.extend(grouped.to_dict('records'))
            baseline=current[current.strategy=='random']
            for strategy in config['acquisition']['strategies']:
                if strategy=='random':continue
                other=current[current.strategy==strategy]
                joined=other.merge(baseline,on=['repeat','partition','evaluator',axis],suffixes=('','_random'))
                for _,row in joined.iterrows():paired.append({'repeat':int(row['repeat']),'partition':row['partition'],'evaluator':row['evaluator'],
                    'strategy':strategy,'axis':axis,'budget':row[axis],'delta_MAE_eV':row['MAE_eV']-row['MAE_eV_random']})
    outputs['learning_summary.csv']=_frame_bytes(pd.DataFrame(summaries));outputs['paired_differences.csv']=_frame_bytes(pd.DataFrame(paired))
    if config['target_mae_ev'] is not None:
        for keys,group in metrics[metrics.partition=='validation'].groupby(['repeat','strategy','evaluator']):
            reached=group.sort_values('round');reached=reached[reached.MAE_eV<=config['target_mae_ev']]
            threshold.append({'repeat':keys[0],'strategy':keys[1],'evaluator':keys[2],
                'threshold_eV':config['target_mae_ev'],'status':'first observed checkpoint' if len(reached) else 'not reached',
                'label_count':int(reached.iloc[0].label_count) if len(reached) else None,
                'attempt_count':int(reached.iloc[0].attempt_count) if len(reached) else None})
    outputs['threshold.csv']=_frame_bytes(pd.DataFrame(threshold))
    partitions=sorted(p.partition.unique());evaluators=['mean','ridge','rf','committee']
    for partition in partitions:
        for axis,title in [('label_count','Training labels (including initial)'),('attempt_count','Attempted calculations'),('allocated_cpu_hours','Measured allocated CPU-hours (complete coverage only)')]:
            fig,axes=plt.subplots(2,2,figsize=(11,8),squeeze=False)
            for evaluator,ax in zip(evaluators,axes.flat):
                subset=metrics[(metrics.partition==partition)&(metrics.evaluator==evaluator)].dropna(subset=[axis]).sort_values('round').drop_duplicates(['repeat','strategy',axis],keep='last')
                if subset.empty:_draw_empty(ax,'No complete measured cost' if axis=='allocated_cpu_hours' else 'No saved predictions');continue
                for strategy in config['acquisition']['strategies']:
                    data=subset[subset.strategy==strategy].groupby(axis)['MAE_eV'].agg(['mean','std']).reset_index().sort_values(axis)
                    if data.empty:continue
                    color=COLORS[strategy];ax.plot(data[axis],data['mean'],marker='o',color=color,label=strategy)
                    if data['std'].notna().any():ax.fill_between(data[axis],data['mean']-data['std'].fillna(0),data['mean']+data['std'].fillna(0),color=color,alpha=.15)
                    for _,rep in subset[subset.strategy==strategy].groupby('repeat'):ax.plot(rep[axis],rep.MAE_eV,color=color,alpha=.25,linewidth=.6)
                ax.set(title=evaluator,xlabel=title,ylabel='MAE (eV)');ax.legend(fontsize=7)
            fig.suptitle(f'{partition}: mean ± sample SD across seeded repeats; individual runs faint')
            fig.tight_layout();_savefig(fig,f'{partition}_learning_{axis}',outputs)
    # Latest checkpoint per repeat/strategy/evaluator. No model is opened here.
    latest=p.groupby(['repeat','strategy','evaluator','partition'])['round'].transform('max')==p['round'];last=p[latest].copy()
    diagnostics=[];selective=[];subgroups=[];worst=[];chemistry=[]
    for (partition,strategy),group in last[last.evaluator=='rf'].groupby(['partition','strategy']):
        fig,axes=plt.subplots(2,2,figsize=(10,8));y=group.observed_gap_ev;pred=group.predicted_gap_ev
        ax=axes[0,0]
        if len(group)>2000:ax.hexbin(y,pred,gridsize=40,mincnt=1)
        else:ax.scatter(y,pred,s=13,alpha=.6,color=COLORS[strategy])
        low=min(y.min(),pred.min());high=max(y.max(),pred.max());pad=max((high-low)*.05,.05)
        ax.plot([low-pad,high+pad],[low-pad,high+pad],'k--',linewidth=.8);ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),xlabel='Observed gap (eV)',ylabel='Predicted gap (eV)');ax.set_aspect('equal',adjustable='box')
        axes[0,1].scatter(pred,group.residual_ev,s=13,alpha=.6,color=COLORS[strategy]);axes[0,1].axhline(0,color='black',linewidth=.8);axes[0,1].set(xlabel='Predicted gap (eV)',ylabel='Predicted − observed (eV)')
        axes[1,0].hist(group.residual_ev,bins=min(25,max(3,len(group)//3)),color=COLORS[strategy]);axes[1,0].set(xlabel='Residual (eV)',ylabel='Count')
        axes[1,1].hist(group.absolute_error_ev,bins=min(25,max(3,len(group)//3)),color=COLORS[strategy]);axes[1,1].set(xlabel='Absolute error (eV)',ylabel='Count')
        fig.suptitle(f'{partition}: {strategy}, RF, latest checkpoints (repeats may share molecules)');fig.tight_layout();_savefig(fig,f'{partition}_{strategy}_errors',outputs)
        worst.extend(group.sort_values(['absolute_error_ev','source_record_id'],ascending=[False,True]).head(12).to_dict('records'))
        for _,r in group.iterrows():
            f,mol=input_features(r.input_smiles)
            tags={'all':True,'cyclic':mol.GetRingInfo().NumRings()>0,'acyclic':mol.GetRingInfo().NumRings()==0,
                'heavy_atoms_'+str(mol.GetNumHeavyAtoms()):True,
                'elements_'+','.join(sorted({a.GetSymbol() for a in mol.GetAtoms()})):True}
            tags.update({name:f[name]>0 for name in ['n_nitroso','n_amine_noxide','n_carbonyl','n_conj_carbonyl_v2','n_azo']})
            subgroups.extend({'partition':partition,'strategy':strategy,'repeat':r['repeat'],'subgroup':tag,'absolute_error_ev':r.absolute_error_ev} for tag,include in tags.items() if include)
    for keys,group in last[last.evaluator=='committee'].groupby(['partition','strategy','repeat']):
        ordered=group.sort_values(['uncertainty_ev','source_record_id'])
        for fraction in [.1,.25,.5,.75,1.]:
            n=max(1,int(np.ceil(len(ordered)*fraction)));selective.append({'partition':keys[0],'strategy':keys[1],'repeat':keys[2],'retained_fraction':fraction,'count':n,'MAE_eV':ordered.head(n).absolute_error_ev.mean()})
        for i,indices in enumerate(np.array_split(np.arange(len(ordered)),min(5,len(ordered)))):
            sub=ordered.iloc[indices];diagnostics.append({'partition':keys[0],'strategy':keys[1],'repeat':keys[2],'bin':i,'count':len(sub),'mean_uncertainty_ev':sub.uncertainty_ev.mean(),'MAE_eV':sub.absolute_error_ev.mean()})
    outputs['uncertainty_bins.csv']=_frame_bytes(pd.DataFrame(diagnostics));outputs['selective_prediction.csv']=_frame_bytes(pd.DataFrame(selective))
    for partition in partitions:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for strategy in config['acquisition']['strategies']:
            d=pd.DataFrame(diagnostics);d=d[(d.partition==partition)&(d.strategy==strategy)]
            if not d.empty:axes[0].scatter(d.mean_uncertainty_ev,d.MAE_eV,s=20,label=strategy,color=COLORS[strategy])
            d=pd.DataFrame(selective);d=d[(d.partition==partition)&(d.strategy==strategy)]
            if not d.empty:
                d=d.groupby('retained_fraction').MAE_eV.mean();axes[1].plot(d.index,d.values,label=strategy,color=COLORS[strategy])
        axes[0].set(xlabel='Mean disagreement (eV)',ylabel='Bin MAE (eV)');axes[1].set(xlabel='Retained fraction (least uncertain first)',ylabel='Selective MAE (eV)');axes[1].legend(fontsize=7)
        fig.suptitle(f'{partition}: ranking diagnostics, not calibrated coverage');fig.tight_layout();_savefig(fig,f'{partition}_uncertainty',outputs)
    sg=pd.DataFrame(subgroups)
    if not sg.empty:sg=sg.groupby(['partition','strategy','repeat','subgroup']).absolute_error_ev.agg(['count','mean']).reset_index().rename(columns={'mean':'MAE_eV'})
    outputs['subgroups.csv']=_frame_bytes(sg);outputs['worst_predictions.csv']=_frame_bytes(pd.DataFrame(worst))
    if worst:
        unique={r['source_record_id']:r for r in sorted(worst,key=lambda r:-r['absolute_error_ev'])};depicted=list(unique.values())[:12]
        grid=Draw.MolsToGridImage([Chem.MolFromSmiles(r['input_smiles']) for r in depicted],molsPerRow=4,subImgSize=(240,180),legends=[r['source_record_id'][:14] for r in depicted],useSVG=True)
        outputs['worst_molecules.svg']=str(grid).encode()
    molecular={r['id']:json.loads(r['record']) for r in s.rows('SELECT * FROM molecules')}
    selection_diversity=[]
    from .experiment import read_features
    from .acquisition import distances
    features=read_features(s)
    batch_references={}
    for _,row in selections.sort_values(['repeat','strategy','round','selection_order']).iterrows() if not selections.empty else []:
        molrow=molecular[row['id']];f,mol=input_features(molrow['input_smiles'])
        chemistry.append({'repeat':row['repeat'],'strategy':row['strategy'],'round':row['round'],'source_record_id':row['id'],**{k:f[k] for k in ['HeavyAtomCount','n_nitroso','n_amine_noxide','n_carbonyl','n_conj_carbonyl_v2']}})
        batch=(int(row['repeat']),row['strategy'],int(row['round']))
        if batch not in batch_references:
            batch_references[batch]=[r['id'] for r in s.rows("SELECT id FROM attempts WHERE repeat=? AND strategy=? AND round<? AND status='accepted' ORDER BY id",batch)]
        refs=batch_references[batch]
        fp=features.rows([row['id']],True)[0]
        diversity=min((float(distances(features.rows([ref],True),fp,config['acquisition']['chunk_size'])[0]) for ref in refs),default=1.)
        selection_diversity.append({'repeat':row['repeat'],'strategy':row['strategy'],'round':row['round'],'diversity_distance':diversity,'uncertainty_ev':row.get('uncertainty_ev')})
        refs.append(row['id'])
    outputs['selected_chemistry.csv']=_frame_bytes(pd.DataFrame(chemistry));outputs['selected_diversity.csv']=_frame_bytes(pd.DataFrame(selection_diversity))
    if selection_diversity:
        d=pd.DataFrame(selection_diversity)
        fig,ax=plt.subplots(figsize=(8,4))
        for strategy,group in d.groupby('strategy'):
            summary=group.groupby('round').diversity_distance.mean()
            ax.plot(summary.index,summary.values,marker='o',label=strategy,color=COLORS[strategy])
        ax.set(xlabel='Acquisition round',ylabel='Mean input-fingerprint minimum distance at selection');ax.legend(fontsize=7)
        fig.tight_layout();_savefig(fig,'selected_diversity',outputs)
    if chemistry:
        comp=pd.DataFrame(chemistry).groupby('strategy')[['n_nitroso','n_amine_noxide','n_carbonyl','n_conj_carbonyl_v2']].mean()
        fig,ax=plt.subplots(figsize=(9,4));comp.plot.bar(ax=ax);ax.set(ylabel='Mean count per selected molecule',xlabel='Strategy');fig.tight_layout();_savefig(fig,'selected_composition',outputs)
    # Paired individual differences; no assertion of independent datasets or significance.
    if paired:
        differences=pd.DataFrame(paired);fig,ax=plt.subplots(figsize=(8,4))
        for strategy in config['acquisition']['strategies']:
            d=differences[(differences.axis=='label_count')&(differences.evaluator=='rf')&(differences.partition=='validation')&(differences.strategy==strategy)]
            if not d.empty:ax.scatter(d.budget,d.delta_MAE_eV,label=strategy,color=COLORS[strategy],alpha=.6)
        ax.axhline(0,color='black',linewidth=.8);ax.set(xlabel='Matched training labels',ylabel='Paired MAE − random MAE (eV)');ax.legend(fontsize=8);fig.tight_layout();_savefig(fig,'paired_differences',outputs)
    cost=coverage(s.rows('SELECT * FROM attempts'));sizes=[p['sizes'] for p in s.meta('partitions')]
    text=f'''# DFT orbital-gap active learning\n\nQuestion: can acquisition reduce labels or measured calculation cost needed to predict orbital gaps (eV), especially on unfamiliar structures?\n\nDataset: {config['dataset']}; source SHA-256 `{config['source']['sha256']}`. Cohort: {s.meta('attrition')}. Split: {config['split']['mode']} / {config['split']['policy']}. Achieved sizes by repeat: `{canonical(sizes)}`.\n\nInitial labels count toward the training budget. Validation/final labels are fixed additional overhead; tuning pilot labels: 0. Evaluators: mean, training-scaled Ridge, RF and molecule-bootstrap RF committee. Fixed settings; input-only 2D features; disagreement is sample SD in eV. Capped acquisition: {config['acquisition']['candidate_cap'] is not None}.\n\nFinalization: {s.meta('finalization',{}).get('status','not finalized')}. Final errors are present only after explicit finalization.\n\nFigures use consistent strategy colors, individual seeded repeats and mean ± sample SD (ddof=1). This is conditional variability across seeded runs; repeats may share molecules and are not independent datasets. Tiny synthetic smoke runs provide software acceptance evidence, not AL superiority. The group policy is a topology heuristic, not guaranteed distribution shift. Retrospective budgets simulate requested labels, not measured DFT savings. No monotonic curve, interpolation or threshold advantage is assumed.\n\nMeasured cost coverage: `{canonical(cost)}`. Unknown costs remain null; only complete-cost checkpoints enter cost curves. Allocated CPU-hours differ from consumed CPU seconds. Acquisition and model overhead are in saved tables; diagnostic committees on random/diversity count as overhead. Failed and retried attempts remain visible.\n\nUncertainty plots test error ranking, not calibrated intervals. Subgroups use input graph counts/chemistry and include sample counts. Missing subgroup cells are absent, not zero-error. Worst predictions include short-ID depictions and full SMILES in tables.\n\nAt duplicate label/attempt/cost budgets within a repeat, paired and aggregate curves use the latest recorded checkpoint; raw per-round metrics remain saved. All plots/tables are generated from saved artifacts without fitting. Consult `metrics.csv`, `learning_summary.csv`, `paired_differences.csv`, `threshold.csv`, `attempts.csv`, `cost_coverage.json`, `attrition.csv`, `selected_chemistry.csv`, `subgroups.csv` and `worst_predictions.csv`. Round reservations, candidate hashes/caps, degeneracy warnings, seeds and acquisition elapsed seconds are in `rounds.csv`. No scientific superiority or complete-cost claim is made by this report.\n\n![Validation learning](validation_learning_label_count.png)\n\n![Paired differences](paired_differences.png)\n\n![Uncertainty](validation_uncertainty.png)\n'''
    if 'paired_differences.png' not in outputs:text=text.replace('![Paired differences](paired_differences.png)','No paired random comparison is available for this configuration.')
    outputs['report.md']=text.encode()
    inputs=hashed([REPORT_VERSION,s.meta('config_hash'),[cp['predictions'] for cp in checkpoints],s.meta('finalization'),s.rows('SELECT * FROM attempts')])
    relative='reports/'+inputs[:16]
    with s.transaction():
        for name,content in outputs.items():s.artifact(relative+'/'+name,content,'report')
    return {'report':str(s.root/relative/'report.md'),'tables_and_figures':len(outputs),'version':REPORT_VERSION}
