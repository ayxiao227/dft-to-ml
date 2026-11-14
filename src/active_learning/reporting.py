"""Figures and tables exclusively from saved predictions, selections and audit state."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from .config import canonical,hashed
from .costs import coverage
from .report_storage import Predictions,Outputs
from .report_cache import StageCache, SETTINGS, log
from .report_diagnostics import Chemistry, batch_diversity, ranking, rf_summary, read_fingerprints, FingerprintLookup
from .streaming import meta_values,rows_hash,prediction_records
from .artifacts import publish_file,file_hash
from .state import crash
from .finalization import checkpoint_refs

COLORS={'random':'#737373','diversity':'#0072B2','uncertainty':'#D55E00','uncertainty_diversity':'#009E73'}
REPORT_VERSION='saved_diagnostics_v3'


def _frame_bytes(frame):return frame


def _savefig(fig,name,outputs):
    for extension in ['png','svg']:
        path=outputs.path(name+'.'+extension)
        fig.savefig(path,format=extension,dpi=150,bbox_inches='tight',metadata={'Date':None} if extension=='svg' else None)
        outputs[name+'.'+extension]=path
    plt.close(fig)


def _draw_empty(axis,message):
    axis.text(.5,.5,message,ha='center',va='center',transform=axis.transAxes);axis.set_axis_off()


def report_identity(s):
    # Exclude published reports/events, so publication itself never invalidates work.
    sources={table:rows_hash(dict(r) for r in s.db.execute('SELECT * FROM '+table+' ORDER BY '+order))
             for table,order in [('molecules','id'),('partitions','repeat,id'),('attempts','repeat,strategy,id,attempt'),
                                 ('labels','repeat,strategy,id'),('selections','repeat,strategy,round,id'),
                                 ('rounds','repeat,strategy,round'),('checkpoints','repeat,strategy,round'),
                                 ('metrics','repeat,strategy,round,evaluator,partition')]}
    artifacts=s.rows("SELECT * FROM artifacts WHERE kind!='report' ORDER BY path")
    implementation={name:file_hash(Path(__file__).parent/name) for name in
                    ['reporting.py','report_storage.py','report_cache.py','report_diagnostics.py','features.py','acquisition.py','finalization.py','streaming.py','costs.py','evaluation.py','artifacts.py',
                     '../descriptors.py','../feature_registry.py']}
    return hashed([REPORT_VERSION,implementation,SETTINGS,s.meta('config_hash'),s.meta('environment'),
                   s.meta('finalization'),s.meta('features'),sources,artifacts])


def report(s,failpoint=None):
    start=time.monotonic();log('verifying reporting source identity',start)
    identity=report_identity(s);cache=StageCache(s.root,identity,failpoint)
    chemistry=None;predictions=None
    try:
        chemistry=Chemistry(cache.root/'chemistry.sqlite3')
        def build(work):
            nonlocal predictions
            predictions=Predictions(s,work,cache)
            result,outputs=_report(s,work,predictions,cache,chemistry,identity)
            return result,dict(outputs)
        result,outputs=cache.run('complete-report',build)
        log(f'table/figure publication: {len(outputs)} files',start)
        relative='reports/'+identity[:16]
        remnants=list((s.root/relative).glob('.*'))
        if remnants:raise ValueError(f'Interrupted report publication evidence retained at {s.root/relative}: {remnants}. Move these temporary files aside after checking that no writer is active, then retry.')
        # Publication may have stopped before or after ledger commit. Exact hashes
        # permit idempotent orphan adoption; conflicts are never overwritten.
        with s.transaction():
            for name,path in outputs.items():
                target=relative+'/'+name
                publish_file(s.root/target,path)
                old=s.db.execute('SELECT sha256 FROM artifacts WHERE path=?',(target,)).fetchone()
                if old is None:s.artifact_file(target,'report')
                elif old[0]!=file_hash(s.root/target):raise ValueError('Committed report hash conflict: '+target)
            crash('before_report_ledger_commit',failpoint)
        crash('after_report_ledger_commit',failpoint)
        log(f'completed: {result["report"]}',start)
        return result
    finally:
        if predictions is not None:predictions.close()
        if chemistry is not None:chemistry.close()
        cache.close();plt.close('all')


def _report(s,work,predictions,cache,chemistry_cache,inputs):
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'dft-to-ml-active-learning-v1'})
    config=s.meta('config')
    checkpoint_keys={(cp['repeat'],cp['strategy'],cp['round']) for cp in checkpoint_refs(s)}
    outputs=Outputs(work);m=[]
    log('metric aggregation: saved checkpoint metrics')
    for row in s.rows('SELECT * FROM metrics ORDER BY repeat,strategy,round,evaluator,partition'):
        if row['partition']=='final' and s.meta('finalization',{}).get('status')!='completed':continue
        if (row['repeat'],row['strategy'],row['round']) not in checkpoint_keys:raise ValueError('Report metric without saved checkpoint')
        keys=['label_count','attempt_count','evaluator_seconds','committee_seconds','prediction_seconds']
        data=dict(zip(keys,s.db.execute('SELECT '+','.join("json_extract(metadata,'$."+key+"')" for key in keys)+' FROM checkpoints WHERE repeat=? AND strategy=? AND round=?',(row['repeat'],row['strategy'],row['round'])).fetchone()))
        cost=coverage(dict(a) for a in s.db.execute('SELECT * FROM attempts WHERE repeat=? AND strategy=? AND round<=?',(row['repeat'],row['strategy'],row['round'])))
        m.append({**{k:v for k,v in row.items() if k!='data'},**json.loads(row['data']),
                  'label_count':data['label_count'],'attempt_count':data['attempt_count'],
                  'allocated_cpu_hours':cost['allocated_cpu_hours']['measured_sum'] if cost['allocated_cpu_hours']['complete'] else None,
                  'cost_known_attempts':cost['allocated_cpu_hours']['known_attempts'],
                  'cost_total_attempts':cost['allocated_cpu_hours']['total_attempts'],
                  'evaluator_seconds':data['evaluator_seconds'],'committee_seconds':data['committee_seconds'],'prediction_seconds':data['prediction_seconds']})
    metrics=pd.DataFrame(m);outputs['predictions.csv']=predictions.csv;outputs['metrics.csv']=_frame_bytes(metrics)
    # Final heldout label evidence is never exposed in a development report.
    complete=s.meta('finalization',{}).get('status')=='completed'
    outputs.rows('attempts.csv',({k:v for k,v in dict(r).items() if complete or k not in ('evidence','log_hash')}
        for r in s.db.execute('SELECT * FROM attempts ORDER BY repeat,strategy,round,id,attempt')))
    selections=pd.DataFrame([{**r,**json.loads(r['detail'])} for r in s.rows('SELECT * FROM selections ORDER BY repeat,strategy,round,id')])
    outputs['selections.csv']=_frame_bytes(selections.drop(columns=['detail'],errors='ignore'))
    outputs.rows('attrition.csv',prediction_records(s.root/'curation.json'))
    outputs.rows('rounds.csv',({**dict(r),**json.loads(r['metadata'])} for r in s.db.execute('SELECT * FROM rounds ORDER BY repeat,strategy,round')))
    outputs['cost_coverage.json']=canonical(coverage(dict(r) for r in s.db.execute('SELECT * FROM attempts'))).encode()
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
    partitions=predictions.partitions();evaluators=['mean','ridge','rf','committee']
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
    diagnostics=[];selective=[];subgroups={};worst=[];chemistry=[]
    plot_samples=[]
    log('error plots and subgroup diagnostics')
    for partition,strategy in predictions.keys('rf'):
        keys=(partition,strategy)
        def rf_build(directory):
            return rf_summary(predictions,keys,chemistry_cache,SETTINGS['plot_limit'],SETTINGS['plot_seed']),{}
        summary,_=cache.run('rf/'+canonical(keys),rf_build)
        log(f"RF {keys}: {summary['population']:,} rows; plotting {len(summary['sample']):,}; {len(summary['subgroups'])} subgroup cells")
        group=pd.DataFrame(summary['sample'])
        plot_samples.append({k:summary[k] for k in ['population','seed','limit']}|{'partition':partition,'strategy':strategy})
        fig,axes=plt.subplots(2,2,figsize=(10,8));y=group.observed_gap_ev;pred=group.predicted_gap_ev
        ax=axes[0,0]
        if len(group)>2000:ax.hexbin(y,pred,gridsize=40,mincnt=1)
        else:ax.scatter(y,pred,s=13,alpha=.6,color=COLORS[strategy])
        low=min(y.min(),pred.min());high=max(y.max(),pred.max());pad=max((high-low)*.05,.05)
        ax.plot([low-pad,high+pad],[low-pad,high+pad],'k--',linewidth=.8);ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),xlabel='Observed gap (eV)',ylabel='Predicted gap (eV)');ax.set_aspect('equal',adjustable='box')
        axes[0,1].scatter(pred,group.residual_ev,s=13,alpha=.6,color=COLORS[strategy],rasterized=True);axes[0,1].axhline(0,color='black',linewidth=.8);axes[0,1].set(xlabel='Predicted gap (eV)',ylabel='Predicted − observed (eV)')
        axes[1,0].hist(group.residual_ev,bins=min(25,max(3,len(group)//3)),color=COLORS[strategy]);axes[1,0].set(xlabel='Residual (eV)',ylabel='Count')
        axes[1,1].hist(group.absolute_error_ev,bins=min(25,max(3,len(group)//3)),color=COLORS[strategy]);axes[1,1].set(xlabel='Absolute error (eV)',ylabel='Count')
        fig.suptitle(f'{partition}: {strategy}, RF, latest checkpoints (repeats may share molecules)');fig.tight_layout();_savefig(fig,f'{partition}_{strategy}_errors',outputs)
        worst.extend(summary['worst'])
        for repeat,tag,count,total in summary['subgroups']:
            subgroups[(partition,strategy,repeat,tag)]=(count,total)
        del group,y,pred,summary
    log('exact uncertainty ranking: SQL sorts on disk')
    for keys in predictions.keys('committee',by_repeat=True):
        def rank_build(directory):
            d,p=ranking(predictions,keys)
            return {'diagnostics':d,'selective':p},{}
        summary,_=cache.run('ranking/'+canonical(keys),rank_build)
        log(f"ranking {keys}: {sum(row['count'] for row in summary['diagnostics']):,} rows; {len(summary['diagnostics'])} exact bins")
        diagnostics.extend(summary['diagnostics']);selective.extend(summary['selective'])
    outputs['plot_sampling.json']=canonical({'method':'deterministic reservoir; error panels only; numeric tables use full population','groups':plot_samples}).encode()
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
    sg=pd.DataFrame([dict(zip(['partition','strategy','repeat','subgroup'],key),count=count,MAE_eV=total/count) for key,(count,total) in sorted(subgroups.items())])
    outputs['subgroups.csv']=_frame_bytes(sg);outputs['worst_predictions.csv']=_frame_bytes(pd.DataFrame(worst))
    if worst:
        unique={r['source_record_id']:r for r in sorted(worst,key=lambda r:-r['absolute_error_ev'])};depicted=list(unique.values())[:12]
        grid=Draw.MolsToGridImage([Chem.MolFromSmiles(r['input_smiles']) for r in depicted],molsPerRow=4,subImgSize=(240,180),legends=[r['source_record_id'][:14] for r in depicted],useSVG=True)
        outputs['worst_molecules.svg']=str(grid).encode()
    selection_diversity=[]
    log('chemistry and exact selected-diversity diagnostics')
    features=None;fingerprints=None
    if not selections.empty:
        ordered=selections.sort_values(['repeat','strategy','round','selection_order'])
        for batch,group in ordered.groupby(['repeat','strategy','round'],sort=False):
            batch=(int(batch[0]),batch[1],int(batch[2]))
            def diversity_build(directory):
                nonlocal features,fingerprints
                if features is None:
                    features=read_fingerprints(s);fingerprints=FingerprintLookup(features)
                ids=group['id'].tolist()
                # Only accepted attempts from earlier rounds; current batch order
                # includes every selected row even if its later calculation failed.
                refs=[r[0] for r in s.db.execute("SELECT id FROM attempts WHERE repeat=? AND strategy=? AND round<? AND status='accepted' ORDER BY id",batch)]
                log(f'diversity {batch}: {len(ids):,} selections, {len(refs):,} accepted earlier-round references')
                values=batch_diversity(features,ids,refs,SETTINGS['fingerprint_chunk'],fingerprints)
                chem=[];div=[]
                for (_,row),distance in zip(group.iterrows(),values):
                    record=json.loads(s.db.execute('SELECT record FROM molecules WHERE id=?',(row['id'],)).fetchone()[0])
                    chem.append({'repeat':batch[0],'strategy':batch[1],'round':batch[2],'source_record_id':row['id'],**chemistry_cache.get(record['input_smiles'])['features']})
                    uncertainty=row.get('uncertainty_ev')
                    div.append({'repeat':batch[0],'strategy':batch[1],'round':batch[2],'diversity_distance':distance,
                                'uncertainty_ev':None if pd.isna(uncertainty) else uncertainty})
                return {'chemistry':chem,'diversity':div},{}
            result,_=cache.run('diversity/'+canonical(batch),diversity_build)
            chemistry.extend(result['chemistry']);selection_diversity.extend(result['diversity'])
    del features,fingerprints
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
    cost=coverage(dict(r) for r in s.db.execute('SELECT * FROM attempts'));sizes=list(meta_values(s,'partitions','item.sizes'))
    text=f'''# DFT orbital-gap active learning\n\nQuestion: can acquisition reduce labels or measured calculation cost needed to predict orbital gaps (eV), especially on unfamiliar structures?\n\nDataset: {config['dataset']}; source SHA-256 `{config['source']['sha256']}`. Cohort: {s.meta('attrition')}. Split: {config['split']['mode']} / {config['split']['policy']}. Achieved sizes by repeat: `{canonical(sizes)}`.\n\nInitial labels count toward the training budget. Validation/final labels are fixed additional overhead; tuning pilot labels: 0. Evaluators: mean, training-scaled Ridge, RF and molecule-bootstrap RF committee. Fixed settings; input-only 2D features; disagreement is sample SD in eV. Capped acquisition: {config['acquisition']['candidate_cap'] is not None}.\n\nFinalization: {s.meta('finalization',{}).get('status','not finalized')}. Partial final metrics remain in the ledger during finalization, but final errors are included here only after all frozen checkpoints complete.\n\nFigures use consistent strategy colors, individual seeded repeats and mean ± sample SD (ddof=1). This is conditional variability across seeded runs; repeats may share molecules and are not independent datasets. Tiny synthetic smoke runs provide software acceptance evidence, not AL superiority. The group policy is a topology heuristic, not guaranteed distribution shift. Retrospective budgets simulate requested labels, not measured DFT savings. No monotonic curve, interpolation or threshold advantage is assumed.\n\nMeasured cost coverage: `{canonical(cost)}`. Unknown costs remain null; only complete-cost checkpoints enter cost curves. Allocated CPU-hours differ from consumed CPU seconds. Acquisition and model overhead are in saved tables; diagnostic committees on random/diversity count as overhead. Failed and retried attempts remain visible.\n\nUncertainty plots test error ranking, not calibrated intervals. Subgroups use input graph counts/chemistry and include sample counts. Missing subgroup cells are absent, not zero-error. Worst predictions include short-ID depictions and full SMILES in tables.\n\nAt duplicate label/attempt/cost budgets within a repeat, paired and aggregate curves use the latest recorded checkpoint; raw per-round metrics remain saved. All plots/tables are generated from saved artifacts without fitting. Consult `metrics.csv`, `learning_summary.csv`, `paired_differences.csv`, `threshold.csv`, `attempts.csv`, `cost_coverage.json`, `attrition.csv`, `selected_chemistry.csv`, `subgroups.csv` and `worst_predictions.csv`. Round reservations, candidate hashes/caps, degeneracy warnings, seeds and acquisition elapsed seconds are in `rounds.csv`. No scientific superiority or complete-cost claim is made by this report.\n\n![Validation learning](validation_learning_label_count.png)\n\n![Paired differences](paired_differences.png)\n\n![Uncertainty](validation_uncertainty.png)\n'''
    if 'paired_differences.png' not in outputs:text=text.replace('![Paired differences](paired_differences.png)','No paired random comparison is available for this configuration.')
    text+='\nLarge error panels use a deterministic reservoir sample of at most '+str(SETTINGS['plot_limit'])+' rows (seed '+str(SETTINGS['plot_seed'])+'). Population sizes are recorded in `plot_sampling.json`. All numeric tables use full membership; stored metric quantiles and ranks are unchanged.\n'
    outputs['report.md']=text.encode()
    outputs['provenance.json']=canonical({'version':REPORT_VERSION,'inputs_hash':inputs,'settings':SETTINGS,'finalization':s.meta('finalization'),'prediction_artifacts':s.rows("SELECT * FROM artifacts WHERE kind IN ('validation_predictions','final_predictions','final_checkpoint_manifest','final_manifest') ORDER BY path")}).encode()
    relative='reports/'+inputs[:16]
    return {'report':str(s.root/relative/'report.md'),'tables_and_figures':len(outputs),'version':REPORT_VERSION},outputs
