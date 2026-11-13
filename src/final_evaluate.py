"""Explicit one-time final evaluation after freezing model/feature choices."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.ml_analysis import BASE_FEATURES_GLOBAL, CONJ_FEATURES_GLOBAL
from src.run_metadata import check_molecular_identity
from src.schema import canonical_smiles
from src.gaussian_records import digest


def evaluate(development, final, output, features='conj', trees=200):
    train, test = pd.read_csv(development), pd.read_csv(final)
    for df in (train,test):
        check_molecular_identity(df)
        if 'protocol_id' not in df or df.protocol_id.isna().any() or df.protocol_id.nunique()!=1:
            raise ValueError('One verified protocol required')
    if train.protocol_id.iloc[0] != test.protocol_id.iloc[0]:
        raise ValueError('Development and final protocols differ')
    if set(train.SMILES.map(canonical_smiles)) & set(test.SMILES.map(canonical_smiles)):
        raise ValueError('Development/final molecular overlap')
    columns=BASE_FEATURES_GLOBAL + (CONJ_FEATURES_GLOBAL if features=='conj' else [])
    for df in (train,test):
        if not np.isfinite(df[columns+['gap_ev']].to_numpy(dtype=float)).all():
            raise ValueError('Incomplete/nonfinite data')
    if len(test)<2 or trees<1:
        raise ValueError('Need two final molecules and positive trees')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    model=RandomForestRegressor(n_estimators=trees,max_depth=20,min_samples_leaf=2,
        max_features='sqrt',random_state=42,n_jobs=1)
    model.fit(train[columns],train.gap_ev)
    predictions=model.predict(test[columns])
    result=test[['SMILES','gap_ev']].copy();result['predicted_gap_ev']=predictions
    result.to_csv(root/'predictions.csv',index=False)
    report={'MAE_eV':mean_absolute_error(test.gap_ev,predictions),
        'RMSE_eV':float(np.sqrt(mean_squared_error(test.gap_ev,predictions))),
        'R2':r2_score(test.gap_ev,predictions),'features':columns,'trees':trees,
        'model_parameters':model.get_params(),
        'development_sha256':digest(Path(development).read_bytes()),
        'final_sha256':digest(Path(final).read_bytes())}
    (root/'metrics.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('development');p.add_argument('final');p.add_argument('output')
    p.add_argument('--features',choices=['base','conj'],default='conj')
    p.add_argument('--trees',type=int,default=200)
    a=p.parse_args();evaluate(a.development,a.final,a.output,a.features,a.trees)
