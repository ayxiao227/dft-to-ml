"""Dataset selector: QM9/tmQM labeled benchmarks and GDB-13 candidates."""
import argparse
import csv
import gzip
import json
import math
import random
from pathlib import Path
import pandas as pd
from rdkit import Chem
from src.qm9 import prepare as prepare_qm9, download as download_qm9, file_hash, HARTREE_TO_EV
from src.schema import canonical_smiles, molecule_key
from src.descriptors import calculate_descriptors, extra_features


def text_input(path):
    return gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()


def reservoir_add(records, record, accepted, sample, rng):
    if len(records)<sample:records.append(record)
    else:
        i=rng.randrange(accepted)
        if i<sample:records[i]=record


def prepare_gdb13(source, output, sample=5000, seed=42):
    """Bounded-memory sampling of SMILES rows; no quantum labels are invented."""
    if sample<1:raise ValueError('Sample must be positive')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    records=[];rng=random.Random(seed);accepted=0
    with text_input(source) as stream,(root/'rejections.csv').open('w',newline='') as reject:
        writer=csv.writer(reject);writer.writerow(['line','reason'])
        for line_number,line in enumerate(stream,1):
            tokens=line.split()
            if not tokens or tokens[0].startswith('#'):continue
            try:
                smi=canonical_smiles(tokens[0]);mol=Chem.MolFromSmiles(smi)
                if mol.GetNumHeavyAtoms()>13 or Chem.GetFormalCharge(mol)!=0 or len(Chem.GetMolFrags(mol))!=1:
                    raise ValueError('Not a neutral connected GDB-13 candidate')
                if any(a.GetSymbol() not in {'C','H','N','O','S','Cl'} or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
                    raise ValueError('Unsupported GDB-13 composition')
                accepted+=1
                reservoir_add(records,{'molecule_id':'gdb13_'+molecule_key(smi),'SMILES':smi,
                    'Name':'gdb13_'+molecule_key(smi),'dataset':'GDB-13','label_status':'unlabeled',
                    'source_line':line_number},accepted,sample,rng)
            except ValueError as exc:writer.writerow([line_number,str(exc)])
    if accepted<sample:raise ValueError('Requested sample exceeds valid rows')
    # Input distributions normally contain unique structures; sampled duplicates
    # are removed without retaining a global set for ~1 billion molecules.
    df=pd.DataFrame(records).drop_duplicates('molecule_id')
    df.to_csv(root/'candidates.csv',index=False)
    metadata={'dataset':'GDB-13','source_sha256':file_hash(source),'seed':seed,
        'sampling':'uniform reservoir over valid input rows; deduplicated after sampling',
        'valid_rows':accepted,'selected_unique':len(df),'label_status':'unlabeled',
        'source':'https://gdb.unibe.ch/downloads/'}
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return root/'candidates.csv'


def prepare_tmqm(source, output, sample=5000, seed=42, units='hartree', fingerprints=False):
    """2024 tmQM semicolon CSV; use supplied SMILES, never infer metal bonding."""
    if sample<1 or units not in {'hartree','ev'}:raise ValueError('Invalid sample or units')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    records=[];seen=set();codes=set();accepted=0;rng=random.Random(seed)
    with text_input(source) as stream,(root/'rejections.csv').open('w',newline='') as reject:
        reader=csv.DictReader(stream,delimiter=';');writer=csv.writer(reject);writer.writerow(['CSD_code','reason'])
        if not {'CSD_code','SMILES','HOMO_Energy','LUMO_Energy','HL_Gap'}.issubset(reader.fieldnames or []):
            raise ValueError('Expected 2024 tmQM_y.csv schema including supplied SMILES')
        for raw in reader:
            try:
                code=raw['CSD_code'].strip()
                if not code or code in codes:raise ValueError('Missing or duplicate CSD code')
                codes.add(code)
                smi=canonical_smiles(raw['SMILES']);mol=Chem.MolFromSmiles(smi)
                metals=[a for a in mol.GetAtoms() if 21<=a.GetAtomicNum()<=30 or 39<=a.GetAtomicNum()<=48 or 72<=a.GetAtomicNum()<=80]
                if len(metals)!=1:raise ValueError('Expected one transition metal')
                if smi in seen:raise ValueError('Duplicate canonical structure')
                homo,lumo,gap=[float(raw[k]) for k in ('HOMO_Energy','LUMO_Energy','HL_Gap')]
                if not all(math.isfinite(v) for v in (homo,lumo,gap)) or gap<0 or abs(lumo-homo-gap)>2e-4:
                    raise ValueError('Invalid or inconsistent orbital energies')
                features=calculate_descriptors(smi)
                features.update(extra_features(smi,include_fp=fingerprints))
                features.update({'MetalAtomicNumber':metals[0].GetAtomicNum(),'FormalCharge':Chem.GetFormalCharge(mol)})
                if not all(math.isfinite(v) for v in features.values()):raise ValueError('Nonfinite descriptors')
                factor=HARTREE_TO_EV if units=='hartree' else 1
                row={'molecule_id':'tmqm_'+code,'run_id':'tmqm_2024','CSD_code':code,'dataset':'tmQM',
                    'SMILES':smi,'optimized_smiles':smi,'accepted':True,
                    'protocol_id':'tmQM-2024-TPSSh-D3BJ-def2SVP-on-GFN2xTB',
                    'homo_ev':homo*factor,'lumo_ev':lumo*factor,'gap_ev':gap*factor,**features}
                seen.add(smi);accepted+=1
                reservoir_add(records,row,accepted,sample,rng)
            except (ValueError,RuntimeError,KeyError,TypeError) as exc:writer.writerow([raw.get('CSD_code',''),str(exc)])
    if accepted<sample:raise ValueError('Requested sample exceeds accepted unique records')
    pd.DataFrame(records).to_csv(root/'features.csv',index=False)
    (root/'metadata.json').write_text(json.dumps({'dataset':'tmQM','release':'2024','source_sha256':file_hash(source),
        'input_energy_units':units,'conversion_to_ev':HARTREE_TO_EV if units=='hartree' else 1,
        'accepted_unique':accepted,'selected':len(records),'seed':seed,
        'source':'https://github.com/uiocompcat/tmQM','identity_source':'publisher supplied Huckel SMILES',
        'limitations':'Graph baseline; metal bonding not reconstructed or independently validated'},indent=2))
    return root/'features.csv'


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='dataset',required=True)
    q=sub.add_parser('qm9');qsub=q.add_subparsers(dest='action',required=True)
    d=qsub.add_parser('download');d.add_argument('directory')
    qprep=qsub.add_parser('prepare');qprep.add_argument('archive');qprep.add_argument('exclusions');qprep.add_argument('output')
    t=sub.add_parser('tmqm');t.add_argument('source');t.add_argument('output');t.add_argument('--units',choices=['hartree','ev'],default='hartree')
    g=sub.add_parser('gdb13');g.add_argument('source');g.add_argument('output')
    for parser in (qprep,t,g):
        parser.add_argument('--sample',type=int,default=5000);parser.add_argument('--seed',type=int,default=42)
    for parser in (qprep,t):parser.add_argument('--fingerprints',action='store_true')
    a=p.parse_args()
    if a.dataset=='qm9':
        if a.action=='download':download_qm9(a.directory)
        else:print(prepare_qm9(a.archive,a.exclusions,a.output,a.sample,a.seed,a.fingerprints))
    elif a.dataset=='tmqm':print(prepare_tmqm(a.source,a.output,a.sample,a.seed,a.units,a.fingerprints))
    else:print(prepare_gdb13(a.source,a.output,a.sample,a.seed))

if __name__=='__main__':main()
