"""Prepare an isolated QM9 HOMO–LUMO benchmark from the original XYZ archive."""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import tarfile
import urllib.request
from pathlib import Path
from rdkit import Chem
from src.descriptors import calculate_descriptors, extra_features
from src.schema import canonical_smiles, molecule_key

HARTREE_TO_EV = 27.211386245988
SOURCE = 'https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904'
PROTOCOL = 'QM9-original-B3LYP-6-31G(2df,p)-gas-v1'


def file_hash(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def download(directory):
    """Discover original Figshare files and verify published MD5 before use."""
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    for article in (1057646,1057644):
        with urllib.request.urlopen(f'https://api.figshare.com/v2/articles/{article}',timeout=60) as f:
            metadata=json.load(f)
        for item in metadata['files']:
            name=Path(item['name']).name
            path=root/name
            if path.exists() and file_hash(path,'md5')==item['computed_md5']:
                continue
            temp=path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(item['download_url'],timeout=120) as response,temp.open('wb') as out:
                for chunk in iter(lambda:response.read(1024*1024),b''):out.write(chunk)
            if file_hash(temp,'md5')!=item['computed_md5']:
                temp.unlink();raise ValueError(f'Checksum mismatch: {name}')
            temp.replace(path)
        (root/f'figshare_{article}.json').write_text(json.dumps(metadata,indent=2))


def excluded_ids(path):
    ids=set()
    for line in Path(path).read_text().splitlines():
        fields=line.split()
        if len(fields)>=2 and fields[0].isdigit():ids.add(int(fields[0]))
    if not ids:
        raise ValueError('Exclusion list has no recognized molecule IDs')
    return ids


def parse_xyz(text):
    lines=text.splitlines();n=int(lines[0]);p=lines[1].split()
    if n<1 or len(p)!=17 or p[0] not in {'gdb','gdb9'}:
        raise ValueError('Unexpected original QM9 property header')
    ident=int(p[1]);values=[float(v.replace('*^','e')) for v in p[2:]]
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite properties')
    homo,lumo,gap=values[5:8]
    if gap<0 or abs(lumo-homo-gap)>2e-4:raise ValueError('Inconsistent orbital gap')
    coordinates=[]
    for line in lines[2:n+2]:
        atom=line.replace('*^','e').split()
        if len(atom)!=5 or atom[0] not in {'C','H','N','O','F'}:raise ValueError('Invalid atom row')
        if not all(math.isfinite(float(x)) for x in atom[1:]):raise ValueError('Invalid coordinates')
        coordinates.append(' '.join(atom[:4]))
    if len(coordinates)!=n:raise ValueError('Incomplete coordinates')
    frequencies=[float(v.replace('*^','e')) for v in lines[n+2].split()]
    if not frequencies or not all(math.isfinite(v) and v >= -10 for v in frequencies):
        raise ValueError('Invalid or imaginary frequencies')
    smiles=lines[n+3].split()
    if len(smiles)!=2:raise ValueError('Expected GDB and optimized SMILES')
    original,optimized=smiles
    mol=Chem.MolFromSmiles(optimized)
    if mol is None or mol.GetNumHeavyAtoms()>9 or len(Chem.GetMolFrags(mol))!=1:
        raise ValueError('Invalid optimized molecular graph')
    if Chem.GetFormalCharge(mol)!=0 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('Non-neutral or radical graph')
    if Chem.AddHs(mol).GetNumAtoms()!=n:raise ValueError('Graph/coordinate atom count mismatch')
    from collections import Counter
    if Counter(line.split()[0] for line in coordinates) != Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()):
        raise ValueError('Graph/coordinate element counts differ')
    optimized=canonical_smiles(optimized)
    row={'molecule_id':'qm9_'+str(ident),'qm9_id':ident,'run_id':'qm9_original',
         'input_smiles':original,'optimized_smiles':optimized,'SMILES':optimized,
         'structure_key':molecule_key(optimized),'protocol_id':PROTOCOL,'dataset':'QM9',
         'accepted':True,'homo_ev':homo*HARTREE_TO_EV,'lumo_ev':lumo*HARTREE_TO_EV,
         'gap_ev':gap*HARTREE_TO_EV,'homo_hartree':homo,'lumo_hartree':lumo,'gap_hartree':gap}
    return row,str(n)+'\nQM9 '+str(ident)+'\n'+'\n'.join(coordinates)+'\n'


def prepare(archive, exclusions, output, sample=None, seed=42, fingerprints=False):
    if sample is not None and sample<1:raise ValueError('Sample must be positive')
    excluded=excluded_ids(exclusions)
    # Original README flags these as difficult/low-threshold/saddlepoint cases.
    excluded.update({21725,87037,59827,117523,128113,129053,129152,129158,130535,6620,59818})
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    rng=random.Random(seed);reservoir=[];seen=set();ids=set();accepted=0;rejections=[]
    # Read members as streams, never extract archive paths onto the filesystem.
    with tarfile.open(archive,'r|*') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.xyz'):continue
            try:
                if member.size>1_000_000:raise ValueError('Oversized molecule record')
                record=tar.extractfile(member).read().decode()
                rawid=int(record.splitlines()[1].split()[1])
                if rawid in excluded:raise ValueError('Official uncharacterized exclusion')
                row,xyz=parse_xyz(record)
                if row['qm9_id'] in ids:raise ValueError('Duplicate QM9 ID')
                ids.add(row['qm9_id'])
                if row['structure_key'] in seen:raise ValueError('Duplicate canonical structure')
                seen.add(row['structure_key']);accepted+=1
                item=(row,xyz)
                if sample is None or len(reservoir)<sample:reservoir.append(item)
                else:
                    position=rng.randrange(accepted)
                    if position<sample:reservoir[position]=item
            except (ValueError,IndexError,UnicodeError,RuntimeError) as exc:
                rejections.append({'member':member.name,'reason':str(exc)})
    import pandas as pd
    pd.DataFrame(rejections,columns=['member','reason']).to_csv(root/'rejections.csv',index=False)
    if not reservoir:raise ValueError('No accepted molecules; inspect rejections.csv')
    if sample is not None and accepted<sample:raise ValueError('Requested sample exceeds accepted unique molecules')
    (root/'xyz').mkdir()
    fields=None
    with (root/'features.csv').open('w',newline='') as out:
        for row,xyz in sorted(reservoir,key=lambda item:item[0]['qm9_id']):
            xyz_path=root/'xyz'/f"{row['molecule_id']}.xyz";xyz_path.write_text(xyz)
            row['xyz_path']=str(xyz_path)
            row.update(calculate_descriptors(row['SMILES']))
            row.update(extra_features(row['SMILES'],include_fp=fingerprints))
            if fields is None:
                fields=csv.DictWriter(out,fieldnames=list(row));fields.writeheader()
            fields.writerow(row)
    (root/'metadata.json').write_text(json.dumps({'source':SOURCE,'protocol_id':PROTOCOL,
        'archive_sha256':file_hash(archive),'exclusions_sha256':file_hash(exclusions),
        'accepted_unique':accepted,'selected':len(reservoir),'rejected':len(rejections),
        'sample':sample,'seed':seed,'fingerprints':fingerprints,'hartree_to_ev':HARTREE_TO_EV,
        'additional_exclusions':'11 convergence exceptions listed in original README',
        'identity_source':'QM9 supplied B3LYP optimized SMILES; not our Gaussian parser'},indent=2))
    return root/'features.csv'


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('download');d.add_argument('directory')
    q=sub.add_parser('prepare');q.add_argument('archive');q.add_argument('exclusions');q.add_argument('output')
    q.add_argument('--sample',type=int);q.add_argument('--seed',type=int,default=42);q.add_argument('--fingerprints',action='store_true')
    a=p.parse_args()
    if a.command=='download':download(a.directory)
    else:print(prepare(a.archive,a.exclusions,a.output,a.sample,a.seed,a.fingerprints))

if __name__=='__main__':main()
