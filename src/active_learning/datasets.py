"""Curate hidden labels separately from input-only records. Legacy imports unchanged."""
import csv
import json
import math
import tarfile
from pathlib import Path
from src.schema import canonical_smiles
from src.csv_conversion import passes_filters
from src.gaussian_records import identity_matches
from src.qm9 import parse_xyz, excluded_ids, PROTOCOL, HARTREE_TO_EV
from .records import MolecularRecord, check_collisions
from .artifacts import file_hash

CONVERGENCE_EXCEPTIONS = {21725,87037,59827,117523,128113,129053,129152,129158,130535,6620,59818}
# Source printed gap precision is 1e-4 Ha; tolerate one last printed unit, not identity changes.
LABEL_TOLERANCE_EV = 1e-4 * HARTREE_TO_EV


def verify_sources(config):
    source=config['source']
    if file_hash(source['path']) != source['sha256']:
        raise ValueError('Source hash changed')
    if source['exclusions_path'] and file_hash(source['exclusions_path']) != source['exclusions_sha256']:
        raise ValueError('Exclusions hash changed')


def validate_labeled(rows, expected=None):
    protocols={r.get('observed_protocol_id') for r in rows}
    if not rows or None in protocols or '' in protocols or len(protocols)!=1:
        raise ValueError('Labels require one explicit observed protocol')
    if expected is not None and protocols!={expected}:
        raise ValueError('Labels violate expected protocol')
    for r in rows:
        if r.get('units')!='eV' or r.get('provenance_type') not in ('publisher_qm9','verified_gaussian','synthetic_fixture') or not math.isfinite(r['gap_ev']) or r['gap_ev']<0:
            raise ValueError('Invalid label provenance, units or target')


def deduplicate(records, labels):
    """Quarantine connected duplicate components with conflicts; never average."""
    check_collisions(records)
    by_id={r.source_record_id:r for r in records}; parent={i:i for i in by_id}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for r in records:
        for kind,key in [('input',r.input_structure_key),('optimized',r.optimized_structure_key)]:
            if key is None:continue
            k=(kind,key)
            if k in seen:parent[find(r.source_record_id)]=find(seen[k])
            else:seen[k]=r.source_record_id
    groups={}
    for i in sorted(by_id):groups.setdefault(find(i),[]).append(i)
    kept=[];ledger=[];kept_labels={}
    for ids in groups.values():
        members=[by_id[i] for i in ids]
        values=[labels[i]['gap_ev'] for i in ids if i in labels]
        protocols={labels[i]['observed_protocol_id'] for i in ids if i in labels}
        inputs={r.input_smiles for r in members};outputs={r.optimized_structure_key for r in members}
        conflict=(len(inputs)>1 or len(outputs)>1 or len(protocols)>1 or (values and max(values)-min(values)>LABEL_TOLERANCE_EV))
        representative=min(ids)
        if conflict:
            ledger.extend({'source_record_id':i,'reason':'conflicting duplicate identity/label/protocol','stage':'dedup','representative':None} for i in ids)
            continue
        kept.append(by_id[representative])
        if representative in labels:kept_labels[representative]=labels[representative]
        ledger.extend({'source_record_id':i,'reason':'consistent duplicate','stage':'dedup','representative':representative} for i in ids if i!=representative)
    return sorted(kept,key=lambda r:r.source_record_id),kept_labels,ledger


def load_source(config):
    verify_sources(config)
    source=config['source'];dataset=config['dataset'];records=[];labels={};ledger=[]
    def add(ident, original, optimized=None, gap=None, protocol=None, provenance=None, evidence=None):
        # Sanitize both before identity_matches; do not normalize tautomers or strip stereo.
        original_c=canonical_smiles(original)
        optimized_c=canonical_smiles(optimized) if optimized is not None else None
        if not passes_filters(original_c):raise ValueError('Input fails organic eligibility')
        if optimized_c is not None and (not passes_filters(optimized_c) or not identity_matches(original_c,optimized_c)):
            raise ValueError('Input/optimized graph or specified stereo mismatch')
        if not isinstance(ident,str) or not ident.strip():raise ValueError('Explicit stable source record ID required')
        if gap is not None and (not math.isfinite(float(gap)) or float(gap)<0):raise ValueError('Invalid finite label')
        record=MolecularRecord.create(dataset,str(ident),original,source['sha256'],optimized)
        records.append(record)
        if gap is not None:
            labels[record.source_record_id]={'source_record_id':record.source_record_id,'gap_ev':float(gap),'units':'eV',
                'observed_protocol_id':protocol,'provenance_type':provenance,'evidence':evidence}
    if dataset=='qm9':
        exclusions=excluded_ids(source['exclusions_path'])
        with tarfile.open(source['path'],'r|*') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('.xyz'):continue
                ident=member.name
                try:
                    if member.size>1_000_000:raise ValueError('Oversized record')
                    text=archive.extractfile(member).read().decode();ident=int(text.splitlines()[1].split()[1])
                    if ident in exclusions:raise ValueError(source['exclusions_kind']+' uncharacterized exclusion')
                    if ident in CONVERGENCE_EXCEPTIONS:raise ValueError('Additional convergence exception')
                    row,_=parse_xyz(text)
                    if abs(row['gap_ev']-row['gap_hartree']*HARTREE_TO_EV)>1e-12:raise ValueError('Unit conversion mismatch')
                    add(row['molecule_id'],row['input_smiles'],row['optimized_smiles'],row['gap_ev'],PROTOCOL,'publisher_qm9',
                        {'member':member.name,'exclusions_sha256':source['exclusions_sha256'],'exclusions_kind':source['exclusions_kind']})
                except (ValueError,IndexError,UnicodeError,RuntimeError) as exc:
                    ledger.append({'source_record_id':str(ident),'reason':str(exc),'stage':'curation'})
    else:
        with Path(source['path']).open() as stream:
            reader=csv.DictReader(stream)
            for index,row in enumerate(reader):
                ident=row.get('source_record_id',row.get('molecule_id'))
                try:
                    original=row.get('input_smiles',row.get('SMILES'))
                    if dataset=='synthetic':
                        add(ident,original,row.get('optimized_smiles') or original,float(row['gap_ev']),'synthetic_fixture_v1','synthetic_fixture')
                    elif config['mode']=='prospective':
                        if any(row.get(k) for k in ['gap_ev','accepted','observed_protocol_id']):
                            raise ValueError('Prospective candidate source must be unlabeled')
                        add(ident,original)
                    else:
                        # An accepted=True column is never sufficient. Reinspect raw evidence.
                        from .gaussian import verify_external_label
                        verified=verify_external_label(row, Path(source['path']).parent,config)
                        add(ident,original,verified['optimized_smiles'],verified['gap_ev'],verified['observed_protocol_id'],'verified_gaussian',verified)
                except (ValueError,TypeError,KeyError,RuntimeError) as exc:
                    ledger.append({'source_record_id':str(ident),'reason':str(exc),'stage':'curation'})
    if labels:validate_labeled(list(labels.values()),config['protocol']['observed_protocol_id'] if dataset=='gaussian' else None)
    elif config['mode']=='retrospective':raise ValueError('No valid verified labels')
    records,labels,duplicates=deduplicate(records,labels);ledger.extend(duplicates)
    if not records:raise ValueError('No eligible unique records')
    return records,labels,ledger
