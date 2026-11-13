"""Shared ordered feature registry. Legacy semantics are explicitly frozen."""
BASE_FEATURES = ['MolWt', 'HeavyAtomCount', 'NumC', 'NumN', 'NumO', 'RingCount',
                 'AromaticRingCount', 'RotatableBondCount', 'HBD', 'HBA', 'TPSA',
                 'FractionCSP3', 'DoubleBondCount', 'TripleBondCount']
LEGACY_CONJ_FEATURES = ['n_conj_bonds', 'frac_conj_bonds', 'n_carbonyl',
                        'n_conj_carbonyl', 'n_conj_diene', 'n_nitroso_noxide']
INPUT_FEATURES = BASE_FEATURES + ['NumH', 'NumS', 'NumP', 'NumF', 'NumCl',
    'n_conj_bonds', 'frac_conj_bonds', 'n_carbonyl', 'n_conj_carbonyl_v2',
    'n_conj_diene', 'n_ene_yne', 'n_nitroso', 'n_amine_noxide', 'n_azo']
DEFINITIONS = {
    'MolWt': ('RDKit average molecular weight', 'g/mol'),
    'TPSA': ('RDKit topological polar surface area', 'angstrom^2'),
    'FractionCSP3': ('fraction of carbons with sp3 hybridization', 'fraction'),
    'frac_conj_bonds': ('RDKit conjugated bonds divided by max(bond count,1)', 'fraction'),
    'n_nitroso': ('unique neutral N centers double bonded to O, excluding nitro/azo', 'count'),
    'n_amine_noxide': ('unique tertiary aliphatic or aromatic N+ centers bonded to O-', 'count'),
    'n_azo': ('unique neutral N=N bonds', 'count'),
    'n_conj_carbonyl_v2': ('unique C=O carbon centers single-bonded to aromatic C or C with adjacent C=C/C=N/C#C/C#N; excludes amide/ester resonance alone', 'count'),
    'n_conj_carbonyl': ('legacy v1 SMARTS [OX1]=[CX3]-[#6,#7]=,#[#6,#7,#8]', 'count'),
    'n_nitroso_noxide': ('legacy v1 [#7]=[#8,#7], includes azo and misses charge-separated N-oxide', 'count'),
    'NumH': ('all H including implicit and graph-explicit attached hydrogen', 'count'),
    'n_conj_diene': ('unique RDKit embeddings of C=C-C=C', 'count'),
    'n_ene_yne': ('unique RDKit embeddings of C=C-C#C; excluded from legacy analysis, included in input v2', 'count'),
}
for name, definition in {
    'HeavyAtomCount':'non-hydrogen atoms', 'RingCount':'RDKit ring count',
    'AromaticRingCount':'RDKit aromatic ring count', 'RotatableBondCount':'Lipinski rotatable bonds',
    'HBD':'Lipinski H-bond donors', 'HBA':'Lipinski H-bond acceptors',
    'DoubleBondCount':'graph double bonds', 'TripleBondCount':'graph triple bonds',
    'n_conj_bonds':'RDKit bonds flagged conjugated', 'n_carbonyl':'unique C=O centers',
}.items(): DEFINITIONS[name] = (definition, 'count')
for element in ['C','N','O','S','P','F','Cl']:
    DEFINITIONS['Num'+element] = ('graph '+element+' atoms', 'count')


def registry(names, version):
    return [{'name': n, 'definition': DEFINITIONS[n][0], 'units': DEFINITIONS[n][1],
             'availability': 'input_2d', 'version': version, 'missingness': 'typed failure; no imputation'} for n in names]


def fingerprint_columns(columns):
    return sorted((c for c in columns if c.startswith('morgan_')), key=lambda c: int(c.split('_')[1]))
