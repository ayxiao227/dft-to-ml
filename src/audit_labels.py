"""Compare old labels to accepted rebuilt labels without modifying either."""
import argparse
from pathlib import Path
import pandas as pd


def compare(old_path, new_path, output):
    old, new = pd.read_csv(old_path), pd.read_csv(new_path)
    keys = ['molecule_id','run_id']
    if not all(set(keys).issubset(df.columns) for df in (old,new)):
        raise ValueError('Both files require explicit molecule_id/run_id mappings; no name guessing')
    result = old[keys+['gap_ev']].merge(new[keys+['gap_ev']], on=keys,
                how='outer', validate='one_to_one', suffixes=('_old','_new'), indicator=True)
    result['gap_change_ev'] = result.gap_ev_new - result.gap_ev_old
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False)
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('old');p.add_argument('new');p.add_argument('output')
    a=p.parse_args();compare(a.old,a.new,a.output)
