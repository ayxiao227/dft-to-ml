"""Program-level label isolation, not a security boundary against file owners."""
import json
from pathlib import Path

class RetrospectiveOracle:
    def __init__(self, path, partitions, reservation_lookup):
        # This file contains only initial/pool labels, never validation/final labels.
        self.__labels=json.loads(Path(path).read_text())
        self.__roles={r['source_record_id']:r['origin'] for r in partitions}
        self.__reservations=reservation_lookup

    def initial(self, ids):
        if len(ids)!=len(set(ids)) or any(self.__roles.get(i)!='initial' for i in ids):
            raise ValueError('Initial oracle access requires unique initial IDs')
        return {i:self.__labels[i] for i in ids}

    def query(self, reservation_id, ids):
        allowed=self.__reservations(reservation_id)
        if len(ids)!=len(set(ids)) or set(ids)!=set(allowed) or any(self.__roles.get(i)!='pool' for i in ids):
            raise ValueError('Only this committed unfulfilled pool reservation may be queried')
        return {i:self.__labels[i] for i in ids}

class EvaluationOracle:
    def __init__(self, path, expected_ids):
        self.__path=Path(path);self.__ids=tuple(expected_ids)

    def read(self, ids):
        if tuple(ids)!=self.__ids:raise ValueError('Evaluation membership differs')
        table=json.loads(self.__path.read_text())
        if set(table)!=set(ids):raise ValueError('Evaluation table membership differs')
        return {i:table[i] for i in ids}
