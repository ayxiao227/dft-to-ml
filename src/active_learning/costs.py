"""Explicit attempt accounting. Consumed CPU time is not allocated CPU-hours."""
import math

FIELDS={'job_id','scope','cpu_seconds','wall_seconds','allocated_cpus','allocated_cpu_hours','source'}


def validate_cost(value):
    if not isinstance(value,dict) or set(value)-FIELDS:raise ValueError('Unknown resource-usage fields')
    out={k:value.get(k) for k in FIELDS}
    if not all(isinstance(out[k],str) and out[k] for k in ['job_id','source']):raise ValueError('Cost needs job_id and source')
    if out['scope'] not in ('job','step'):raise ValueError('Cost scope must be job or step')
    for k in ['cpu_seconds','wall_seconds','allocated_cpu_hours']:
        v=out[k]
        if v is not None and (type(v) not in (int,float) or not math.isfinite(v) or v<0):raise ValueError('Invalid resource units/value')
    if out['allocated_cpus'] is not None and (type(out['allocated_cpus']) is not int or out['allocated_cpus']<1):raise ValueError('Invalid allocated CPU count')
    return out


def coverage(attempts):
    import json
    fields=['cpu_seconds','wall_seconds','allocated_cpu_hours']
    totals={f:0 for f in fields};known={f:0 for f in fields};count=0
    for attempt in attempts:
        cost=json.loads(attempt['cost']) if attempt.get('cost') else {};count+=1
        for field in fields:
            if cost.get(field) is not None:totals[field]+=cost[field];known[field]+=1
    result={'attempts':count}
    for field in fields:
        result[field]={'measured_sum':totals[field] if known[field] else None,'known_attempts':known[field],
            'total_attempts':count,'complete':bool(count) and known[field]==count}
    return result
