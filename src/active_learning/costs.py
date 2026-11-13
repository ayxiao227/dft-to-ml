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
    costs=[json.loads(a['cost']) if a.get('cost') else {} for a in attempts]
    result={'attempts':len(attempts)}
    for field in ['cpu_seconds','wall_seconds','allocated_cpu_hours']:
        values=[c[field] for c in costs if c.get(field) is not None]
        result[field]={'measured_sum':sum(values) if values else None,'known_attempts':len(values),
            'total_attempts':len(attempts),'complete':bool(attempts) and len(values)==len(attempts)}
    return result
