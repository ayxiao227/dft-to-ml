"""Cached acquisition with bounded retries and durable failure records."""
import hashlib
import json
import time
from pathlib import Path
from urllib.error import URLError
import pubchempy as pcp


def fetch(properties, query, classifier, cache_dir='data/pubchem_cache', attempts=3):
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([properties,str(query),classifier]).encode()).hexdigest()
    path = root / (key + '.json')
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(attempts):
        try:
            records = pcp.get_properties(properties, query, classifier)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(records))
            temporary.replace(path)
            return records
        except (pcp.PubChemHTTPError, URLError, TimeoutError, ConnectionError) as exc:
            status = getattr(exc, 'code', None)
            retry = status is None or status in {429,500,502,503,504}
            if not retry or attempt + 1 == attempts:
                with (root / 'failures.jsonl').open('a') as out:
                    out.write(json.dumps({'query':str(query),'classifier':classifier,'error':str(exc)})+'\n')
                raise
            time.sleep(2 ** attempt)
