"""Same-filesystem durable artifacts. SQLite references are committed afterwards."""
import hashlib
import os
from pathlib import Path
import tempfile


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def atomic_write(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as out:
            out.write(content);out.flush();os.fsync(out.fileno())
        os.replace(temp,path)
        directory=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return file_hash(path)


def publish_stream(path, write, failpoint=None):
    """Durably publish a streamed artifact, preserving conflicting/orphan evidence."""
    from .state import crash
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    # Interrupted temporary files remain evidence; they are never completion records.
    with os.fdopen(fd,'w',encoding='utf-8',newline='') as out:
        write(out);out.flush();os.fsync(out.fileno())
    sha=file_hash(temp)
    crash('before_final_publication',failpoint)
    if path.exists():
        if file_hash(path)!=sha:
            raise ValueError(f'Conflicting/corrupt artifact retained: {path}; candidate retained: {temp}. Restore the verified original or investigate before retrying.')
        os.unlink(temp)
    else:
        os.replace(temp,path)
    directory=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    crash('after_artifact_rename',failpoint)
    return sha


def publish_file(path,source):
    """Copy a staged binary/text report in bounded blocks, durably and immutably."""
    import shutil
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    sha=file_hash(source)
    if path.exists():
        if file_hash(path)!=sha:raise ValueError('Report artifact conflict retained: '+str(path))
        return sha
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    with os.fdopen(fd,'wb') as out,Path(source).open('rb') as inp:
        shutil.copyfileobj(inp,out,1024*1024);out.flush();os.fsync(out.fileno())
    os.replace(temp,path)
    directory=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return sha
