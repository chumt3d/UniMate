"""Bundle bounded retained research evidence; omit duplicated per-window skin readbacks."""
import hashlib
import json
import tarfile
import time
from pathlib import Path
import modal

image = modal.Image.from_id('im-KhepHmwYkODnf6SDKdLAjZ')
volume = modal.Volume.from_name('tin88-unimate-eval', environment_name='main')
app = modal.App('tin88-unimate-escalation')


@app.function(image=image, volumes={'/evaluation': volume}, cpu=(1, 1), memory=(1536, 1536),
              timeout=180, retries=0, max_containers=1, scaledown_window=2)
def pack():
    base = Path('/evaluation')
    output = base / 'unimate-20261021-escalation-final-bundles'
    output.mkdir(exist_ok=False)
    started = time.time()
    groups = [(f'{case}-{seed}', [base / 'unimate-20261021-escalation-long-skin' / case / str(seed)])
              for case in ('v3-walk', 'truebones-forward') for seed in (1729, 2718, 31415)]
    groups += [(arm, [base / 'unimate-20261021-escalation-gait' / arm]) for arm in ('procedural', 'hybrid')]
    groups += [(label, [base / name]) for label, name in (
        ('generation-c', 'unimate-20261021-escalation-c'),
        ('generation-d', 'unimate-20261021-escalation-d'),
        ('generation-e', 'unimate-20261021-escalation-e'),
        ('metrics', 'unimate-20261021-escalation-skin-metrics'))]
    groups += [('initial-skin', [base / 'unimate-20261021-escalation-skin/initial'])]
    manifest = {'scope': 'Retained research evidence, not acceptance', 'bundles': [], 'missingRoots': []}
    for label, roots in groups:
        files = []
        for root in roots:
            if not root.is_dir():
                manifest['missingRoots'].append(str(root)); continue
            for file in sorted(root.rglob('*')):
                if file.is_symlink(): raise ValueError(f'Unexpected symlink: {file}')
                if not file.is_file(): continue
                relative = file.relative_to(root)
                if any(part.startswith('window-') for part in relative.parts[:-1]): continue
                if file.stat().st_size > 32_000_000: raise ValueError(f'Unexpected large evidence file: {file}')
                files.append(file)
        archive = output / f'{label}.tar.gz'
        with tarfile.open(archive, 'w:gz', compresslevel=1) as bundle:
            for file in files: bundle.add(file, arcname=str(file.relative_to(base)), recursive=False)
        digest = hashlib.sha256()
        with archive.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''): digest.update(chunk)
        if archive.stat().st_size >= 32_000_000: raise ValueError(f'Bundle exceeds recovery envelope: {archive}')
        manifest['bundles'].append({'path':str(archive.relative_to(base)), 'bytes':archive.stat().st_size,
            'sha256':digest.hexdigest(), 'files':[str(file.relative_to(base)) for file in files]})
        volume.commit()
    manifest['seconds'] = time.time() - started
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    volume.commit()
    return {'output':str(output), 'bundles':len(manifest['bundles']), 'missingRoots':manifest['missingRoots'],
            'bytes':sum(item['bytes'] for item in manifest['bundles']), 'seconds':manifest['seconds']}
