"""Self-contained source/data/toolkit bundle; recreate, never copy, virtualenvs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

from rental.common import ROOT, output, read, relative, sha, validate_hashes, write

ENVIRONMENTS = {
    'training': ('.venv', 'deploy/rental/training.lock'),
    'mini': ('runs/envs/mini-sglang', 'config/delivery/requirements.txt'),
    'nano': ('runs/envs/nano-vllm', 'deploy/rental/nano.lock'),
}


def verify_assets(root):
    model = read(root / 'config/delivery/model.json')
    validate_hashes(root / 'runs/models/Qwen3-0.6B', model['files'])
    config = read(root / 'config/f_ep_rental_requirements.json')
    for name, key in (('train.bin', 'train_sha256'), ('val.bin', 'validation_sha256')):
        if sha(root / 'data/tinystories_full' / name) != config['data'][key]:
            raise ValueError('TinyStories hash mismatch: ' + name)
    nano = read(root / 'config/serving/h1_nanovllm.json')['environment']
    wheel = root / 'runs/downloads' / nano['flash_attn_wheel'].rsplit('/', 1)[1]
    if sha(wheel) != nano['flash_attn_wheel_sha256']:
        raise ValueError('FlashAttention wheel mismatch')
    return wheel


def build_bundle(destination, root=ROOT):
    if output(['git', 'status', '--porcelain'], root):
        raise ValueError('Commit source before packaging; clean source checkout required')
    wheel = verify_assets(root)
    destination.mkdir(parents=True, exist_ok=False)
    repositories = {'project': root, 'nano': root / 'runs/vendor/nano-vllm',
                    'mini': root / 'runs/vendor/mini-sglang-i2'}
    commits = {}
    for name, repo in repositories.items():
        if output(['git', 'status', '--porcelain'], repo):
            raise ValueError('Dirty repository: ' + name)
        commits[name] = output(['git', 'rev-parse', 'HEAD'], repo)
        subprocess.run(['git', 'bundle', 'create', str(destination / f'{name}.bundle'), 'HEAD'],
                       cwd=repo, check=True, timeout=180)
    if commits['nano'] != read(root / 'config/delivery/tp_matrix.json')['frameworks']['nano']['commit']:
        raise ValueError('Wrong nano-vLLM source')
    mini_tree = output(['git', 'rev-parse', 'HEAD^{tree}'], repositories['mini'])
    if mini_tree != read(root / 'config/delivery/tp_matrix.json')['frameworks']['mini']['tree']:
        raise ValueError('Wrong Mini-SGLang tree')
    assets = destination / 'assets'
    names = ['runs/models/Qwen3-0.6B/' + name for name in read(root / 'config/delivery/model.json')['files']]
    names += ['data/tinystories_full/' + name for name in ('train.bin', 'val.bin', 'manifest.json')]
    names += [str(wheel.relative_to(root))]
    for name in names:
        target = assets / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    # Dereference the local toolkit overlay: its absolute /usr/local links are
    # not portable. The packaged tree must have no dangling host dependencies.
    toolkit = 'runs/toolchains/cuda-13.0-i1'
    shutil.copytree(root / toolkit, assets / toolkit, symlinks=False)
    files = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    manifest = dict(schema_version=1, commits=commits, mini_tree=mini_tree, files=files,
                    bytes=sum((destination / name).stat().st_size for name in files),
                    virtualenvs_included=False, python='3.11.16', platform='Linux x86_64',
                    dependencies_require_uv_cache_or_network=True)
    write(destination / 'bundle.json', manifest)
    return manifest


def verify_bundle(bundle):
    manifest = read(bundle / 'bundle.json')
    if any(p.is_symlink() for p in bundle.rglob('*')):
        raise ValueError('Portable bundle may not contain symlinks')
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()} - {'bundle.json'}
    if actual != set(manifest['files']):
        raise ValueError('Bundle file set differs from manifest')
    validate_hashes(bundle, manifest['files'])
    return manifest


def restore_bundle(bundle, target):
    manifest = verify_bundle(bundle)
    if bundle == target or bundle in target.parents:
        raise ValueError('Restore target must be outside the bundle')
    if target.exists():
        raise ValueError('Restore requires a fresh target directory')
    destinations = {'project': target, 'nano': target / 'runs/vendor/nano-vllm',
                    'mini': target / 'runs/vendor/mini-sglang-i2'}
    for name, repo in destinations.items():
        subprocess.run(['git', 'clone', str(bundle / f'{name}.bundle'), str(repo)], check=True, timeout=180)
        subprocess.run(['git', 'checkout', '--detach', manifest['commits'][name]], cwd=repo, check=True, timeout=30)
    shutil.copytree(bundle / 'assets', target, dirs_exist_ok=True)
    verify_assets(target)
    for name, digest in manifest['files'].items():
        if name.startswith('assets/') and sha(target / relative(name[7:])) != digest:
            raise ValueError('Restored file mismatch: ' + name)
    write(target / 'runs/rental/deployment.json', dict(bundle_sha256=sha(bundle / 'bundle.json'),
          commits=manifest['commits'], assets_verified=True, environment_install_pending=True,
          assets_sha256={name[7:]: digest for name, digest in manifest['files'].items() if name.startswith('assets/')}))
    return dict(passed=True, files=len(manifest['files']), bytes=manifest['bytes'])


def environment_report(root):
    program = ('import importlib.metadata as m,json,platform; '
               'print(json.dumps(dict(python=platform.python_version(),packages='
               '{d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()})))')
    reports = {}
    for name, (directory, lock) in ENVIRONMENTS.items():
        observed = json.loads(output([root / directory / 'bin/python', '-c', program], root))
        expected = dict(line.split('==', 1) for line in (root / lock).read_text().splitlines()
                        if line and not line.startswith('#'))
        if name == 'nano':
            expected['flash-attn'] = read(root / 'config/serving/h1_nanovllm.json')['environment']['flash-attn']
        if observed['python'] != '3.11.16' or any(observed['packages'].get(k) != v for k, v in expected.items()):
            raise ValueError(f'{name} differs from pinned Python/package versions')
        reports[name] = dict(python=observed['python'], packages=expected, lock_sha256=sha(root / lock))
    return reports


def install_environments(root, offline=False):
    wheel = verify_assets(root)
    logdir = root / 'runs/rental/install'
    logdir.mkdir(parents=True, exist_ok=False)
    for name, (directory, lock) in ENVIRONMENTS.items():
        target = root / directory
        if target.exists():
            raise ValueError('Refusing to replace environment: ' + str(target))
        python = target / 'bin/python'
        commands = [['uv', 'venv', '--python', '3.11.16', str(target)],
                    ['uv', 'pip', 'sync', '--torch-backend', 'cu129', '--python', str(python), str(root / lock)]]
        if name == 'nano':
            commands.append(['uv', 'pip', 'install', '--no-deps', '--python', str(python), str(wheel)])
        commands.append(['uv', 'pip', 'check', '--python', str(python)])
        with (logdir / f'{name}.log').open('x') as log:
            for command in commands:
                if offline:
                    command.insert(1, '--offline')
                log.write(json.dumps(command) + '\n'); log.flush()
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800,
                               env={**os.environ, 'UV_LINK_MODE': 'clone'})
    result = environment_report(root)
    (root / 'runs/tmp').mkdir(parents=True, exist_ok=True)
    write(logdir / 'environments.json', result)
    return result
