"""Lightweight preparation admission shared by training and public data checks.

Legacy pools remain usable with unverified provenance. A modern manifest must
declare completed preparation; this does not certify its tensor hash claims.
"""
import hashlib
import json
from pathlib import Path


def pool_paths(directory):
    paths = {}
    for split in ('train', 'val'):
        found = [Path(directory) / name for name in (f'{split}.pt', f'fineweb_{split}.pt')
                 if (Path(directory) / name).is_file()]
        if not found:
            raise ValueError(f'missing {split}.pt in {directory}')
        if len(found) != 1:
            raise ValueError(f'ambiguous {split} pool filenames in {directory}')
        paths[split] = found[0]
    return paths


def read_preparation_manifest(directory, expected_corpus=None):
    path = Path(directory) / 'corpus.json'
    provenance = {'manifest_path': str(path.resolve()), 'manifest_sha256': None,
                  'status': 'legacy_provenance_unverified',
                  'full_tensor_hashes_verified': False,
                  'verification_scope': 'preparation status only; no tensor hashes verified'}
    if not path.exists():
        provenance['reason'] = 'no corpus manifest'
        return None, provenance
    raw = path.read_bytes()
    manifest = json.loads(raw)
    provenance['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    if not isinstance(manifest, dict) or (expected_corpus is not None and manifest.get('corpus') != expected_corpus):
        raise ValueError(f'wrong or missing corpus label in {path}')
    modern = {'schema_version', 'status', 'revision', 'requested_revision', 'repo_id',
              'tokenizer', 'packages', 'outputs', 'selected_shards', 'consumed_shards'}
    if not modern.intersection(manifest):
        provenance['reason'] = 'legacy corpus label without modern provenance'
        return None, provenance
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError(f'unsupported or missing modern corpus schema in {path}')
    if manifest.get('status') != 'complete':
        raise ValueError(f'corpus preparation is not complete in {path}')
    provenance.update(status='complete_manifest_status_checked',
                      corpus=manifest.get('corpus'), revision=manifest.get('revision'))
    return manifest, provenance
