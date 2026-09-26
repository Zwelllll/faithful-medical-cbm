"""Pre-test metadata/binary attestation. Never imports or constructs a data loader."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = 'artifacts/final_protocol/development_freeze_manifest.json'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def frozen_state(root: Path) -> tuple[dict, dict]:
    """Check only frozen source/model metadata and aggregate split metadata."""
    manifest = json.loads((root / MANIFEST).read_bytes())
    mismatch = []
    for group in ('source_config_docs_sha256', 'model_metadata_sha256'):
        for name, expected in manifest[group].items():
            path = root / name
            if not path.is_file() or sha(path) != expected:
                mismatch.append(name)
    split = manifest['split']
    if sha(root / split['metadata_path']) != split['metadata_sha256']:
        mismatch.append(split['metadata_path'])
    if mismatch:
        raise ValueError(f'Frozen files changed; do not update freeze: {mismatch}')
    report = {'current_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'freeze_base_commit': manifest['git']['commit'], 'frozen_hashes_match': True,
              'frozen_documentation_changes': [], 'manifest_sha256': sha(root / MANIFEST)}
    return manifest, report


def evaluator_hashes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): sha(p)
            for p in sorted((root/'src/faithful_medical_cbm/evaluation').glob('final_*.py'))}


def development_folds(root: Path, manifest: dict) -> dict[str, int]:
    path = root / 'data/splits/development_folds.csv'
    if sha(path) != manifest['split']['file_sha256'][path.name]:
        raise ValueError('Frozen development folds changed')
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    folds = {r['case_num']: int(r['validation_fold']) for r in rows}
    if len(folds) != 658 or len(rows) != 658 or set(folds.values()) != set(range(4)):
        raise ValueError('Invalid frozen development fold metadata')
    return folds


def validate_payload(payload: dict, family: str, fold: int, epoch: int,
                     manifest: dict, folds: dict[str, int]) -> dict:
    """No inference. Reject missing fields rather than infer fold from a filename."""
    cfg, provenance = payload['model_config'], payload['provenance']
    if payload['epoch'] != epoch or provenance['validation_fold'] != fold:
        raise ValueError('Checkpoint epoch/fold mismatch')
    order = manifest['protocol']['concept_order']
    architecture = {'black_box': 'efficientnet_b0', 'seven_concept': 'efficientnet_b0_concept',
                    'joint_soft': 'efficientnet_b0_joint_cbm', 'joint_hard_ste': 'efficientnet_b0_joint_cbm'}[family]
    if cfg['architecture'] != architecture or cfg['pretrained'] is not True:
        raise ValueError('Checkpoint architecture/pretraining mismatch')
    if family == 'black_box':
        if cfg.get('concept_names') or cfg.get('concept_order') or cfg.get('diagnosis_inputs'):
            raise ValueError('Black box must be concept-free')
        if cfg.get('model_type', 'black_box') != 'black_box':
            raise ValueError('Black-box model type mismatch')
    else:
        key = 'concept_names' if family == 'seven_concept' else 'concept_order'
        if cfg[key] != order or provenance['concept_order'] != order:
            raise ValueError('Frozen concept order mismatch')
    if family.startswith('joint_'):
        if cfg['model_type'] != family or cfg['diagnosis_inputs'] != 7:
            raise ValueError('Joint model family/inputs mismatch')
        if payload['loss_weights'] != {'concept': 1., 'diagnosis': 1.}:
            raise ValueError('Joint losses changed')
    flags = {}
    for owner, record in (('payload', payload), ('provenance', provenance)):
        for key in ('locked_test_used', 'locked_test_images_accessed', 'locked_test_accessed'):
            if key in record:
                flags[f'{owner}.{key}'] = record[key]
                if record[key] is not False:
                    raise ValueError('Checkpoint reports test access')
    if not flags:
        raise ValueError('No recorded false test-access provenance')
    expected_val = {i for i, f in folds.items() if f == fold}
    expected_train = set(folds) - expected_val
    for key, expected in (('training_ids', expected_train), ('validation_ids', expected_val)):
        ids = provenance[key]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError(f'Checkpoint {key} differs from frozen development fold')
    if provenance['cohort_sha256'] != manifest['split']['cohort_sha256'] or provenance['split_metadata_sha256'] != manifest['split']['metadata_sha256']:
        raise ValueError('Checkpoint cohort/split provenance mismatch')
    if provenance['seed'] != 42:
        raise ValueError('Checkpoint seed mismatch')
    settings = payload['config']
    preprocessing = manifest['protocol']['preprocessing']
    if settings['experiment']['image_size'] != 224 or settings['normalization']['mean'] != preprocessing['normalization_mean'] or settings['normalization']['std'] != preprocessing['normalization_std']:
        raise ValueError('Checkpoint preprocessing mismatch')
    # Metadata explicitly records the baseline architecture; strict state loading
    # below proves that its weights have no concept bottleneck or extra head.
    return {'family': family, 'fold': fold, 'epoch': epoch, 'model_config': cfg,
            'test_access_flags': flags, 'git_commit': provenance.get('git_commit'),
            'training_config_sha256': provenance.get('config_sha256'), 'provenance': provenance,
            'model_type_evidence': 'model_config.architecture + strict frozen state-dict compatibility' if family == 'black_box' else 'model_config and frozen concept order',
            'training_cases': len(expected_train), 'validation_cases': len(expected_val)}


def construct_model(family: str):
    """Construct without weights download or a forward pass."""
    from ..models.black_box import BlackBoxEfficientNet
    from ..models.seven_concept import SevenConceptEfficientNet
    from ..models.joint_cbm import JointCBM
    if family == 'black_box':
        return BlackBoxEfficientNet(pretrained=False)
    if family == 'seven_concept':
        return SevenConceptEfficientNet(pretrained=False)
    if family in ('joint_soft', 'joint_hard_ste'):
        return JointCBM(family, pretrained=False)
    raise ValueError('Unknown model family')


def load_verified_model(path: Path, family: str, fold: int, epoch: int,
                        manifest: dict, folds: dict, expected_sha: str | None = None):
    import torch
    before = sha(path)
    if expected_sha is not None and before != expected_sha:
        raise ValueError('Checkpoint differs from attested bytes')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    evidence = validate_payload(payload, family, fold, epoch, manifest, folds)
    model = construct_model(family)
    model.load_state_dict(payload['model_state'], strict=True)
    if any(not torch.isfinite(value).all().item() for value in model.state_dict().values() if value.is_floating_point()):
        raise ValueError('Nonfinite checkpoint parameters/buffers')
    model.eval()
    if sha(path) != before:
        raise ValueError('Checkpoint changed during loading')
    evidence.update(sha256=before, size_bytes=path.stat().st_size, strict_state_dict_compatible=True)
    return model, evidence


def verify_checkpoints(root: Path, drive_outputs: Path) -> dict:
    manifest, state = frozen_state(root)
    folds = development_folds(root, manifest)
    records = []
    for family, checkpoints in manifest['checkpoints'].items():
        for selected in checkpoints:
            relative = selected['expected_path_relative_to_drive_outputs']
            path = drive_outputs / relative
            record = {'family': family, 'fold': selected['fold'], 'epoch': selected['epoch'], 'relative_path': relative,
                      'exists': path.is_file(), 'sha256': None, 'size_bytes': None}
            try:
                if not path.is_file():
                    raise FileNotFoundError(f'Missing checkpoint: {path}')
                record.update(sha256=sha(path), size_bytes=path.stat().st_size)
                model, evidence = load_verified_model(path, family, selected['fold'], selected['epoch'], manifest, folds, record['sha256'])
                del model
                run_dir = manifest['protocol']['source_run_directories'][family]
                if run_dir:
                    saved = json.loads((root/run_dir/f"fold_{selected['fold']}/run.json").read_bytes())
                    if saved['provenance'] != evidence['provenance']:
                        raise ValueError('Checkpoint provenance differs from frozen development run.json')
                record.update(evidence, status='verified')
            except Exception as error:
                # Retain one explicit failed record even for truncated/unsafe-pickle
                # files or unavailable dependencies; never fall back to unsafe load.
                record.update(status='failed', error=f'{type(error).__name__}: {error}')
            records.append(record)
    count = sum(r['status'] == 'verified' for r in records)
    return {'stage': '15A', 'created_utc': datetime.now(timezone.utc).isoformat(), **state,
            'expected_count': 16, 'verified_count': count, 'status': 'passed' if count == 16 else 'blocked',
            'drive_outputs': str(drive_outputs), 'records': records, 'python': platform.python_version(),
            'evaluator_sha256': evaluator_hashes(root),
            'locked_test_accessed': False, 'final_test_executed': False, 'inference_performed': False}


def validate_attestation(report: dict, root: Path, drive_outputs: Path, manifest: dict) -> None:
    if report['status'] != 'passed' or report['verified_count'] != 16 or report['expected_count'] != 16:
        raise ValueError('All 16 checkpoints must pass verification before test access')
    if report['manifest_sha256'] != sha(root / MANIFEST) or report['locked_test_accessed'] is not False:
        raise ValueError('Attestation does not match the untouched frozen protocol')
    if report['evaluator_sha256'] != evaluator_hashes(root) or report['final_test_executed'] is not False or report['inference_performed'] is not False:
        raise ValueError('Evaluator changed since pre-execution attestation')
    records = report['records']
    expected = {(family, r['fold']): r for family, rows in manifest['checkpoints'].items() for r in rows}
    if len(records) != 16 or {(r['family'], r['fold']) for r in records} != set(expected):
        raise ValueError('Missing/duplicate checkpoint attestation')
    for record in records:
        selected = expected[record['family'], record['fold']]
        relative = selected['expected_path_relative_to_drive_outputs']
        if record['relative_path'] != relative or record['epoch'] != selected['epoch'] or record['status'] != 'verified' or record['strict_state_dict_compatible'] is not True:
            raise ValueError('Attestation selection mismatch')
        path = drive_outputs / relative
        if sha(path) != record['sha256'] or path.stat().st_size != record['size_bytes']:
            raise ValueError('Attested checkpoint bytes changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--drive-outputs', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Attestation exists; refusing overwrite')
    if 'final_test' in args.report.parts:
        raise ValueError('Verification reports belong outside the final-test result namespace')
    result = verify_checkpoints(args.root.resolve(), args.drive_outputs.resolve())
    write_json(args.report, result)
    print(f"{result['verified_count']}/16 verified; {result['status']}")
    if result['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
