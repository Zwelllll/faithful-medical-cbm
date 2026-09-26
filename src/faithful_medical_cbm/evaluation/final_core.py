"""Pure array calculations and output schemas; no dataset or checkpoint access."""
from __future__ import annotations
import csv
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from .diagnosis_metrics import diagnosis_metrics, probability_arrays

MODEL_ORDER = ('black_box', 'sequential_soft', 'sequential_hard', 'joint_soft', 'joint_hard_ste', 'oracle')
BOOT_METRICS = ('auroc', 'macro_f1', 'accuracy', 'sensitivity', 'specificity', 'brier')


def binary(values):
    return (np.asarray(values) >= .5).astype(int)


def probabilities(values, shape=None):
    a = np.asarray(values, dtype=np.float64)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError('Invalid probability shape/value')
    return a


def ensemble(values, concepts=False):
    a = probabilities(values)
    if a.ndim != (3 if concepts else 2) or a.shape[1] != 4 or (concepts and a.shape[2] != 7):
        raise ValueError('Require exactly four fold probabilities in axis 1')
    return a.mean(axis=1)


def head_probability(head: dict, inputs):
    x = np.asarray(inputs, dtype=np.float64)
    w = np.asarray(head['coefficients'], dtype=np.float64)
    if x.shape[-1] != 7 or w.shape != (7,) or not np.isfinite(x).all() or not np.isfinite(w).all() or not np.isfinite(head['intercept']):
        raise ValueError('Invalid seven-input logistic head')
    return expit(x @ w + float(head['intercept']))


def route_predictions(folds: dict, heads: dict, truths):
    """Joint diagnosis stays on each model's end-to-end route."""
    truth = np.asarray(truths)
    if truth.ndim != 2 or truth.shape[1] != 7 or not np.isin(truth, [0, 1]).all():
        raise ValueError('Seven binary concept truths required')
    n = len(truth)
    q = ensemble(folds['seven_concept']['concept_probability'], True)
    if q.shape != (n, 7):
        raise ValueError('Case alignment mismatch')
    diagnoses = {'black_box': ensemble(folds['black_box']['diagnosis_probability']),
                 'sequential_soft': head_probability(heads['sequential_soft'], q),
                 'sequential_hard': head_probability(heads['sequential_hard'], binary(q)),
                 'oracle': head_probability(heads['oracle'], truth)}
    concepts = {'sequential_concept_ensemble': q}
    for family in ('joint_soft', 'joint_hard_ste'):
        diagnoses[family] = ensemble(folds[family]['diagnosis_probability'])
        concepts[family + '_concept_ensemble'] = ensemble(folds[family]['concept_probability'], True)
    for p in diagnoses.values():
        probabilities(p, (n,))
    for p in concepts.values():
        probabilities(p, (n, 7))
    return diagnoses, concepts


def metrics(y, p):
    # Final ranking is on probabilities, including intervention/ensemble metrics.
    return diagnosis_metrics(y, p, p, bins=10)


def concept_metrics(truth, p, order):
    truth = np.asarray(truth)
    p = probabilities(p, truth.shape)
    if truth.ndim != 2 or truth.shape[1] != 7 or not np.isin(truth, [0, 1]).all() or len(order) != 7:
        raise ValueError('Invalid concept labels/order')
    rows = []
    for j, concept in enumerate(order):
        y = truth[:, j]
        pred = binary(p[:, j])
        tp = ((y == 1) & (pred == 1)).sum(); tn = ((y == 0) & (pred == 0)).sum()
        fp = ((y == 0) & (pred == 1)).sum(); fn = ((y == 1) & (pred == 0)).sum()
        safe = lambda a, b: a / b if b else 0.
        auc = metrics(y, p[:, j])['auroc'] if len(set(y)) == 2 else None
        rows.append({'concept': concept, 'auroc': auc, 'macro_f1': float((safe(2*tp, 2*tp+fp+fn)+safe(2*tn, 2*tn+fp+fn))/2),
                     'reason': '' if auc is not None else 'AUROC undefined: one target class'})
    rows.append({'concept': 'macro_mean', 'auroc': float(np.mean([r['auroc'] for r in rows])) if all(r['auroc'] is not None for r in rows) else None,
                 'macro_f1': float(np.mean([r['macro_f1'] for r in rows])),
                 'reason': '' if all(r['auroc'] is not None for r in rows) else 'At least one concept AUROC undefined; none omitted'})
    return rows


def bootstrap_indices(ids, targets):
    y = np.asarray(targets)
    if len(ids) != 165 or len(set(ids)) != 165 or y.shape != (165,) or not np.isin(y, [0, 1]).all() or int(y.sum()) != 50:
        raise ValueError('Frozen bootstrap requires 165 unique cases: 50 positive/115 negative')
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    pos = np.array([i for i in order if y[i] == 1]); neg = np.array([i for i in order if y[i] == 0])
    rng = np.random.Generator(np.random.PCG64(42))
    return np.array([np.concatenate([rng.choice(pos, 50, replace=True), rng.choice(neg, 115, replace=True)]) for _ in range(2000)])


def bootstrap_statistics(ids, y, predictions, indices):
    """Vectorized identical paired resamples; no refitting or mutable settings."""
    y = np.asarray(y)
    if not np.array_equal(indices, bootstrap_indices(ids, y)):
        raise ValueError('Bootstrap indices differ from frozen draws')
    if set(predictions) != set(MODEL_ORDER):
        raise ValueError('All six diagnosis models required')
    truth = y[indices]; samples = {}
    for model in MODEL_ORDER:
        _, p = probability_arrays(y, predictions[model]); p = p[indices]; pred = p >= .5
        tp = ((truth == 1) & pred).sum(axis=1); tn = ((truth == 0) & ~pred).sum(axis=1)
        fp = 115-tn; fn = 50-tp
        ranks = rankdata(p, axis=1, method='average')
        samples[model] = {'auroc': ((ranks*(truth == 1)).sum(axis=1)-50*51/2)/(50*115),
            'macro_f1': (2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2,
            'accuracy': (tp+tn)/165, 'sensitivity': tp/50, 'specificity': tn/115,
            'brier': ((p-truth)**2).mean(axis=1)}
    interval = lambda a: dict(zip(('lower_95', 'upper_95'), map(float, np.percentile(a, [2.5, 97.5], method='linear'))))
    individual = [{'model': m, 'metric': k, **interval(samples[m][k])} for m in MODEL_ORDER for k in BOOT_METRICS]
    paired = [{'model_a': a, 'model_b': b, 'metric': k, **interval(samples[a][k]-samples[b][k])}
              for a, b in combinations(MODEL_ORDER, 2) for k in BOOT_METRICS]
    return individual, paired


def schemas(order):
    base = ['case_num', 'diagnosis_binary']
    targets = [c+'_target' for c in order]
    fold_d = [f'fold_{f}_diagnosis_{kind}' for f in range(4) for kind in ('logit', 'probability')]
    fold_c = [f'fold_{f}_{c}_{kind}' for f in range(4) for kind in ('logit', 'probability') for c in order]
    mean_c = [c+'_probability' for c in order]
    final = ['melanoma_probability', 'predicted_diagnosis', 'provenance_ref']
    result = {'black_box_predictions.csv': base+fold_d+final,
              'sequential_concept_predictions.csv': base+targets+fold_c+mean_c+['provenance_ref']}
    for mode in ('soft', 'hard'):
        result[f'sequential_{mode}_predictions.csv'] = base+targets+mean_c+[c+'_head_input' for c in order]+final
    result['oracle_predictions.csv'] = base+targets+['non_deployable']+final
    for name in ('joint_soft', 'joint_hard'):
        result[name+'_predictions.csv'] = base+targets+fold_d+fold_c+mean_c+final
    result.update({'diagnosis_metrics.csv': ['model','auroc','macro_f1','accuracy','sensitivity','specificity','brier','ece'],
                   'concept_metrics.csv': ['model','concept','auroc','macro_f1','reason'],
                   'bootstrap_intervals.csv': ['model','metric','lower_95','upper_95'],
                   'paired_bootstrap_differences.csv': ['model_a','model_b','metric','lower_95','upper_95']})
    return result


def write_csv(path: Path, rows, columns=None):
    iterator = iter(rows); first = next(iterator, None)
    if first is None:
        raise ValueError('Refuse empty output')
    columns = columns or list(first)
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n', extrasaction='raise')
        writer.writeheader(); writer.writerow(first); writer.writerows(iterator)


def prediction_tables(ids, y, truth, folds, heads, order):
    diagnoses, concepts = route_predictions(folds, heads, truth)
    schema = schemas(order)
    tables = {key: [] for key in schema if key.endswith('_predictions.csv')}
    q = concepts['sequential_concept_ensemble']
    for i, case in enumerate(ids):
        base = {'case_num': case, 'diagnosis_binary': int(y[i])}
        targets = {c+'_target': int(truth[i,j]) for j,c in enumerate(order)}
        for filename in tables:
            row = {**base, 'provenance_ref': 'integrity.json'}
            if filename != 'black_box_predictions.csv': row.update(targets)
            family = {'black_box_predictions.csv':'black_box', 'sequential_concept_predictions.csv':'seven_concept',
                      'joint_soft_predictions.csv':'joint_soft', 'joint_hard_predictions.csv':'joint_hard_ste'}.get(filename)
            if family:
                values = folds[family]
                for f in range(4):
                    for kind in ('logit','probability'):
                        if 'diagnosis_'+kind in values: row[f'fold_{f}_diagnosis_{kind}'] = float(values['diagnosis_'+kind][i,f])
                        if 'concept_'+kind in values:
                            row.update({f'fold_{f}_{c}_{kind}': float(values['concept_'+kind][i,f,j]) for j,c in enumerate(order)})
            if filename == 'oracle_predictions.csv': row['non_deployable'] = True
            elif filename != 'black_box_predictions.csv':
                cq = concepts[family+'_concept_ensemble'] if family in ('joint_soft','joint_hard_ste') else q
                row.update({c+'_probability': float(cq[i,j]) for j,c in enumerate(order)})
                if filename in ('sequential_soft_predictions.csv','sequential_hard_predictions.csv'):
                    inputs = q if filename.startswith('sequential_soft') else binary(q)
                    row.update({c+'_head_input': float(inputs[i,j]) for j,c in enumerate(order)})
            if filename != 'sequential_concept_predictions.csv':
                name = filename.removesuffix('_predictions.csv')
                if name == 'joint_hard': name = 'joint_hard_ste'
                row.update(melanoma_probability=float(diagnoses[name][i]), predicted_diagnosis=int(binary(diagnoses[name][i])))
            if set(row) != set(schema[filename]): raise ValueError('Prediction schema mismatch')
            tables[filename].append(row)
    return tables, diagnoses, concepts
