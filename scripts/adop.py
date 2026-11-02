"""Evidence-constrained ADOP on frozen predictions; standard library only.

Input JSON: {"train_counts": {"student_id": count}, "validation":
[{"user_id": "...", "label": 0, "score": 0.3}], "test": [same schema]}.
Scores MUST come from a predictor whose checkpoints and committee were selected
using only the students in manifest['selection']. Use `split` before training.
"""
import argparse
import json
import math
import random
from pathlib import Path

GRID = tuple((250 + 5 * j) / 1000 for j in range(91))
ETAS = (0, 0.01, 0.1, 1, 10)
ROLES = ('selection', 'fit', 'tune')


def group(count):
    return 'low' if count <= 10 else 'medium' if count <= 30 else 'high'


def split_students(user_ids, seed=2024):
    users = sorted({str(u) for u in user_ids})
    if len(users) < 4:
        raise ValueError('ADOP unavailable: at least four validation students required')
    random.Random(seed).shuffle(users)
    a, b = len(users) // 2, len(users) // 2 + len(users) // 4
    return dict(seed=seed, selection=users[:a], fit=users[a:b], tune=users[b:])


def validate_manifest(manifest, rows):
    roles = [{str(u) for u in manifest[r]} for r in ROLES]
    if any(not r for r in roles) or any(roles[i] & roles[j]
            for i in range(3) for j in range(i)):
        raise ValueError('Validation roles must be nonempty and student-disjoint')
    if set.union(*roles) != {str(r['user_id']) for r in rows}:
        raise ValueError('Manifest must cover exactly the validation students')


def check_rows(rows):
    if not rows:
        raise ValueError('Empty prediction partition')
    for r in rows:
        if r['label'] not in (0, 1) or not math.isfinite(r['score']) or not 0 <= r['score'] <= 1:
            raise ValueError('Expected binary labels and finite scores in [0,1]')


def accuracy(rows, threshold):
    return sum((r['score'] > threshold(r)) == r['label'] for r in rows) / len(rows)


def fit_adop(validation, counts, manifest):
    check_rows(validation)
    validate_manifest(manifest, validation)
    fit_users, tune_users = set(manifest['fit']), set(manifest['tune'])
    fit = [r for r in validation if str(r['user_id']) in fit_users]
    tune = [r for r in validation if str(r['user_id']) in tune_users]
    tau0 = max(GRID, key=lambda t: accuracy(fit, lambda r: t))
    candidates = [('global', None, {g: tau0 for g in ('low', 'medium', 'high')})]
    groups = {g: [r for r in fit if group(counts[str(r['user_id'])]) == g]
              for g in ('low', 'medium', 'high')}
    for eta in ETAS:
        thresholds = {}
        for g, rows in groups.items():
            n = len({str(r['user_id']) for r in rows})
            thresholds[g] = max(GRID, key=lambda t: (
                accuracy(rows, lambda r: t) - eta / (n + 1) * (t - tau0) ** 2,
                -abs(t - tau0), -t)) if rows else tau0
        candidates.append(('group', eta, thresholds))
    scored = []
    for kind, eta, thresholds in candidates:
        acc = accuracy(tune, lambda r: thresholds[group(counts[str(r['user_id'])])])
        scored.append((acc, kind == 'global', eta if eta is not None else -1,
                       kind, thresholds))
    acc, _, eta, kind, thresholds = max(scored, key=lambda c: c[:3])
    return dict(global_threshold=tau0, kind=kind, eta=None if kind == 'global' else eta,
                thresholds=thresholds, tuning_accuracy=acc,
                fitting_students={g: len({str(r['user_id']) for r in rows})
                                  for g, rows in groups.items()})


def auc(rows):
    ordered = sorted(rows, key=lambda r: r['score'])
    positives = sum(r['label'] for r in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        return None
    rank_sum, i = 0, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j]['score'] == ordered[i]['score']:
            j += 1
        rank_sum += sum(r['label'] for r in ordered[i:j]) * (i + 1 + j) / 2
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate(test, counts, rule):
    check_rows(test)
    def metrics(rows):
        return dict(n_responses=len(rows), n_students=len({str(r['user_id']) for r in rows}),
                    auc=auc(rows), rmse=math.sqrt(sum((r['score']-r['label'])**2 for r in rows)/len(rows)),
                    acc_05=accuracy(rows, lambda r: 0.5),
                    acc_global=accuracy(rows, lambda r: rule['global_threshold']),
                    acc_adop=accuracy(rows, lambda r: rule['thresholds'][group(counts[str(r['user_id'])])]))
    result = metrics(test)
    result['groups'] = {}
    for g in ('low', 'medium', 'high'):
        rows = [r for r in test if group(counts[str(r['user_id'])]) == g]
        result['groups'][g] = metrics(rows) if rows else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    split = sub.add_parser('split', help='Create student roles BEFORE model selection')
    split.add_argument('validation', help='Grouped val_set.json from preprocess.py')
    split.add_argument('output')
    split.add_argument('--seed', type=int, default=2024)
    run = sub.add_parser('evaluate')
    run.add_argument('predictions')
    run.add_argument('manifest')
    run.add_argument('output')
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Output already exists; choose a new path to preserve artifacts')
    if args.command == 'split':
        rows = json.loads(Path(args.validation).read_text(encoding='utf-8'))
        # Existing grouped validation uses 1-based IDs; training JSON uses 0-based IDs.
        result = split_students([str(int(r['user_id']) - 1) for r in rows], args.seed)
        result['id_convention'] = 'zero_based_training_ids'
    else:
        data = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
        manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        counts = {str(k): v for k, v in data['train_counts'].items()}
        for count in counts.values():
            if not isinstance(count, int) or count < 0:
                raise ValueError('Training counts must be nonnegative integers')
        for row in data['validation'] + data['test']:
            if str(row['user_id']) not in counts:
                raise ValueError('Missing training count; explicitly supply 0 for unseen students')
        rule = fit_adop(data['validation'], counts, manifest)
        result = dict(rule=rule, test=evaluate(data['test'], counts, rule), manifest=manifest)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()

