import copy
import unittest
from adop import GRID, split_students, fit_adop, evaluate, validate_manifest


class ADOPTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(selection=['0', '1'], fit=['2', '3'], tune=['4', '5'])
        self.counts = {str(i): 5 if i % 2 == 0 else 40 for i in range(8)}
        self.rows = [dict(user_id=str(i), label=y, score=s) for i in range(6)
                     for y, s in [(0, .3), (1, .4)]]

    def test_split_is_reproducible_and_disjoint(self):
        m = split_students(range(12))
        self.assertEqual(m, split_students(reversed(range(12))))
        validate_manifest(m, [dict(user_id=str(i)) for i in range(12)])
        self.assertEqual([len(m[r]) for r in ('selection', 'fit', 'tune')], [6, 3, 3])

    def test_global_tie_and_empty_group_fallback(self):
        rule = fit_adop(self.rows, self.counts, self.manifest)
        self.assertEqual(rule['global_threshold'], .3)
        self.assertEqual(rule['kind'], 'global')
        self.assertEqual(rule['thresholds']['medium'], .3)
        self.assertEqual(len(GRID), 91)

    def test_selection_labels_do_not_affect_thresholds(self):
        changed = copy.deepcopy(self.rows)
        for r in changed:
            if r['user_id'] in self.manifest['selection']:
                r['label'] = 1 - r['label']
        self.assertEqual(fit_adop(self.rows, self.counts, self.manifest),
                         fit_adop(changed, self.counts, self.manifest))

    def test_metrics_preserve_scores(self):
        rule = fit_adop(self.rows, self.counts, self.manifest)
        before = copy.deepcopy(self.rows)
        report = evaluate(self.rows, self.counts, rule)
        self.assertEqual(report['auc'], 1)
        self.assertEqual(report['acc_adop'], 1)
        self.assertEqual(report['acc_05'], .5)
        self.assertEqual(before, self.rows)

    def test_invalid_manifest(self):
        self.manifest['fit'].append('0')
        with self.assertRaises(ValueError):
            fit_adop(self.rows, self.counts, self.manifest)

    def test_group_search_can_beat_global_on_tuning_students(self):
        rows = [dict(user_id=str(i), label=y, score=s)
                for i in range(6)
                for y, s in ([(0, .2), (1, .3)] if i % 2 == 0
                             else [(0, .6), (1, .7)])]
        rule = fit_adop(rows, self.counts, self.manifest)
        self.assertEqual(rule['kind'], 'group')
        self.assertLess(rule['thresholds']['low'], rule['thresholds']['high'])
        self.assertEqual(rule['tuning_accuracy'], 1)

    def test_unavailable_split(self):
        with self.assertRaises(ValueError):
            split_students(['0', '1', '2'])


if __name__ == '__main__':
    unittest.main()

