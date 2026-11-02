from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from serving.h3_stability import admission, check_sources, config, cv, plan, summarize
from serving.h3_performance import SCOPES, VARIANTS


class StabilityTests(unittest.TestCase):
    def setUp(self): self.c = config()

    def records(self):
        rows=[]
        for job in plan(self.c):
            factor={'native_kv':1,'paged_eager':1.1,'paged_graph':3}[job['variant']]
            traces=[]
            for _ in range(20):
                traces.append(dict(ttft_ms=2/factor,itl_ms=[1/factor]*3,tpot_ms=1/factor,
                    wall_ms=6/factor,model_execution_ms=5/factor,output_tokens=4))
            measurements={'model_execution':dict(traces=deepcopy(traces)),
                'replay_e2e':dict(traces=[dict(t,model_execution_ms=None) for t in traces])}
            rows.append(dict(job=job,measurements=measurements))
        return rows

    def test_six_complete_permutations_and_position_balance(self):
        jobs=plan(self.c); self.assertEqual(len(jobs),36);self.assertEqual(len({j['label'] for j in jobs}),36)
        for case in self.c['case_ids']:
            orders={tuple(j['variant'] for j in jobs if j['case_id']==case and j['round']==r) for r in range(1,7)}
            self.assertEqual(len(orders),6)
            self.assertEqual(set(Counter((j['variant'],j['position']) for j in jobs if j['case_id']==case).values()),{2})

    def test_scope_and_case_first_position_balanced(self):
        jobs=plan(self.c)
        for variant in VARIANTS:
            for case in self.c['case_ids']:
                first=[j['scopes'][0] for j in jobs if j['variant']==variant and j['case_id']==case]
                self.assertEqual(Counter(first),dict(model_execution=3,replay_e2e=3))
        self.assertEqual(Counter(jobs[i]['case_id'] for i in range(0,36,6)),dict(long_prefill=3,batch_short=3))

    def test_missing_formal_process_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing'): summarize(self.records()[:-1],self.c)

    def test_duplicate_process_cannot_replace_missing(self):
        rows=self.records();rows[-1]=deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'duplicate'): summarize(rows,self.c)

    def test_scopes_have_separate_denominators(self):
        summary=summarize(self.records(),self.c)
        rows=[r for r in summary['statistics'] if r['case_id']=='long_prefill' and r['variant']=='native_kv']
        by={r['measurement_scope']:r for r in rows}
        self.assertEqual(by['model_execution']['output_tokens_per_second'],800)
        self.assertAlmostEqual(by['replay_e2e']['output_tokens_per_second'],4000/6)
        self.assertEqual(len(summary['statistics']),12);self.assertEqual(len(summary['comparisons']),12)

    def test_paired_ratios_use_round_pairing(self):
        summary=summarize(self.records(),self.c)
        p=next(r for r in summary['comparisons'] if r['baseline']=='native_kv' and r['candidate']=='paged_graph')
        self.assertEqual(p['paired_ratios'],[3]*6);self.assertTrue(summary['timing_stable'])

    def test_between_process_instability_retained_not_deleted(self):
        rows=self.records()
        for t in rows[0]['measurements']['replay_e2e']['traces']: t['wall_ms']*=3
        summary=summarize(rows,self.c)
        self.assertEqual(summary['processes'],36);self.assertFalse(summary['timing_stable'])
        self.assertEqual(summary['timing_status'],'variability_remains')
        self.assertFalse(summary['steady_state_claimed'])

    def test_within_process_variability_in_both_time_bases(self):
        rows=self.records();rows[0]['measurements']['model_execution']['traces'][0]['wall_ms']*=8
        summary=summarize(rows,self.c)
        group=next(r for r in summary['statistics'] if r['case_id']=='long_prefill' and r['variant']=='native_kv' and r['measurement_scope']=='model_execution')
        self.assertGreater(group['maximum_within_process_cv'],.15);self.assertFalse(group['stable'])

    def test_nonfinite_nonpositive_or_empty_samples_rejected(self):
        for values in ([],[math.nan],[math.inf],[0],[-1]):
            with self.assertRaises(ValueError): cv(values)

    def test_failed_qualification_gate_rejected(self):
        with patch('serving.h3_stability.check_sources'), patch('serving.h3_stability.qualification_audit',return_value={'passed':False}), patch('serving.h3_stability.read',return_value={'passed':False}):
            with self.assertRaisesRegex(ValueError,'failed'): admission(Path('/unused'))

    def test_stale_qualification_even_if_passed_rejected(self):
        with patch('serving.h3_stability.check_sources'), patch('serving.h3_stability.qualification_audit',return_value={'passed':True,'hash':'new'}), patch('serving.h3_stability.read',return_value={'passed':True,'hash':'old'}):
            with self.assertRaisesRegex(ValueError,'stale'): admission(Path('/unused'))

    def test_mutated_current_or_snapshot_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'source_snapshot').mkdir();(path/'source_snapshot/x').write_text('frozen')
            with patch('serving.h3_stability.read',return_value={'source_sha256':{'x':'expected'}}), patch('serving.h3_stability.digest',return_value='mutated'):
                with self.assertRaisesRegex(ValueError,'source changed'): check_sources(path)


if __name__=='__main__': unittest.main()
