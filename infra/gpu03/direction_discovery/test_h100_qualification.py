"""Authored numerical-report fixtures; never execute generated code or a model."""
import copy
import unittest

from infra.gpu03.direction_discovery import h100_qualification as q


def plan_fixture():
    conditions = {c: {"layers": [] if c == "baseline" else [{"layer": 21}]} for c in q.ORDER}
    requests = [{"request_id": f"authored-{p}-{sample}-{c}", "record_id": f"record-{p}",
                 "problem_id": p, "problem_split": "configuration_validation", "scope": "primary",
                 "sample_index": sample, "seed": p * 4 + sample, "condition_id": c}
                for p in range(37) for sample in range(4) for c in conditions]
    return {"conditions": conditions, "selected_problem_ids": list(range(37)), "requests": requests}


def result_fixture(condition):
    energy = {}
    if condition != "baseline":
        one = {"activation_energy": 10.0, "removed_energy_fp32": 1.0, "actual_change_energy": 1.01,
               "remaining_subspace_energy_fp32": 1e-12, "remaining_subspace_energy_native": .001,
               "forward_calls": 1, "selected_tokens": 1}
        total = {k: v * 2 for k, v in one.items()}
        energy = {"21": {**total, "rank": 1, "projection_dtype": "float32",
                         "scopes": {"prefill": copy.deepcopy(one), "decode": copy.deepcopy(one)}}}
    return {"generated_token_ids": [12, 151645], "completion_token_ids": [12, 151645],
            "completion": "authored inert fixture", "fixed_completion_prefix_token_count": 0,
            "stop_reason": "eos", "elapsed_seconds": .5, "energy": energy}


def rows_fixture(plan):
    return [{**r, "result": result_fixture(r["condition_id"])} for r in q.qualification_requests(plan)]


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_fixture(); self.rows = rows_fixture(self.plan)

    def test_full_success_and_exact_six_scientific_cells(self):
        requests = q.qualification_requests(self.plan)
        self.assertEqual([r["condition_id"] for r in requests], list(q.ORDER))
        self.assertEqual(len({r["request_id"] for r in requests}), 6)
        self.assertEqual(len({r["seed"] for r in requests}), 1)
        self.assertFalse(set(r["request_id"] for r in requests) & set(r["request_id"] for r in self.plan["requests"]))
        proof = q.verify_rows(self.plan, self.rows)
        self.assertEqual(proof["generation_requests"], 6)
        self.assertEqual(proof["generated_tokens"], 12)
        self.assertTrue(proof["baseline_repeat_bitwise_equal"])

    def test_baseline_repeat_drift_fails(self):
        self.rows[-1]["result"]["completion"] = "different"
        with self.assertRaisesRegex(ValueError, "Repeated baseline"): q.verify_rows(self.plan, self.rows)

    def test_missing_reordered_or_seed_changed_fails(self):
        for mutate in (lambda r: r.pop(), lambda r: r.reverse(), lambda r: r[2].update(seed=72)):
            rows = copy.deepcopy(self.rows); mutate(rows)
            with self.assertRaises(ValueError): q.verify_rows(self.plan, rows)

    def test_short_length_stop_and_eos_continuation_fail(self):
        for key, value in (("stop_reason", "length"), ("fixed_completion_prefix_token_count", 1),
                           ("generated_token_ids", [151645, 151645]), ("elapsed_seconds", float("nan"))):
            result = result_fixture("baseline"); result[key] = value
            with self.assertRaises(ValueError): q.check_generation(result)

    def test_projection_rank_precision_position_and_nonfinite_fail(self):
        for key, value in (("rank", 2), ("projection_dtype", "bfloat16"), ("selected_tokens", 20),
                           ("remaining_subspace_energy_fp32", .01), ("removed_energy_fp32", float("nan"))):
            rows = copy.deepcopy(self.rows); rows[1]["result"]["energy"]["21"][key] = value
            with self.assertRaises(ValueError): q.verify_rows(self.plan, rows)

    def test_scope_aggregation_or_prefill_all_tokens_fails(self):
        for key, value in (("forward_calls", 2), ("activation_energy", 99), ("removed_energy_fp32", float("inf"))):
            rows = copy.deepcopy(self.rows); rows[1]["result"]["energy"]["21"]["scopes"]["prefill"][key] = value
            with self.assertRaises(ValueError): q.verify_rows(self.plan, rows)

    def test_no_projected_token_difference_requirement(self):
        self.assertEqual(self.rows[0]["result"]["generated_token_ids"], self.rows[1]["result"]["generated_token_ids"])
        self.assertTrue(q.verify_rows(self.plan, self.rows)["projection_energy_verified"])

    def test_reuse_exact_five_preserves_result_and_maps_original_cell(self):
        rows=q.reuse_first_five(self.plan,self.rows[:5])
        self.assertEqual([r['request_id'] for r in rows],[r['request_id'] for r in self.plan['requests'][:5]])
        self.assertEqual([r['result'] for r in rows],[r['result'] for r in self.rows[:5]])
        self.assertEqual([r['physical_request_id'] for r in rows],[r['request_id'] for r in self.rows[:5]])
        for values in (self.rows,self.rows[:4],list(reversed(self.rows[:5]))):
            with self.assertRaises(ValueError):q.reuse_first_five(self.plan,values)

    def test_reuse_changed_seed_or_projection_rejected(self):
        rows=copy.deepcopy(self.rows[:5]);rows[0]['seed']+=1
        with self.assertRaises(ValueError):q.reuse_first_five(self.plan,rows)
        rows=copy.deepcopy(self.rows[:5]);rows[1]['result']['energy']['21']['rank']=2
        with self.assertRaises(ValueError):q.reuse_first_five(self.plan,rows)


if __name__ == "__main__":
    unittest.main()
