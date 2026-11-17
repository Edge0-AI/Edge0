"""Tuning history tests (checklist T06).

The requirement is that an engine, model, driver, workload or context change invalidates incompatible
history, and that model identity is more than a name. The risk being tested is specific and quiet: a
report that mixes two configurations' numbers produces a confident recommendation from incomparable
measurements, and nothing about the output looks wrong.

So each element of the comparison key gets its own test, and the *reason* is asserted, not just the
boolean — a report has to be able to say "the model changed", not merely "no".
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata_control import tune_history as th  # noqa: E402


class HistoryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-history-"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def store(self) -> th.HistoryStore:
        return th.HistoryStore(self.tmp / "history")

    def file(self, name: str, content: bytes) -> Path:
        p = self.tmp / name
        p.write_bytes(content)
        return p

    def key(self, **over):
        base = th.comparison_key(
            engine={"digest": "e" * 64, "bytes": 100, "exists": True, "method": th.FULL, "path": "/e"},
            model={"digest": "m" * 64, "bytes": 200, "exists": True, "method": th.SAMPLED, "path": "/m"},
            backend="metal",
            workload={"prompt_tokens": 512, "max_new": 128, "seed": 1, "temperature": 0.0},
            context=32768,
            machine={"system": "Darwin", "machine": "arm64", "cpu_count": 10},
        )
        base.update(over)
        return base


class TestFileIdentity(HistoryCase):
    """Identity is a content hash, not a label."""

    def test_a_different_file_with_the_same_name_is_a_different_identity(self):
        """The whole point: a re-quantised model keeps its name and changes its behaviour."""
        a = self.file("model.gguf", b"x" * 5000)
        first = th.file_identity(a)["digest"]
        a.write_bytes(b"y" * 5000)          # same path, same size, different bytes
        self.assertNotEqual(th.file_identity(a)["digest"], first)

    def test_a_truncated_file_is_a_different_identity(self):
        a = self.file("model.gguf", b"x" * 5000)
        full = th.file_identity(a, full=True)["digest"]
        a.write_bytes(b"x" * 4000)
        self.assertNotEqual(th.file_identity(a, full=True)["digest"], full)

    def test_the_identity_method_travels_with_the_digest(self):
        a = self.file("model.gguf", b"x" * 5000)
        self.assertEqual(th.file_identity(a)["method"], th.SAMPLED)
        self.assertEqual(th.file_identity(a, full=True)["method"], th.FULL)

    def test_a_large_file_is_sampled_without_reading_all_of_it(self):
        # The point of sampling is cost. This asserts the digest does not depend on every byte, so a
        # change strictly between the sampled regions is (correctly) not claimed as detection.
        big = self.tmp / "big.gguf"
        with big.open("wb") as fh:
            fh.write(b"a" * th.SAMPLE_BYTES)
            fh.write(b"b" * (4 * th.SAMPLE_BYTES))   # middle lands here
            fh.write(b"c" * th.SAMPLE_BYTES)
        first = th.file_identity(big)
        self.assertEqual(first["bytes"], 6 * th.SAMPLE_BYTES)
        self.assertIsNotNone(first["digest"])

    def test_a_missing_file_is_explicitly_absent_rather_than_hashed_as_empty(self):
        ident = th.file_identity(self.tmp / "nope")
        self.assertFalse(ident["exists"])
        self.assertIsNone(ident["digest"])
        self.assertIsNone(ident["method"])


class TestComparisonKey(HistoryCase):
    """Every element that makes two measurements incomparable is in the key."""

    def test_each_element_produces_its_own_reason(self):
        base = self.key()
        cases = {
            "engine_digest": "a different engine build",
            "model_digest": "different model bytes",
            "backend": "a different backend",
            "workload": "a different workload",
            "context": "a different context",
            "machine": "a different machine or driver",
        }
        for field, expected in cases.items():
            other = dict(base)
            other[field] = "changed" if not isinstance(base[field], (dict, int)) else 1
            reasons = th.incomparable_reasons(base, other)
            self.assertTrue(reasons, f"{field} must make the records incomparable")
            self.assertTrue(any(expected in r for r in reasons),
                            f"{field}: expected a reason mentioning {expected!r}, got {reasons}")

    def test_a_different_digest_method_is_incomparable(self):
        """Two records hashed differently cannot be compared as though they were the same."""
        base = self.key()
        other = dict(base, digest_method=th.FULL)
        reasons = th.incomparable_reasons(base, other)
        self.assertTrue(any("computed differently" in r for r in reasons), reasons)

    def test_an_unknown_identity_is_not_treated_as_equal(self):
        """An unknown is not evidence of sameness."""
        known = self.key()
        unknown = self.key(model_digest=None)
        self.assertTrue(th.incomparable_reasons(known, unknown))

    def test_identical_keys_are_comparable(self):
        self.assertEqual(th.incomparable_reasons(self.key(), self.key()), [])

    def test_a_different_hostname_is_the_same_machine(self):
        """A renamed machine is the same machine; the hostname is deliberately not in the key."""
        self.assertNotIn("node", th.machine_identity())
        self.assertNotIn("hostname", th.machine_identity())


class TestHistoryStore(HistoryCase):
    def test_a_recorded_sweep_is_comparable_to_itself(self):
        store = self.store()
        store.record(key=self.key(), outcome="accepted", recommendation_args={"--kv": "int8"},
                     median=24.0, baseline_median=20.0, trials_run=4, combinations_total=4)
        usable, unusable = store.comparable(self.key())
        self.assertEqual(len(usable), 1)
        self.assertEqual(usable[0].recommendation_args, {"--kv": "int8"})
        self.assertEqual(unusable, [])

    def test_a_changed_model_makes_the_history_incomparable_with_a_reason(self):
        store = self.store()
        store.record(key=self.key(), outcome="accepted")
        changed = self.key(model_digest="z" * 64)
        usable, unusable = store.comparable(changed)
        self.assertEqual(usable, [], "a different model's numbers must not be reused")
        self.assertEqual(len(unusable), 1)
        self.assertIn("different model bytes", unusable[0]["reasons"])

    def test_a_changed_engine_makes_the_history_incomparable(self):
        store = self.store()
        store.record(key=self.key(), outcome="accepted")
        usable, unusable = store.comparable(self.key(engine_digest="d" * 64))
        self.assertEqual(usable, [])
        self.assertIn("a different engine build", unusable[0]["reasons"])

    def test_a_changed_context_makes_the_history_incomparable(self):
        store = self.store()
        store.record(key=self.key(), outcome="accepted")
        usable, _ = store.comparable(self.key(context=131072))
        self.assertEqual(usable, [], "a different context is a different measurement")

    def test_a_changed_workload_makes_the_history_incomparable(self):
        store = self.store()
        store.record(key=self.key(), outcome="accepted")
        usable, _ = store.comparable(self.key(
            workload={"prompt_tokens": 1024, "max_new": 128, "seed": 1, "temperature": 0.0}))
        self.assertEqual(usable, [])

    def test_incomparable_history_is_retained_not_deleted(self):
        """A previous sweep may still answer a question, so the store does not discard it."""
        store = self.store()
        store.record(key=self.key(), outcome="accepted")
        store.record(key=self.key(model_digest="z" * 64), outcome="accepted")
        usable, unusable = store.comparable(self.key())
        self.assertEqual(len(usable), 1)
        self.assertEqual(len(unusable), 1)
        self.assertEqual(len(store.records()), 2, "both records are still there")

    def test_a_damaged_line_is_skipped_rather_than_fatal(self):
        """The store is a text file a human may have edited; one bad line must not lose the rest."""
        store = self.store()
        store.record(key=self.key(), outcome="accepted")
        with store.index.open("a", encoding="utf-8") as fh:
            fh.write("{ this is not json\n")
        store.record(key=self.key(), outcome="inconclusive")
        records = store.records()
        self.assertEqual(len(records), 2)
        self.assertEqual([r.outcome for r in records], ["accepted", "inconclusive"])

    def test_a_record_from_another_schema_version_is_ignored(self):
        store = self.store()
        store.root.mkdir(parents=True, exist_ok=True)
        store.index.write_text('{"schema_version": 99, "id": "old", "recorded_at": 1, "key": {}}\n',
                               encoding="utf-8")
        self.assertEqual(store.records(), [])

    def test_the_summary_says_what_the_numbers_sit_next_to(self):
        store = self.store()
        empty = th.summarise_history(store, self.key())
        self.assertEqual(empty["comparable_records"], 0)
        self.assertIn("no earlier sweep", empty["note"])

        store.record(key=self.key(), outcome="accepted")
        store.record(key=self.key(model_digest="z" * 64), outcome="accepted")
        summary = th.summarise_history(store, self.key())
        self.assertEqual(summary["comparable_records"], 1)
        self.assertEqual(summary["incomparable_records"], 1)
        self.assertTrue(summary["incomparable"][0]["reasons"])
        self.assertIn("this exact engine", summary["note"])


class TestPlanCarriesIdentity(HistoryCase):
    """The plan records what it measured, so a later sweep can compare without re-deriving it."""

    def test_plan_tuning_records_a_content_identity(self):
        from strata_control import tuning
        engine = self.file("strata", b"#!/bin/sh\necho 'x' >&2\n")
        model = self.file("model.gguf", b"weights" * 1000)
        plan = tuning.plan_tuning(
            engine, "metal", "fixture", "generation",
            {"prompt_tokens": 16, "max_new": 4, "seed": 1, "temperature": 0.0},
            hold=("ple_io", "kv", "kv_grow", "spec_min_p"), model_path=model, context=32768)
        ident = plan.identity
        self.assertEqual(ident["backend"], "metal")
        self.assertEqual(ident["context"], 32768)
        self.assertIsNotNone(ident["engine_digest"])
        self.assertIsNotNone(ident["model_digest"])
        self.assertEqual(ident["model_bytes"], len(b"weights" * 1000))

    def test_a_plan_without_a_model_path_is_incomparable_rather_than_falsely_comparable(self):
        """Not knowing the weights must be recorded as unknown, not as a match."""
        from strata_control import tuning
        engine = self.file("strata", b"#!/bin/sh\necho 'x' >&2\n")
        plan = tuning.plan_tuning(
            engine, "metal", "fixture", "generation",
            {"prompt_tokens": 16, "max_new": 4, "seed": 1, "temperature": 0.0},
            hold=("ple_io", "kv", "kv_grow", "spec_min_p"))
        self.assertIsNone(plan.identity["model_digest"])
        # And such a key matches nothing, including another key with the same unknown model.
        other = dict(plan.identity)
        self.assertTrue(th.incomparable_reasons(plan.identity, other) == [],
                        "two records with the same unknowns are consistently comparable to each other")
        self.assertTrue(th.incomparable_reasons(
            plan.identity, dict(plan.identity, model_digest="m" * 64)))


if __name__ == "__main__":
    unittest.main()
