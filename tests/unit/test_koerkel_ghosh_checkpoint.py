"""
Unit tests for Körkel-Ghosh run checkpointing.

WHY THIS EXISTS
    A 750x750 rung takes hours. The sidecar used to be written only after the
    final instance, so a machine reboot four instances into a six-instance rung
    discarded every one of them. That happened, and cost roughly five hours of
    compute.

    A checkpoint is written after each instance so a restart picks up where the
    crash left off. The delicate part is not saving -- it is refusing to resume
    from a checkpoint that belongs to a *different* experiment, because blending
    32-restart records into a 4-restart report would corrupt the results
    silently rather than loudly.
"""

import json

import pytest

from scripts.run_koerkel_ghosh import (
    CHECKPOINT_NAME,
    _run_signature,
    load_checkpoint,
    save_checkpoint,
)


class _Args:
    """Stand-in for the parsed CLI namespace."""

    def __init__(self, **kw):
        defaults = dict(size=250, classes=["a", "b", "c"], instances=2, restarts=32,
                        alpha=0.2, ip_time_limit=1800.0, no_exact_ip=False)
        defaults.update(kw)
        self.__dict__.update(defaults)


RECORDS = [{"name": "gs250a-1", "value": 1}, {"name": "gs250a-2", "value": 2}]


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / CHECKPOINT_NAME)


class TestSignature:
    def test_identical_configurations_share_a_signature(self):
        assert _run_signature(_Args()) == _run_signature(_Args())

    @pytest.mark.parametrize("field,value", [
        ("size", 500), ("instances", 1), ("restarts", 4),
        ("alpha", 0.5), ("ip_time_limit", 60.0), ("no_exact_ip", True),
        ("classes", ["a"]),
    ])
    def test_anything_that_changes_the_numbers_changes_the_signature(self, field, value):
        """
        Each of these alters what a record contains. None may be resumable
        across, or the report would mix two experiments.
        """
        assert _run_signature(_Args()) != _run_signature(_Args(**{field: value}))

    def test_the_exact_solve_flag_is_recorded_as_attempt_not_as_skip(self):
        assert _run_signature(_Args(no_exact_ip=False))["attempt_exact"] is True
        assert _run_signature(_Args(no_exact_ip=True))["attempt_exact"] is False


class TestRoundTrip:
    def test_records_survive_a_save_and_load(self, path):
        sig = _run_signature(_Args())
        save_checkpoint(path, sig, RECORDS)
        assert load_checkpoint(path, sig) == RECORDS

    def test_saving_is_atomic_leaving_no_temp_file_behind(self, path, tmp_path):
        save_checkpoint(path, _run_signature(_Args()), RECORDS)
        assert not (tmp_path / (CHECKPOINT_NAME + ".tmp")).exists()

    def test_a_later_save_supersedes_an_earlier_one(self, path):
        sig = _run_signature(_Args())
        save_checkpoint(path, sig, RECORDS[:1])
        save_checkpoint(path, sig, RECORDS)
        assert len(load_checkpoint(path, sig)) == 2


class TestRefusingToResume:
    def test_a_missing_checkpoint_yields_nothing(self, path):
        assert load_checkpoint(path, _run_signature(_Args())) == []

    def test_a_checkpoint_from_a_different_experiment_is_ignored(self, path):
        """The failure this guards: silently mixing two configurations."""
        save_checkpoint(path, _run_signature(_Args(restarts=4)), RECORDS)
        assert load_checkpoint(path, _run_signature(_Args(restarts=32))) == []

    def test_a_truncated_checkpoint_is_ignored_rather_than_crashing(self, path):
        """A reboot mid-write is exactly the scenario this feature exists for."""
        with open(path, "w") as fh:
            fh.write('{"signature": {"size": 250}, "reco')
        assert load_checkpoint(path, _run_signature(_Args())) == []

    def test_a_checkpoint_with_no_records_key_yields_nothing(self, path):
        with open(path, "w") as fh:
            json.dump({"signature": _run_signature(_Args())}, fh)
        assert load_checkpoint(path, _run_signature(_Args())) == []

    def test_an_unreadable_path_yields_nothing(self, tmp_path):
        """A directory where a file should be: report empty, do not raise."""
        d = tmp_path / CHECKPOINT_NAME
        d.mkdir()
        assert load_checkpoint(str(d), _run_signature(_Args())) == []


class TestResumeSkipsOnlyWhatIsDone:
    def test_a_resumed_run_recomputes_exactly_the_missing_instances(
            self, tmp_path, monkeypatch):
        """
        End to end: seed a checkpoint with two of the six instances, then run
        and assert only the other four are computed -- and that all six reach
        the report, in the canonical order.
        """
        import scripts.run_koerkel_ghosh as mod

        computed = []

        def fake_run_one(size, klass, symmetric, index, seeds, **kw):
            name = mod.instance_name(size, klass, symmetric, index)
            computed.append(name)
            return _stub_record(name, klass, symmetric, index, size)

        monkeypatch.setattr(mod, "run_one_instance", fake_run_one)

        done = [_stub_record(mod.instance_name(100, "a", True, 1), "a", True, 1, 100),
                _stub_record(mod.instance_name(100, "a", False, 1), "a", False, 1, 100)]
        args = _Args(size=100, instances=1, restarts=2, ip_time_limit=1.0)
        save_checkpoint(str(tmp_path / CHECKPOINT_NAME), _run_signature(args), done)

        mod.main(["--size", "100", "--instances", "1", "--restarts", "2",
                  "--ip-time-limit", "1.0", "--out-dir", str(tmp_path)])

        assert computed == ["gs100b-1", "ga100b-1", "gs100c-1", "ga100c-1"]
        payload = json.loads((tmp_path / "koerkel_ghosh_results.json").read_text())
        assert [r["name"] for r in payload["records"]] == [
            "gs100a-1", "ga100a-1", "gs100b-1", "ga100b-1", "gs100c-1", "ga100c-1"]

    def test_no_resume_recomputes_everything(self, tmp_path, monkeypatch):
        """COUNTERWEIGHT: the escape hatch must actually bypass the checkpoint."""
        import scripts.run_koerkel_ghosh as mod
        computed = []
        monkeypatch.setattr(mod, "run_one_instance",
                            lambda size, klass, sym, index, seeds, **kw: (
                                computed.append(mod.instance_name(size, klass, sym, index))
                                or _stub_record(mod.instance_name(size, klass, sym, index),
                                                klass, sym, index, size)))
        args = _Args(size=100, instances=1, restarts=2, ip_time_limit=1.0)
        save_checkpoint(str(tmp_path / CHECKPOINT_NAME), _run_signature(args),
                        [_stub_record("gs100a-1", "a", True, 1, 100)])

        mod.main(["--size", "100", "--instances", "1", "--restarts", "2",
                  "--ip-time-limit", "1.0", "--no-resume", "--out-dir", str(tmp_path)])
        assert len(computed) == 6

    def test_the_checkpoint_is_cleared_once_the_sidecar_is_written(
            self, tmp_path, monkeypatch):
        """
        Leaving it would make a later deliberate re-run silently return stale
        results instead of recomputing.
        """
        import scripts.run_koerkel_ghosh as mod
        monkeypatch.setattr(mod, "run_one_instance",
                            lambda size, klass, sym, index, seeds, **kw: _stub_record(
                                mod.instance_name(size, klass, sym, index),
                                klass, sym, index, size))
        mod.main(["--size", "100", "--instances", "1", "--restarts", "2",
                  "--ip-time-limit", "1.0", "--out-dir", str(tmp_path)])
        assert not (tmp_path / CHECKPOINT_NAME).exists()
        assert (tmp_path / "koerkel_ghosh_results.json").exists()


def _stub_record(name, klass, symmetric, index, size):
    """Minimal record with every field the report renderer reads."""
    def arm(cost, multistart=False):
        base = {"method": "m", "initial_cost": cost, "initial_gap": 0.0,
                "initial_open": 1, "final_open": 1, "iterations": 1, "moves": {},
                "construct_seconds": 0.0, "search_seconds": 0.0, "lp_seconds": 0.0}
        if multistart:
            base.update({"best_cost": cost, "best_gap": 0.0, "best_open": 1,
                         "best_found_at": 1, "seeds": [1, 2], "trajectory": [cost],
                         "trajectory_costs": [cost], "per_restart_gaps": [0.0],
                         "total_iterations": 1})
        else:
            base.update({"final_cost": cost, "final_gap": 0.0, "seed": None})
        return base

    return {
        "name": name, "size": size, "klass": klass, "symmetric": symmetric,
        "index": index, "best_cost_found": 100.0,
        "lp": {"bound": 100.0, "n_facilities": size, "n_fractional": 1,
               "n_integral_open": 1, "n_near_zero": 1, "sum_y": 1.0,
               "rounded_open_count": 1, "solve_seconds": 0.0},
        "reference": {"value": 100.0, "lp_bound": 100.0, "proven": False,
                      "ip_status": "not attempted", "ip_seconds": 0.0, "incumbent": None},
        "arms": {
            "lp_rounding_control": arm(110.0),
            "lp_rounding_plus_search": arm(101.0),
            "local_search_only": arm(102.0),
            "lp_biased_multistart": arm(100.0, True),
            "alpha_grasp_multistart": arm(103.0, True),
        },
    }


class TestRerenderFromSidecar:
    """
    A rung costs hours. When the prose is corrected -- as it was after the §3
    verdict defect -- the already-measured rungs must be re-rendered from their
    sidecars, never re-run and never hand-edited.
    """

    def _sidecar(self, tmp_path, records):
        payload = {
            "generated_at": "2026-09-13 00:00 UTC",
            "generated_by": "run_koerkel_ghosh.py::render_report",
            "git_commit": "abc1234", "environment": {"python": "3.14", "numpy": "1",
                                                      "scipy": "1"},
            "size": 250, "restarts": 2, "alpha": 0.2, "seeds": [1, 2],
            "records": records, "wall_seconds": 99.0,
        }
        path = tmp_path / "koerkel_ghosh_results.json"
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return str(path)

    def test_it_rewrites_the_markdown_without_running_anything(self, tmp_path,
                                                               monkeypatch):
        import scripts.run_koerkel_ghosh as mod
        monkeypatch.setattr(mod, "run_one_instance", lambda *a, **k: (
            _ for _ in ()).throw(AssertionError("--from-json must not run a benchmark")))
        path = self._sidecar(tmp_path, [_stub_record("gs250a-1", "a", True, 1, 250)])

        mod.main(["--from-json", path, "--out-dir", str(tmp_path)])
        md = (tmp_path / "koerkel_ghosh_results.md").read_text()
        assert "Körkel-Ghosh Benchmark" in md
        assert "250×250" in md

    def test_the_measurements_come_from_the_sidecar_not_from_a_rerun(self, tmp_path):
        """The metadata in the output must be the sidecar's, verbatim."""
        import scripts.run_koerkel_ghosh as mod
        path = self._sidecar(tmp_path, [_stub_record("gs250a-1", "a", True, 1, 250)])
        mod.main(["--from-json", path, "--out-dir", str(tmp_path)])
        md = (tmp_path / "koerkel_ghosh_results.md").read_text()
        assert "2026-09-13 00:00 UTC" in md
        assert "abc1234" in md

    def test_rerendering_twice_is_byte_identical(self, tmp_path):
        """COUNTERWEIGHT: re-rendering must be a pure function of the sidecar."""
        import scripts.run_koerkel_ghosh as mod
        path = self._sidecar(tmp_path, [_stub_record("gs250a-1", "a", True, 1, 250)])
        mod.main(["--from-json", path, "--out-dir", str(tmp_path)])
        first = (tmp_path / "koerkel_ghosh_results.md").read_text()
        mod.main(["--from-json", path, "--out-dir", str(tmp_path)])
        assert (tmp_path / "koerkel_ghosh_results.md").read_text() == first
