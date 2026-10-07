"""Failure propagation without rebuilding artifacts inside tests."""

import subprocess
import sys

import run_all


def test_failed_step_reports_output_and_uses_current_python(monkeypatch, capsys):
    def fail(command, **kwargs):
        assert command == [sys.executable, "builder.py"]
        assert kwargs["cwd"] == run_all.ROOT
        return subprocess.CompletedProcess(command, 7, "builder output", "broken input")

    monkeypatch.setattr(run_all.subprocess, "run", fail)
    assert not run_all.run_step("example", ("builder.py",))
    output = capsys.readouterr().out
    assert "FAIL example: exit status 7" in output
    assert "broken input" in output and "builder output" in output


def test_launch_failure_is_reported(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError("cannot launch interpreter")

    monkeypatch.setattr(run_all.subprocess, "run", fail)
    assert not run_all.run_step("example", ("builder.py",))
    assert "FAIL example: cannot launch interpreter" in capsys.readouterr().out


def test_runner_keeps_order_and_failure_with_optional_tests(monkeypatch):
    calls = []

    def run(name, arguments):
        calls.append((name, arguments))
        return name != "README blocks"

    monkeypatch.setattr(run_all, "run_step", run)
    assert run_all.main(["--tests"]) == 1
    assert calls == [*run_all.STEPS, ("pytest", ("-m", "pytest", "tests", "-q"))]
    calls.clear()
    monkeypatch.setattr(run_all, "run_step", lambda name, args: calls.append((name, args)) or True)
    assert run_all.main([]) == 0
    assert calls == list(run_all.STEPS)
