import subprocess
import sys
import textwrap

from hydra import compose, initialize

from nucleus.training.run import make_run_directories

# printed after the redirect, through Python and straight to the file descriptors as C code would
REDIRECTED_PROGRAM = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    from nucleus.training.output import redirect_output

    print("before")
    redirect_output(Path(sys.argv[1]))
    print("python stdout")
    print("python stderr", file=sys.stderr)
    os.write(1, b"descriptor stdout\\n")
    os.write(2, b"descriptor stderr\\n")
    """
)


def test_redirect_output_sends_stdout_and_stderr_to_the_file(tmp_path):
    path = tmp_path / "output.log"
    result = subprocess.run(
        [sys.executable, "-c", REDIRECTED_PROGRAM, str(path)], capture_output=True, text=True, check=True
    )
    assert result.stdout == "before\n"
    assert result.stderr == ""
    lines = path.read_text().splitlines()
    assert sorted(lines) == sorted(["python stdout", "python stderr", "descriptor stdout", "descriptor stderr"])


def test_redirect_output_appends_to_an_existing_file(tmp_path):
    path = tmp_path / "output.log"
    path.write_text("earlier run\n")
    subprocess.run([sys.executable, "-c", REDIRECTED_PROGRAM, str(path)], capture_output=True, check=True)
    assert path.read_text().startswith("earlier run\n")


def test_run_directories_share_a_name_and_each_hold_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    overrides = [f"log_dir={tmp_path / 'logs'}", f"checkpoint_dir={tmp_path / 'checkpoints'}", "split_dir=x"]
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="train_nucleus2_moe", overrides=overrides)
    log_directory, checkpoint_directory = make_run_directories(cfg)
    assert log_directory.parent == tmp_path / "logs"
    assert checkpoint_directory.parent == tmp_path / "checkpoints"
    assert log_directory.name == checkpoint_directory.name
    assert log_directory.name.startswith("nucleus2_moe_") and log_directory.name.endswith("_1234")
    assert (log_directory / "config.yaml").is_file()
    assert (checkpoint_directory / "config.yaml").is_file()
