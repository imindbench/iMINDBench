"""Exercise interrupted result writes and direct-evaluator resume behavior."""

import json
import stat

import pytest
from omegaconf import OmegaConf

from imindbench.utils import result_io
from imindbench.utils.logging_utils import save_results, should_skip_existing_output


def test_result_writes_preserve_normal_file_permissions(tmp_path):
    ordinary = tmp_path / "ordinary.json"
    ordinary.write_text("{}")
    destination = tmp_path / "population.json"
    save_results({"new": True}, destination)
    assert stat.S_IMODE(destination.stat().st_mode) == stat.S_IMODE(
        ordinary.stat().st_mode
    )
    # Overwriting a shared result must retain its explicitly configured mode.
    destination.chmod(0o640)
    save_results({"replacement": True}, destination)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_write_preserves_previous_result(tmp_path, monkeypatch, existing):
    destination = tmp_path / "population.json"
    previous = '{"previous": true}'
    if existing:
        destination.write_text(previous)

    def interrupted_dump(results, handle, **kwargs):
        handle.write('{"partial":')
        handle.flush()
        if existing:
            assert destination.read_text() == previous
        else:
            assert not destination.exists()
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(result_io.json, "dump", interrupted_dump)
        with pytest.raises(KeyboardInterrupt):
            save_results({"new": True}, destination)
    assert list(tmp_path.glob("*.tmp")) == []
    assert destination.exists() is existing
    if existing:
        assert destination.read_text() == previous
    save_results({"new": True}, destination)
    assert json.loads(destination.read_text()) == {"new": True}
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("contents", [b"", b'{"partial":', b"\xff", b"null", b"[]"])
def test_direct_evaluator_retries_invalid_results(tmp_path, contents):
    destination = tmp_path / "population.json"
    cfg = OmegaConf.create({"runtime": {"overwrite": False}})
    assert not should_skip_existing_output(cfg, str(destination))
    destination.write_bytes(contents)
    assert not should_skip_existing_output(cfg, str(destination))
    save_results({"evaluation_results": {}}, destination)
    assert should_skip_existing_output(cfg, str(destination))
    cfg.runtime.overwrite = True
    assert not should_skip_existing_output(cfg, str(destination))
