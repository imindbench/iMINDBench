"""Exercise interrupted result writes and direct-evaluator resume behavior."""

import json

import pytest
from omegaconf import OmegaConf

from imindbench.utils import result_io
from imindbench.utils.logging_utils import save_results, should_skip_existing_output


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
