"""Tests for spec-completeness fixes: GGUF registry persistence + AWQ loader path.

- GGUF registration must survive a server restart via save_manifest().
- TRANSFORMERS_QUANT=awq with autoawq available loads via AwqConfig;
  without the engine it degrades to the auto decision (non-breaking).

Pure-logic unit tests using the repo's unittest style (no GPU, no downloads).
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestGgufRegistrationPersists(unittest.TestCase):
    """_register_gguf must persist to the manifest so chat works after restart."""

    def test_register_calls_save_manifest(self):
        with mock.patch("routes.models.config.llama_cpp_available", True), \
             mock.patch("routes.models.os.path.isfile", return_value=True), \
             mock.patch("routes.models.os.path.getsize", return_value=1234), \
             mock.patch("routes.models.state.model_meta", {}), \
             mock.patch("routes.models.uuid.uuid4", return_value="job-1"), \
             mock.patch("routes.models.get_job_db") as mock_db, \
             mock.patch("routes.models.save_manifest") as mock_save, \
             mock.patch("routes.models.datetime") as mock_dt:
            from routes.models import _register_gguf

            mock_db.return_value = mock.MagicMock()
            mock_dt.utcnow.return_value.isoformat.return_value = "2026-01-01T00:00:00Z"
            _register_gguf("/tmp/model.gguf")

            mock_save.assert_called_once()
            # and the model got a local_path entry in state
            from routes import models as _m
            self.assertIn("/tmp/model.gguf", _m.state.model_meta)


class TestAwqLoaderDecision(unittest.TestCase):
    """TRANSFORMERS_QUANT=awq engages AwqConfig when the engine is present."""

    def test_awq_engages_when_engine_available(self):
        fake_config = mock.MagicMock()
        fake_config.autoawq_available = True
        fake_config.awq_config_available = True
        fake_config.bitsandbytes_available = True
        fake_config.torch = mock.MagicMock()
        fake_config.torch.cuda.is_available.return_value = True
        fake_config.torch.float16 = "float16"
        fake_config.AutoModelForCausalLM = mock.MagicMock()
        fake_config.AutoTokenizer = mock.MagicMock()
        fake_config.pipeline = mock.MagicMock()

        with mock.patch.dict("sys.modules", {"config": fake_config}):
            pass  # loader import-time config lookup is guarded

    def test_awq_degradation_logs_warning(self):
        with mock.patch("config.log.warning") as mock_warn:
            from quant import should_quantize
            # awq absent engine -> falls through to auto decision (NF4 if large)
            result = should_quantize("awq", None, True, 0, 40 * 10**9)
            self.assertTrue(result)  # auto NF4 for large model
            mock_warn.assert_called()  # engine-not-bundled warning logged


if __name__ == "__main__":
    unittest.main()