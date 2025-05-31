import unittest
from unittest.mock import patch, MagicMock
import os
import requests # For requests.exceptions.ConnectionError

# Add project root to sys.path to allow direct import of termipals
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from termipals.cli.main import Termipals
from termipals.llm_provider import HuggingFaceProvider, OllamaProvider

# Default config values used in Termipals.__init__ for testing
DEFAULT_HF_MODEL_ID = 'distilgpt2'
DEFAULT_HF_REVISION = 'main'
DEFAULT_OLLAMA_HOST = 'http://localhost:11434'
DEFAULT_OLLAMA_MODEL = 'llama2:7b'
DEFAULT_CONNECTION_TIMEOUT = 5
DEFAULT_REQUEST_TIMEOUT = 60
DEFAULT_OLLAMA_TEMPERATURE = 0.7

class TestProviderSelection(unittest.TestCase):

    def create_provider_config(self, hf_model_id=DEFAULT_HF_MODEL_ID, hf_revision=DEFAULT_HF_REVISION,
                               ollama_host=DEFAULT_OLLAMA_HOST, ollama_model=DEFAULT_OLLAMA_MODEL,
                               conn_timeout=DEFAULT_CONNECTION_TIMEOUT, req_timeout=DEFAULT_REQUEST_TIMEOUT,
                               ollama_temp=DEFAULT_OLLAMA_TEMPERATURE):
        return {
            "huggingface_model_id": hf_model_id,
            "huggingface_model_revision": hf_revision,
            "ollama_host": ollama_host,
            "ollama_model_name": ollama_model,
            "connection_timeout": conn_timeout,
            "request_timeout": req_timeout,
            "ollama_options": {"temperature": ollama_temp}
        }

    @patch.dict(os.environ, {}, clear=True)
    @patch('termipals.llm_provider.OllamaProvider.is_available', return_value=False)
    def test_default_to_huggingface(self, mock_ollama_available):
        """Test that HuggingFaceProvider is default when no env vars and Ollama not available."""
        termipals_app = Termipals()
        self.assertIsInstance(termipals_app.llm_provider, HuggingFaceProvider)
        self.assertEqual(termipals_app.llm_provider.model_id, DEFAULT_HF_MODEL_ID)
        self.assertEqual(termipals_app.llm_provider.revision, DEFAULT_HF_REVISION)
        mock_ollama_available.assert_called_once() # Ensure auto-detection was attempted

    @patch.dict(os.environ, {"TERMI_PROVIDER": "local"}, clear=True)
    @patch('termipals.llm_provider.OllamaProvider.is_available') # Mock to prevent actual calls
    def test_explicit_huggingface_provider(self, mock_ollama_is_available):
        """Test explicit selection of HuggingFaceProvider via TERMI_PROVIDER=local."""
        termipals_app = Termipals()
        self.assertIsInstance(termipals_app.llm_provider, HuggingFaceProvider)
        self.assertEqual(termipals_app.llm_provider.model_id, DEFAULT_HF_MODEL_ID)
        mock_ollama_is_available.assert_not_called() # No auto-detection if explicit

    @patch.dict(os.environ, {"TERMI_PROVIDER": "huggingface", "HUGGINGFACE_MODEL_ID": "custom/model", "HUGGINGFACE_MODEL_REVISION": "custom_rev"}, clear=True)
    @patch('termipals.llm_provider.OllamaProvider.is_available')
    def test_explicit_huggingface_custom_model(self, mock_ollama_is_available):
        """Test explicit HuggingFace with custom model ID and revision."""
        termipals_app = Termipals()
        self.assertIsInstance(termipals_app.llm_provider, HuggingFaceProvider)
        self.assertEqual(termipals_app.llm_provider.model_id, "custom/model")
        self.assertEqual(termipals_app.llm_provider.revision, "custom_rev")
        mock_ollama_is_available.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    @patch('termipals.llm_provider.OllamaProvider.is_available', return_value=True)
    def test_autodetect_ollama(self, mock_ollama_available):
        """Test auto-detection of OllamaProvider when it's available."""
        termipals_app = Termipals()
        self.assertIsInstance(termipals_app.llm_provider, OllamaProvider)
        self.assertEqual(termipals_app.llm_provider.host, DEFAULT_OLLAMA_HOST)
        self.assertEqual(termipals_app.llm_provider.model_name, DEFAULT_OLLAMA_MODEL)
        mock_ollama_available.assert_called_once()

    @patch.dict(os.environ, {"TERMI_PROVIDER": "ollama", "OLLAMA_HOST": "http://customhost:1234", "OLLAMA_MODEL": "custom_ollama_model"}, clear=True)
    @patch('termipals.llm_provider.OllamaProvider.is_available', return_value=True)
    def test_explicit_ollama_provider(self, mock_ollama_available):
        """Test explicit selection of OllamaProvider with custom settings."""
        termipals_app = Termipals()
        self.assertIsInstance(termipals_app.llm_provider, OllamaProvider)
        self.assertEqual(termipals_app.llm_provider.host, "http://customhost:1234")
        self.assertEqual(termipals_app.llm_provider.model_name, "custom_ollama_model")
        mock_ollama_available.assert_called_once() # is_available is checked even if explicit

    @patch.dict(os.environ, {"TERMI_PROVIDER": "ollama"}, clear=True)
    @patch('termipals.llm_provider.OllamaProvider.is_available', return_value=False)
    @patch('termipals.llm_provider.HuggingFaceProvider.is_available', return_value=True) # Assuming HF is always available for fallback
    def test_explicit_ollama_unavailable_fallback(self, mock_hf_available, mock_ollama_available):
        """Test fallback to HuggingFace when explicit Ollama is unavailable."""
        termipals_app = Termipals()
        self.assertIsInstance(termipals_app.llm_provider, HuggingFaceProvider)
        self.assertEqual(termipals_app.llm_provider.model_id, DEFAULT_HF_MODEL_ID)
        mock_ollama_available.assert_called_once()

    @patch.dict(os.environ, {"TERMI_PROVIDER": "unknown_provider"}, clear=True)
    @patch('termipals.llm_provider.OllamaProvider.is_available', return_value=False) # Ensure Ollama auto-detect also fails
    @patch('termipals.llm_provider.HuggingFaceProvider.is_available', return_value=True)
    def test_unrecognized_provider_fallback(self, mock_hf_available, mock_ollama_available):
        """Test fallback to HuggingFace for an unrecognized TERMI_PROVIDER."""
        termipals_app = Termipals()
        self.assertIsInstance(termipals_app.llm_provider, HuggingFaceProvider)
        self.assertEqual(termipals_app.llm_provider.model_id, DEFAULT_HF_MODEL_ID)
        # No OllamaProvider.is_available call if provider_name is unrecognized and not empty
        # The logic is: if provider_name -> try that provider. If fails -> fallback.
        # If not provider_name -> try auto-detect Ollama. If fails -> fallback.
        # In this case, provider_name is "unknown_provider", so OllamaProvider.is_available shouldn't be called for auto-detection.
        # However, the current implementation in Termipals.__init__ might call it if the unknown provider init fails and self.llm_provider is None.
        # Let's verify the calls based on current Termipals.__init__ logic:
        # 1. Tries "unknown_provider" - fails (no such class) -> self.llm_provider is None
        # 2. Then, because provider_name is not empty, it SKIPS auto-detection block.
        # 3. Goes to final fallback.
        # So, mock_ollama_available should NOT be called.
        mock_ollama_available.assert_not_called()


class TestOllamaProvider(unittest.TestCase):

    def setUp(self):
        self.sample_config = {
            "ollama_host": "http://mockhost:11434",
            "ollama_model_name": "mockmodel:latest",
            "connection_timeout": 1,
            "request_timeout": 5,
            "ollama_options": {"temperature": 0.5}
        }
        # Ensure that HuggingFaceProvider doesn't try to download models during these tests
        # by mocking its is_available to True and setup_model to do nothing.
        # This is relevant if OllamaProvider tests inadvertently cause HF fallback logging.
        patcher = patch('termipals.llm_provider.HuggingFaceProvider.is_available', return_value=True)
        self.addCleanup(patcher.stop)
        patcher.start()

        patcher_setup = patch('termipals.llm_provider.HuggingFaceProvider.setup_model', return_value=None)
        self.addCleanup(patcher_setup.stop)
        patcher_setup.start()


    @patch('requests.get')
    def test_ollama_is_available_success(self, mock_get):
        """Test OllamaProvider.is_available success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        provider = OllamaProvider(config=self.sample_config)
        self.assertTrue(provider.is_available())
        mock_get.assert_called_once_with(f"{self.sample_config['ollama_host']}/api/tags", timeout=self.sample_config['connection_timeout'])

    @patch('requests.get', side_effect=requests.exceptions.ConnectionError("Test connection error"))
    def test_ollama_is_available_connection_error(self, mock_get):
        """Test OllamaProvider.is_available handles ConnectionError."""
        provider = OllamaProvider(config=self.sample_config)
        self.assertFalse(provider.is_available())
        mock_get.assert_called_once_with(f"{self.sample_config['ollama_host']}/api/tags", timeout=self.sample_config['connection_timeout'])

    @patch('requests.post')
    @patch('termipals.llm_provider.OllamaProvider.is_available', return_value=True) # Assume available for generate_art tests
    def test_ollama_generate_art_success(self, mock_is_available, mock_post):
        """Test OllamaProvider.generate_art success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "  mocked art  "} # With spaces for strip test
        mock_post.return_value = mock_response

        provider = OllamaProvider(config=self.sample_config)
        prompt_messages = [{"role": "user", "content": "draw a cat"}]
        art = provider.generate_art(prompt_messages)

        self.assertEqual(art, "mocked art")
        expected_payload = {
            "model": self.sample_config['ollama_model_name'],
            "prompt": "draw a cat", # From _build_ollama_prompt
            "stream": False,
            "options": self.sample_config['ollama_options']
        }
        mock_post.assert_called_once_with(
            f"{self.sample_config['ollama_host']}/api/generate",
            json=expected_payload,
            timeout=self.sample_config['request_timeout']
        )

    @patch('requests.post', side_effect=requests.exceptions.Timeout("Test timeout"))
    @patch('termipals.llm_provider.OllamaProvider.is_available', return_value=True)
    def test_ollama_generate_art_timeout(self, mock_is_available, mock_post):
        """Test OllamaProvider.generate_art handles Timeout."""
        provider = OllamaProvider(config=self.sample_config)
        prompt_messages = [{"role": "user", "content": "draw a dog"}]
        art = provider.generate_art(prompt_messages)
        self.assertTrue(art.startswith("Error: Ollama request timed out."))

    @patch('requests.post')
    @patch('termipals.llm_provider.OllamaProvider.is_available', return_value=True)
    def test_ollama_generate_art_api_error(self, mock_is_available, mock_post):
        """Test OllamaProvider.generate_art handles API error response."""
        mock_response = MagicMock()
        mock_response.status_code = 404 # Not Found
        mock_response.json.return_value = {"error": "model 'testmodel' not found"}
        mock_response.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError(response=mock_response))
        mock_post.return_value = mock_response

        provider = OllamaProvider(config=self.sample_config)
        prompt_messages = [{"role": "user", "content": "draw something"}]
        art = provider.generate_art(prompt_messages)
        self.assertTrue(art.startswith("Error: Ollama API request failed with status 404"))


if __name__ == '__main__':
    unittest.main()
