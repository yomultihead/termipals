import abc
import logging
from pathlib import Path
import re

# Import Hugging Face specific libraries
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    # This will be handled by is_available() in HuggingFaceProvider
    # and prevent its use if dependencies are missing.
    pass

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import importlib.util # Moved importlib.util to the top
import requests # Added for OllamaProvider
import json # Added for OllamaProvider


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: dict):
        """
        Initializes the LLM provider with configuration parameters.

        Args:
            config: A dictionary containing configuration parameters.
        """
        self.config = config

    @abc.abstractmethod
    def generate_art(self, prompt_messages: list) -> str:
        """
        Generates art based on the given prompt messages.

        Args:
            prompt_messages: A list of messages representing the prompt.

        Returns:
            A string representing the generated art.
        """
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """
        Checks if the LLM provider is available.

        Returns:
            True if the provider is available, False otherwise.
        """
        pass


class HuggingFaceProvider(LLMProvider):
    """LLM provider for Hugging Face models."""

    def __init__(self, config: dict):
        """
        Initializes the HuggingFaceProvider.

        Args:
            config: A dictionary containing configuration parameters.
                    Expected keys: "huggingface_model_id", "huggingface_model_revision".
        """
        super().__init__(config)
        self.model_id = self.config.get("huggingface_model_id", "distilgpt2")
        self.revision = self.config.get("huggingface_model_revision", "main")
        self.pipe = None
        # Sanitize model_id for directory creation
        safe_model_id_path = self.model_id.replace("/", "_")
        self.model_dir = Path.home() / '.cache' / 'termipals' / 'models' / safe_model_id_path
        self.logger = logging.getLogger(__name__)
        self.tokenizer = None
        self.model = None

    def setup_model(self):
        """
        Downloads (if necessary) and loads the Hugging Face model and tokenizer.
        Initializes the generation pipeline.
        """
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created model directory: {self.model_dir}")

        try:
            self.logger.info(f"Loading tokenizer for {self.model_id} (revision: {self.revision}) from {self.model_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=True
            )
            self.logger.info("Tokenizer loaded successfully from local cache.")
        except OSError:
            self.logger.info(f"Tokenizer not found locally. Downloading and caching to {self.model_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True
            )
            self.tokenizer.save_pretrained(str(self.model_dir))
            self.logger.info("Tokenizer downloaded and cached.")

        try:
            self.logger.info(f"Loading model {self.model_id} (revision: {self.revision}) from {self.model_dir}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=True
            )
            self.logger.info("Model loaded successfully from local cache.")
        except OSError:
            self.logger.info(f"Model not found locally. Downloading and caching to {self.model_dir}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True
            )
            self.model.save_pretrained(str(self.model_dir))
            self.logger.info("Model downloaded and cached.")

        # Ensure model is on the correct device (e.g., GPU if available)
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            self.logger.info("CUDA is available. Using GPU.")
            if self.model.device.type != "cuda": # Only move if not already on CUDA
                 self.model.to("cuda")
        else:
            self.logger.info("CUDA not available. Using CPU.")


        self.logger.info("Initializing text generation pipeline...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # device=device # device argument for pipeline
        )
        self.logger.info("Text generation pipeline initialized.")


    def _clean_art(self, generated_text: str) -> str:
        """Cleans the generated text to extract ASCII art."""
        # Remove chatml markers if present
        cleaned_text = generated_text.replace("<|assistant|>", "").replace("<|user|>", "").replace("<|system|>", "")

        # Find the start of the ASCII art
        start_marker = "```ascii"
        start_index = cleaned_text.find(start_marker)
        if start_index != -1:
            cleaned_text = cleaned_text[start_index + len(start_marker):]
            # Remove the marker itself and any preceding newlines
            cleaned_text = cleaned_text.lstrip('\n')

        # Find the end of the ASCII art
        end_marker = "```"
        end_index = cleaned_text.rfind(end_marker)
        if end_index != -1:
            cleaned_text = cleaned_text[:end_index]

        # Also remove common model apologies or refusals if they appear before art
        apology_patterns = [
            r"I apologize, but I cannot fulfill this request.*?\n",
            r"I'm sorry, I can't create ASCII art.*?\n",
            r"I am unable to generate images or ASCII art.*?\n",
            r"As a large language model, I don't have the ability to create images.*?\n"
        ]
        for pattern in apology_patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE | re.DOTALL)

        return cleaned_text.strip()

    def generate_art(self, prompt_messages: list) -> str:
        """
        Generates art using the Hugging Face model.

        Args:
            prompt_messages: A list of messages representing the prompt.
                           Example: [{"role": "user", "content": "Draw a cat."}]

        Returns:
            A string representing the generated art.
        """
        if self.pipe is None:
            try:
                self.setup_model()
            except Exception as e:
                self.logger.error(f"Error setting up Hugging Face model: {e}")
                return f"Error: Could not load model {self.model_id}. Check logs."

        if not self.pipe:
             return "Error: Text generation pipeline not available."

        generation_args = {
            "max_new_tokens": 500,
            "temperature": 0.7, # Adjusted for creativity
            "top_k": 50,
            "top_p": 0.95,
            "do_sample": True, # Important for creative tasks
            "pad_token_id": self.tokenizer.eos_token_id if self.tokenizer else 50256, # Use EOS token for padding
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else 50256,
        }

        self.logger.info(f"Generating art with prompt: {prompt_messages}")

        # The pipeline expects a list of conversations or a single string.
        # If prompt_messages is a list of dicts, format it.
        # For many models, a simple concatenation of "role: content" is not ideal.
        # It's better to use the tokenizer's apply_chat_template if available,
        # or format as a single string prompt.
        # For now, assuming prompt_messages is a list of dicts like:
        # [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "response (optional)"}]
        # We will take the last user message as the primary prompt for basic text-generation.
        # More complex chat handling would require apply_chat_template.

        input_text = ""
        if isinstance(prompt_messages, list) and prompt_messages:
            # Attempt to use apply_chat_template if available and it's a newer feature
            if hasattr(self.tokenizer, 'apply_chat_template') and callable(self.tokenizer.apply_chat_template):
                try:
                    input_text = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    self.logger.info(f"Using chat template. Input text: {input_text}")
                except Exception as e:
                    self.logger.warning(f"Failed to apply chat template: {e}. Falling back to simpler prompt string.")
                    # Fallback for models that don't support it well or if there's an error
                    last_user_message = next((msg['content'] for msg in reversed(prompt_messages) if msg['role'] == 'user'), None)
                    if last_user_message:
                        input_text = last_user_message
                    else: # if no user message, concatenate all
                        input_text = " ".join([msg['content'] for msg in prompt_messages])

            else: # Older tokenizer or simple case: concatenate content from user messages
                last_user_message = next((msg['content'] for msg in reversed(prompt_messages) if msg['role'] == 'user'), None)
                if last_user_message:
                    input_text = last_user_message
                else: # if no user message, concatenate all
                    input_text = " ".join([msg['content'] for msg in prompt_messages])
        elif isinstance(prompt_messages, str): # If it's already a string
            input_text = prompt_messages
        else:
            self.logger.error("Invalid prompt_messages format. Expected list of dicts or string.")
            return "Error: Invalid prompt format."

        if not input_text:
            self.logger.warning("Prompt is empty after processing.")
            return "Error: Prompt is empty."

        self.logger.info(f"Processed input text for pipeline: {input_text[:200]}...") # Log snippet

        try:
            outputs = self.pipe(input_text, **generation_args)
            generated_text = outputs[0]['generated_text']

            # The actual prompt might be included in the generated_text by some models
            # We should try to remove it if apply_chat_template was not used or was not effective
            if hasattr(self.tokenizer, 'apply_chat_template') and input_text in generated_text and not input_text.endswith(generated_text):
                 # Check if input_text is a prefix of generated_text and not the whole string
                if generated_text.startswith(input_text):
                    generated_text = generated_text[len(input_text):]

            self.logger.info(f"Raw generated text: {generated_text[:300]}...") # Log snippet
            art = self._clean_art(generated_text)
            return art
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"Error: Failed to generate art. {e}"


    def is_available(self) -> bool:
        """
        Checks if the Hugging Face provider is available (e.g., dependencies installed).
        """
        try:
            importlib.util.find_spec("torch")
            importlib.util.find_spec("transformers")
            return True
        except ImportError:
            self.logger.warning("HuggingFaceProvider: Missing torch or transformers library.")
            return False


class OllamaProvider(LLMProvider):
    """LLM provider for Ollama models."""

    def __init__(self, config: dict):
        """
        Initializes the OllamaProvider.

        Args:
            config: A dictionary containing configuration parameters.
                    Expected keys: "ollama_host", "ollama_model_name".
        """
        super().__init__(config)
        self.host = self.config.get("ollama_host", "http://localhost:11434").rstrip('/') # Ensure no trailing slash
        self.model_name = self.config.get("ollama_model_name", "llama2:7b")
        self.api_url = f"{self.host}/api/generate"
        self.tags_url = f"{self.host}/api/tags"
        self.logger = logging.getLogger(__name__)

    def _build_ollama_prompt(self, prompt_messages: list) -> str:
        """
        Builds a single prompt string from a list of message dictionaries
        for Ollama.
        """
        full_prompt_parts = []
        system_prompt = ""
        user_prompts = []

        for message in prompt_messages:
            role = message.get("role")
            content = message.get("content")
            if not content:
                continue
            if role == "system":
                system_prompt = content # Ollama typically takes system prompt separately or at the start
            elif role == "user":
                user_prompts.append(content)
            # Assistant messages could be used for few-shot prompting if needed
            # else:
            #    full_prompt_parts.append(f"{role}: {content}")

        if system_prompt: # Prepend system prompt if it exists
            full_prompt_parts.append(system_prompt)

        full_prompt_parts.extend(user_prompts) # Add all user prompts

        return "\n".join(full_prompt_parts)


    def generate_art(self, prompt_messages: list) -> str:
        """
        Generates art using an Ollama model.

        Args:
            prompt_messages: A list of messages representing the prompt.

        Returns:
            A string representing the generated art or an error message.
        """
        if not self.is_available(): # Check availability before trying to generate
            return "Error: Ollama provider is not available. Please check connection and configuration."

        combined_prompt = self._build_ollama_prompt(prompt_messages)
        if not combined_prompt:
            return "Error: Prompt is empty after processing messages."

        payload = {
            "model": self.model_name,
            "prompt": combined_prompt,
            "stream": False, # Keep it simple for now
            "options": self.config.get("ollama_options", {"temperature": 0.7}) # Get options from config or default
        }

        self.logger.info(f"Sending request to Ollama API: {self.api_url} with model {self.model_name}")
        self.logger.debug(f"Ollama payload: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(self.api_url, json=payload, timeout=self.config.get("request_timeout", 60))
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            response_data = response.json()
            self.logger.debug(f"Ollama raw response: {response_data}")

            generated_art = response_data.get("response")
            if generated_art:
                # As per subtask, direct response for now. Cleaning can be added later.
                # art = self._clean_art(generated_art) # If a shared/simple cleaner is available
                return generated_art.strip()
            else:
                error_message = response_data.get("error", "Unknown error from Ollama")
                self.logger.error(f"Ollama API error: {error_message}")
                # Check if the model is available if a "model not found" type error occurs
                if "model not found" in error_message.lower() or (response_data.get("status_code") == 404 and "model" in error_message.lower()):
                     self.logger.error(f"Model '{self.model_name}' might not be available on Ollama host '{self.host}'. Please pull it with 'ollama pull {self.model_name}'.")
                     return f"Error: Model '{self.model_name}' not found on Ollama host. Pull the model."

                return f"Error: Ollama API did not return a response. Details: {error_message}"

        except requests.exceptions.Timeout:
            self.logger.error(f"Ollama API request timed out to {self.api_url}.")
            return "Error: Ollama request timed out."
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Ollama API connection error to {self.api_url}. Is Ollama running?")
            return "Error: Could not connect to Ollama server."
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"Ollama API HTTP error: {e.response.status_code} - {e.response.text}")
            return f"Error: Ollama API request failed with status {e.response.status_code}."
        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode JSON response from Ollama: {response.text}")
            return "Error: Invalid JSON response from Ollama."
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while communicating with Ollama: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"Error: An unexpected error occurred with Ollama. {e}"


    def is_available(self) -> bool:
        """
        Checks if the Ollama provider is available by trying to connect to its API.
        It checks the /api/tags endpoint.
        """
        try:
            response = requests.get(self.tags_url, timeout=self.config.get("connection_timeout", 5)) # 5s timeout
            if response.status_code == 200:
                self.logger.info(f"Ollama provider detected and available at {self.host}")
                # Optionally, check if self.model_name is in response.json()['models']
                # For now, just confirming the server is up.
                return True
            else:
                self.logger.warning(f"Ollama provider at {self.host} responded with status {response.status_code}. Not available.")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Ollama provider not available at {self.host}. Connection error: {e}")
            return False
