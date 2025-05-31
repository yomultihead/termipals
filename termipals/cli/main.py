#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import random
# Removed: from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# Removed: import torch
from dotenv import load_dotenv
# Removed: from tqdm import tqdm

from termipals.llm_provider import HuggingFaceProvider, OllamaProvider # Updated import
import os # Added import for environment variables

# Ensure .env is loaded early
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

class Termipals:
    def __init__(self):
        # Get the project root directory (where setup.py is located)
        self.project_root = Path(__file__).parent.parent.parent
        self.home_dir = Path.home()
        self.termipals_dir = self.home_dir / '.local' / 'share' / 'termipals'
        self.assets_dir = self.termipals_dir / 'assets' / 'animals'
        # self.model_dir = self.project_root / 'llm' / 'models' # Removed, managed by provider

        # Create necessary directories
        self.termipals_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        # self.model_dir.mkdir(parents=True, exist_ok=True) # Removed

        # LLM Provider Initialization
        self.llm_provider = None
        self.logger = logging.getLogger(__name__) # Ensure logger is available in __init__

        provider_name = os.getenv('TERMI_PROVIDER', '').lower()
        hf_model_id = os.getenv('HUGGINGFACE_MODEL_ID', 'distilgpt2')
        hf_revision = os.getenv('HUGGINGFACE_MODEL_REVISION', 'main')
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        ollama_model_name = os.getenv('OLLAMA_MODEL', 'llama2:7b') # Consider orca-mini or tinyllama for wider default use

        provider_config = {
            "huggingface_model_id": hf_model_id,
            "huggingface_model_revision": hf_revision,
            "ollama_host": ollama_host,
            "ollama_model_name": ollama_model_name,
            "connection_timeout": int(os.getenv('TERMI_CONNECTION_TIMEOUT', 5)),
            "request_timeout": int(os.getenv('TERMI_REQUEST_TIMEOUT', 60)),
            "ollama_options": {"temperature": float(os.getenv('OLLAMA_TEMPERATURE', 0.7))}
            # Add other shared configs if any, e.g., logging level for providers
        }

        if provider_name == 'ollama':
            self.logger.info("Attempting to use Ollama provider (specified by TERMI_PROVIDER)...")
            self.llm_provider = OllamaProvider(config=provider_config)
            if not self.llm_provider.is_available():
                self.logger.warning("Ollama provider specified but not available. Falling back.")
                self.llm_provider = None # Fallback will be handled below
            else:
                self.logger.info("Using Ollama provider (specified by TERMI_PROVIDER).")
        elif provider_name in ['local', 'huggingface']:
            self.logger.info("Using local HuggingFace provider (specified by TERMI_PROVIDER)...")
            self.llm_provider = HuggingFaceProvider(config=provider_config)
            if not self.llm_provider.is_available(): # Should generally be true unless deps are missing
                self.logger.error("HuggingFace provider specified but not available (check dependencies: torch, transformers).")
                # Critical error, perhaps exit or raise exception? For now, will fallback.
                self.llm_provider = None
        elif provider_name: # Unrecognized provider_name
            self.logger.warning(f"Unrecognized TERMI_PROVIDER value: '{provider_name}'. Proceeding with auto-detection.")

        # Auto-detection if no explicit provider set or if specified was unavailable
        if self.llm_provider is None and not provider_name : # Only auto-detect if TERMI_PROVIDER was empty
            self.logger.info("TERMI_PROVIDER not set. Attempting to auto-detect Ollama provider...")
            ollama_candidate = OllamaProvider(config=provider_config)
            if ollama_candidate.is_available():
                self.llm_provider = ollama_candidate
                self.logger.info("Ollama provider auto-detected and available. Using Ollama provider.")
            else:
                self.logger.info("Ollama provider not detected or not available during auto-detection.")
        
        # Default to HuggingFace if no provider has been successfully initialized yet
        if self.llm_provider is None:
            if provider_name and provider_name not in ['ollama', 'local', 'huggingface']: # if specified but unrecognized
                 self.logger.info(f"Falling back to local HuggingFace provider (due to unrecognized TERMI_PROVIDER: '{provider_name}').")
            elif provider_name == 'ollama': # if specified ollama but it failed
                 self.logger.info("Falling back to local HuggingFace provider (Ollama specified but was not available).")
            elif provider_name in ['local', 'huggingface']: # if specified HF but it failed (e.g. missing deps)
                 self.logger.critical("HuggingFace provider specified but failed to initialize. Termipals might not function correctly.")
                 # This is a critical state. The HuggingFaceProvider's is_available() should return false.
                 # self.llm_provider will be assigned in the final default block if it's still None.
                 # For now, we let it fall through to the final default assignment.
            else: # Default case (no TERMI_PROVIDER, Ollama not auto-detected)
                self.logger.info("Using local HuggingFace provider (default).")
            # This final assignment ensures self.llm_provider is always set.
            self.llm_provider = HuggingFaceProvider(config=provider_config)

    # Removed setup_model(self) method

    def generate_art(self, animal: str) -> str:
        """Generate ASCII art for an animal"""
        messages = [
            {"role": "system", "content": "You are an ASCII art generator. You create simple and clear ASCII art using standard characters."},
            {"role": "user", "content": f"""Create an ASCII art of a {animal}. 
Rules:
1. Use only standard ASCII characters
2. Keep it simple and clear
3. Maximum size: 80x24 characters
4. Only output the ASCII art itself, no additional text

Example of good ASCII art:
    ^__^
   (oo)\_______
   (__)\       )\/\
       ||----w |
       ||     ||

Now create the ASCII art:"""}
        ]

        # generation_args are now handled by the provider
        # output = self.pipe(messages, **generation_args) # Old call
        
        art = self.llm_provider.generate_art(messages)
        
        # Cleaning logic is now primarily in the provider's _clean_art method.
        # The provider should return already cleaned art.
        # If additional cleaning specific to this CLI context is needed, it can be added here.
        # For now, assume provider's cleaning is sufficient.
        return art

    def save_art(self, animal: str, art: str):
        """Save generated art to assets directory"""
        art_file = self.assets_dir / f"{animal}.txt"
        art_file.write_text(art)
        logger.info(f"💾 Saved to: {art_file}")

    def get_random_animal(self) -> Optional[str]:
        """Get a random animal art from assets"""
        files = list(self.assets_dir.glob("*.txt"))
        if not files:
            return None
        return random.choice(files).read_text()

    def get_animal(self, name: str) -> Optional[str]:
        """Get specific animal art from assets"""
        art_file = self.assets_dir / f"{name}.txt"
        if art_file.exists():
            return art_file.read_text()
        return None

    def list_animals(self):
        """List all available animals"""
        logger.info("🐾 Available animals:")
        for file in sorted(self.assets_dir.glob("*.txt")):
            logger.info(f"  • {file.stem}")

    def inject_to_file(self, animal: str, target_file: Path):
        """Inject ASCII art into a file"""
        art = self.get_animal(animal)
        if not art:
            logger.error(f"❌ Animal '{animal}' not found")
            return
        
        content = target_file.read_text()
        new_content = f"{art}\n{content}"
        target_file.write_text(new_content)
        logger.info(f"✨ Added {animal} to {target_file}")

def main():
    parser = argparse.ArgumentParser(description="🐮 Termipals - ASCII art animals for your terminal")
    parser.add_argument("animal", nargs="?", default="random", help="Animal to display")
    parser.add_argument("file", nargs="?", type=Path, help="File to inject the ASCII art into")
    parser.add_argument("-l", "--list", action="store_true", help="List available animals")
    parser.add_argument("--create", action="store_true", help="Generate new animal ASCII art")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Moved load_dotenv() to the top of the script

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    app = Termipals()

    if args.list:
        app.list_animals()
        return

    if args.create:
        if args.animal == "random": # Cannot create a random animal
            logger.error("❌ Please specify an animal name to create. Example: --create cat")
            return
        logger.info(f"🎨 Generating ASCII art for '{args.animal}'...")
        # app.setup_model() # Removed call, model setup is on-demand by provider
        art = app.generate_art(args.animal)
        print("\nGenerated ASCII Art:")
        print(art)
        app.save_art(args.animal, art)
        return

    if args.file:
        app.inject_to_file(args.animal, args.file)
        return

    # Display animal
    if args.animal == "random":
        art = app.get_random_animal()
    else:
        art = app.get_animal(args.animal)

    if art:
        print(art)
    else:
        logger.error(f"❌ Animal '{args.animal}' not found. Use --create to generate it!")

if __name__ == "__main__":
    main()
