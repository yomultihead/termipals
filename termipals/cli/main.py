#!/usr/bin/env python3

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from dotenv import load_dotenv
from tqdm import tqdm

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
        self.model_dir = self.project_root / 'llm' / 'models'
        
        # Create necessary directories
        self.termipals_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Model configuration
        self.model_name = os.getenv('HUGGINGFACE_MODEL_ID')  # Using the original GPT-2 model
        self.model_revision = "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"  # Specific revision for stability
        
    def setup_model(self):
        """Initialize the model for generation"""
        logger.info("🤖 Loading model...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            device_map="auto",
            torch_dtype=torch.float32,
            cache_dir=str(self.model_dir)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            cache_dir=str(self.model_dir)
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
        logger.info("✨ Model loaded successfully!")

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

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.7,
            "do_sample": True,
        }

        output = self.pipe(messages, **generation_args)
        art = output[0]['generated_text']
        
        # Clean up the output
        art_lines = []
        capture = False
        for line in art.split('\n'):
            if not capture and any(c in line for c in r'/\|_-~^'):
                capture = True
            if capture and line.strip():
                art_lines.append(line.rstrip())
            elif capture and not line.strip() and art_lines:
                break
        
        return '\n'.join(art_lines)

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

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    app = Termipals()

    if args.list:
        app.list_animals()
        return

    if args.create:
        logger.info(f"🎨 Generating ASCII art for '{args.animal}'...")
        app.setup_model()
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