# 🐮 Termipals

Add some joy to your terminal with friendly ASCII art animals! Termipals uses a small local LLM to generate unique ASCII art animals on demand.

## 🚀 Features

- Generate unique ASCII art animals using a local LLM
- Save and reuse generated art
- Display random animals
- Inject animals into files
- List available animals
- Lightweight and fast

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/termipals.git
cd termipals
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the package in editable mode:
```bash
pip install -e .
```

The first time you run a generation command, it will download the SmolLM model (about 250MB) to the `llm/models` directory.

## 🎮 Usage

### Display an animal
```bash
# Show a random animal
termipals

# Show a specific animal
termipals cat
```

### Generate new art
```bash
# Generate ASCII art for a new animal
termipals --create elephant
```

### List available animals
```bash
termipals -l
```

### Inject art into a file
```bash
# Add an animal to the top of a file
termipals cat myfile.txt
```

### Debug mode
```bash
# Enable debug output
termipals --debug --create unicorn
```

## 📁 Project Structure

```
termipals/
├── llm/
│   └── models/          # LLM model storage
├── termipals/
│   ├── assets/
│   │   └── animals/    # Generated ASCII art
│   └── cli/           # CLI implementation
└── docs/              # Documentation
```

## 🛠️ Development

1. Clone the repository
2. Create a virtual environment
3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.