# 🐮 Termipals

Add some joy to your terminal with friendly ASCII art animals! Termipals uses a flexible LLM backend (defaulting to a small local model, with support for Ollama) to generate unique ASCII art animals on demand.

## 🚀 Features

- Generate unique ASCII art animals.
- Flexible LLM backend: Uses a small local Hugging Face model by default, with automatic detection for Ollama, and manual configuration options.
- Save and reuse generated art.
- Display random animals.
- Inject animals into files.
- List available animals.
- Lightweight and aims to be fast.

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
    This will install Termipals and its dependencies, including `transformers`, `torch`, and `requests`.

The first time you run a generation command with the default local Hugging Face provider, it will download the model (e.g., `distilgpt2`, approx. 300-400MB) to a cache directory (typically `~/.cache/termipals/models/`).

## ⚙️ LLM Configuration

Termipals intelligently selects an LLM provider. By default, it uses a built-in small Hugging Face model (`distilgpt2`). If you have Ollama running, it may be auto-detected and preferred. You can also explicitly configure the provider using environment variables.

The selection priority is as follows:
1.  **Explicit Provider**: Defined by the `TERMI_PROVIDER` environment variable.
2.  **Auto-detection**: If `TERMI_PROVIDER` is not set, Termipals checks for a running Ollama instance.
3.  **Default**: If no explicit provider is set and Ollama is not detected, Termipals defaults to using a local Hugging Face model.

### Environment Variables

You can control Termipals' LLM behavior using the following environment variables. You can set them in your shell or place them in a `.env` file in the project root.

| Variable                     | Description                                                                                                | Default Value               |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------- |
| `TERMI_PROVIDER`             | Specifies the LLM provider. Values: `local` (or `huggingface`) for local Hugging Face, `ollama` for Ollama.  | (empty, for auto-detect)  |
| `HUGGINGFACE_MODEL_ID`       | The Hugging Face model ID to use when `TERMI_PROVIDER` is `local`.                                         | `distilgpt2`                |
| `HUGGINGFACE_MODEL_REVISION` | The model revision (branch, tag, commit hash) for the Hugging Face model.                                  | `main`                      |
| `OLLAMA_HOST`                | The host address for your Ollama server (e.g., `http://localhost:11434`).                                   | `http://localhost:11434`    |
| `OLLAMA_MODEL`               | The model name to use with Ollama (must be pulled in Ollama first). Examples: `llama2:7b`, `orca-mini`.      | `llama2:7b`                 |
| `OLLAMA_TEMPERATURE`         | Sets the temperature for Ollama generation (float, e.g., `0.7`).                                             | `0.7`                       |
| `TERMI_CONNECTION_TIMEOUT`   | Timeout in seconds for establishing connections (e.g., to Ollama).                                         | `5`                         |
| `TERMI_REQUEST_TIMEOUT`      | Timeout in seconds for the entire LLM request.                                                              | `60`                        |

**Example `.env` file**:
```env
# Example .env file for Termipals

# Uncomment and set the provider you want to use explicitly
# TERMI_PROVIDER=ollama
# TERMI_PROVIDER=local

# Hugging Face specific settings (if using 'local' provider)
# HUGGINGFACE_MODEL_ID=gpt2
# HUGGINGFACE_MODEL_REVISION=main

# Ollama specific settings (if using 'ollama' provider)
# OLLAMA_HOST=http://127.0.0.1:11434
# OLLAMA_MODEL=mistral:7b # Make sure you have pulled this model, e.g., ollama pull mistral:7b
# OLLAMA_TEMPERATURE=0.6

# Advanced: Timeouts (in seconds)
# TERMI_CONNECTION_TIMEOUT=10
# TERMI_REQUEST_TIMEOUT=180
```
*Note: Create a file named `.env` in the root of the project and add your configurations there. Termipals will automatically load it if `python-dotenv` is installed and `load_dotenv()` is called.*

### Using Ollama

To use Ollama with Termipals:
1.  **Install Ollama**: Visit [ollama.com](https://ollama.com) and follow the installation instructions for your operating system.
2.  **Ensure Ollama is running**: Typically, Ollama runs as a background service.
3.  **Pull a model**: Before Termipals can use an Ollama model, you need to pull it. For example, to pull `orca-mini`:
    ```bash
    ollama pull orca-mini
    ```
    Other models like `llama2:7b`, `mistral:7b`, or `codellama:7b` can also be used.
4.  **Configure Termipals**:
    *   **Auto-detection**: If Ollama is running on the default host (`http://localhost:11434`), Termipals should detect it automatically if `TERMI_PROVIDER` is not set.
    *   **Manual Configuration**: Set `TERMI_PROVIDER=ollama` in your environment or `.env` file. You can also specify `OLLAMA_HOST` and `OLLAMA_MODEL` if they differ from the defaults.

## 🎮 Usage

### Display an animal
```bash
# Show a random animal
termipals

# Show a specific animal (if previously generated and saved)
termipals cat
```

### Generate new art
```bash
# Generate ASCII art for a new animal (e.g., an elephant)
termipals --create elephant
```
*This will use the configured or auto-detected LLM provider.*

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
# Enable debug output, useful for seeing provider selection and generation details
termipals --debug --create unicorn
```

## 📁 Project Structure

```
termipals/
├── .env.example         # Example environment file
├── llm/                 # (No longer for models, cache is ~/.cache/termipals/models/)
├── termipals/
│   ├── __init__.py
│   ├── llm_provider.py  # LLM provider implementations (HuggingFace, Ollama)
│   ├── assets/
│   │   └── animals/     # Saved generated ASCII art
│   └── cli/
│       ├── __init__.py
│       └── main.py      # CLI implementation
├── README.md
├── setup.py
└── ... (other project files)
```
*Note: The primary Hugging Face model cache is now typically `~/.cache/termipals/models/`.*

## 🛠️ Development

1.  Clone the repository.
2.  Create a virtual environment (see Installation).
3.  Install development dependencies:
    ```bash
    pip install -e ".[dev]"
    ```
    *(Ensure your `setup.py` includes a `[dev]` extra with tools like `pytest`, `flake8` etc.)*

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
If you encounter any issues or have feature suggestions, please open an issue on GitHub.