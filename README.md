# 🐮 Termipals

Add some joy to your terminal with friendly ASCII art animals! Termipals brings life to your command line by letting you summon cute animal companions using simple commands.

```bash
$ termipals moo

------------ < hello friend! > ------------
                                   \
                                    \
                                     \    ^__^
                                      \  (oo)\_______
                                         (__)\       )\/\
                                             ||----w |
                                             ||     ||
```

## 🚀 Features

- **Easy to use**: Just type `termipals <animal>` to summon your friends
- **File injection**: Use `termipals <animal> file.md` to add ASCII art to your files
- **Lightweight**: Uses a tiny local LLM for generating new animals
- **Extensible**: Easy to add new animals and customize existing ones
- **Fast**: Written in Rust for blazing-fast performance

## 🛠 Installation

```bash
cargo install termipals
```

## 🎮 Usage

```bash
# Show a random animal
termipals

# Show specific animals
termipals moo    # Show a cow
termipals meow   # Show a cat
termipals woof   # Show a dog

# Add to files
termipals moo README.md  # Add cow to start of README.md

# Generate new animals
termipals create "hamster"  # Generate ASCII art for a hamster
```

## 🏗 Architecture

Termipals is built with performance and extensibility in mind:

- **Core**: Written in Rust for speed and reliability
- **LLM**: Uses a small local model for generating new animals
- **Storage**: Animals are stored as simple text files for easy customization
- **CLI**: Simple and intuitive command-line interface

Check out [ARCHITECTURE.md](./docs/ARCHITECTURE.md) for more details.

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for guidelines.

## 📜 License

MIT