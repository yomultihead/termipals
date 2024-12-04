use clap::Parser;
use colored::*;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "termipals",
    about = "🐮 Your friendly neighborhood terminal companions!"
)]
struct Cli {
    /// The animal to show (e.g., moo, meow)
    #[arg(default_value = "random")]
    animal: String,

    /// Optional file to inject the ASCII art into
    #[arg(value_parser = clap::value_parser!(PathBuf))]
    file: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    println!("{} Welcome to Termipals!", "🐮".green());
    
    // TODO: Implement core functionality
    Ok(())
}