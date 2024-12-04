use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

#[derive(Debug)]
pub struct Storage {
    base_path: PathBuf,
}

impl Storage {
    pub fn new() -> Result<Self> {
        let base_path = dirs::data_local_dir()
            .context("Failed to get local data directory")?            .join("termipals")
            .join("animals");

        std::fs::create_dir_all(&base_path)
            .context("Failed to create storage directory")?;

        Ok(Self { base_path })
    }

    pub fn get_animal(&self, name: &str) -> Result<Option<String>> {
        let file_path = self.base_path.join(format!("{}.txt", name));
        if file_path.exists() {
            let content = std::fs::read_to_string(&file_path)
                .context("Failed to read animal file")?;
            Ok(Some(content))
        } else {
            Ok(None)
        }
    }

    pub fn save_animal(&self, name: &str, art: &str) -> Result<()> {
        let file_path = self.base_path.join(format!("{}.txt", name));
        std::fs::write(&file_path, art)
            .context("Failed to write animal file")?;
        Ok(())
    }
}
