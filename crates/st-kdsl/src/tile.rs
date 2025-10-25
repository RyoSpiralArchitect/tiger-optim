//! Tile template bookkeeping utilities used by the Tiger kernel DSL.

use std::fmt;

/// Describes a single tile configuration that can be emitted by the DSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemplateConfig {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub stages: u32,
    pub warps: u32,
}

impl TemplateConfig {
    /// Creates a new tile configuration instance.
    pub const fn new(tile_m: u32, tile_n: u32, tile_k: u32, stages: u32, warps: u32) -> Self {
        Self {
            tile_m,
            tile_n,
            tile_k,
            stages,
            warps,
        }
    }
}

impl fmt::Display for TemplateConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{}x{} ({} stages / {} warps)",
            self.tile_m, self.tile_n, self.tile_k, self.stages, self.warps
        )
    }
}

/// Pre-computed statistics for frequently used tile templates.
#[derive(Debug, Clone, Default)]
pub struct TemplateStats {
    configs: Vec<TemplateConfig>,
}

impl TemplateStats {
    /// Constructs the statistics container with the built-in configuration set.
    pub fn new() -> Self {
        Self {
            configs: Self::default_configs(),
        }
    }

    /// Returns a shared view of the known template configurations.
    pub fn configs(&self) -> &[TemplateConfig] {
        &self.configs
    }

    /// Internal helper that lists all built-in configurations.
    fn default_configs() -> Vec<TemplateConfig> {
        // These presets are intentionally conservative so they remain valid across
        // a wide range of GPU architectures. They are mirrored from the Python
        // implementation used by the optimizer's Triton kernels.
        let mut configs = Vec::with_capacity(4);

        configs.push(TemplateConfig::new(16, 16, 32, 2, 4));
        configs.push(TemplateConfig::new(32, 32, 32, 3, 4));
        configs.push(TemplateConfig::new(64, 32, 32, 4, 8));
        configs.push(TemplateConfig::new(64, 64, 32, 5, 8));

        configs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_configs_are_populated() {
        let stats = TemplateStats::new();
        assert!(!stats.configs().is_empty());
    }
}
