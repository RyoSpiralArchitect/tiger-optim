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
#[derive(Debug, Clone, Copy)]
pub struct TemplateStats {
    configs: &'static [TemplateConfig],
}

impl Default for TemplateStats {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateStats {
    /// Constructs the statistics container with the built-in configuration set.
    pub const fn new() -> Self {
        Self {
            configs: DEFAULT_CONFIGS,
        }
    }

    /// Returns a shared view of the known template configurations.
    pub fn configs(&self) -> &'static [TemplateConfig] {
        self.configs
    }
}

const DEFAULT_CONFIGS: &[TemplateConfig] = &[
    // These presets are intentionally conservative so they remain valid across a
    // wide range of GPU architectures. They are mirrored from the Python
    // implementation used by the optimizer's Triton kernels.
    TemplateConfig::new(16, 16, 32, 2, 4),
    TemplateConfig::new(32, 32, 32, 3, 4),
    TemplateConfig::new(64, 32, 32, 4, 8),
    TemplateConfig::new(64, 64, 32, 5, 8),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configs_accessor_preserves_contents() {
        let stats = TemplateStats::from_configs([
            TileConfig::new("a", 4, 4),
            TileConfig::new("b", 8, 8),
        ]);

        assert_eq!(stats.configs().len(), 2);
        assert_eq!(stats.configs()[0].name, "a");
        assert_eq!(stats.configs()[1].width, 8);

        let owned = stats.clone().into_configs();
        assert_eq!(owned.len(), 2);
    }
}
