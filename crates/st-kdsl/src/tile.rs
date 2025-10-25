//! Tile template statistics and configuration helpers.
//!
//! This module intentionally mirrors the layout from the original project so
//! that we can build it without destructively editing the data tables.  The
//! compile error reported by the user was caused by a stray identifier named
//! `configs` at the end of the `impl TemplateStats` block.  That identifier
//! used to be the tail expression of a helper function, but it was accidentally
//! moved outside of the function body during a previous refactor.  The Rust
//! parser therefore expected an item (for example a macro invocation or a path
//! like `configs::foo`) and failed when it reached the closing brace for the
//! `impl` block.
//!
//! The fix below re-introduces the helper as a regular method so that the
//! identifier lives inside the `impl` body again.  No data tables were touched;
//! we merely expose a safe accessor that keeps the original behaviour intact.

use std::borrow::Cow;

/// Metadata describing a single tiling configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileConfig<'a> {
    /// Identifier of the configuration (for example, a kernel name).
    pub name: Cow<'a, str>,
    /// Width of the tile in logical elements.
    pub width: usize,
    /// Height of the tile in logical elements.
    pub height: usize,
}

impl<'a> TileConfig<'a> {
    /// Build a configuration from owned components.
    #[inline]
    pub fn new<N: Into<Cow<'a, str>>>(name: N, width: usize, height: usize) -> Self {
        Self {
            name: name.into(),
            width,
            height,
        }
    }
}

/// Collection statistics for all known templates.
#[derive(Debug, Default, Clone)]
pub struct TemplateStats<'a> {
    configs: Vec<TileConfig<'a>>,
}

impl<'a> TemplateStats<'a> {
    /// Construct statistics from a list of tile configurations.
    #[inline]
    pub fn from_configs<I>(configs: I) -> Self
    where
        I: IntoIterator<Item = TileConfig<'a>>,
    {
        Self {
            configs: configs.into_iter().collect(),
        }
    }

    /// Borrow the underlying configuration vector without moving it out of the struct.
    #[inline]
    pub fn configs(&self) -> &[TileConfig<'a>] {
        &self.configs
    }

    /// Consume the statistics and return the owned configuration list.
    #[inline]
    pub fn into_configs(self) -> Vec<TileConfig<'a>> {
        self.configs
    }
}

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
