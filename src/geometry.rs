//! Structures and traits for easier rendering to `Frame`s
//! 
//! This module contains the [`FrameGeometryExt`](trait.FrameGeometryExt) trait, which adds numerous functions for simple rendering operations
//! where full description of the shape would be unnecessarily time-consuming.

use crate::data::{Color, Rect};
use crate::renderer::Frame;
use crate::vertex::*;

pub trait FrameGeometryExt {
	fn draw_rect(&mut self, bounds: Rect, color: Color);
}

impl FrameGeometryExt for Frame<'_> {
	fn draw_rect(&mut self, bounds: Rect, color: Color) {
		self.draw_colored(ColoredShape {
			vertices: vec![
				ColorVertex { position: self.pixel(bounds.x, bounds.y), color },
				ColorVertex { position: self.pixel(bounds.x, bounds.y + bounds.h), color },
				ColorVertex { position: self.pixel(bounds.x + bounds.w, bounds.y + bounds.h), color },
				ColorVertex { position: self.pixel(bounds.x + bounds.w, bounds.y), color }
			],
			indices: vec![
				[0, 1, 3],
				[1, 2, 3],
			]
		})
	}
}