//! Structures and traits for easier rendering to `Frame`s
//! 
//! This module contains the [`FrameGeometryExt`](trait.FrameGeometryExt) trait, which adds numerous functions for simple rendering operations
//! where full description of the shape would be unnecessarily time-consuming.

use crate::data::{GpuPos, Color, Rect};
use crate::renderer::Frame;
use crate::texture::Texture;
use crate::vertex::*;

pub trait FrameGeometryExt<'a> {
	fn draw_rect(&mut self, bounds: Rect, color: Color);
	fn draw_texture(&mut self, bounds: Rect, texture: &'a Texture);
	fn draw_texture_cropped(&mut self, source: Rect, destination: Rect, texture: &'a Texture);
}

impl<'a> FrameGeometryExt<'a> for Frame<'a> {
	fn draw_rect(&mut self, bounds: Rect, color: Color) {
		self.draw_colored(ColoredShape {
			vertices: &[
				ColorVertex { position: self.pixel(bounds.x, bounds.y), color },
				ColorVertex { position: self.pixel(bounds.x, bounds.y + bounds.h), color },
				ColorVertex { position: self.pixel(bounds.x + bounds.w, bounds.y + bounds.h), color },
				ColorVertex { position: self.pixel(bounds.x + bounds.w, bounds.y), color }
			],
			indices: &[
				[0, 1, 3],
				[1, 2, 3],
			]
		});
	}

	fn draw_texture(&mut self, bounds: Rect, texture: &'a Texture) {
		self.draw_textured(TexturedShape {
			vertices: &[
				TextureVertex { position: self.pixel(bounds.x, bounds.y), tex_coords: GpuPos { x: 0.0, y: 0.0 } },
				TextureVertex { position: self.pixel(bounds.x, bounds.y + bounds.h), tex_coords: GpuPos { x: 0.0, y: 1.0 } },
				TextureVertex { position: self.pixel(bounds.x + bounds.w, bounds.y + bounds.h), tex_coords: GpuPos { x: 1.0, y: 1.0 } },
				TextureVertex { position: self.pixel(bounds.x + bounds.w, bounds.y), tex_coords: GpuPos { x: 1.0, y: 0.0 } }
			],
			indices: &[
				[0, 1, 3],
				[1, 2, 3],
			]
		}, texture);
	}

	fn draw_texture_cropped(&mut self, source: Rect, destination: Rect, texture: &'a Texture) {
		self.draw_textured(TexturedShape {
			vertices: &[
				TextureVertex { position: self.pixel(destination.x, destination.y), tex_coords: texture.pixel(source.x, source.y) },
				TextureVertex { position: self.pixel(destination.x, destination.y + destination.h), tex_coords: texture.pixel(source.x, source.y + source.h) },
				TextureVertex { position: self.pixel(destination.x + destination.w, destination.y + destination.h), tex_coords: texture.pixel(source.x + source.w, source.y + source.h) },
				TextureVertex { position: self.pixel(destination.x + destination.w, destination.y), tex_coords: texture.pixel(source.x + source.w, source.y) }
			],
			indices: &[
				[0, 1, 3],
				[1, 2, 3],
			]
		}, texture);
	}
}