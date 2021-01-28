//! Structures and traits for easier rendering to `Frame`s
//! 
//! This module contains the [`FrameGeometryExt`](trait.FrameGeometryExt) trait, which adds numerous functions for simple rendering operations
//! where full description of the shape would be unnecessarily time-consuming.

use crate::data::{GpuVec2, Color, Rect};
use crate::Frame;
use crate::Texture;
use crate::vertex::*;

const QUAD_INCICES: [[u16; 3]; 2] = [[0, 1, 3], [1, 2, 3]];

/// This trait contains numerous functions to make rendering to a [`Frame`](../struct.Frame.html) easier.
pub trait FrameGeometryExt<'a> {
	/// Draw and fill a `Rect` with the specified `Color`.
	fn draw_rect(&mut self, bounds: Rect, color: Color);

	/// Draw a rectangular texture at a 1:1 scale at the specified coordinates.
	fn draw_texture(&mut self, x: i32, y: i32, texture: &'a Texture);
	
	/// Draw a rectangular texture to fit the specified `Rect`
	fn draw_texture_scaled(&mut self, destination: Rect, texture: &'a Texture);

	/// Draw a rectangular section of a texture at a 1:1 scale at the specified coordinates.
	/// 
	/// # Arguments
	/// * `source`: The section of the texture to draw. Sampling beyond the bounds of the texture will
	///             result in the texture being repeated
	/// * `x`, `y`: The coordinates of the top-left corner of the destination.
	/// * `texture`: A reference to the texture to be drawn
	fn draw_texture_cropped(&mut self, source: Rect, x: i32, y: i32, texture: &'a Texture);

	/// Draw a rectangular section of a texture to fit the specified destination `Rect`.
	/// 
	/// # Arguments
	/// * `source`: The section of the texture to draw. Sampling beyond the bounds of the texture will
	///             result in the texture being repeated
	/// * `destination`: The position and size at which to draw the texture
	/// * `texture`: The texture to be drawn
	fn draw_texture_cropped_scaled(&mut self, source: Rect, destination: Rect, texture: &'a Texture);
}

impl<'a> FrameGeometryExt<'a> for Frame<'a> {
	fn draw_rect(&mut self, bounds: Rect, color: Color) {
		self.draw_colored(ColoredShape {
			vertices: &[
				ColorVertex { position: self.pixel(bounds.x, bounds.y).with_height(0.0), color },
				ColorVertex { position: self.pixel(bounds.x, bounds.y + bounds.h).with_height(0.0), color },
				ColorVertex { position: self.pixel(bounds.x + bounds.w, bounds.y + bounds.h).with_height(0.0), color },
				ColorVertex { position: self.pixel(bounds.x + bounds.w, bounds.y).with_height(0.0), color }
			],
			indices: &QUAD_INCICES
		}, Matrix4::identity());
	}

	fn draw_texture(&mut self, x: i32, y: i32, texture: &'a Texture) {
		self.draw_textured(TexturedShape {
			vertices: &[
				TextureVertex { position: self.pixel(x, y).with_height(0.0), tex_coords: GpuVec2 { x: 0.0, y: 0.0 } },
				TextureVertex { position: self.pixel(x, y + texture.height() as i32).with_height(0.0), tex_coords: GpuVec2 { x: 0.0, y: 1.0 } },
				TextureVertex { position: self.pixel(x + texture.width() as i32, y + texture.height() as i32).with_height(0.0), tex_coords: GpuVec2 { x: 1.0, y: 1.0 } },
				TextureVertex { position: self.pixel(x + texture.width() as i32, y).with_height(0.0), tex_coords: GpuVec2 { x: 1.0, y: 0.0 } }
			],
			indices: &QUAD_INCICES
		}, texture, Matrix4::identity());
	}

	fn draw_texture_scaled(&mut self, destination: Rect, texture: &'a Texture) {
		self.draw_textured(TexturedShape {
			vertices: &[
				TextureVertex { position: self.pixel(destination.x, destination.y).with_height(0.0), tex_coords: GpuVec2 { x: 0.0, y: 0.0 } },
				TextureVertex { position: self.pixel(destination.x, destination.y + destination.h).with_height(0.0), tex_coords: GpuVec2 { x: 0.0, y: 1.0 } },
				TextureVertex { position: self.pixel(destination.x + destination.w, destination.y + destination.h).with_height(0.0), tex_coords: GpuVec2 { x: 1.0, y: 1.0 } },
				TextureVertex { position: self.pixel(destination.x + destination.w, destination.y).with_height(0.0), tex_coords: GpuVec2 { x: 1.0, y: 0.0 } }
			],
			indices: &QUAD_INCICES
		}, texture, Matrix4::identity());
	}

	fn draw_texture_cropped(&mut self, source: Rect, x: i32, y: i32, texture: &'a Texture) {
		self.draw_textured(TexturedShape {
			vertices: &[
				TextureVertex { position: self.pixel(x, y).with_height(0.0), tex_coords: texture.pixel(source.x, source.y) },
				TextureVertex { position: self.pixel(x, y + texture.height() as i32 - source.h).with_height(0.0), tex_coords: texture.pixel(source.x, source.y + source.h) },
				TextureVertex { position: self.pixel(x + texture.width() as i32 - source.w, y + texture.height() as i32 - source.h).with_height(0.0), tex_coords: texture.pixel(source.x + source.w, source.y + source.h) },
				TextureVertex { position: self.pixel(x + texture.width() as i32 - source.w, y).with_height(0.0), tex_coords: texture.pixel(source.x + source.w, source.y) }
			],
			indices: &QUAD_INCICES
		}, texture, Matrix4::identity());
	}

	fn draw_texture_cropped_scaled(&mut self, source: Rect, destination: Rect, texture: &'a Texture) {
		self.draw_textured(TexturedShape {
			vertices: &[
				TextureVertex { position: self.pixel(destination.x, destination.y).with_height(0.0), tex_coords: texture.pixel(source.x, source.y) },
				TextureVertex { position: self.pixel(destination.x, destination.y + destination.h).with_height(0.0), tex_coords: texture.pixel(source.x, source.y + source.h) },
				TextureVertex { position: self.pixel(destination.x + destination.w, destination.y + destination.h).with_height(0.0), tex_coords: texture.pixel(source.x + source.w, source.y + source.h) },
				TextureVertex { position: self.pixel(destination.x + destination.w, destination.y).with_height(0.0), tex_coords: texture.pixel(source.x + source.w, source.y) }
			],
			indices: &QUAD_INCICES
		}, texture, Matrix4::identity());
	}
}