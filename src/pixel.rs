//! Methods for converting between screen space and pixel coordinates
//! 
//! This module defines the [`PixelTranslator`] type, which is created from a [`Renderer`] and translates
//! coordinates based on the size of the renderer's viewport

use std::rc::Rc;

use crate::{RendererContext, Texture};
use crate::vertex::{Color, ColorVertex, Rect, TextureVertex, Vector2};
use crate::vertex::with_height;

pub struct PixelTranslator {
	context: Rc<RendererContext>,
}

impl PixelTranslator {
	pub(crate) fn new(context: Rc<RendererContext>) -> PixelTranslator {
		PixelTranslator {
			context
		}
	}

	/// Converts a pixel value into an absolute screen space position. Doing operations with absolute screen space positions
	/// will not give expected values. See the [`pixel_offset`](#method.pixel_offset) method to create screen space offsets
	pub fn pixel_position(&self, x: i32, y: i32) -> Vector2 {
		let extent = self.context.extent.get();
		Vector2 {
			x: (x * 2) as f32 / extent.width as f32 - 1.0,
			y: -((y * 2) as f32 / extent.height as f32 - 1.0),
		}
	}

	/// Converts a pixel value into a screen space offset.
	pub fn pixel_offset(&self, x: i32, y: i32) -> Vector2 {
		let extent = self.context.extent.get();
		Vector2 {
			x: (x * 2) as f32 / extent.width as f32,
			y: -((y * 2) as f32 / extent.height as f32),
		}
	}

	/// Converts a `Rect` into a set of `ColorVertex`es with the given `Color` and height `0.0`.
	/// 
	/// The indices of the four corners of the `Rect` are as follows:
	/// ```
	/// (0)---(1)
	///  |     |
	/// (3)---(2)
	/// ```
	pub fn colored_rect(&self, rect: Rect, color: Color) -> [ColorVertex; 4] {
		[
			ColorVertex { position: with_height(self.pixel_offset(rect.x, rect.y), 0.0), color },
			ColorVertex { position: with_height(self.pixel_offset(rect.x + rect.w, rect.y), 0.0), color },
			ColorVertex { position: with_height(self.pixel_offset(rect.x + rect.w, rect.y + rect.h), 0.0), color },
			ColorVertex { position: with_height(self.pixel_offset(rect.x, rect.y + rect.h), 0.0), color },
		]
	}

	/// Converts a `Rect` into a set of `TextureVertex`es with height `0.0`.
	/// 
	/// The texture will be scaled to fit entirely onto the `Rect`. If that is not expected, use one of
	/// [`texture_at`], [`texture_scaled`], [`texture_cropped`], or [`texture_scaled_cropped`] instead.
	/// 
	/// The indices of the four corners of the `Rect` are as follows:
	/// ```
	/// (0)---(1)
	///  |     |
	/// (3)---(2)
	/// ```
	pub fn textured_rect(&self, rect: Rect) -> [TextureVertex; 4] {
		[
			TextureVertex { position: with_height(self.pixel_offset(rect.x, rect.y), 0.0), tex_coords: Vector2 { x: 0.0, y: 0.0 } },
			TextureVertex { position: with_height(self.pixel_offset(rect.x + rect.w, rect.y), 0.0), tex_coords: Vector2 { x: 1.0, y: 0.0 } },
			TextureVertex { position: with_height(self.pixel_offset(rect.x + rect.w, rect.y + rect.h), 0.0), tex_coords: Vector2 { x: 1.0, y: 1.0 } },
			TextureVertex { position: with_height(self.pixel_offset(rect.x, rect.y + rect.h), 0.0), tex_coords: Vector2 { x: 0.0, y: 1.0 } },
		]
	}

	/// Creates a set of `TextureVertex`es with the width and height of the passed `Texture` and height `0.0`.
	/// 
	/// The indices of the four corners of the `Rect` are as follows:
	/// ```
	/// (0)---(1)
	///  |     |
	/// (3)---(2)
	/// ```
	pub fn texture_at(&self, texture: &Texture, x: i32, y: i32) -> [TextureVertex; 4] {
		[
			TextureVertex { position: with_height(self.pixel_offset(x, y), 0.0), tex_coords: Vector2 { x: 0.0, y: 0.0 } },
			TextureVertex { position: with_height(self.pixel_offset(x + texture.width() as i32, y), 0.0), tex_coords: Vector2 { x: 1.0, y: 0.0 } },
			TextureVertex { position: with_height(self.pixel_offset(x + texture.width() as i32, y + texture.height() as i32), 0.0), tex_coords: Vector2 { x: 1.0, y: 1.0 } },
			TextureVertex { position: with_height(self.pixel_offset(x, y + texture.height() as i32), 0.0), tex_coords: Vector2 { x: 0.0, y: 1.0 } },
		]
	}

	/// Creates a set of `TextureVertex`es with the width and height of the passed `Texture` and height `0.0`.
	/// 
	/// The dimensions of the texture will be scaled by the provided `scale` factor. The `x` and `y` positions will not change.
	/// 
	/// The indices of the four corners of the `Rect` are as follows:
	/// ```
	/// (0)---(1)
	///  |     |
	/// (3)---(2)
	/// ```
	pub fn texture_scaled(&self, texture: &Texture, x: i32, y: i32, scale: f32) -> [TextureVertex; 4] {
		[
			TextureVertex { position: with_height(self.pixel_offset(x, y), 0.0), tex_coords: Vector2 { x: 0.0, y: 0.0 } },
			TextureVertex { position: with_height(self.pixel_offset(x + (texture.width() as f32 * scale) as i32, y), 0.0), tex_coords: Vector2 { x: 1.0, y: 0.0 } },
			TextureVertex { position: with_height(self.pixel_offset(x + (texture.width() as f32 * scale) as i32, y + (texture.height() as f32 * scale) as i32), 0.0), tex_coords: Vector2 { x: 1.0, y: 1.0 } },
			TextureVertex { position: with_height(self.pixel_offset(x, y + (texture.height() as f32 * scale) as i32), 0.0), tex_coords: Vector2 { x: 0.0, y: 1.0 } },
		]
	}
}
