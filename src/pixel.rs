//! Methods for converting between screen space and pixel coordinates
//! 
//! This module defines the [`PixelTranslator`] type, which is created from a [`WindowTarget`](crate::WindowTarget)
//! and translates coordinates based on the most recently given size (See [`WindowTarget::resize()`](crate::WindowTarget::resize))

use std::rc::Rc;

use crate::{RenderSize, Texture};
use crate::math::{Color, Rect, Vector2};
use crate::gon::{ColorVertex, TextureVertex};

/// When constructed from a [`WindowTarget`](crate::WindowTarget), tracks the window's size and provides methods which
/// convert between pixel space and screen space for that window.
/// 
/// # Rectangles
/// The various rectangle methods defined on this struct output vertices in the following layout:
/// ```text
/// (0)---(1)
///  |     |
/// (3)---(2)
/// ```
pub struct PixelTranslator {
	extent: Rc<RenderSize>,
}

impl PixelTranslator {
	pub(crate) fn new(extent: Rc<RenderSize>) -> PixelTranslator {
		PixelTranslator {
			extent
		}
	}

	/// Converts a pixel value into an absolute screen space position. Doing arithmetic operations with absolute screen space
	/// positions will not give expected values. See the [`pixel_offset`](PixelTranslator::pixel_offset) method to create screen space
	/// offsets
	pub fn pixel_position(&self, x: i32, y: i32) -> Vector2 {
		let extent = self.extent.get();
		Vector2::new((x * 2) as f32 / extent.width as f32 - 1.0, -((y * 2) as f32 / extent.height as f32 - 1.0))
	}

	/// Converts a pixel value into a screen space offset.
	pub fn pixel_offset(&self, x: i32, y: i32) -> Vector2 {
		let extent = self.extent.get();
		Vector2::new((x * 2) as f32 / extent.width as f32, -((y * 2) as f32 / extent.height as f32))
	}

	/// Converts a `Rect` into a set of `ColorVertex`es with the given `Color` and height `0.0`.
	pub fn colored_rect(&self, rect: Rect, color: Color) -> [ColorVertex; 4] {
		[
			ColorVertex { position: Vector2::with_height(self.pixel_position(rect.x, rect.y), 0.0), color },
			ColorVertex { position: Vector2::with_height(self.pixel_position(rect.x + rect.w, rect.y), 0.0), color },
			ColorVertex { position: Vector2::with_height(self.pixel_position(rect.x + rect.w, rect.y + rect.h), 0.0), color },
			ColorVertex { position: Vector2::with_height(self.pixel_position(rect.x, rect.y + rect.h), 0.0), color },
		]
	}

	//TODO: Better names for these functions

	/// Converts a `Rect` into a set of `TextureVertex`es with height `0.0`.
	/// 
	/// The texture will be scaled to fit entirely onto the `Rect`.
	pub fn textured_rect(&self, rect: Rect) -> [TextureVertex; 4] {
		[
			TextureVertex { position: Vector2::with_height(self.pixel_position(rect.x, rect.y), 0.0), tex_coords: Vector2::new(0.0, 0.0) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(rect.x + rect.w, rect.y), 0.0), tex_coords: Vector2::new(1.0, 0.0) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(rect.x + rect.w, rect.y + rect.h), 0.0), tex_coords: Vector2::new(1.0, 1.0) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(rect.x, rect.y + rect.h), 0.0), tex_coords: Vector2::new(0.0, 1.0) },
		]
	}

	/// Creates a set of `TextureVertex`es with the width and height of the passed `Texture` and height `0.0`.
	pub fn texture_at(&self, texture: &Texture, x: i32, y: i32) -> [TextureVertex; 4] {
		[
			TextureVertex { position: Vector2::with_height(self.pixel_position(x, y), 0.0), tex_coords: Vector2::new(0.0, 0.0) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(x + texture.width() as i32, y), 0.0), tex_coords: Vector2::new(1.0, 0.0) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(x + texture.width() as i32, y + texture.height() as i32), 0.0), tex_coords: Vector2::new(1.0, 1.0) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(x, y + texture.height() as i32), 0.0), tex_coords: Vector2::new(0.0, 1.0) },
		]
	}

	/// Creates a set of `TextureVertex`es with the width and height of the passed `Texture` and height `0.0`.
	/// 
	/// The dimensions of the texture will be scaled by the provided `scale` factor. The `x` and `y` positions will not change.
	pub fn texture_scaled(&self, texture: &Texture, x: i32, y: i32, scale: f32) -> [TextureVertex; 4] {
		[
			TextureVertex { position: Vector2::with_height(self.pixel_position(x, y), 0.0), tex_coords: Vector2::new(0.0, 0.0) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(x + (texture.width() as f32 * scale) as i32, y), 0.0), tex_coords: Vector2::new(1.0, 0.0) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(x + (texture.width() as f32 * scale) as i32, y + (texture.height() as f32 * scale) as i32), 0.0), tex_coords: Vector2::new(1.0, 1.0) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(x, y + (texture.height() as f32 * scale) as i32), 0.0), tex_coords: Vector2::new(0.0, 1.0) },
		]
	}

	/// Creates a set of `TextureVertex`es with the width and height of the passed `Texture` and height `0.0`.
	/// 
	/// Only the part of the texture inside the passed `crop` rectangle is shown. The top-left corner of the crop rectangle
	/// is drawn at (`x`, `y`)
	pub fn texture_cropped(&self, texture: &Texture, x: i32, y: i32, crop: Rect) -> [TextureVertex; 4] {
		[
			TextureVertex { position: Vector2::with_height(self.pixel_position(x, y), 0.0), tex_coords: texture.pixel(crop.x, crop.y) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(x + texture.width() as i32, y), 0.0), tex_coords: texture.pixel(crop.x + crop.w, crop.y) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(x + texture.width() as i32, y + texture.height() as i32), 0.0), tex_coords: texture.pixel(crop.x + crop.w, crop.y + crop.h) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(x, y + texture.height() as i32), 0.0), tex_coords: texture.pixel(crop.x, crop.y + crop.h) },
		]
	}

	/// Creates a set of `TextureVertex`es with the width and height of the passed `Texture` and height `0.0`.
	/// 
	/// Only the part of the texture inside the passed `crop` rectangle is shown. The top-left corner of the crop rectangle
	/// is drawn at (`x`, `y`)
	pub fn texture_scaled_cropped(&self, texture: &Texture, destination: Rect, crop: Rect) -> [TextureVertex; 4] {
		[
			TextureVertex { position: Vector2::with_height(self.pixel_position(destination.x, destination.y), 0.0), tex_coords: texture.pixel(crop.x, crop.y) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(destination.x + destination.w, destination.y), 0.0), tex_coords: texture.pixel(crop.x + crop.w, crop.y) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(destination.x + destination.w, destination.y + destination.h), 0.0), tex_coords: texture.pixel(crop.x + crop.w, crop.y + crop.h) },
			TextureVertex { position: Vector2::with_height(self.pixel_position(destination.x, destination.y + destination.h), 0.0), tex_coords: texture.pixel(crop.x, crop.y + crop.h) },
		]
	}
}
