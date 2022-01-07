//! Methods for converting between screen space and pixel coordinates
//!
//! This module defines the [`PixelTranslator`] type, which is created from a [`WindowTarget`](crate::WindowTarget)
//! and translates coordinates based on the most recently given size (See [`WindowTarget::resize()`](crate::WindowTarget::resize))

use std::rc::Rc;

use crate::math::{Color, Rect, Vector2, Matrix4};
use crate::{RenderSize, Texture};

#[cfg(feature = "gon")]
use crate::gon::{GpuColorVertex, GpuTextureVertex};

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
		PixelTranslator { extent }
	}

	/// Converts a pixel value into an absolute screen space position. Doing arithmetic operations with absolute screen space
	/// positions will not give expected values. See the [`pixel_offset`](PixelTranslator::pixel_offset) method to create screen space
	/// offsets
	pub fn pixel_position(&self, x: i32, y: i32) -> Vector2 {
		let (width, height) = self.extent.get();
		Vector2::new((x * 2) as f32 / width as f32 - 1.0, -((y * 2) as f32 / height as f32 - 1.0))
	}

	/// Converts a pixel value into a screen space offset.
	pub fn pixel_offset(&self, x: i32, y: i32) -> Vector2 {
		let (width, height) = self.extent.get();
		Vector2::new((x * 2) as f32 / width as f32, -((y * 2) as f32 / height as f32))
	}

	pub fn transform_rect(&self, Rect { x, y, w, h }: Rect) -> Matrix4 {
		let (width, height) = self.extent.get();
		let x = (x * 2) as f32 / width as f32 - 1.0;
		let y = -((y * 2) as f32 / height as f32 - 1.0);
		let w = (w * 2) as f32 / width as f32;
		let h = -((h * 2) as f32 / height as f32);
		Matrix4::from([
			[w, 0.0, 0.0, 0.0],
			[0.0, h, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0],
			[x, y, 0.0, 1.0],
		])
	}
}
