//! Helper structures for coordination of data

use std::fmt::Debug;

/// 2D coordinates in GPU space, represented with `f32`s.
/// 
/// # Screen space
/// `(0.0, 0.0)` is the screen center. `(1.0, 1.0)` is the top-right corner.
/// `(-1.0, -1.0)` is the bottom-left corner.
/// 
/// # Texture space
/// `(0.0, 0.0)` is the top-left corner
/// `(1.0, 1.0)` is the bottom-right corner
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct GpuVec2 {
	pub x: f32,
	pub y: f32,
}

impl GpuVec2 {
	pub fn new(x: f32, y: f32) -> GpuVec2 {
		GpuVec2 { x, y }
	}

	pub fn with_height(self, h: f32) -> GpuVec3 {
		GpuVec3 { x: self.x, y: self.y, h }
	}
}

/// 3D coordinates in GPU space, represented with `f32`s.
/// 
/// # Height
/// The third coordinate, `h`, is uncapped non-linear height, used to affect the render output.
/// Out of a set of shapes drawn at the same height, the one drawn last appears on top.
/// Out of a set of shapes drawn at different heights, however, the one with the greatest height appears on top.
/// 
/// Additionally, height interpolates linearly between vertices.
/// 
/// # Screen space
/// `(0.0, 0.0)` is the screen center. `(1.0, 1.0)` is the top-right corner.
/// `(-1.0, -1.0)` is the bottom-left corner.
/// 
/// # Texture space
/// `(0.0, 0.0)` is the top-left corner
/// `(1.0, 1.0)` is the bottom-right corner
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct GpuVec3 {
	pub x: f32,
	pub y: f32,
	pub h: f32,
}

impl GpuVec3 {
	pub fn new(x: f32, y: f32, h: f32) -> GpuVec3 {
		GpuVec3 { x, y, h }
	}
}

/// A color in the sRGB color space, with red, green, blue, and alpha components all represented with `u8`s
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct Color {
	pub r: u8,
	pub g: u8,
	pub b: u8,
	pub a: u8,
}

impl Color {
	pub fn new(r: u8, g: u8, b: u8, a: u8) -> Color {
		Color { r, g, b, a }
	}
}

/// A rectangle in pixel coordinates. (x, y) is the top-left corner; (w, h) expanding rightward and downward.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct Rect {
	pub x: i32,
	pub y: i32,
	pub w: i32,
	pub h: i32,
}

impl Rect {
	pub fn new(x: i32, y: i32, w: i32, h: i32) -> Rect {
		Rect { x, y, w, h }
	}
}