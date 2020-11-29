//! Helper structures for coordination of data

use std::fmt::Debug;

/// Coordinates in GPU space, represented with `f32`s.
/// 
/// `(0.0, 0.0)` is the screen center. `(1.0, 1.0)` is the top-right corner.
/// `(-1.0, -1.0)` is the bottom-left corner.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct GpuPos {
	pub x: f32,
	pub y: f32,
}

impl GpuPos {
	pub fn new(x: f32, y: f32) -> GpuPos {
		GpuPos { x, y }
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