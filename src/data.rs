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

/// A color in the sRGB color space, with red, green, blue, and alpha components all represented with `u8`s
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct Color {
	pub r: u8,
	pub g: u8,
	pub b: u8,
	pub a: u8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct Rect {
	pub x: i32,
	pub y: i32,
	pub w: i32,
	pub h: i32,
}