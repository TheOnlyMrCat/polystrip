use std::fmt::Debug;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec2 {
	pub x: f32,
	pub y: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Color {
	pub r: u8,
	pub g: u8,
	pub b: u8,
}