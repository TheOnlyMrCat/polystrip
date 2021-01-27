//! Vertices and shapes, the core of the rendering process.

use crate::data::*;

/// A vertex describing a position and a position on a texture.
/// 
/// Texture coordinates are interpolated linearly between vertices.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TextureVertex {
	pub position: GpuVec3,
	pub tex_coords: GpuVec2,
}

unsafe impl bytemuck::Pod for TextureVertex {}
unsafe impl bytemuck::Zeroable for TextureVertex {}

impl TextureVertex {
	pub(crate) fn desc<'a>() -> &'a [gfx_hal::pso::AttributeDesc] {
		use std::mem::size_of;
		
		&[
			gfx_hal::pso::AttributeDesc {
				location: 0,
				binding: 0,
				element: gfx_hal::pso::Element {
					format: gfx_hal::format::Format::Rgb32Sfloat,
					offset: 0,
				},
			},
			gfx_hal::pso::AttributeDesc {
				location: 1,
				binding: 0,
				element: gfx_hal::pso::Element {
					format: gfx_hal::format::Format::Rg32Sfloat,
					offset: size_of::<[f32; 3]>() as u32,
				},
			},
		]
	}
}

/// A vertex describing a position and a colour.
/// 
/// Colours are interpolated linearly between vertices.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct ColorVertex {
	pub position: GpuVec3,
	pub color: Color,
}

unsafe impl bytemuck::Pod for ColorVertex {}
unsafe impl bytemuck::Zeroable for ColorVertex {}

impl ColorVertex {
	pub(crate) fn desc<'a>() -> &'a [gfx_hal::pso::AttributeDesc] {
		use std::mem::size_of;
		
		&[
			gfx_hal::pso::AttributeDesc {
				location: 0,
				binding: 0,
				element: gfx_hal::pso::Element {
					format: gfx_hal::format::Format::Rgb32Sfloat,
					offset: 0,
				},
			},
			gfx_hal::pso::AttributeDesc {
				location: 1,
				binding: 0,
				element: gfx_hal::pso::Element {
					format: gfx_hal::format::Format::Rgba8Unorm,
					offset: size_of::<[f32; 3]>() as u32,
				},
			},
		]
	}
}

/// A set of vertices and indices describing an outlined geometric shape as a set of lines.
///
/// The colors of the lines are determined by interpolating the colours at each
/// [`ColorVertex`](struct.ColorVertex).
#[derive(Clone, Copy, Debug)]
pub struct StrokedShape<'a> {
	pub vertices: &'a [ColorVertex],
	/// A list of pairs of vertices which specify which vertices should have lines drawn between them
	pub indices: &'a [[u16; 2]],
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
///
/// The color of the shape is determined by interpolating the colours at each
/// [`ColorVertex`](struct.ColorVertex).
#[derive(Clone, Copy, Debug)]
pub struct ColoredShape<'a> {
	pub vertices: &'a [ColorVertex],
	/// A list of sets of three vertices which specify how the vertices should be rendered as triangles.
	pub indices: &'a [[u16; 3]], //TODO: Work out if it should need to be CCW or not
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
/// 
/// The color of the shape is determined by interpolating the texture coordinates at each
/// [`TextureVertex`](struct.TextureVertex).
/// 
/// A `TexturedShape` does not store the texture it is to draw. This must be specified in the
/// arguments to [`Frame::draw_textured`](../renderer/struct.Frame#method.draw_textured)
#[derive(Clone, Copy, Debug)]
pub struct TexturedShape<'a> {
	pub vertices: &'a [TextureVertex],
	/// A list of sets of three vertices which specify how the vertices should be rendered as triangles.
	pub indices: &'a [[u16; 3]], //TODO: As above
}

/// A 3 x 3, column major matrix
/// 
/// If the `cgmath` feature is enabled, this is instead a type alias to `cgmath::Matrix3<f32>`
#[cfg(not(feature="cgmath"))]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Matrix3 {
	/// The first column of the matrix
	pub x: [f32; 3],
	/// The second column of the matrix
	pub y: [f32; 3],
	/// The third column of the matrix
	pub z: [f32; 3],
}

#[cfg(not(feature="cgmath"))]
impl Matrix3 {
	/// Returns the [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix).
	pub fn identity() -> Matrix3 {
		Matrix3 {
			x: [1.0, 0.0, 0.0],
			y: [0.0, 1.0, 0.0],
			z: [0.0, 0.0, 1.0],
		}
	}

	pub fn row(&self, i: usize) -> [f32; 3] {
		[self.x[i], self.y[i], self.z[i]]
	}
}

fn dot(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
	lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

#[cfg(not(feature="cgmath"))]
impl std::ops::Mul<Matrix3> for Matrix3 {
	type Output = Matrix3;
	fn mul(self, rhs: Matrix3) -> Matrix3 {
		Matrix3 {
			x: [dot(self.row(0), rhs.x), dot(self.row(1), rhs.x), dot(self.row(2), rhs.x)],
			y: [dot(self.row(0), rhs.y), dot(self.row(1), rhs.y), dot(self.row(2), rhs.y)],
			z: [dot(self.row(0), rhs.z), dot(self.row(1), rhs.z), dot(self.row(2), rhs.z)],
		}
	}
}

#[cfg(not(feature="cgmath"))]
impl Into<[[f32; 3]; 3]> for Matrix3 {
	fn into(self) -> [[f32; 3]; 3] {
		[self.x, self.y, self.z]
	}
}

#[cfg(feature="cgmath")]
pub type Matrix3 = cgmath::Matrix3<f32>;