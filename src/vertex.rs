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

mod matrix {
	/// A 4 x 4, column major matrix
	/// 
	/// If the `cgmath` feature is enabled, this is instead a type alias to `cgmath::Matrix4<f32>`
	#[repr(C)]
	#[derive(Clone, Copy, Debug)]
	pub struct Matrix4 {
		/// The first column of the matrix
		pub x: [f32; 4],
		/// The second column of the matrix
		pub y: [f32; 4],
		/// The third column of the matrix
		pub z: [f32; 4],
		/// The fourth column of the matrix
		pub w: [f32; 4],
	}

	impl Matrix4 {
		/// Returns the [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix).
		pub fn identity() -> Matrix4 {
			Matrix4 {
				x: [1.0, 0.0, 0.0, 0.0],
				y: [0.0, 1.0, 0.0, 0.0],
				z: [0.0, 0.0, 1.0, 0.0],
				w: [0.0, 0.0, 0.0, 1.0],
			}
		}

		pub fn translate(x: f32, y: f32) -> Matrix4 {
			Matrix4 {
				x: [1.0, 0.0, 0.0, 0.0],
				y: [0.0, 1.0, 0.0, 0.0],
				z: [0.0, 0.0, 1.0, 0.0],
				w: [ x ,  y , 0.0, 1.0],
			}
		}

		pub fn row(&self, i: usize) -> [f32; 4] {
			[self.x[i], self.y[i], self.z[i], self.w[i]]
		}
	}

	fn dot(lhs: [f32; 4], rhs: [f32; 4]) -> f32 {
		lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2] + lhs[3] * rhs[3]
	}

	impl std::ops::Mul<Matrix4> for Matrix4 {
		type Output = Matrix4;
		fn mul(self, rhs: Matrix4) -> Matrix4 {
			Matrix4 {
				x: [dot(self.row(0), rhs.x), dot(self.row(1), rhs.x), dot(self.row(2), rhs.x), dot(self.row(3), rhs.x)],
				y: [dot(self.row(0), rhs.y), dot(self.row(1), rhs.y), dot(self.row(2), rhs.y), dot(self.row(3), rhs.y)],
				z: [dot(self.row(0), rhs.z), dot(self.row(1), rhs.z), dot(self.row(2), rhs.z), dot(self.row(3), rhs.z)],
				w: [dot(self.row(0), rhs.w), dot(self.row(1), rhs.w), dot(self.row(2), rhs.w), dot(self.row(3), rhs.w)],
			}
		}
	}

	impl Into<[[f32; 4]; 4]> for Matrix4 {
		fn into(self) -> [[f32; 4]; 4] {
			[self.x, self.y, self.z, self.w]
		}
	}
}

#[cfg(not(feature="cgmath"))]
pub type Matrix4 = matrix::Matrix4;

#[cfg(feature="cgmath")]
pub type Matrix4 = cgmath::Matrix4<f32>;