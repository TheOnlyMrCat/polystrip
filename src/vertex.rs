//! Vertices and shapes, the core of the rendering process.

use crate::data::*;

/// A vertex describing a position and a position on a texture.
/// 
/// Texture coordinates are interpolated linearly between vertices.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TextureVertex {
	pub position: GpuPos,
	pub tex_coords: GpuPos,
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
					format: gfx_hal::format::Format::Rg32Sfloat,
					offset: 0,
				},
			},
			gfx_hal::pso::AttributeDesc {
				location: 1,
				binding: 0,
				element: gfx_hal::pso::Element {
					format: gfx_hal::format::Format::Rg32Sfloat,
					offset: size_of::<[f32; 2]>() as u32,
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
	pub position: GpuPos,
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
					format: gfx_hal::format::Format::Rg32Sfloat,
					offset: 0,
				},
			},
			gfx_hal::pso::AttributeDesc {
				location: 1,
				binding: 0,
				element: gfx_hal::pso::Element {
					format: gfx_hal::format::Format::Rgba8Unorm,
					offset: size_of::<[f32; 2]>() as u32,
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
	pub indices: &'a [[u16; 3]], //TODO: Work out if it needs to be CCW or not
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