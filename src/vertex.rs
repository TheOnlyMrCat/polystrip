//! Vertices and shapes, the core of the rendering process.

use std::rc::Rc;

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

pub struct ShapePool {
	context: Rc<crate::RendererContext>,
}

impl ShapePool {
	/// Creates a [`ColoredShape`](struct.ColoredShape) from raw vertex and index data.
	/// 
	/// * `indices`: indices into `vertices` describing how the vertices arrange into triangles
	fn raw_colored(vertices: &[ColorVertex], indices: &[[u16; 3]]) -> ColoredShape {
		todo!();
	}

	fn raw_textured(vertices: &[TextureVertex], indices: &[[u16; 3]]) -> TexturedShape {
		todo!();
	}
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
///
/// The color of the shape is determined by interpolating the colours at each
/// [`ColorVertex`](struct.ColorVertex).
/// 
/// See also [`TexturedShape`](struct.TexturedShape)
pub struct ColoredShape {
	pub(crate) vertex_buffer: crate::backend::Buffer,
	pub(crate) index_buffer: crate::backend::Buffer,
	pub(crate) index_count: u32,
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
/// 
/// The color of the shape is determined by interpolating the texture coordinates at each
/// [`TextureVertex`](struct.TextureVertex).
/// 
/// A `TexturedShape` does not store the texture it is to draw. This must be specified in the
/// arguments to [`Frame::draw_textured`](../renderer/struct.Frame#method.draw_textured)
/// 
/// See also [`ColoredShape`](struct.ColoredShape)
pub struct TexturedShape {
	pub(crate) vertex_buffer: crate::backend::Buffer,
	pub(crate) index_buffer: crate::backend::Buffer,
	pub(crate) index_count: u32,
}