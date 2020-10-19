//! Vertices and shapes, the core of the rendering process.

use crate::data::*;
use crate::texture::Texture;

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
	pub(crate) fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
		use std::mem::size_of;
		
		wgpu::VertexBufferDescriptor {
			stride: size_of::<TextureVertex>() as wgpu::BufferAddress,
			step_mode: wgpu::InputStepMode::Vertex,
			attributes: &[
				wgpu::VertexAttributeDescriptor {
					offset: 0,
					shader_location: 0,
					format: wgpu::VertexFormat::Float2,
				},
				wgpu::VertexAttributeDescriptor {
					offset: size_of::<[f32; 2]>() as wgpu::BufferAddress,
					shader_location: 1,
					format: wgpu::VertexFormat::Float2,
				},
			]
		}
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
	pub(crate) fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
		use std::mem::size_of;
		
		wgpu::VertexBufferDescriptor {
			stride: size_of::<ColorVertex>() as wgpu::BufferAddress,
			step_mode: wgpu::InputStepMode::Vertex,
			attributes: &[
				wgpu::VertexAttributeDescriptor {
					offset: 0,
					shader_location: 0,
					format: wgpu::VertexFormat::Float2,
				},
				wgpu::VertexAttributeDescriptor {
					offset: size_of::<[f32; 2]>() as wgpu::BufferAddress,
					shader_location: 1,
					format: wgpu::VertexFormat::Uchar4Norm,
				},
			]
		}
	}
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
///
/// The color of the shape is determined by interpolating the colours at each
/// [`ColorVertex`](struct.ColorVertex).
/// 
/// See also [`TexturedShape`](struct.TexturedShape)
#[derive(Clone, Debug)]
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
/// A `TexturedShape` does not store the texture it is to draw. This must be specified in the [`ShapeSet`](struct.ShapeSet)
/// or in the arguments to [`Frame::add_textured`](../renderer/struct.Frame#method.add_textured)
/// 
/// See also [`ColoredShape`](struct.ColoredShape)
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TexturedShape<'a> {
	pub vertices: &'a [TextureVertex],
	/// A list of sets of three vertices which specify how the vertices should be rendered as triangles.
	pub indices: &'a [[u16; 3]], //TODO: As above
}

/// A set of [`ColoredShape`](struct.ColoredShape)s or [`TexturedShape`](struct.TexturedShape)s.
#[derive(Debug)]
pub enum ShapeSet<'a> {
	/// Multiple [`ColoredShape`](struct.ColoredShape)s.
	Colored(&'a [ColoredShape<'a>]),
	/// Multiple [`TexturedShape`](struct.TexturedShape)s, with a reference to the texture to draw to them.
	Textured(&'a [TexturedShape<'a>], &'a Texture),
}