//! Vertices and shapes, the core of the rendering process.

use crate::data::*;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct Vertex {
	position: [f32; 3],
	color: [f32; 3],
	tex_coords: [f32; 2],
	texture_index: u32,
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

impl Vertex {
	pub fn from_texture(vertex: TextureVertex, texture_index: u32, depth: f32) -> Vertex {
		Vertex {
			position: [vertex.position.x, vertex.position.y, depth],
			color: [0.0; 3],
			tex_coords: [vertex.tex_coords.x, vertex.tex_coords.y],
			texture_index,
		}
	}

	pub fn from_color(vertex: ColorVertex, depth: f32) -> Vertex {
		Vertex {
			position: [vertex.position.x, vertex.position.y, depth],
			//TODO: This is not how to convert srgb to linear colour
			color: [f32::from(vertex.color.r) / 255.0, f32::from(vertex.color.g) / 255.0, f32::from(vertex.color.b) / 255.0],
			tex_coords: [0.0; 2],
			texture_index: u32::MAX,
		}
	}

	pub(crate) fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
		wgpu::VertexBufferDescriptor {
			stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
			step_mode: wgpu::InputStepMode::Vertex,
			attributes: &[
				wgpu::VertexAttributeDescriptor {
					offset: 0,
					shader_location: 0,
					format: wgpu::VertexFormat::Float3,
				},
				wgpu::VertexAttributeDescriptor {
					offset: size_of::<[f32; 3]>() as wgpu::BufferAddress,
					shader_location: 1,
					format: wgpu::VertexFormat::Float3,
				},
				wgpu::VertexAttributeDescriptor {
					offset: (size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
					shader_location: 2,
					format: wgpu::VertexFormat::Float2,
				},
				wgpu::VertexAttributeDescriptor {
					offset: (size_of::<[f32; 3]>() * 2 + size_of::<[f32; 2]>()) as wgpu::BufferAddress,
					shader_location: 3,
					format: wgpu::VertexFormat::Uint,
				},
			]
		}
	}
}

/// A vertex describing a position and a position on a texture.
/// 
/// Texture coordinates are interpolated linearly between vertices.
#[derive(Copy, Clone, Debug)]
pub struct TextureVertex {
	pub position: Vec2,
	pub tex_coords: Vec2,
}

/// A vertex describing a position and a colour.
/// 
/// Colours are interpolated linearly between vertices.
#[derive(Copy, Clone, Debug)]
pub struct ColorVertex {
	pub position: Vec2,
	pub color: Color,
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
///
/// The `vertices` field describes the points on the shape, the `indices` field describes how to convert the vertices into
/// triangles. The number of indices must be a mutiple of 3.
#[derive(Clone, Debug)]
pub enum Shape {
	/// A shape whose color is determined by interpolating the colours at each vertex.
	Colored {
		vertices: Vec<ColorVertex>,
		indices: Vec<(u16, u16, u16)>,
	},
	/// A shape whose color is determined by sampling a texture at specified points.
	Textured {
		vertices: Vec<TextureVertex>,
		indices: Vec<(u16, u16, u16)>,
		texture_index: u32,
	},
}