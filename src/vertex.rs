//! Vertices and shapes, the core of the rendering process.

use std::cell::RefCell;
use std::rc::Rc;

use gfx_hal::prelude::*;
use gpu_alloc::{MemoryBlock, Request, UsageFlags};

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
	allocated_memory: RefCell<Vec<MemoryBlock<std::sync::Arc<crate::backend::Memory>>>>,
}

impl ShapePool {
	fn create_buffers(&self, vertices: &[u8], indices: &[u8]) -> (crate::backend::Buffer, crate::backend::Buffer) {
		let mut vertex_buffer = unsafe {
			self.context.gpu.device.create_buffer(vertices.len() as u64, gfx_hal::buffer::Usage::VERTEX)
		}.unwrap();
		let mut index_buffer = unsafe {
			self.context.gpu.device.create_buffer(indices.len() as u64, gfx_hal::buffer::Usage::INDEX)
		}.unwrap();
		let vertex_mem_req = unsafe { self.context.gpu.device.get_buffer_requirements(&vertex_buffer) };
		let index_mem_req = unsafe { self.context.gpu.device.get_buffer_requirements(&index_buffer) };

		let memory_device = self.context.get_memory_device();
		let vertex_block = unsafe {
			self.context.allocator.borrow_mut().alloc(
				memory_device,
				Request {
					size: vertex_mem_req.size,
					align_mask: vertex_mem_req.alignment,
					memory_types: vertex_mem_req.type_mask,
					usage: UsageFlags::UPLOAD, // Implies host-visible
				}
			)
		}.unwrap();
		let index_block = unsafe {
			self.context.allocator.borrow_mut().alloc(
				memory_device,
				Request {
					size: index_mem_req.size,
					align_mask: index_mem_req.alignment,
					memory_types: index_mem_req.type_mask,
					usage: UsageFlags::UPLOAD,
				}
			)
		}.unwrap();
		unsafe {
			vertex_block.write_bytes(memory_device, 0, vertices);
			index_block.write_bytes(memory_device, 0, indices);
			self.context.gpu.device.bind_buffer_memory(&vertex_block.memory(), vertex_block.offset(), &mut vertex_buffer);
			self.context.gpu.device.bind_buffer_memory(&index_block.memory(), index_block.offset(), &mut index_buffer);
		}
		let vec = self.allocated_memory.borrow_mut();
		vec.push(vertex_block);
		vec.push(index_block);
		(vertex_buffer, index_buffer)
	}

	/// Creates a [`ColoredShape`](struct.ColoredShape) from raw vertex and index data.
	/// 
	/// * `vertices`: a list of all distinct vertices in the shape, and their colors
	/// * `indices`: indices into `vertices` describing how the vertices arrange into triangles
	pub fn raw_colored(&self, vertices: &[ColorVertex], indices: &[[u16; 3]]) -> ColoredShape {
		let (vertex_buffer, index_buffer) = self.create_buffers(bytemuck::cast_slice(vertices), bytemuck::cast_slice(indices));
		ColoredShape {
			vertex_buffer,
			index_buffer,
			index_count: indices.len() as u32 * 3,
			_marker: std::marker::PhantomData,
		}
	}

	/// Creates a [`TexturedShape`](struct.TexturedShape) from raw vertex and index data
	/// 
	/// * `vertices`: a list of all distinct vertices in the shape, and their colors
	/// * `indices`: indices into `vertices` describing how the vertices arrange into triangles
	pub fn raw_textured(&self, vertices: &[TextureVertex], indices: &[[u16; 3]]) -> TexturedShape {
		let (vertex_buffer, index_buffer) = self.create_buffers(bytemuck::cast_slice(vertices), bytemuck::cast_slice(indices));
		TexturedShape {
			vertex_buffer,
			index_buffer,
			index_count: indices.len() as u32 * 3,
			_marker: std::marker::PhantomData,
		}
	}
}

impl Drop for ShapePool {
	fn drop(&mut self) {
		let allocator = self.context.allocator.borrow_mut();
		let memory_device = self.context.get_memory_device();
		for block in self.allocated_memory.get_mut().drain(..) {
			allocator.dealloc(memory_device, block);
		}
	}
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
///
/// The color of the shape is determined by interpolating the colours at each
/// [`ColorVertex`](struct.ColorVertex).
/// 
/// See also [`TexturedShape`](struct.TexturedShape)
pub struct ColoredShape<'a> {
	pub(crate) vertex_buffer: crate::backend::Buffer,
	pub(crate) index_buffer: crate::backend::Buffer,
	pub(crate) index_count: u32,
	_marker: std::marker::PhantomData<&'a ShapePool>,
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
pub struct TexturedShape<'a> {
	pub(crate) vertex_buffer: crate::backend::Buffer,
	pub(crate) index_buffer: crate::backend::Buffer,
	pub(crate) index_count: u32,
	_marker: std::marker::PhantomData<&'a ShapePool>,
}