//! Images to be rendered onto shapes
use gfx_hal::prelude::*;

use crate::data::GpuPos;
use crate::{Renderer, RendererContext};

/// A texture which can be copied to and rendered by a [`Frame`](../renderer/struct.Frame.html).
/// 
/// It must be used only with the `Renderer` which created it.
#[derive(Debug)]
pub struct Texture {
	image: super::backend::Image,
	view: super::backend::ImageView,
	sampler: super::backend::Sampler,
	pub(super) descriptor_set: super::backend::DescriptorSet,
	width: u32,
	height: u32,
}

impl Texture {
	/// Create a new texture from the given rgba data, associated with the given `Renderer`.
	/// 
	/// # Arguments
	/// * `renderer`: The `Renderer` to create this texture for. 
	/// * `data`: A reference to a byte array containing the pixel data. The data must be formatted to `Rgba8` in
	///           the sRGB color space, in row-major order.
	/// * `size`: The size of the texture, in pixels, in (width, height) order.
	pub fn new_from_rgba(renderer: &mut Renderer, data: &[u8], (width, height): (u32, u32)) -> Texture {
		// Do this before grabbing the gpu to avoid borrow checker
		let descriptor_set = unsafe { renderer.descriptor_pool.allocate_set(&renderer.texture_descriptor_set_layout) }.unwrap();
		let mut command_buffer = unsafe { renderer.command_pool().allocate_one(gfx_hal::command::Level::Primary) };
		let memory_types = renderer.physical_device().memory_properties().memory_types;

		let gpu = renderer.device();

		let mut image = unsafe { gpu.device.create_image(
			gfx_hal::image::Kind::D2(width, height, 1, 1),
			1,
			gfx_hal::format::Format::Rgba8Srgb,
			gfx_hal::image::Tiling::Optimal,
			gfx_hal::image::Usage::TRANSFER_DST | gfx_hal::image::Usage::SAMPLED,
			gfx_hal::image::ViewCapabilities::empty(),
		)}.unwrap();


		let req = unsafe { gpu.device.get_image_requirements(&image) };
		let memory_type = memory_types.iter()
			.enumerate()
			.find(|(id, memory_type)| {
				req.type_mask & (1_u32 << id) != 0 &&
				memory_type.properties.contains(gfx_hal::memory::Properties::DEVICE_LOCAL)
			})
			.map(|(id, _)| gfx_hal::MemoryTypeId(id))
			.unwrap();

		let image_memory = unsafe { gpu.device.allocate_memory(
			memory_type,
			req.size,
		)}.unwrap();

		unsafe {
			gpu.device.bind_image_memory(&image_memory, 0, &mut image).unwrap();
		}

		let mut image_buffer = unsafe { gpu.device.create_buffer(req.size, gfx_hal::buffer::Usage::TRANSFER_SRC) }.unwrap();
		let req = unsafe { gpu.device.get_buffer_requirements(&image_buffer) };
		let memory_type = memory_types.iter()
			.enumerate()
			.find(|(id, memory_type)| {
				req.type_mask & (1_u32 << id) != 0 &&
				memory_type.properties.contains(gfx_hal::memory::Properties::CPU_VISIBLE)
			})
			.map(|(id, _)| gfx_hal::MemoryTypeId(id))
			.unwrap();
		
		let buffer_memory = unsafe { gpu.device.allocate_memory(
			memory_type,
			req.size,
		)}.unwrap();

		unsafe {
			gpu.device.bind_buffer_memory(&buffer_memory, 0, &mut image_buffer).unwrap();

			let mapped_memory = gpu.device.map_memory(&buffer_memory, gfx_hal::memory::Segment::ALL).unwrap();
			std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_memory, data.len());
			gpu.device.flush_mapped_memory_ranges(&[(&buffer_memory, gfx_hal::memory::Segment::ALL)]).unwrap();
			gpu.device.unmap_memory(&buffer_memory);

			command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TOP_OF_PIPE..gfx_hal::pso::PipelineStage::TRANSFER,
				gfx_hal::memory::Dependencies::empty(),
				&[gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::empty(), gfx_hal::image::Layout::Undefined)
						..
						(gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal),
					target: &image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
			command_buffer.copy_buffer_to_image(
				&image_buffer,
				&image,
				gfx_hal::image::Layout::TransferDstOptimal,
				&[gfx_hal::command::BufferImageCopy {
					buffer_offset: 0,
					buffer_width: width,
					buffer_height: height,
					image_layers: gfx_hal::image::SubresourceLayers {
						aspects: gfx_hal::format::Aspects::COLOR,
						level: 1,
						layers: 0..1,
					},
					image_offset: gfx_hal::image::Offset::ZERO,
					image_extent: gfx_hal::image::Extent {
						width, height,
						depth: 1,
					}
				}]
			);
			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TRANSFER..gfx_hal::pso::PipelineStage::FRAGMENT_SHADER,
				gfx_hal::memory::Dependencies::empty(),
				&[gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal)
						..
						(gfx_hal::image::Access::SHADER_READ, gfx_hal::image::Layout::ShaderReadOnlyOptimal),
					target: &image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
			command_buffer.finish();

			let fence = gpu.device.create_fence(false).unwrap();
			gpu.queue_groups[0].queues[0].submit_without_semaphores(&[command_buffer], Some(&fence));
			gpu.device.wait_for_fence(&fence, u64::MAX).unwrap();

			gpu.device.destroy_fence(fence);
		}

		let view = unsafe { gpu.device.create_image_view(
			&image,
			gfx_hal::image::ViewKind::D2,
			gfx_hal::format::Format::Rgba8Srgb,
			gfx_hal::format::Swizzle::NO,
			gfx_hal::image::SubresourceRange {
				aspects: gfx_hal::format::Aspects::COLOR,
				level_start: 1,
				level_count: None,
				layer_start: 1,
				layer_count: None,
			},
		)}.unwrap();

		let sampler = unsafe { gpu.device.create_sampler(&gfx_hal::image::SamplerDesc::new(gfx_hal::image::Filter::Nearest, gfx_hal::image::WrapMode::Tile)) }.unwrap();

		unsafe {
			gpu.device.write_descriptor_sets(vec![gfx_hal::pso::DescriptorSetWrite {
				set: &descriptor_set,
				binding: 0,
				array_offset: 0,
				descriptors: &[
					gfx_hal::pso::Descriptor::Image(&view, gfx_hal::image::Layout::General),
					gfx_hal::pso::Descriptor::Sampler(&sampler),
				]
			}])
		}

		Texture {
			image, view, sampler, descriptor_set,
			width, height,
		}
	}

	/// Get the dimensions of this texture, in (width, height) order.
	pub fn dimensions(&self) -> (u32, u32) {
		(self.width, self.height)
	}

	pub fn width(&self) -> u32 {
		self.width
	}

	pub fn height(&self) -> u32 {
		self.height
	}

	/// Converts pixel coordinates to Gpu coordinates
	pub fn pixel(&self, x: i32, y: i32) -> GpuPos {
		GpuPos {
			x: x as f32 / self.width as f32,
			y: y as f32 / self.height as f32,
		}
	}
}