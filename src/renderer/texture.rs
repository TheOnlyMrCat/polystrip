//! Images to be rendered onto shapes
use gfx_hal::prelude::*;

use crate::data::GpuPos;
use crate::renderer::Renderer;

/// A texture which can be copied to and rendered by a [`Frame`](../renderer/struct.Frame.html).
/// 
/// It must be used only with the `Renderer` which created it.
#[derive(Debug)]
pub struct Texture {
	texture: super::backend::Image,
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
	///           the sRGB color space.
	/// * `size`: The size of the texture, in pixels, in (width, height) order.
	pub fn new_from_rgba(renderer: &mut Renderer, data: &[u8], (width, height): (u32, u32)) -> Texture {
		// Do this before grabbing the gpu to avoid borrow checker
		let descriptor_set = unsafe { renderer.descriptor_pool.allocate_set(&renderer.texture_descriptor_set_layout) }.unwrap();

		let gpu = renderer.device();

		let mut texture = unsafe { gpu.device.create_image(
			gfx_hal::image::Kind::D2(width, height, 1, 1),
			1,
			gfx_hal::format::Format::Rgba8Srgb,
			gfx_hal::image::Tiling::Linear,
			gfx_hal::image::Usage::SAMPLED,
			gfx_hal::image::ViewCapabilities::empty(),
		)}.unwrap();

		let memory_types = renderer.physical_device().memory_properties().memory_types;

		let req = unsafe { gpu.device.get_image_requirements(&texture) };
		let memory_type = memory_types.iter()
			.enumerate()
			.find(|(id, memory_type)| {
				req.type_mask & (1_u32 << id) != 0 &&
				memory_type.properties.contains(gfx_hal::memory::Properties::CPU_VISIBLE)
			})
			.map(|(id, _)| gfx_hal::MemoryTypeId(id))
			.unwrap();

		let memory = unsafe {gpu.device.allocate_memory(
			memory_type,
			req.size,
		)}.unwrap();

		unsafe {
			let image = gpu.device.map_memory(&memory, gfx_hal::memory::Segment::ALL).unwrap();
			std::ptr::copy_nonoverlapping(data.as_ptr(), image, data.len());
			gpu.device.unmap_memory(&memory);

			gpu.device.bind_image_memory(&memory, 0, &mut texture);
		}

		let view = unsafe { gpu.device.create_image_view(
			&texture,
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
			texture, view, sampler, descriptor_set,
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