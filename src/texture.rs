//! Images to be rendered onto shapes

use crate::renderer::Renderer;

/// A texture which can be copied to and rendered by a [`Frame`](../renderer/struct.Frame.html).
/// 
/// It must be used only with the `Renderer` which created it.
#[derive(Debug)]
pub struct Texture {
	texture: wgpu::Texture,
	view: wgpu::TextureView,
	sampler: wgpu::Sampler,
	pub(crate) bind_group: wgpu::BindGroup,
}

impl Texture {
	/// Create a new texture from the given rgba data, associated with the given `Renderer`.
	/// 
	/// # Arguments
	/// * `renderer`: The `Renderer` to create this texture for. 
	/// * `data`: A reference to a byte array containing the pixel data. The data must be formatted to `Rgba8` in
	///           the sRGB color space.
	/// * `size`: The size of the texture, in pixels, in (width, height) order.
	pub fn new_from_rgba(renderer: &Renderer, data: &[u8], size: (u32, u32)) -> Texture {
		let extent = wgpu::Extent3d {
			width: size.0,
			height: size.1,
			depth: 1,
		};

		let texture = renderer.device.create_texture(&wgpu::TextureDescriptor {
			label: None,
			size: extent,
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Rgba8UnormSrgb,
			usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
		});

		renderer.queue.write_texture(
			wgpu::TextureCopyView {
				texture: &texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
			},
			data.as_ref(),
			wgpu::TextureDataLayout {
				offset: 0,
				bytes_per_row: 4 * size.0,
				rows_per_image: size.1,
			},
			extent,
		);

		let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
		let sampler = renderer.device.create_sampler(&wgpu::SamplerDescriptor {
			label: None,
			address_mode_u: wgpu::AddressMode::Repeat,
			address_mode_v: wgpu::AddressMode::Repeat,
			address_mode_w: wgpu::AddressMode::Repeat,
			mag_filter: wgpu::FilterMode::Nearest,
			min_filter: wgpu::FilterMode::Nearest,
			mipmap_filter: wgpu::FilterMode::Nearest,
			..Default::default()
		});

		let bind_group = renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &renderer.texture_bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: wgpu::BindingResource::TextureView(&view),
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: wgpu::BindingResource::Sampler(&sampler),
				}
			]
		});

		Texture {
			texture, view, sampler, bind_group
		}
	}
}