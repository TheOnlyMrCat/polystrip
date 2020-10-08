use crate::renderer::Renderer;

pub struct Texture {
	texture: wgpu::Texture,
	view: wgpu::TextureView,
	sampler: wgpu::Sampler,
	bind_group: wgpu::BindGroup,
}

impl Texture {
	fn new_from_rgba(renderer: &Renderer, data: impl AsRef<[u8]>, size: (u32, u32)) {
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
			usage: wgpu::TextureUsage::SAMPLED,
		});

		renderer.queue.write_texture(
			wgpu::TextureCopyView {
				texture: &diffuse_texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
			},
			data.as_ref(),
			wgpu::TextureDataLayout {
				offset: 0,
				bytes_per_row: 4 * dimensions.0,
				rows_per_image: dimensions.1,
			},
			extent,
		);

		let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
		let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			label: None,
			address_mode_u: wgpu::AddressMode::Repeat,
			address_mode_v: wgpu::AddressMode::Repeat,
			address_mode_w: wgpu::AddressMode::Repeat,
			mag_filter: wgpu::FilterMode::Nearest,
			min_filter: wgpu::FilterMode::Nearest,
			mipmap_filter: wgpu::FilterMode::Nearest,
			..Default::default()
		});

		let bind_group = device.create_bind_group(
			&wgpu::BindGroupDescriptor {
				label: None,
				layout: &texture_bind_group_layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
					}
				],
			}
		);

		Texture {
			texture, view, sampler, bind_group
		}
	}
}