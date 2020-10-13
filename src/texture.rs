use crate::renderer::Renderer;

/// A texture which can be copied to and rendered by a [`Frame`](../renderer/struct.Frame.html). It should be used
/// only with the renderer which created it. (It is not known what happens otherwise, but it is not undefined)
#[repr(transparent)]
pub struct Texture(u32);

impl Texture {
	pub fn new_from_rgba(renderer: &mut Renderer, data: impl AsRef<[u8]>, size: (u32, u32)) -> Texture {
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
		
		let idx = renderer.textures.len();
		if idx != renderer.texture_views.len() {
			panic!("Renderer texture state not self-consistent. This is a bug in polystrip; please contact the developer");
		}

		renderer.textures.push(texture);
		renderer.texture_views.push(view);

		Texture(idx as u32)
	}
}