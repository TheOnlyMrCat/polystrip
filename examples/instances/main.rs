use polystrip::{PolystripDevice, RenderPipeline, RenderSize, ImageTexture, WindowTarget};
use polystrip::gon::{GonPipeline, PixelTextureVertex, PixelTexturedShape};
use polystrip::math::{Color, Matrix4, Vector2, Vector3};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() {
	let el = EventLoop::new();
	let window = WindowBuilder::new().with_title("Polystrip example (Instanced shapes)").build(&el).unwrap();

	let size = window.inner_size();
	let size_handle = RenderSize::new(size.width, size.height).wrap();
	let mut renderer =
		unsafe { WindowTarget::new(pollster::block_on(PolystripDevice::new()).wrap(), &window, &size_handle) };
	let mut pipeline = GonPipeline::new(&renderer, &renderer);
	let pixel_translator = renderer.pixel_translator();

	let mut shape = PixelTexturedShape::new(
		&renderer,
		vec![
			PixelTextureVertex {
				position: Vector3::new(50.0, 50.0, 0.0),
				tex_coord: Vector2::new(0.0, 0.0),
			},
			PixelTextureVertex {
				position: Vector3::new(150.0, 50.0, 0.0),
				tex_coord: Vector2::new(1.0, 0.0),
			},
			PixelTextureVertex {
				position: Vector3::new(150.0, 150.0, 0.0),
				tex_coord: Vector2::new(1.0, 1.0),
			},
			PixelTextureVertex {
				position: Vector3::new(50.0, 150.0, 0.0),
				tex_coord: Vector2::new(0.0, 1.0),
			},
		],
		vec![[0, 1, 2], [0, 2, 3]],
	);

	let sandstone_img = image::load_from_memory(include_bytes!("sandstone3.png")).unwrap().to_rgba8();
	let sandstone = ImageTexture::new_from_rgba(&renderer, &*sandstone_img, sandstone_img.dimensions());
	let squares_img = image::load_from_memory(include_bytes!("squares.png")).unwrap().to_rgba8();
	let squares = ImageTexture::new_from_rgba(&renderer, &*squares_img, squares_img.dimensions());

	let mut sandstone_matrices = Vec::new();
	let mut squares_matrices = Vec::new();
	for y in 0..10 {
		for x in 0..10 {
			if (x + y) % 2 == 0 {
				squares_matrices.push(Matrix4::translate(pixel_translator.pixel_offset(x * 100, y * 100)))
			} else {
				sandstone_matrices.push(Matrix4::translate(pixel_translator.pixel_offset(x * 100, y * 100)))
			}
		}
	}

	el.run(move |event, _, control_flow| match event {
		Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
			*control_flow = ControlFlow::Exit;
		}
		Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
			size_handle.set(new_size.width, new_size.height);
			sandstone_matrices.clear();
			squares_matrices.clear();
			for y in 0..10 {
				for x in 0..10 {
					if (x + y) % 2 == 0 {
						squares_matrices.push(Matrix4::translate(pixel_translator.pixel_offset(x * 100, y * 100)))
					} else {
						sandstone_matrices.push(Matrix4::translate(pixel_translator.pixel_offset(x * 100, y * 100)))
					}
				}
			}
		}
		Event::MainEventsCleared => {
			let mut frame = renderer.next_frame();
			let mut frame = pipeline.render_to(&mut frame);
			frame.clear(Color { r: 128, g: 128, b: 128, a: 255 });
			frame.draw(&shape.gpu_shape(&frame).with_instances(&[(sandstone.sampled(), &sandstone_matrices), (squares.sampled(), &squares_matrices)]));
		}
		_ => {}
	});
}

#[cfg(test)]
#[test]
fn instanced_drawing() {
	use polystrip::RenderTarget;

	let expected_output = image::load_from_memory(include_bytes!("expected.png")).unwrap().to_rgba();
	let renderer = PolystripDevice::new().wrap();
	let size_handle = RenderSize::new(1100, 1100).wrap();
	let mut pipeline = GonPipeline::new(&renderer, &size_handle);
	let mut texture = ImageTexture::new_solid_color(&renderer, Color::ZERO, (1100, 1100));
	let pixel_translator = texture.pixel_translator();

	let shape = renderer.create_textured(
		&[
			GpuTextureVertex { position: frame.pixel(50, 50).with_height(0.0), tex_coord: Vector2::new(0.0, 0.0) },
			GpuTextureVertex { position: frame.pixel(50, 150).with_height(0.0), tex_coord: Vector2::new(0.0, 1.0) },
			GpuTextureVertex { position: frame.pixel(150, 150).with_height(0.0), tex_coord: Vector2::new(1.0, 1.0) },
			GpuTextureVertex { position: frame.pixel(150, 50).with_height(0.0), tex_coord: Vector2::new(1.0, 0.0) },
		][..],
		&[[0, 1, 3], [1, 2, 3]][..],
	);

	let sandstone_img = image::load_from_memory(include_bytes!("sandstone3.png")).unwrap().to_rgba();
	let sandstone = ImageTexture::new_from_rgba(&renderer, &*sandstone_img, sandstone_img.dimensions());
	let squares_img = image::load_from_memory(include_bytes!("squares.png")).unwrap().to_rgba();
	let squares = ImageTexture::new_from_rgba(&renderer, &*squares_img, squares_img.dimensions());

	let mut sandstone_matrices = Vec::new();
	let mut squares_matrices = Vec::new();
	for y in 0..10 {
		for x in 0..10 {
			if (x + y) % 2 == 0 {
				squares_matrices.push(Matrix4::translate(pixel_translator.pixel_offset(x * 100, y * 100)))
			} else {
				sandstone_matrices.push(Matrix4::translate(pixel_translator.pixel_offset(x * 100, y * 100)))
			}
		}
	}

	let mut frame = texture.create_frame().render_with(&mut pipeline);
	frame.draw_textured(&shape, &[(&sandstone, &sandstone_matrices), (&squares, &squares_matrices)]);
	frame.present();

	assert_eq!(*texture.get_data(), *expected_output);
}
