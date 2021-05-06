use polystrip::{Renderer, Texture, WindowTarget};
use polystrip::vertex::{TexturedShape, TextureVertex, Color, Vector2, Matrix4};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::WindowBuilder;

fn main() {
	let el = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Polystrip example (textures)")
		.build(&el).unwrap();

	let size = window.inner_size().to_logical(window.scale_factor());
	let mut renderer = WindowTarget::new(Renderer::new().wrap(), &window, (size.width, size.height));
	let pixel_translator = renderer.pixel_translator();

	let sandstone_img = image::load_from_memory(include_bytes!("sandstone3.png")).unwrap().to_rgba();
	let sandstone = Texture::new_from_rgba(&renderer, &*sandstone_img, sandstone_img.dimensions());
	let squares_img = image::load_from_memory(include_bytes!("squares.png")).unwrap().to_rgba();
	let squares = Texture::new_from_rgba(&renderer, &*squares_img, squares_img.dimensions());

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

	el.run(move |event, _, control_flow| {
		match event {
			Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
				*control_flow = ControlFlow::Exit;
			},
			Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
				let window_size = new_size.to_logical(window.scale_factor());
				renderer.resize((window_size.width, window_size.height));
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
			},
			Event::MainEventsCleared => {
				let mut frame = renderer.next_frame_clear(Color { r: 128, g: 128, b: 128, a: 255 });
				frame.draw_textured(
					TexturedShape {
						vertices: [
							TextureVertex { position: frame.pixel(50, 50).with_height(0.0), tex_coords: Vector2::new(0.0, 0.0) },
							TextureVertex { position: frame.pixel(50, 150).with_height(0.0), tex_coords: Vector2::new(0.0, 1.0) },
							TextureVertex { position: frame.pixel(150, 150).with_height(0.0), tex_coords: Vector2::new(1.0, 1.0) },
							TextureVertex { position: frame.pixel(150, 50).with_height(0.0), tex_coords: Vector2::new(1.0, 0.0) },
						][..].into(),
						indices: [
							[0, 1, 3],
							[1, 2, 3],
						][..].into()
					},
					&[
						(&sandstone, &sandstone_matrices),
						(&squares, &squares_matrices),
					]
				);
			},
			_ => {}
		}
	});
}

#[cfg(test)]
#[test]
fn instanced_drawing() {
	use polystrip::RenderTarget;

	let expected_output = image::load_from_memory(include_bytes!("expected.png")).unwrap().to_rgba();
	let renderer = Renderer::new().wrap();
	let mut texture = Texture::new_solid_color(&renderer, Color::ZERO, (1100, 1100));
	let pixel_translator = texture.pixel_translator();

	let sandstone_img = image::load_from_memory(include_bytes!("sandstone3.png")).unwrap().to_rgba();
	let sandstone = Texture::new_from_rgba(&renderer, &*sandstone_img, sandstone_img.dimensions());
	let squares_img = image::load_from_memory(include_bytes!("squares.png")).unwrap().to_rgba();
	let squares = Texture::new_from_rgba(&renderer, &*squares_img, squares_img.dimensions());

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

	let mut frame = texture.create_frame();
	frame.draw_textured(
		TexturedShape {
			vertices: [
				TextureVertex { position: frame.pixel(50, 50).with_height(0.0), tex_coords: Vector2::new(0.0, 0.0) },
				TextureVertex { position: frame.pixel(50, 150).with_height(0.0), tex_coords: Vector2::new(0.0, 1.0) },
				TextureVertex { position: frame.pixel(150, 150).with_height(0.0), tex_coords: Vector2::new(1.0, 1.0) },
				TextureVertex { position: frame.pixel(150, 50).with_height(0.0), tex_coords: Vector2::new(1.0, 0.0) },
			][..].into(),
			indices: [
				[0, 1, 3],
				[1, 2, 3],
			][..].into()
		},
		&[
			(&sandstone, &sandstone_matrices),
			(&squares, &squares_matrices),
		]
	);
	frame.present();

	image::ImageBuffer::<image::Rgba<_>, _>::from_raw(1100, 1100, texture.get_data()).unwrap().save("expected.png").unwrap();

	assert_eq!(
		*texture.get_data(),
		*expected_output
	);
}