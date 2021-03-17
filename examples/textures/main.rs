use polystrip::{Renderer, Texture, WindowTarget};
use polystrip::vertex::{TexturedShape, TextureVertex, Color, Vector2, Matrix4};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::WindowBuilder;

fn main() {
	let el = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Textures")
		.build(&el).unwrap();

	let size = window.inner_size().to_logical(window.scale_factor());
	let mut renderer = WindowTarget::new(Renderer::new().wrap(), &window, (size.width, size.height));

	let sandstone_img = image::load_from_memory(include_bytes!("sandstone3.png")).unwrap().to_rgba();
	let sandstone = Texture::new_from_rgba(&renderer, &*sandstone_img, sandstone_img.dimensions());

	assert_eq!(*sandstone_img, *sandstone.get_data());

	let mut matrices = Vec::new();
	for y in 0..10 {
		for x in 0..16 {
			matrices.push(Matrix4::translate(renderer.pixel(x * 100, y * 100)))
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
				matrices = Vec::new();
				for y in 0..10 {
					for x in 0..16 {
						matrices.push(Matrix4::translate(renderer.pixel(x * 100, y * 100)))
					}
				}
			},
			Event::MainEventsCleared => {
				let mut frame = renderer.next_frame_clear(Color { r: 128, g: 128, b: 128, a: 255 });
				frame.draw_textured(TexturedShape {
					vertices: &[
						TextureVertex { position: frame.pixel(50, 50).with_height(0.0), tex_coords: Vector2::new(0.0, 0.0) },
						TextureVertex { position: frame.pixel(50, 150).with_height(0.0), tex_coords: Vector2::new(0.0, 1.0) },
						TextureVertex { position: frame.pixel(150, 150).with_height(0.0), tex_coords: Vector2::new(1.0, 1.0) },
						TextureVertex { position: frame.pixel(150, 50).with_height(0.0), tex_coords: Vector2::new(1.0, 0.0) },
					],
					indices: &[
						[0, 1, 3],
						[1, 2, 3],
					]
				}, &sandstone, &matrices);
			},
			_ => {}
		}
	});
}