use polystrip::{Renderer, Texture, WindowTarget};
use polystrip::vertex::{Color, ColoredShape, Matrix4, Rect, TexturedShape};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::WindowBuilder;

const RECT_INDICES: [[u16; 3]; 2] = [[0, 3, 1], [1, 3, 2]];

fn main() {
	let el = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Geometry")
		.build(&el).unwrap();

	let size = window.inner_size().to_logical(window.scale_factor());
	let mut renderer = WindowTarget::new(Renderer::new().wrap(), &window, (size.width, size.height));
	let pixel_translator = renderer.pixel_translator();

	let sandstone_img = image::load_from_memory(include_bytes!("sandstone3.png")).unwrap().to_rgba();
	let sandstone = Texture::new_from_rgba(&renderer, &*sandstone_img, sandstone_img.dimensions());

	let player_img = image::load_from_memory(include_bytes!("player.png")).unwrap().to_rgba();
	let player = Texture::new_from_rgba(&renderer, &*player_img, player_img.dimensions());
	
	el.run(move |event, _, control_flow| {
		match event {
			Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
				*control_flow = ControlFlow::Exit;
			},
			Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
				let window_size = new_size.to_logical(window.scale_factor());
				renderer.resize((window_size.width, window_size.height));
			},
			Event::MainEventsCleared => {
				let mut frame = renderer.next_frame();
				frame.draw_colored(ColoredShape {
					vertices: &pixel_translator.colored_rect(Rect { x: 50, y: 50, w: 100, h: 60 }, Color { r: 255, g: 0, b: 0, a: 255 }),
					indices: &RECT_INDICES,
				}, &[Matrix4::identity()]);
				frame.draw_textured(TexturedShape {
					vertices: &pixel_translator.textured_rect(Rect { x: 70, y: 200, w: 80, h: 120 }),
					indices: &RECT_INDICES,
				}, &sandstone, &[Matrix4::identity()]);
				frame.draw_textured(TexturedShape {
					vertices: &pixel_translator.texture_scaled(&player, 70, 200, 6.0),
					indices: &RECT_INDICES,
				}, &player, &[Matrix4::identity()]);
				frame.draw_textured(TexturedShape {
					vertices: &pixel_translator.texture_scaled_cropped(&sandstone, Rect { x: 160, y: 200, w: 70, h: 160 }, Rect { x: 7, y: 0, w: 7, h: 16 }),
					indices: &RECT_INDICES,
				}, &sandstone, &[Matrix4::identity()]);
			},
			_ => {}
		}
	});
}