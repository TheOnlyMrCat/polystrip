use polystrip::prelude::*;
use polystrip::renderer::Texture;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::WindowBuilder;

fn main() {
	let el = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Geometry")
		.build(&el).unwrap();

	let size = window.inner_size().to_logical(window.scale_factor());
	let mut renderer = Renderer::new(&window, (size.width, size.height));

	let sandstone_img = image::load_from_memory(include_bytes!("sandstone3.png")).unwrap().to_rgba();
	let sandstone = Texture::new_from_rgba(&mut renderer, &*sandstone_img, sandstone_img.dimensions());

	let player_img = image::load_from_memory(include_bytes!("player.png")).unwrap().to_rgba();
	let player = Texture::new_from_rgba(&mut renderer, &*player_img, player_img.dimensions());
	
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
				let mut frame = renderer.get_next_frame();
				frame.draw_rect(Rect { x: 50, y: 50, w: 100, h: 60 }, Color { r: 255, g: 0, b: 0, a: 255 });
				frame.draw_texture_scaled(Rect { x: 70, y: 200, w: 80, h: 120 }, &sandstone);
				frame.draw_texture_scaled(Rect { x: 70, y: 200, w: 80, h: 120 }, &player);
				frame.draw_texture_cropped_scaled(Rect { x: 7, y: 0, w: 7, h: 16 }, Rect { x: 160, y: 200, w: 70, h: 160 }, &sandstone);
			},
			_ => {}
		}
	});
}