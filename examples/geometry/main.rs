use polystrip::prelude::*;
use polystrip::vertex::{Shape, ColorVertex};

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
	
	el.run(move |event, _, control_flow| {
		match event {
			Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
				*control_flow = ControlFlow::Exit
			},
			Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
				let window_size = new_size.to_logical(window.scale_factor());
				renderer.resize((window_size.width, window_size.height));
			}
			Event::MainEventsCleared => {
				let mut frame = Frame::new();
				frame.set_clear(Color { r: 128, g: 128, b: 128 });
				frame.push_shape(Shape::Colored {
					vertices: vec![
						ColorVertex { position: Vec2 { x: -0.5, y: 0.5 }, color: Color { r: 255, g: 0, b: 0 }},
						ColorVertex { position: Vec2 { x: -0.75, y: -0.5 }, color: Color { r: 0, g: 255, b: 0 }},
						ColorVertex { position: Vec2 { x: -0.25, y: -0.5 }, color: Color { r: 0, g: 0, b: 255 }},
					],
					indices: vec![(0, 1, 2)]
				});
				frame.push_shape(Shape::Colored {
					vertices: vec![
						ColorVertex { position: Vec2 { x: 0.5, y: 0.5 }, color: Color { r: 255, g: 0, b: 0 }},
						ColorVertex { position: Vec2 { x: 0.125, y: 0.125 }, color: Color { r: 255, g: 255, b: 0 }},
						ColorVertex { position: Vec2 { x: 0.25, y: -0.5 }, color: Color { r: 0, g: 255, b: 0 }},
						ColorVertex { position: Vec2 { x: 0.75, y: -0.5 }, color: Color { r: 0, g: 0, b: 255 }},
						ColorVertex { position: Vec2 { x: 0.875, y: 0.125 }, color: Color { r: 255, g: 0, b: 255 }},
					],
					indices: vec![
						(0, 1, 4),
						(1, 2, 4),
						(2, 3, 4),
					]
				});
				renderer.render_frame(&frame);
			},
			_ => {}
		}
	});
}