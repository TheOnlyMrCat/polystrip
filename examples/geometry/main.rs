use polystrip::Renderer;
use polystrip::data::GpuPos;
use polystrip::vertex::{ColoredShape, ColorVertex};

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
	let shape_pool = renderer.create_shape_pool();
	
	let pentagon = shape_pool.raw_colored(
		&[
			ColorVertex { position: GpuPos { x: 0.5, y: 0.5 }, color: Color { r: 255, g: 0, b: 0, a: 255 }},
			ColorVertex { position: GpuPos { x: 0.125, y: 0.125 }, color: Color { r: 255, g: 255, b: 0, a: 255 }},
			ColorVertex { position: GpuPos { x: 0.25, y: -0.5 }, color: Color { r: 0, g: 255, b: 0, a: 255 }},
			ColorVertex { position: GpuPos { x: 0.75, y: -0.5 }, color: Color { r: 0, g: 0, b: 255, a: 255 }},
			ColorVertex { position: GpuPos { x: 0.875, y: 0.125 }, color: Color { r: 255, g: 0, b: 255, a: 255 }},
		],
		// Note the vertices are specified going counter-clockwise
		&[
			[0, 1, 4],
			[1, 2, 4],
			[2, 3, 4],
		]
	);
	
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
				frame.clear(Color { r: 128, g: 128, b: 128, a: 255 });
				frame.draw_colored();
				frame.draw_colored(ColoredShape {
					vertices: &[
						ColorVertex { position: GpuPos { x: -0.5, y: 0.5 }, color: Color { r: 255, g: 0, b: 0, a: 255 }},
						ColorVertex { position: GpuPos { x: -0.75, y: -0.5 }, color: Color { r: 0, g: 255, b: 0, a: 255 }},
						ColorVertex { position: GpuPos { x: -0.25, y: -0.5 }, color: Color { r: 0, g: 0, b: 255, a: 255 }},
					],
					indices: &[[0, 1, 2]]
				});
			},
			_ => {}
		}
	});
}