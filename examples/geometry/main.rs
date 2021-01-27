use polystrip::Renderer;
use polystrip::data::{GpuVec3, Color};
use polystrip::vertex::{StrokedShape, ColoredShape, ColorVertex, Matrix3};

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
				*control_flow = ControlFlow::Exit;
			},
			Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
				let window_size = new_size.to_logical(window.scale_factor());
				renderer.resize((window_size.width, window_size.height));
			},
			Event::MainEventsCleared => {
				let mut frame = renderer.next_frame_clear(Color { r: 128, g: 128, b: 128, a: 255 });
				frame.draw_stroked(StrokedShape {
					vertices: &[
						ColorVertex { position: GpuVec3 { x: 0.5, y: 0.5, h: 0.0 }, color: Color { r: 255, g: 255, b: 255, a: 255 }},
						ColorVertex { position: GpuVec3 { x: 0.125, y: 0.125, h: 1.0 }, color: Color { r: 255, g: 255, b: 255, a: 255 }},
						ColorVertex { position: GpuVec3 { x: 0.25, y: -0.5, h: 1.0 }, color: Color { r: 255, g: 255, b: 255, a: 255 }},
						ColorVertex { position: GpuVec3 { x: 0.75, y: -0.5, h: 1.0 }, color: Color { r: 255, g: 255, b: 255, a: 255 }},
						ColorVertex { position: GpuVec3 { x: 0.875, y: 0.125, h: 1.0 }, color: Color { r: 255, g: 255, b: 255, a: 255 }},
					],
					indices: &[
						[0, 1], [0, 2], [0, 3], [0, 4],
						[1, 2], [1, 3], [1, 4],
						[2, 3], [2, 4],
						[3, 4],
					]
				}, Matrix3::identity());
				frame.draw_colored(ColoredShape {
					vertices: &[
						ColorVertex { position: GpuVec3 { x: 0.5, y: 0.5, h: 1.0 }, color: Color { r: 255, g: 0, b: 0, a: 255 }},
						ColorVertex { position: GpuVec3 { x: 0.125, y: 0.125, h: 0.0 }, color: Color { r: 255, g: 255, b: 0, a: 255 }},
						ColorVertex { position: GpuVec3 { x: 0.25, y: -0.5, h: 0.0 }, color: Color { r: 0, g: 255, b: 0, a: 255 }},
						ColorVertex { position: GpuVec3 { x: 0.75, y: -0.5, h: 0.0 }, color: Color { r: 0, g: 0, b: 255, a: 255 }},
						ColorVertex { position: GpuVec3 { x: 0.875, y: 0.125, h: 0.0 }, color: Color { r: 255, g: 0, b: 255, a: 255 }},
					],
					// Note the vertices are specified going counter-clockwise
					indices: &[
						[0, 1, 4],
						[1, 2, 4],
						[2, 3, 4],
					]
				}, Matrix3::identity());
				frame.draw_colored(ColoredShape {
					vertices: &[
						ColorVertex { position: GpuVec3 { x: -0.5, y: 0.5, h: 0.0 }, color: Color { r: 255, g: 0, b: 0, a: 255 }},
						ColorVertex { position: GpuVec3 { x: -0.75, y: -0.5, h: 0.0 }, color: Color { r: 0, g: 255, b: 0, a: 255 }},
						ColorVertex { position: GpuVec3 { x: -0.25, y: -0.5, h: 0.0 }, color: Color { r: 0, g: 0, b: 255, a: 255 }},
					],
					indices: &[[0, 1, 2]]
				}, Matrix3::identity());
			},
			_ => {}
		}
	});
}