use polystrip::{RendererBuilder, WindowTarget};
use polystrip::vertex::{StrokedShape, ColoredShape, ColorVertex, Color, Vector2, Vector3, Matrix4};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::WindowBuilder;

fn main() {
	let el = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Geometry")
		.build(&el).unwrap();

	let size = window.inner_size().to_logical(window.scale_factor());
	let mut renderer = WindowTarget::new(RendererBuilder::new().real_3d(true).build_rc(), &window, (size.width, size.height));
	
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
						ColorVertex { position: Vector3::new(0.0, 0.5, 0.0), color: Color::WHITE},
						ColorVertex { position: Vector3::new(-0.375, 0.125, 2.0), color: Color::WHITE},
						ColorVertex { position: Vector3::new(-0.25, -0.5, 2.0), color: Color::WHITE},
						ColorVertex { position: Vector3::new(0.25, -0.5, 2.0), color: Color::WHITE},
						ColorVertex { position: Vector3::new(0.375, 0.125, 2.0), color: Color::WHITE},
					],
					indices: &[
						[0, 1], [0, 2], [0, 3], [0, 4],
						[1, 2], [1, 3], [1, 4],
						[2, 3], [2, 4],
						[3, 4],
					]
				}, &[Matrix4::translate(Vector2::new(0.5, 0.0))]);
				frame.set_global_transform(Matrix4::rotate(std::f32::consts::PI));
				frame.draw_colored(ColoredShape {
					vertices: &[
						ColorVertex { position: Vector3::new(0.0, 0.5, 1.0), color: Color::RED },
						ColorVertex { position: Vector3::new(-0.375, 0.125, 0.0), color: Color::YELLOW },
						ColorVertex { position: Vector3::new(-0.25, -0.5, 0.0), color: Color::GREEN },
						ColorVertex { position: Vector3::new(0.25, -0.5, 0.0), color: Color::BLUE },
						ColorVertex { position: Vector3::new(0.375, 0.125, 0.0), color: Color::MAGENTA },
					],
					// Note the vertices are specified going counter-clockwise
					indices: &[
						[0, 1, 4],
						[1, 2, 4],
						[2, 3, 4],
					]
				}, &[Matrix4::translate(Vector2::new(0.5, 0.0))]);
				frame.draw_colored(ColoredShape {
					vertices: &[
						ColorVertex { position: Vector3::new(0.0, 0.5, 0.0), color: Color::RED },
						ColorVertex { position: Vector3::new(-0.25, -0.5, 0.0), color: Color::GREEN },
						ColorVertex { position: Vector3::new(0.25, -0.5, 0.0), color: Color::BLUE },
					],
					indices: &[[0, 1, 2]]
				}, &[Matrix4::translate(Vector2::new(-0.5, 0.0))]);
			},
			_ => {}
		}
	});
}