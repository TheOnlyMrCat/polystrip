use polystrip::{RenderSize, Renderer, WindowTarget};
use polystrip::vertex::{StrokedShape, ColoredShape, ColorVertex, Color, Vector2, Vector3, Matrix4};
use polystrip::pipeline::{StandardPipeline, StandardFrame};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::WindowBuilder;

fn main() {
	let el = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Polystrip example (Colored shapes)")
		.build(&el).unwrap();

	let size = window.inner_size();
	let size_handle = RenderSize::new(size.width, size.height);
	let mut renderer = WindowTarget::new(Renderer::new().wrap(), &window, &size_handle, 3);
	let mut pipeline = StandardPipeline::new(&renderer, &renderer);
	
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
				let mut frame = renderer.next_frame().render_with(&mut pipeline);
				frame.clear(Color { r: 128, g: 128, b: 128, a: 255 });
				render_frame(&mut frame);
			},
			_ => {}
		}
	});
}

// The rendering logic is extracted to a separate function for testing purposes. This would work the same inline.
fn render_frame(frame: &mut StandardFrame<'_>) {
	// This stroked shape is drawn before the colored shape on top of it, but will still appear on top due to the height given to it
	frame.draw_stroked(
		StrokedShape {
			vertices: [
				ColorVertex { position: Vector3::new(0.0, 0.5, 1.0), color: Color::WHITE },
				ColorVertex { position: Vector3::new(-0.375, 0.125, 1.0), color: Color::WHITE },
				ColorVertex { position: Vector3::new(-0.25, -0.5, 1.0), color: Color::WHITE },
				ColorVertex { position: Vector3::new(0.25, -0.5, 1.0), color: Color::WHITE },
				ColorVertex { position: Vector3::new(0.375, 0.125, 1.0), color: Color::WHITE },
			][..].into(),
			indices: [
				[0, 1], [0, 2], [0, 3], [0, 4],
				[1, 2], [1, 3], [1, 4],
				[2, 3], [2, 4],
				[3, 4],
			][..].into()
		},
		&[Matrix4::translate(Vector2::new(0.5, 0.0))]
	);
	frame.draw_colored(
		ColoredShape {
			vertices: [
				ColorVertex { position: Vector3::new(0.0, 0.5, 0.0), color: Color::RED },
				ColorVertex { position: Vector3::new(-0.375, 0.125, 0.0), color: Color::YELLOW },
				ColorVertex { position: Vector3::new(-0.25, -0.5, 0.0), color: Color::GREEN },
				ColorVertex { position: Vector3::new(0.25, -0.5, 0.0), color: Color::BLUE },
				ColorVertex { position: Vector3::new(0.375, 0.125, 0.0), color: Color::MAGENTA },
			][..].into(),
			// Note the vertices are specified going counter-clockwise
			indices: [
				[0, 1, 4],
				[1, 2, 4],
				[2, 3, 4],
			][..].into()
		},
		&[Matrix4::translate(Vector2::new(0.5, 0.0))]
	);
	frame.draw_colored(
		ColoredShape {
			vertices: [
				ColorVertex { position: Vector3::new(0.0, 0.5, 0.0), color: Color::RED },
				ColorVertex { position: Vector3::new(-0.25, -0.5, 0.0), color: Color::GREEN },
				ColorVertex { position: Vector3::new(0.25, -0.5, 0.0), color: Color::BLUE },
			][..].into(),
			indices: [[0, 1, 2]][..].into()
		},
		&[Matrix4::translate(Vector2::new(-0.5, 0.0))]
	);
}


#[cfg(test)]
#[test]
fn shapes() {
	use polystrip::{RenderTarget, Texture};
	use image::ImageFormat;

	let expected_output = image::load_from_memory_with_format(include_bytes!("expected.png"), ImageFormat::Png).unwrap().to_rgba();
	let renderer = Renderer::new().wrap();
	let mut texture = Texture::new_solid_color(&renderer, Color::BLACK, (640, 480));

	render_frame(&mut texture.create_frame());

	assert_eq!(
		*texture.get_data(),
		*expected_output
	);
}