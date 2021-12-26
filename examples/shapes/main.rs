use polystrip::gon::{ColorVertex, ColoredShape, GonFrame, GonPipeline, PolystripShapeExt, StrokedShape};
use polystrip::math::{Color, Matrix4, Vector2, Vector3};
use polystrip::{HasRenderer, PolystripDevice, RenderPipeline, RenderSize, WindowTarget};

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() {
	let el = EventLoop::new();
	let window = WindowBuilder::new().with_title("Polystrip example (Colored shapes)").build(&el).unwrap();

	let size = window.inner_size();
	let size_handle = RenderSize::new(size.width, size.height).wrap();
	let mut renderer =
		unsafe { WindowTarget::new(pollster::block_on(PolystripDevice::new()).wrap(), &window, &size_handle) };
	let mut pipeline = GonPipeline::new(&renderer, &renderer);

	let shapes = create_shapes(&renderer);

	el.run(move |event, _, control_flow| match event {
		Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
			*control_flow = ControlFlow::Exit;
		}
		Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
			size_handle.set(new_size.width, new_size.height);
		}
		Event::MainEventsCleared => {
			let mut frame = renderer.next_frame();
			let mut frame = pipeline.render_to(&mut frame);
			frame.clear(Color { r: 128, g: 128, b: 128, a: 255 });
			render_frame(&mut frame, &shapes);
		}
		_ => {}
	});
}

fn create_shapes(context: &impl HasRenderer) -> (StrokedShape, ColoredShape, ColoredShape) {
	(
		context.create_stroked(
			&[
				ColorVertex { position: Vector3::new(0.0, 0.5, 1.0), color: Color::WHITE },
				ColorVertex { position: Vector3::new(-0.375, 0.125, 1.0), color: Color::WHITE },
				ColorVertex { position: Vector3::new(-0.25, -0.5, 1.0), color: Color::WHITE },
				ColorVertex { position: Vector3::new(0.25, -0.5, 1.0), color: Color::WHITE },
				ColorVertex { position: Vector3::new(0.375, 0.125, 1.0), color: Color::WHITE },
			][..],
			&[[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]][..],
		),
		context.create_colored(
			&[
				ColorVertex { position: Vector3::new(0.0, 0.5, 0.0), color: Color::RED },
				ColorVertex { position: Vector3::new(-0.375, 0.125, 0.0), color: Color::YELLOW },
				ColorVertex { position: Vector3::new(-0.25, -0.5, 0.0), color: Color::GREEN },
				ColorVertex { position: Vector3::new(0.25, -0.5, 0.0), color: Color::BLUE },
				ColorVertex { position: Vector3::new(0.375, 0.125, 0.0), color: Color::MAGENTA },
			][..],
			// Note the vertices are specified going counter-clockwise
			&[[0, 1, 4], [1, 2, 4], [2, 3, 4]][..],
		),
		context.create_colored(
			&[
				ColorVertex { position: Vector3::new(0.0, 0.5, 0.0), color: Color::RED },
				ColorVertex { position: Vector3::new(-0.25, -0.5, 0.0), color: Color::GREEN },
				ColorVertex { position: Vector3::new(0.25, -0.5, 0.0), color: Color::BLUE },
			][..],
			&[[0, 1, 2]][..],
		),
	)
}

// The rendering logic is extracted to a separate function for testing purposes. This would work the same inline.
fn render_frame<'a>(
	frame: &mut GonFrame<'a>,
	(stroked, pentagon, triangle): &'a (StrokedShape, ColoredShape, ColoredShape),
) {
	// This stroked shape is drawn before the colored shape on top of it, but will still appear on top due to the height given to it
	frame.draw_stroked(stroked, &[Matrix4::translate(Vector2::new(0.5, 0.0))]);
	frame.draw_colored(pentagon, &[Matrix4::translate(Vector2::new(0.5, 0.0))]);
	frame.draw_colored(triangle, &[Matrix4::translate(Vector2::new(-0.5, 0.0))]);
}

#[cfg(test)]
#[test]
fn shapes() {
	use image::ImageFormat;
	use polystrip::{RenderTarget, Texture};

	let expected_output =
		image::load_from_memory_with_format(include_bytes!("expected.png"), ImageFormat::Png).unwrap().to_rgba();
	let renderer = PolystripDevice::new().wrap();
	let size_handle = RenderSize::new(640, 480).wrap();
	let mut pipeline = GonPipeline::new(&renderer, &size_handle);
	let mut texture = Texture::new_solid_color(&renderer, Color::BLACK, (640, 480));

	let shapes = create_shapes(&renderer);

	let frame = texture.create_frame();
	render_frame(&mut pipeline.render_to(&mut frame), &shapes);
	drop(frame);

	assert_eq!(*texture.get_data(), *expected_output);
}
