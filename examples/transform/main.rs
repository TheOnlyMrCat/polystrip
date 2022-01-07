use polystrip::math::{Color, Rect};
use polystrip::{RenderSize, WindowTarget, PolystripDevice, RenderPipeline};
use polystrip::gon::{GonPipeline, GpuColorVertex, PolystripShapeExt};

use winit::event::{WindowEvent, Event};
use winit::event_loop::ControlFlow;
use winit::{event_loop::EventLoop, window::WindowBuilder};

fn main() {
	let el = EventLoop::new();
	let window = WindowBuilder::new().with_title("Polystrip example (textures)").build(&el).unwrap();

	let size = window.inner_size();
	let size_handle = RenderSize::new(size.width, size.height).wrap();
	let mut renderer =
		unsafe { WindowTarget::new(pollster::block_on(PolystripDevice::new()).wrap(), &window, &size_handle) };
	let mut pipeline = GonPipeline::new(&renderer, &renderer);
	let pixel_translator = renderer.pixel_translator();

    let mut moving_position = Rect::new(0, 0, 100, 100);
    let movement_speed = 5;
    let mut moving_up = false;
    let mut moving_down = false;
    let mut moving_left = false;
    let mut moving_right = false;
    let moving_shape = pipeline.create_colored(
        &[
            GpuColorVertex {
                position: [0.0, 0.0, 0.0].into(),
                color: Color::RED,
            },
            GpuColorVertex {
                position: [0.0, 1.0, 0.0].into(),
                color: Color::GREEN,
            },
            GpuColorVertex {
                position: [1.0, 1.0, 0.0].into(),
                color: Color::BLUE,
            },
            GpuColorVertex {
                position: [1.0, 0.0, 0.0].into(),
                color: Color::YELLOW,
            },
        ],
        &[[0, 1, 2], [0, 2, 3]],
    );

    let static_shape = pipeline.create_colored(
        &[
            GpuColorVertex {
                position: [0.0, 0.0, 0.0].into(),
                color: Color::BLUE,
            },
            GpuColorVertex {
                position: [0.0, 1.0, 0.0].into(),
                color: Color::BLUE,
            },
            GpuColorVertex {
                position: [1.0, 1.0, 0.0].into(),
                color: Color::BLUE,
            },
            GpuColorVertex {
                position: [1.0, 0.0, 0.0].into(),
                color: Color::BLUE,
            },
        ],
        &[[0, 1, 2], [0, 2, 3]],
    );

    el.run(move |event, _, control_flow| match event {
		Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
			*control_flow = ControlFlow::Exit;
		}
		Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
			size_handle.set(new_size.width, new_size.height);
		}
        Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
            if let Some(key) = input.virtual_keycode {
                match key {
                    winit::event::VirtualKeyCode::Up => {
                        moving_up = matches!(input.state, winit::event::ElementState::Pressed);
                    }
                    winit::event::VirtualKeyCode::Down => {
                        moving_down = matches!(input.state, winit::event::ElementState::Pressed);
                    }
                    winit::event::VirtualKeyCode::Left => {
                        moving_left = matches!(input.state, winit::event::ElementState::Pressed);
                    }
                    winit::event::VirtualKeyCode::Right => {
                        moving_right = matches!(input.state, winit::event::ElementState::Pressed);
                    }
                    _ => {}
                }
            }
        }
		Event::MainEventsCleared => {
            if moving_up {
                moving_position.y -= movement_speed;
            }
            if moving_down {
                moving_position.y += movement_speed;
            }
            if moving_left {
                moving_position.x -= movement_speed;
            }
            if moving_right {
                moving_position.x += movement_speed;
            }

			let mut frame = renderer.next_frame();
			let mut frame = pipeline.render_to(&mut frame);
			frame.clear(Color { r: 128, g: 128, b: 128, a: 255 });
			frame.draw(&static_shape.with_instances(&[pixel_translator.transform_rect(Rect::new(100, 200, 50, 50))]));
            frame.draw(&moving_shape.with_instances(&[pixel_translator.transform_rect(moving_position)]))
		}
		_ => {}
	});
}