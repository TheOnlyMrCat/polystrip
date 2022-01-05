use glyph_brush::ab_glyph::FontArc;
use glyph_brush::GlyphBrushBuilder;
use glyph_brush::Section;
use glyph_brush::Text;
use polystrip::gon::{GlyphBrush, GonPipeline};
use polystrip::math::Color;
use polystrip::RenderPipeline;
use polystrip::{PolystripDevice, RenderSize, WindowTarget};

use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::TextChar;

fn main() {
	let el = EventLoop::new();
	let window = WindowBuilder::new().with_title("Polystrip example (Text rendering)").build(&el).unwrap();
	let mut input_helper = winit_input_helper::WinitInputHelper::new();

	let size = window.inner_size();
	let size_handle = RenderSize::new(size.width, size.height).wrap();
	let mut renderer =
		unsafe { WindowTarget::new(pollster::block_on(PolystripDevice::new()).wrap(), &window, &size_handle) };
	let mut pipeline = GonPipeline::new(&renderer, &renderer);

	let mut has_typed_yet = false;
	let mut text_buffer = String::new();
	let open_sans = FontArc::try_from_slice(include_bytes!("OpenSans-Regular.ttf")).unwrap();
	let mut glyph_brush = GlyphBrush::from_glyph_brush(&renderer, GlyphBrushBuilder::using_font(open_sans).build());

	el.run(move |event, _, control_flow| {
		if input_helper.update(&event) {
			if input_helper.quit() {
				*control_flow = ControlFlow::Exit;
			}

			if let Some(new_size) = input_helper.window_resized() {
				size_handle.set(new_size.width, new_size.height);
			}

			for c in input_helper.text() {
				has_typed_yet = true;
				match c {
					TextChar::Char(c) => {
						if !c.is_control() {
							text_buffer.push(c)
						}
					}
					TextChar::Back => {
						text_buffer.pop();
					}
				}
			}

			if input_helper.key_pressed(winit::event::VirtualKeyCode::Return) {
				has_typed_yet = true;
				text_buffer.push('\n');
			}

			if has_typed_yet {
				glyph_brush.queue(Section::default().with_text(vec![Text::default().with_text(&text_buffer)]));
			} else {
				glyph_brush.queue(
					Section::default()
						.with_text(vec![Text::default().with_text("Start typing and your message will show up here!")]),
				)
			}
			glyph_brush.process_queued(&size_handle);

			let mut frame = renderer.next_frame();
			let mut frame = pipeline.render_to(&mut frame);
			frame.clear(Color { r: 0, g: 0, b: 0, a: 255 });
			frame.draw(&glyph_brush.place());
		}
	});
}
