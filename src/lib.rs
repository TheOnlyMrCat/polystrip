//! Polystrip is an accelerated 2D graphics library with similar capabilities to SDL's graphics system, which
//! it intends to be a replacement for.
//! 
//! # The `Renderer`
//! The [`Renderer`](renderer/struct.Renderer.html) is the core of `polystrip`. It can be built on top of any window compatible with `raw_window_handle`,
//! but must have its size managed manually. A `Renderer` allows one to create [`Frame`](renderer/struct.Frame.html)s,
//! which can be drawn onto and will present themselves when they are dropped.
//! 
//! # Example with `winit`
//! ```no_run
//! # use winit::event::{Event, WindowEvent};
//! # use polystrip::prelude::Renderer;
//! let event_loop = winit::event_loop::EventLoop::new();
//! let window = winit::window::Window::new(&event_loop).unwrap();
//! 
//! let window_size = window.inner_size().to_logical(window.scale_factor());
//! let mut renderer = Renderer::new(&window, (window_size.width, window_size.height));
//! 
//! event_loop.run(move |event, _, control_flow| {
//!     match event {
//!         Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
//!             let window_size = new_size.to_logical(window.scale_factor());
//!             renderer.resize((window_size.width, window_size.height));
//!         },
//!         Event::MainEventsCleared => {
//!             let mut frame = renderer.begin_frame();
//!             // Render in here
//!         },
//!         _ => {}
//!     }
//! });
//! ```

pub mod data;
pub mod renderer;
pub mod texture;
pub mod vertex;

pub mod prelude {
	pub use crate::renderer::{Renderer, Frame};
	pub use crate::data::{GpuPos, Color};
}