//! Polystrip is an accelerated 2D graphics library built on `gfx_hal`, which intends to be a pure-rust
//! replacement for SDL2.
//!
//! # Quick breakdown
//! - [`Renderer`]: Contains data types from the `gfx_hal` backend.
//! - [`WindowTarget`]: Holds data for rendering to a `raw_window_handle` window.
//! - [`Frame`]: The struct everything is drawn onto, generally created from a `WindowTarget`
//! - [`*Shape`](vertex): Primitives to be rendered to `Frame`s
//! - [`Texture`]: An image in GPU memory, ready to be rendered to a frame
//!
//! # Quick example
//! An example with `winit` is available in the documentation for `WindowTarget`.

pub mod math;
pub mod pixel;

#[cfg(feature = "gon")]
pub mod gon;

use std::cell::Cell;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::pin::Pin;
use std::rc::Rc;

use raw_window_handle::HasRawWindowHandle;

use crate::math::*;
use crate::pixel::PixelTranslator;

bitflags::bitflags! {
	pub struct CompatiblePipeline: u32 {
		const GON = 1 << 0;
	}
}

impl CompatiblePipeline {
	pub fn required_features(&self) -> wgpu::Features {
		if self.contains(CompatiblePipeline::GON) {
			wgpu::Features::PUSH_CONSTANTS
		} else {
			wgpu::Features::empty()
		}
	}

	pub fn required_limits(&self, adapter_limits: wgpu::Limits) -> wgpu::Limits {
		let mut max_push_constant_size = 0;
		if self.contains(CompatiblePipeline::GON) {
			max_push_constant_size = adapter_limits.max_push_constant_size;
		}
		wgpu::Limits {
			max_push_constant_size,
			..Default::default()
		}
	}
}

/// Customization options for building a Renderer. Options are detailed on builder methods.
///
/// ```no_run
/// # use polystrip::RendererBuilder;
/// let renderer = RendererBuilder::new()
///     .build();
/// ```
pub struct DeviceBuilder {
	compatible_pipelines: CompatiblePipeline,
}

impl DeviceBuilder {
	/// Creates a new `RendererBuilder` with default values
	pub fn new() -> DeviceBuilder {
		DeviceBuilder {
			#[cfg(feature = "gon")]
			compatible_pipelines: CompatiblePipeline::GON,
			#[cfg(not(feature = "gon"))]
			compatible_pipelines: CompatiblePipeline::empty(),
		}
	}

	pub fn compatible_with(mut self, compatible_pipeline: CompatiblePipeline) -> Self {
		self.compatible_pipelines |= compatible_pipeline;
		self
	}

	/// Builds the renderer, initialising the `wgpu` backend.
	pub async fn build(self) -> PolystripDevice {
		PolystripDevice::with_config(self).await
	}

	/// Builds the renderer, initialising the `wgpu` backend, returning a `Rc<Renderer>` which can be
	/// used more easily with the rest of the API.
	///
	/// See also [`Renderer::wrap`]
	pub async fn build_rc(self) -> Rc<PolystripDevice> {
		Rc::new(PolystripDevice::with_config(self).await)
	}
}

impl Default for DeviceBuilder {
	fn default() -> DeviceBuilder {
		DeviceBuilder::new()
	}
}

pub struct PolystripDevice {
	pub instance: wgpu::Instance,
	pub adapter: wgpu::Adapter,
	pub device: wgpu::Device,
	pub queue: wgpu::Queue,
	pub texture_bind_group_layout: wgpu::BindGroupLayout,
}

impl PolystripDevice {
	pub async fn new() -> PolystripDevice {
		PolystripDevice::with_config(DeviceBuilder::new()).await
	}

	async fn with_config(config: DeviceBuilder) -> PolystripDevice {
		let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);

		let adapter = instance.request_adapter(&Default::default()).await.unwrap();

		let (device, queue) = adapter
			.request_device(
				&wgpu::DeviceDescriptor {
					label: Some("Polystrip"),
					features: config.compatible_pipelines.required_features(),
					limits: config.compatible_pipelines.required_limits(adapter.limits()),
				},
				None,
			)
			.await
			.unwrap();

		let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("Polystrip Textures"),
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Texture {
						sample_type: wgpu::TextureSampleType::Float { filterable: false },
						view_dimension: wgpu::TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
					count: None,
				},
			],
		});

		PolystripDevice { instance, adapter, device, queue, texture_bind_group_layout }
	}

	/// Convenience method to create an `Rc<Renderer>` in a builder method chain.
	/// See also [`RendererBuilder::build_rc`]
	pub fn wrap(self) -> Rc<PolystripDevice> {
		Rc::new(self)
	}
}

/// Holds a shared `Renderer`
pub trait HasRenderer {
	fn context_ref(&self) -> &PolystripDevice;
	fn clone_context(&self) -> Rc<PolystripDevice>;
}

impl HasRenderer for Rc<PolystripDevice> {
	fn context_ref(&self) -> &PolystripDevice {
		self
	}

	fn clone_context(&self) -> Rc<PolystripDevice> {
		self.clone()
	}
}

pub trait HasRenderSize {
	fn clone_size_handle(&self) -> Rc<RenderSize>;
}

impl HasRenderSize for Rc<RenderSize> {
	fn clone_size_handle(&self) -> Rc<RenderSize> {
		self.clone()
	}
}

/// Can be rendered to by a `Frame`
pub trait RenderTarget {
	fn create_frame(&mut self) -> BaseFrame<'_>;
}

pub trait RenderDrop<'a> {
	fn initialize(&mut self, context: &PolystripDevice, command_encoder: &mut wgpu::CommandEncoder);
	fn finalize(&mut self, context: &PolystripDevice, command_encoder: &mut wgpu::CommandEncoder);
	fn cleanup(&mut self, context: &PolystripDevice);
}

pub trait RenderPipeline<'a> {
	type Frame: 'a;
	//? What if the Renderers are different?
	fn render_to(&'a mut self, base: &'a mut BaseFrame<'_>) -> Self::Frame;
}

#[derive(Clone, Debug)]
pub struct RenderSize {
	size: Cell<(u32, u32)>,
}

impl RenderSize {
	pub fn new(width: u32, height: u32) -> RenderSize {
		RenderSize { size: Cell::new((width, height)) }
	}

	pub fn get(&self) -> (u32, u32) {
		self.size.get()
	}

	pub fn set(&self, width: u32, height: u32) {
		self.size.set((width, height));
	}

	pub fn wrap(self) -> Rc<RenderSize> {
		Rc::new(self)
	}
}

impl From<(u32, u32)> for RenderSize {
	fn from(extent: (u32, u32)) -> Self {
		RenderSize { size: Cell::new(extent) }
	}
}

/// A target for drawing to a `raw_window_handle` window.
///
/// A `WindowTarget` can be created for any window compatible with `raw_window_handle`. The size of this window must be updated
/// in the event loop, and specified on creation.
pub struct WindowTarget {
	pub context: Rc<PolystripDevice>,
	surface: wgpu::Surface,
	surface_config: wgpu::SurfaceConfiguration,
	size: Rc<RenderSize>,
}

impl WindowTarget {
	/// Creates a new window target for the given `Renderer`.
	///
	/// # Safety
	///
	/// `window` must be a valid object to create a `wgpu::Surface` from, and must remain valid for the lifetime of the `WindowTarget`.
	/// Most likely, the library you use to create the window from will ensure the window is valid. Keep track of the lifetime, though.
	///
	/// ```no_run
	/// # use std::rc::Rc;
	/// # use polystrip::{Renderer, RenderSize, WindowTarget};
	/// # let event_loop = winit::event_loop::EventLoop::new();
	/// # let window = winit::window::Window::new(&event_loop).unwrap();
	/// # let window_size = window.inner_size().to_logical(window.scale_factor());
	/// let renderer = unsafe { WindowTarget::new(
	///     Renderer::new().wrap(),
	///     &window,
	///     &RenderSize::new(window_size.width, window_size.height).wrap(),
	/// ) };
	/// ```
	pub unsafe fn new(
		context: Rc<PolystripDevice>, //TODO: This is inconsistent with the rest of the API
		window: &impl HasRawWindowHandle,
		size: &impl HasRenderSize,
	) -> WindowTarget {
		let extent = size.clone_size_handle();
		let (width, height) = extent.get();
		let surface = context.instance.create_surface(window);
		let surface_config = wgpu::SurfaceConfiguration {
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			format: wgpu::TextureFormat::Bgra8UnormSrgb,
			width,
			height,
			present_mode: wgpu::PresentMode::Mailbox,
		};
		surface.configure(&context.device, &surface_config);

		WindowTarget { context, surface, surface_config, size: extent }
	}

	/// Returns the next `Frame`, which can be drawn to and will present on drop. The frame will contain the data from the
	/// previous frame. This `Renderer` is borrowed mutably while the `Frame` is alive.
	pub fn next_frame(&mut self) -> BaseFrame<'_> {
		// If the size has changed, resize the surface
		let (width, height) = self.size.get();
		if self.surface_config.width != width || self.surface_config.height != height {
			self.surface_config.width = width;
			self.surface_config.height = height;
			self.reconfigure();
		}
		let image = self.acquire_image();
		self.generate_frame(image)
	}

	fn acquire_image(&mut self) -> wgpu::SurfaceTexture {
		match self.surface.get_current_texture() {
			Ok(image) => image,
			Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
				self.reconfigure();
				match self.surface.get_current_texture() {
					Ok(image) => image,
					Err(e) => panic!("{}", e),
				}
			}
			Err(e) => panic!("{}", e),
		}
	}

	fn generate_frame(&mut self, frame: wgpu::SurfaceTexture) -> BaseFrame<'_> {
		let view = frame
			.texture
			.create_view(&wgpu::TextureViewDescriptor { label: Some("Surface View"), ..Default::default() });

		BaseFrame::new(&self.context, WindowFrame { frame: Some(frame), view }, |drop| FrameResources {
			image: &drop.view,
		})
	}

	fn reconfigure(&mut self) {
		self.surface.configure(&self.context.device, &self.surface_config);
	}

	/// Gets the width of the internal swapchain, which is updated every time [`resize`](#method.resize) is called
	pub fn width(&self) -> u32 {
		self.surface_config.width
	}

	/// Gets the height of the internal swapchain, which is updated every time [`resize`](#method.resize) is called
	pub fn height(&self) -> u32 {
		self.surface_config.height
	}

	/// Converts pixel coordinates to screen space coordinates. Alternatively, a [`PixelTranslator`] can be constructed
	/// with the [`pixel_translator`](WindowTarget::pixel_translator) method.
	pub fn pixel(&self, x: i32, y: i32) -> Vector2 {
		let (width, height) = self.size.get();
		Vector2::new((x * 2) as f32 / width as f32 - 1.0, -((y * 2) as f32 / height as f32 - 1.0))
	}

	/// Creates a `PixelTranslator` for this window's size. The `PixelTranslator` will track this `WindowTarget`'s size
	/// even after [`resize`](WindowTarget::resize) calls
	pub fn pixel_translator(&self) -> PixelTranslator {
		PixelTranslator::new(self.size.clone())
	}
}

impl HasRenderer for WindowTarget {
	fn context_ref(&self) -> &PolystripDevice {
		&self.context
	}

	fn clone_context(&self) -> Rc<PolystripDevice> {
		self.context.clone()
	}
}

impl HasRenderSize for WindowTarget {
	fn clone_size_handle(&self) -> Rc<RenderSize> {
		self.size.clone()
	}
}

impl RenderTarget for WindowTarget {
	fn create_frame(&mut self) -> BaseFrame<'_> {
		self.next_frame()
	}
}

struct WindowFrame {
	frame: Option<wgpu::SurfaceTexture>,
	view: wgpu::TextureView,
}

impl<'a> RenderDrop<'a> for WindowFrame {
	fn initialize(&mut self, _context: &PolystripDevice, _command_encoder: &mut wgpu::CommandEncoder) {
		// Nothing to initialize
	}

	fn finalize(&mut self, _context: &PolystripDevice, _command_encoder: &mut wgpu::CommandEncoder) {
		// Nothing to finalize
	}

	fn cleanup(&mut self, _context: &PolystripDevice) {
		if !std::thread::panicking() {
			if let Some(frame) = self.frame.take() {
				frame.present();
			}
		}
	}
}

pub struct BaseFrame<'a> {
	context: Rc<PolystripDevice>,
	encoder: ManuallyDrop<wgpu::CommandEncoder>,
	// SAFETY: To uphold safety guarantees for potential borrows in `resources`, this field must not be modified.
	drop: Pin<Box<dyn RenderDrop<'a> + 'a>>,
	// SAFETY: To uphold safety guarantees for potential borrows from `drop`, this field must not outlive `drop`.
	resources: ManuallyDrop<FrameResources<'a>>,
}

impl<'a> BaseFrame<'a> {
	pub fn new<'b: 'a, D, F>(context: &impl HasRenderer, drop: D, resources: F) -> BaseFrame<'a>
	where
		D: RenderDrop<'a> + 'a + 'b,
		F: FnOnce(&'b D) -> FrameResources<'b>,
	{
		let context = context.clone_context();
		let encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
		let dropbox = Box::pin(drop);
		let drop_reborrow = unsafe { &*(&*dropbox as *const D) };
		let resolved_resources = resources(drop_reborrow);
		BaseFrame {
			context,
			encoder: ManuallyDrop::new(encoder),
			drop: dropbox,
			resources: ManuallyDrop::new(resolved_resources),
		}
	}

	// Render this frame with the given [`Pipeline`]
	// pub fn render_with<P: RenderPipeline<'a>>(self, pipeline: &'a P) -> P::Frame {
	// 	pipeline.render_to(self)
	// }
}

impl<'a> Deref for BaseFrame<'a> {
	type Target = FrameResources<'a>;

	fn deref(&self) -> &FrameResources<'a> {
		&self.resources
	}
}

impl<'a> HasRenderer for BaseFrame<'a> {
	fn context_ref(&self) -> &PolystripDevice {
		&self.context
	}

	fn clone_context(&self) -> Rc<PolystripDevice> {
		self.context.clone()
	}
}

impl<'a> Drop for BaseFrame<'a> {
	fn drop(&mut self) {
		self.context.queue.submit(std::iter::once(unsafe { ManuallyDrop::take(&mut self.encoder) }.finish()));

		// Explicitly drop resources so we can move out of the pin
		let drop = unsafe {
			ManuallyDrop::drop(&mut self.resources);
			Pin::get_unchecked_mut(self.drop.as_mut())
		};
		drop.cleanup(&self.context);
	}
}

pub struct FrameResources<'a> {
	image: &'a wgpu::TextureView,
}

/// A texture which can be copied to and rendered by a [`Frame`](struct.Frame.html).
///
/// It can be used only with the [`Renderer`](struct.Renderer.html) which created it.
pub struct ImageTexture {
	context: Rc<PolystripDevice>,
	texture: wgpu::Texture,
	view: wgpu::TextureView,
	sampler: wgpu::Sampler,
	bind_group: wgpu::BindGroup,
	width: u32,
	height: u32,
}

impl ImageTexture {
	/// Create a new texture from the given rgba data, associated with this `Renderer`.
	///
	/// # Arguments
	/// * `data`: A reference to a byte array containing the pixel data. The data must be formatted to `Rgba8` in
	///           the sRGB color space, in row-major order.
	/// * `size`: The size of the texture, in pixels, in (width, height) order.
	pub fn new_from_rgba(context: &impl HasRenderer, data: &[u8], size: (u32, u32)) -> ImageTexture {
		Self::_from_rgba(context.clone_context(), data, size)
	}

	/// Create a new texture with every pixel initialized to the given color.
	///
	/// # Arguments
	/// * `size`: The size of the texture, in (width, height) order.
	pub fn new_solid_color(context: &impl HasRenderer, color: Color, size: (u32, u32)) -> ImageTexture {
		Self::_solid_color(context.clone_context(), color, size)
	}

	fn _from_rgba(context: Rc<PolystripDevice>, data: &[u8], (width, height): (u32, u32)) -> ImageTexture {
		let texture = context.device.create_texture(&wgpu::TextureDescriptor {
			label: None,
			size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Rgba8UnormSrgb,
			usage: wgpu::TextureUsages::TEXTURE_BINDING
				| wgpu::TextureUsages::COPY_DST,
		});

		context.queue.write_texture(
			texture.as_image_copy(),
			data,
			wgpu::ImageDataLayout {
				offset: 0,
				bytes_per_row: Some(std::num::NonZeroU32::new(width * 4).unwrap()),
				rows_per_image: None,
			},
			wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
		);

		let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
		let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor::default());

		let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &context.texture_bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
				wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
			],
		});

		ImageTexture { context, texture, view, sampler, bind_group, width, height }
	}

	fn _solid_color(context: Rc<PolystripDevice>, color: Color, (width, height): (u32, u32)) -> ImageTexture {
		let texture = context.device.create_texture(&wgpu::TextureDescriptor {
			label: None,
			size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Rgba8UnormSrgb,
			usage: wgpu::TextureUsages::TEXTURE_BINDING
				| wgpu::TextureUsages::COPY_DST,
		});

		let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
		let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor::default());

		let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &context.texture_bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
				wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
			],
		});

		let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
		drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: None,
			color_attachments: &[wgpu::RenderPassColorAttachment {
				view: &view,
				resolve_target: None,
				ops: wgpu::Operations {
					load: wgpu::LoadOp::Clear(wgpu::Color {
						// Convert from sRGB to linear using a gamma of 2.
						r: (color.r as f64 / 255.0).powi(2),
						g: (color.g as f64 / 255.0).powi(2),
						b: (color.b as f64 / 255.0).powi(2),
						a: color.a as f64,
					}),
					store: true,
				},
			}],
			depth_stencil_attachment: None,
		}));
		context.queue.submit([encoder.finish()]);

		ImageTexture { context, texture, view, sampler, bind_group, width, height }
	}

	/// Replaces the data in the given section of the image. The data is interpreted as RGBA8 in row-major order
	///
	/// ```
	/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
	/// # use polystrip::{Renderer, Texture, gon::GonPipeline, math::Rect};
	/// # let texture_data = image::load_from_memory(include_bytes!("../test/res/squares.png"))?.to_rgba();
	/// # let new_data = image::load_from_memory(include_bytes!("../test/res/sandstone3.png"))?.to_rgba();
	/// # let expected_output = image::load_from_memory(include_bytes!("../test/res/expected/texture_write_section.png"))?.to_rgba();
	/// # let mut texture = Texture::new_from_rgba(&Renderer::new().wrap(), &texture_data, (32, 32));
	/// let section = Rect { x: 8, y: 8, w: 16, h: 16 };
	/// texture.write_section(section, &new_data);
	/// assert_eq!(*texture.get_data(), *expected_output);
	/// # Ok(())
	/// # }
	pub fn write_section(&mut self, section: Rect, data: &[u8]) {
		unimplemented!()
	}

	/// Get the dimensions of this texture, in (width, height) order.
	pub fn dimensions(&self) -> (u32, u32) {
		(self.width, self.height)
	}

	pub fn width(&self) -> u32 {
		self.width
	}

	pub fn height(&self) -> u32 {
		self.height
	}

	pub fn sampled(&self) -> SampledTexture<'_> {
		SampledTexture {
			context: &self.context,
			bind_group: &self.bind_group,
		}
	}

	/// Gets the data from this `Texture` in RGBA8 format, in a newly-allocated slice.
	pub fn get_data(&self) -> Box<[u8]> {
		unimplemented!()
	}

	/// Converts pixel coordinates to texture space coordinates
	pub fn pixel(&self, x: i32, y: i32) -> Vector2 {
		Vector2::new(x as f32 / self.width as f32, y as f32 / self.height as f32)
	}

	/// Creates a `PixelTranslator` for this `Texture`, because textures use screen space coords when being rendered to,
	/// not texture space
	pub fn pixel_translator(&self) -> PixelTranslator {
		PixelTranslator::new(Rc::new((self.width, self.height).into()))
	}
}

impl HasRenderer for ImageTexture {
	fn context_ref(&self) -> &PolystripDevice {
		&self.context
	}

	fn clone_context(&self) -> Rc<PolystripDevice> {
		self.context.clone()
	}
}

pub struct RenderTexture {
	context: Rc<PolystripDevice>,
	texture: wgpu::Texture,
	view: wgpu::TextureView,
	sampler: wgpu::Sampler,
	bind_group: wgpu::BindGroup,
	width: u32,
	height: u32,
}

impl RenderTexture {
	/// Creates a new `RenderTexture` with the given dimensions.
	/// 
	/// The contents of the texture are undefined until the first frame is rendered to it.
	/// The texture may only be rendered to or used in a frame by the same `PolystripDevice` that created it.
	pub fn new(context: &impl HasRenderer, (width, height): (u32, u32)) -> RenderTexture {
		let context = context.clone_context();
		let texture = context.device.create_texture(&wgpu::TextureDescriptor {
			label: None,
			size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Bgra8UnormSrgb,
			usage: wgpu::TextureUsages::TEXTURE_BINDING
				| wgpu::TextureUsages::RENDER_ATTACHMENT,
		});

		let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
		let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor::default());

		let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: None,
			layout: &context.texture_bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
				wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
			],
		});

		RenderTexture { context, texture, view, sampler, bind_group, width, height }
	}

	pub fn sampled(&self) -> SampledTexture<'_> {
		SampledTexture {
			context: &self.context,
			bind_group: &self.bind_group,
		}
	}
}

impl HasRenderer for RenderTexture {
	fn context_ref(&self) -> &PolystripDevice {
		&self.context
	}

	fn clone_context(&self) -> Rc<PolystripDevice> {
		self.context.clone()
	}
}

impl RenderTarget for RenderTexture {
	fn create_frame(&mut self) -> BaseFrame<'_> {
		BaseFrame::new(&self.context, TextureFrame { view: &self.view }, |drop| {
			FrameResources { image: drop.view }
		})
	}
}

struct TextureFrame<'a> {
	view: &'a wgpu::TextureView,
}

impl<'a> RenderDrop<'a> for TextureFrame<'a> {
	fn initialize(&mut self, _context: &PolystripDevice, _command_encoder: &mut wgpu::CommandEncoder) {}

	fn finalize(&mut self, _context: &PolystripDevice, _command_encoder: &mut wgpu::CommandEncoder) {}

	fn cleanup(&mut self, _context: &PolystripDevice) {
		// Nothing to clean up
	}
}

pub struct SampledTexture<'a> {
	context: &'a Rc<PolystripDevice>,
	bind_group: &'a wgpu::BindGroup,
}

/// Wrapper for a depth texture, necessary for custom `RenderTarget`s
pub struct DepthTexture {
	context: Rc<PolystripDevice>,
	texture: wgpu::Texture,
	view: wgpu::TextureView,
}

impl DepthTexture {
	//TODO: Inconsistent with rest of API
	pub fn new(context: Rc<PolystripDevice>, (width, height): (u32, u32)) -> DepthTexture {
		let texture = context.device.create_texture(&wgpu::TextureDescriptor {
			label: None,
			size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: wgpu::TextureFormat::Depth32Float,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
		});
		let view = texture.create_view(&wgpu::TextureViewDescriptor {
			label: None,
			aspect: wgpu::TextureAspect::All,
			..Default::default()
		});

		DepthTexture { context, texture, view }
	}
}

impl HasRenderer for DepthTexture {
	fn context_ref(&self) -> &PolystripDevice {
		&self.context
	}

	fn clone_context(&self) -> Rc<PolystripDevice> {
		self.context.clone()
	}
}
