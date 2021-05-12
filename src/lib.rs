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

pub(crate) mod backend;
pub mod pipeline;
pub mod pixel;
pub mod vertex;

pub use gfx_hal;
pub use gpu_alloc;

use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::rc::Rc;

use gpu_alloc::{GpuAllocator, MemoryBlock, Request, UsageFlags};
use gpu_alloc_gfx::GfxMemoryDevice;

use crate::vertex::*;
use crate::pixel::PixelTranslator;

use raw_window_handle::HasRawWindowHandle;

use gfx_hal::prelude::*;

#[macro_export]
macro_rules! iter {
	() => {
		std::iter::empty()
	};
	($e:expr) => {
		std::iter::once($e)
	};
	($($e:expr),+ $(,)?) => {
		arrayvec::ArrayVec::from([$($e),+]).into_iter()
	}
}

pub struct Renderer {
	pub instance: backend::Instance,
	pub device: backend::Device,
	pub queue_groups: RefCell<Vec<gfx_hal::queue::family::QueueGroup<backend::Backend>>>,
	pub adapter: gfx_hal::adapter::Adapter<backend::Backend>,

	render_command_pool: RefCell<ManuallyDrop<backend::CommandPool>>,
	texture_command_pool: RefCell<ManuallyDrop<backend::CommandPool>>,

	texture_descriptor_set_layout: ManuallyDrop<backend::DescriptorSetLayout>,
	texture_descriptor_pool: RefCell<ManuallyDrop<backend::DescriptorPool>>,

	pub allocator: RefCell<GpuAllocator<backend::Memory>>,
}

impl Renderer {
	pub fn new() -> Renderer {
		Renderer::with_config(RendererBuilder::new())
	}

	fn with_config(config: RendererBuilder) -> Renderer {
	// - Physical and logical devices
		//Note: Keep up-to-date.         X0.X6.X0_XX
		const POLYSTRIP_VERSION: u32 = 0x00_06_00_00;
		let instance = backend::Instance::create("polystrip", POLYSTRIP_VERSION).unwrap();

		let adapter = instance.enumerate_adapters().into_iter()
			.find(|adapter| {
				adapter.queue_families.iter()
					.any(|family| family.queue_type().supports_graphics())
			})
			.unwrap();

		let gpu = unsafe {
			adapter.physical_device.open(
				&[(
					adapter.queue_families.iter()
						.find(|family| family.queue_type().supports_graphics()).unwrap(),
					&[0.9]
				)],
				gfx_hal::Features::empty()
			).unwrap()		
		};

	// - Command pools, frame resources
		let texture_command_pool = unsafe { gpu.device.create_command_pool(gpu.queue_groups[0].family, gfx_hal::pool::CommandPoolCreateFlags::TRANSIENT) }.unwrap();
		let render_command_pool = unsafe { gpu.device.create_command_pool(gpu.queue_groups[0].family, gfx_hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL) }.unwrap();

	// - Descriptor set and pool
		let texture_descriptor_set_layout = unsafe { gpu.device.create_descriptor_set_layout(
			iter![
				gfx_hal::pso::DescriptorSetLayoutBinding {
					binding: 0,
					ty: gfx_hal::pso::DescriptorType::Image {
						ty: gfx_hal::pso::ImageDescriptorType::Sampled {
							with_sampler: false,
						},
					},
					count: 1,
					stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
					immutable_samplers: false,
				},
				gfx_hal::pso::DescriptorSetLayoutBinding {
					binding: 1,
					ty: gfx_hal::pso::DescriptorType::Sampler,
					count: 1,
					stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
					immutable_samplers: false,
				}
			],
			iter![]
		)}.unwrap();
		let texture_descriptor_pool = unsafe { gpu.device.create_descriptor_pool(
			config.max_textures,
			iter![
				gfx_hal::pso::DescriptorRangeDesc {
					ty: gfx_hal::pso::DescriptorType::Image {
						ty: gfx_hal::pso::ImageDescriptorType::Sampled {
							with_sampler: false,
						},
					},
					count: config.max_textures,
				},
				gfx_hal::pso::DescriptorRangeDesc {
					ty: gfx_hal::pso::DescriptorType::Sampler,
					count: config.max_textures,
				},
			],
			gfx_hal::pso::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
		)}.unwrap();
	// - Memory allocator
		let alloc_props = gpu_alloc_gfx::gfx_device_properties(&adapter);
		let config_fn = config.alloc_config;
		let allocator = GpuAllocator::new(
			config_fn(&alloc_props),
			alloc_props,
		);
	
	// - Passes and pipelines
	
	// - Final construction
		Renderer {
			instance, adapter,
			device: gpu.device,
			queue_groups: RefCell::new(gpu.queue_groups),

			texture_command_pool: RefCell::new(ManuallyDrop::new(texture_command_pool)),
			render_command_pool: RefCell::new(ManuallyDrop::new(render_command_pool)),

			texture_descriptor_set_layout: ManuallyDrop::new(texture_descriptor_set_layout),
			texture_descriptor_pool: RefCell::new(ManuallyDrop::new(texture_descriptor_pool)),

			allocator: RefCell::new(allocator),
		}
	}

	/// Convenience method to create an `Rc<Renderer>` in a builder method chain.
	/// See also [`RendererBuilder::build_rc`]
	pub fn wrap(self) -> Rc<Renderer> {
		Rc::new(self)
	}

}

impl Drop for Renderer {
	fn drop(&mut self) {
		unsafe {
			self.device.wait_idle().unwrap();

			let allocator = self.allocator.get_mut();
			let mem_device = GfxMemoryDevice::wrap(&self.device);

			allocator.cleanup(mem_device);

			self.device.destroy_command_pool(ManuallyDrop::take(self.render_command_pool.get_mut()));

			self.device.destroy_descriptor_set_layout(ManuallyDrop::take(&mut self.texture_descriptor_set_layout));
			self.device.destroy_descriptor_pool(ManuallyDrop::take(self.texture_descriptor_pool.get_mut()));
		}
	}
}

/// Holds a shared `Renderer`
pub trait HasRenderer {
	fn clone_context(&self) -> Rc<Renderer>;
}

impl HasRenderer for Rc<Renderer> {
	fn clone_context(&self) -> Rc<Renderer> {
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
	fn initialize(&mut self, context: &Renderer, command_buffer: &mut backend::CommandBuffer);
	fn finalize(&mut self, context: &Renderer, command_buffer: &mut backend::CommandBuffer);
	fn cleanup(&mut self, context: &Renderer, wait_semaphore: Option<&mut backend::Semaphore>);
}

pub trait RenderPipeline<'a> {
	type Frame: 'a;
	//? What if the Renderers are different?
	fn render_to(&'a mut self, base: BaseFrame<'a>) -> Self::Frame;
}

#[derive(Clone, Debug)]
pub struct RenderSize {
	size: Cell<gfx_hal::window::Extent2D>,
}

impl RenderSize {
	pub fn new(width: u32, height: u32) -> RenderSize {
		RenderSize {
			size: Cell::new(gfx_hal::window::Extent2D { width, height }),
		}
	}

	pub fn get(&self) -> gfx_hal::window::Extent2D {
		self.size.get()
	}

	pub fn set(&self, width: u32, height: u32) {
		self.size.set(gfx_hal::window::Extent2D { width, height });
	}

	pub fn wrap(self) -> Rc<RenderSize> {
		Rc::new(self)
	}
}

impl From<gfx_hal::window::Extent2D> for RenderSize {
    fn from(extent: gfx_hal::window::Extent2D) -> Self {
        RenderSize {
			size: Cell::new(extent),
		}
    }
}

pub fn default_memory_config(_props: &gpu_alloc::DeviceProperties) -> gpu_alloc::Config {
	gpu_alloc::Config::i_am_prototyping() //TODO: Choose sensible defaults
}

/// Customization options for building a Renderer. Options are detailed on builder methods.
/// 
/// ```no_run
/// # use polystrip::RendererBuilder;
/// let renderer = RendererBuilder::new()
///     .real_3d(true)
///     .max_textures(2048)
///     .build();
/// ```
pub struct RendererBuilder {
	max_textures: usize,
	alloc_config: Box<dyn FnOnce(&gpu_alloc::DeviceProperties) -> gpu_alloc::Config>,
}

impl RendererBuilder {
	/// Creates a new `RendererBuilder` with default values
	pub fn new() -> RendererBuilder {
		RendererBuilder {
			max_textures: 1024,
			alloc_config: Box::new(default_memory_config),
		}
	}

	/// The allocation size of the texture pool.
	/// 
	/// Default: 1024
	pub fn max_textures(mut self, max_textures: usize) -> RendererBuilder {
		self.max_textures = max_textures;
		self
	}

	/// The memory allocator's allocation strategy.
	/// 
	/// Default: [default_memory_config](fn.default_memory_config.html)
	pub fn allocation_config(mut self, alloc_config: impl FnOnce(&gpu_alloc::DeviceProperties) -> gpu_alloc::Config + 'static) -> RendererBuilder {
		self.alloc_config = Box::new(alloc_config);
		self
	}

	/// Builds the renderer, initialising the `gfx_hal` backend.
	pub fn build(self) -> Renderer {
		Renderer::with_config(self)
	}

	/// Builds the renderer, initialising the `gfx_hal` backend, returning a `Rc<Renderer>` which can be
	/// used more easily with the rest of the API.
	/// 
	/// See also [`Renderer::wrap`]
	pub fn build_rc(self) -> Rc<Renderer> {
		Rc::new(Renderer::with_config(self))
	}
}

impl Default for RendererBuilder {
	fn default() -> RendererBuilder {
		RendererBuilder::new()
	}
}

/// A target for drawing to a `raw_window_handle` window.
/// 
/// A `WindowTarget` can be created for any window compatible with `raw_window_handle`. The size of this window must be updated
/// in the event loop, and specified on creation. For example, in `winit`:
/// ```no_run
/// # use winit::event::{Event, WindowEvent};
/// # use polystrip::{Renderer, WindowTarget};
/// # let event_loop = winit::event_loop::EventLoop::new();
/// # let window = winit::window::Window::new(&event_loop).unwrap();
/// let window_size = window.inner_size().to_logical(window.scale_factor());
/// let mut renderer = WindowTarget::new(Renderer::new().wrap(), &window, (window_size.width, window_size.height));
/// 
/// event_loop.run(move |event, _, control_flow| {
///     match event {
///         Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
///             let window_size = new_size.to_logical(window.scale_factor());
///             renderer.resize((window_size.width, window_size.height));
///         },
///         // --snip--
/// #       _ => {}
///     }
/// });
/// ```
pub struct WindowTarget {
	pub context: Rc<Renderer>,
	surface: ManuallyDrop<backend::Surface>,
	swapchain_config: gfx_hal::window::SwapchainConfig,
	extent: Rc<RenderSize>,
}

impl WindowTarget {
	/// Creates a new window target for the given `Renderer`.
	/// 
	/// This method assumes the raw window handle was created legitimately. *Technically*, that's my problem, but if
	/// you're not making your window properly, I'm not going to take responsibility for the resulting crash. (The
	/// only way I'd be able to deal with it anyway would be to mark this method `unsafe`)
	/// 
	/// ```no_run
	/// # use std::rc::Rc;
	/// # use polystrip::{RendererBuilder, WindowTarget};
	/// # let event_loop = winit::event_loop::EventLoop::new();
	/// # let window = winit::window::Window::new(&event_loop).unwrap();
	/// # let window_size = window.inner_size().to_logical(window.scale_factor());
	/// let renderer = WindowTarget::new(
	///     RendererBuilder::new().max_textures(2048).build_rc(),
	///     &window,
	///     (window_size.width, window_size.height)
	/// );
	/// ```
	pub fn new(context: Rc<Renderer>, window: &impl HasRawWindowHandle, size: &impl HasRenderSize, swap_image_count: u32) -> WindowTarget {
		let extent = size.clone_size_handle();
		let mut surface = unsafe { context.instance.create_surface(window).unwrap() };
		let caps = surface.capabilities(&context.adapter.physical_device);
		let swapchain_config = gfx_hal::window::SwapchainConfig::from_caps(
				&caps,
				gfx_hal::format::Format::Bgra8Srgb,
				extent.get(),
			)
			.with_image_count(swap_image_count.max(*caps.image_count.start()).min(*caps.image_count.end()));
		unsafe { surface.configure_swapchain(&context.device, swapchain_config.clone()).unwrap(); }

		WindowTarget {
			context,
			surface: ManuallyDrop::new(surface),
			swapchain_config,
			extent,
		}
	}

	/// Returns the next `Frame`, which can be drawn to and will present on drop. The frame will contain the data from the
	/// previous frame. This `Renderer` is borrowed mutably while the `Frame` is alive.
	pub fn next_frame(&mut self) -> BaseFrame<'_> {
		let image = self.acquire_image();
		self.generate_frame(image)
	}

	fn acquire_image(&mut self) -> backend::SwapchainImage {
		match unsafe { self.surface.acquire_image(u64::MAX) } {
			Ok((image, _)) => image,
			Err(gfx_hal::window::AcquireError::OutOfDate(_)) => {
				self.reconfigure_swapchain();
				match unsafe { self.surface.acquire_image(u64::MAX) } {
					Ok((image, _)) => image,
					Err(e) => panic!("{}", e),
				}
			},
			Err(e) => panic!("{}", e),
		}
	}

	fn generate_frame(&mut self, image: backend::SwapchainImage) -> BaseFrame<'_> {
		use std::borrow::Borrow;

		//TODO: Better, scaleable way to reconfigure swapchain each time?
		let recent_extent = self.extent.get();
		if self.swapchain_config.extent != recent_extent {
			self.swapchain_config.extent = recent_extent;
			self.reconfigure_swapchain();
		}

		let viewport = gfx_hal::pso::Viewport {
			rect: gfx_hal::pso::Rect {
				x: 0,
				y: 0,
				w: self.swapchain_config.extent.width as i16,
				h: self.swapchain_config.extent.height as i16,
			},
			depth: 0.0..1.0,
		};
		
		BaseFrame::new(
			&self.context,
			WindowFrame {
				surface: &mut self.surface,
				swap_chain_frame: ManuallyDrop::new(image),
			},
			|drop | {
				FrameResources {
					image: (*drop.swap_chain_frame).borrow(),
					viewport,
				}
			}
		)
	}

	fn reconfigure_swapchain(&mut self) {
		unsafe { self.surface.configure_swapchain(&self.context.device, self.swapchain_config.clone()) }.unwrap();
	}

	/// Gets the width of the internal swapchain, which is updated every time [`resize`](#method.resize) is called
	pub fn width(&self) -> u32 {
		self.swapchain_config.extent.width
	}

	/// Gets the height of the internal swapchain, which is updated every time [`resize`](#method.resize) is called
	pub fn height(&self) -> u32 {
		self.swapchain_config.extent.height
	}

	/// Converts pixel coordinates to screen space coordinates. Alternatively, a [`PixelTranslator`] can be constructed
	/// with the [`pixel_translator`](WindowTarget::pixel_translator) method.
	pub fn pixel(&self, x: i32, y: i32) -> Vector2 {
		Vector2::new((x * 2) as f32 / self.swapchain_config.extent.width as f32 - 1.0, -((y * 2) as f32 / self.swapchain_config.extent.height as f32 - 1.0))
	}

	/// Creates a `PixelTranslator` for this window's size. The `PixelTranslator` will track this `WindowTarget`'s size
	/// even after [`resize`](WindowTarget::resize) calls
	pub fn pixel_translator(&self) -> PixelTranslator {
		PixelTranslator::new(self.extent.clone())
	}
}

impl HasRenderer for WindowTarget {
	fn clone_context(&self) -> Rc<Renderer> {
		self.context.clone()
	}
}

impl HasRenderSize for WindowTarget {
	fn clone_size_handle(&self) -> Rc<RenderSize> {
		self.extent.clone()
	}
}

impl Drop for WindowTarget {
	fn drop(&mut self) {
		unsafe {
			let mut surface = ManuallyDrop::take(&mut self.surface);
			surface.unconfigure_swapchain(&self.context.device);
			self.context.instance.destroy_surface(surface);
		}
	}
}

impl RenderTarget for WindowTarget {
	fn create_frame(&mut self) -> BaseFrame<'_> {
		self.next_frame()
	}
}

struct WindowFrame<'a> {
	surface: &'a mut backend::Surface,
	swap_chain_frame: ManuallyDrop<<backend::Surface as gfx_hal::window::PresentationSurface<backend::Backend>>::SwapchainImage>,
}

impl<'a> RenderDrop<'a> for WindowFrame<'a> {
	fn initialize(&mut self, _context: &Renderer, _command_buffer: &mut backend::CommandBuffer) {
		// Nothing to initialize
	}

	fn finalize(&mut self, _context: &Renderer, _command_buffer: &mut backend::CommandBuffer) {
		// Nothing to finalize
	}

	fn cleanup(&mut self, context: &Renderer, wait_semaphore: Option<&mut backend::Semaphore>) {
		if !std::thread::panicking() {
			unsafe {
				let mut queue_groups = context.queue_groups.borrow_mut();
				queue_groups[0].queues[0].present(&mut self.surface, ManuallyDrop::take(&mut self.swap_chain_frame), wait_semaphore).unwrap();
			}
		} else {
			unsafe {
				ManuallyDrop::drop(&mut self.swap_chain_frame);
			}
		}
	}
}

pub struct BaseFrame<'a> {
	context: Rc<Renderer>,
	// SAFETY: To uphold safety guarantees for potential borrows in `resources`, this field must not be modified.
	drop: Box<dyn RenderDrop<'a> + 'a>,
	// SAFETY: To uphold safety guarantees for potential borrows from `drop`, this field must not outlive `drop`.
	resources: ManuallyDrop<FrameResources<'a>>,
}

impl<'a> BaseFrame<'a> {
	pub fn new<'b: 'a, D, F>(context: &impl HasRenderer, drop: D, resources: F) -> BaseFrame<'a>
	where
		D: RenderDrop<'a> + 'a + 'b,
		F: FnOnce(&'b D) -> FrameResources<'b>
	{
		let dropbox = Box::new(drop);
		let drop_reborrow = unsafe { &*(&*dropbox as *const D) };
		let resolved_resources = resources(drop_reborrow);
		BaseFrame {
			context: context.clone_context(),
			drop: dropbox,
			resources: ManuallyDrop::new(resolved_resources),
		}
	}

	pub fn render_with<P: RenderPipeline<'a>>(self, pipeline: &'a mut P) -> P::Frame {
		pipeline.render_to(self)
	}

	pub unsafe fn drop_finalize(&mut self, command_buffer: &mut backend::CommandBuffer) {
		self.drop.finalize(&self.context, command_buffer);
	}

	pub unsafe fn drop_cleanup(&mut self, wait_semaphore: Option<&mut backend::Semaphore>) {
		self.drop.cleanup(&self.context, wait_semaphore);
	}
}

impl<'a> Deref for BaseFrame<'a> {
	type Target = FrameResources<'a>;

	fn deref(&self) -> &FrameResources<'a> {
		&self.resources
	}
}

impl<'a> HasRenderer for BaseFrame<'a> {
	fn clone_context(&self) -> Rc<Renderer> {
		self.context.clone()
	}
}

impl<'a> Drop for BaseFrame<'a> {
	fn drop(&mut self) {
		// Explicitly drop resources before fields are deinitialised, so no references are held improperly
		unsafe { ManuallyDrop::drop(&mut self.resources); }
	}
}

pub struct FrameResources<'a> {
	image: &'a backend::ImageView,
	viewport: gfx_hal::pso::Viewport,
}

/// A texture which can be copied to and rendered by a [`Frame`](struct.Frame.html).
/// 
/// It can be used only with the [`Renderer`](struct.Renderer.html) which created it.
pub struct Texture {
	context: Rc<Renderer>,
	image: ManuallyDrop<backend::Image>,
	view: ManuallyDrop<backend::ImageView>,
	sampler: ManuallyDrop<backend::Sampler>,
	descriptor_set: ManuallyDrop<backend::DescriptorSet>,
	memory_block: ManuallyDrop<MemoryBlock<backend::Memory>>,
	extent: gfx_hal::window::Extent2D,
	fence: ManuallyDrop<RefCell<backend::Fence>>,
}

impl Texture {
	/// Create a new texture from the given rgba data, associated with this `Renderer`.
	/// 
	/// # Arguments
	/// * `data`: A reference to a byte array containing the pixel data. The data must be formatted to `Rgba8` in
	///           the sRGB color space, in row-major order.
	/// * `size`: The size of the texture, in pixels, in (width, height) order.
	pub fn new_from_rgba(context: &impl HasRenderer, data: &[u8], size: (u32, u32)) -> Texture {
		Self::_from_rgba(context.clone_context(), data, size)
	}

	/// Create a new texture with every pixel initialized to the given color.
	/// 
	/// # Arguments
	/// * `size`: The size of the texture, in (width, height) order.
	pub fn new_solid_color(context: &impl HasRenderer, color: Color, size: (u32, u32)) -> Texture {
		Self::_solid_color(context.clone_context(), color, size)
	}

	fn _from_rgba(context: Rc<Renderer>, data: &[u8], (width, height): (u32, u32)) -> Texture {
		let mut descriptor_set = unsafe { context.texture_descriptor_pool.borrow_mut().allocate_one(&context.texture_descriptor_set_layout) }.unwrap();
		let memory_device = GfxMemoryDevice::wrap(&context.device);

		let mut image = unsafe { context.device.create_image(
			gfx_hal::image::Kind::D2(width, height, 1, 1),
			1,
			gfx_hal::format::Format::Rgba8Srgb,
			gfx_hal::image::Tiling::Optimal,
			gfx_hal::image::Usage::TRANSFER_SRC | gfx_hal::image::Usage::TRANSFER_DST | gfx_hal::image::Usage::SAMPLED,
			gfx_hal::image::ViewCapabilities::MUTABLE_FORMAT,
		)}.unwrap();
		let img_req = unsafe { context.device.get_image_requirements(&image) };

		//TODO: Use non_coherent_atom_size as well
		let row_alignment_mask = context.adapter.physical_device.limits().optimal_buffer_copy_pitch_alignment as u32 - 1;
        let row_pitch = (width * 4 + row_alignment_mask) & !row_alignment_mask;
		let upload_size = (height * row_pitch) as u64;

		let mut buffer = unsafe { context.device.create_buffer(upload_size, gfx_hal::buffer::Usage::TRANSFER_SRC) }.unwrap();
		let buf_req = unsafe { context.device.get_buffer_requirements(&buffer) };
		let mut buf_block = unsafe { context.allocator.borrow_mut().alloc(
			memory_device,
			Request {
				size: buf_req.size,
				align_mask: buf_req.alignment,
				memory_types: buf_req.type_mask,
				usage: UsageFlags::UPLOAD | UsageFlags::TRANSIENT,
			}
		)}.unwrap();

		unsafe {
			let mapping = buf_block.map(memory_device, 0, upload_size as usize).unwrap();
			for y in 0..height as usize {
                let row = &data[y * (width as usize) * 4..(y + 1) * (width as usize) * 4];
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    mapping.as_ptr().offset(y as isize * row_pitch as isize),
                    width as usize * 4,
                );
            }
			use gpu_alloc::MemoryDevice;
			memory_device.flush_memory_ranges(&[gpu_alloc::MappedMemoryRange {
				memory: &buf_block.memory(),
				offset: buf_block.offset(),
				size: upload_size,
			}]).unwrap();
			context.device.bind_buffer_memory(&buf_block.memory(), buf_block.offset(), &mut buffer).unwrap();
		}

		let img_block = unsafe { context.allocator.borrow_mut().alloc(
			memory_device,
			Request {
				size: img_req.size,
				align_mask: img_req.alignment,
				memory_types: img_req.type_mask,
				usage: UsageFlags::FAST_DEVICE_ACCESS,
			}
		)}.unwrap();

		unsafe {
			context.device.bind_image_memory(&img_block.memory(), img_block.offset(), &mut image).unwrap();
		}

		let view = unsafe { context.device.create_image_view(
			&image,
			gfx_hal::image::ViewKind::D2,
			gfx_hal::format::Format::Bgra8Srgb,
			gfx_hal::format::Swizzle(gfx_hal::format::Component::B, gfx_hal::format::Component::G, gfx_hal::format::Component::R, gfx_hal::format::Component::A),
			gfx_hal::image::SubresourceRange {
				aspects: gfx_hal::format::Aspects::COLOR,
				level_start: 0,
				level_count: None,
				layer_start: 0,
				layer_count: None,
			},
		)}.unwrap();

		let sampler = unsafe { context.device.create_sampler(&gfx_hal::image::SamplerDesc::new(gfx_hal::image::Filter::Nearest, gfx_hal::image::WrapMode::Tile)) }.unwrap();

		unsafe {
			context.device.write_descriptor_set(gfx_hal::pso::DescriptorSetWrite {
				set: &mut descriptor_set,
				binding: 0,
				array_offset: 0,
				descriptors: iter![
					gfx_hal::pso::Descriptor::Image(&view, gfx_hal::image::Layout::ShaderReadOnlyOptimal),
					gfx_hal::pso::Descriptor::Sampler(&sampler),
				]
			});
		}

		let mut fence = context.device.create_fence(false).unwrap();
		unsafe {
			let mut command_buffer = context.texture_command_pool.borrow_mut().allocate_one(gfx_hal::command::Level::Primary);
			command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);

			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TOP_OF_PIPE..gfx_hal::pso::PipelineStage::TRANSFER,
				gfx_hal::memory::Dependencies::empty(),
				iter![gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::empty(), gfx_hal::image::Layout::Undefined)
						..
						(gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal),
					target: &image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
			command_buffer.copy_buffer_to_image(
				&buffer,
				&image,
				gfx_hal::image::Layout::TransferDstOptimal,
				iter![gfx_hal::command::BufferImageCopy {
					buffer_offset: 0,
					buffer_width: width,
					buffer_height: height,
					image_layers: gfx_hal::image::SubresourceLayers {
						aspects: gfx_hal::format::Aspects::COLOR,
						level: 0,
						layers: 0..1,
					},
					image_offset: gfx_hal::image::Offset::ZERO,
					image_extent: gfx_hal::image::Extent {
						width, height,
						depth: 1,
					}
				}]
			);
			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TRANSFER..gfx_hal::pso::PipelineStage::FRAGMENT_SHADER,
				gfx_hal::memory::Dependencies::empty(),
				iter![gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal)
						..
						(gfx_hal::image::Access::SHADER_READ, gfx_hal::image::Layout::ShaderReadOnlyOptimal),
					target: &image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
			command_buffer.finish();

			context.queue_groups.borrow_mut()[0].queues[0].submit(iter![&command_buffer], iter![], iter![], Some(&mut fence));
			context.device.wait_for_fence(&fence, u64::MAX).unwrap();

			context.texture_command_pool.borrow_mut().free(iter![command_buffer]);
		}
		
		unsafe {
			context.device.destroy_buffer(buffer);
			context.allocator.borrow_mut().dealloc(
				GfxMemoryDevice::wrap(&context.device),
				buf_block
			);
		}

		Texture {
			context,
			image: ManuallyDrop::new(image),
			view: ManuallyDrop::new(view),
			sampler: ManuallyDrop::new(sampler),
			descriptor_set: ManuallyDrop::new(descriptor_set),
			memory_block: ManuallyDrop::new(img_block),
			extent: gfx_hal::window::Extent2D { width, height },
			fence: ManuallyDrop::new(RefCell::new(fence)),
		}
	}

	fn _solid_color(context: Rc<Renderer>, color: Color, (width, height): (u32, u32)) -> Texture {
		let mut descriptor_set = unsafe { context.texture_descriptor_pool.borrow_mut().allocate_one(&context.texture_descriptor_set_layout) }.unwrap();
		let memory_device = GfxMemoryDevice::wrap(&context.device);

		let mut image = unsafe { context.device.create_image(
			gfx_hal::image::Kind::D2(width, height, 1, 1),
			1,
			gfx_hal::format::Format::Rgba8Srgb,
			gfx_hal::image::Tiling::Optimal,
			gfx_hal::image::Usage::TRANSFER_SRC | gfx_hal::image::Usage::TRANSFER_DST | gfx_hal::image::Usage::SAMPLED,
			gfx_hal::image::ViewCapabilities::MUTABLE_FORMAT,
		)}.unwrap();
		let img_req = unsafe { context.device.get_image_requirements(&image) };
		let img_block = unsafe { context.allocator.borrow_mut().alloc(
			memory_device,
			Request {
				size: img_req.size,
				align_mask: img_req.alignment,
				memory_types: img_req.type_mask,
				usage: UsageFlags::FAST_DEVICE_ACCESS,
			}
		)}.unwrap();

		unsafe {
			context.device.bind_image_memory(&img_block.memory(), img_block.offset(), &mut image).unwrap();
		}

		let view = unsafe { context.device.create_image_view(
			&image,
			gfx_hal::image::ViewKind::D2,
			gfx_hal::format::Format::Bgra8Srgb,
			gfx_hal::format::Swizzle(gfx_hal::format::Component::B, gfx_hal::format::Component::G, gfx_hal::format::Component::R, gfx_hal::format::Component::A),
			gfx_hal::image::SubresourceRange {
				aspects: gfx_hal::format::Aspects::COLOR,
				level_start: 0,
				level_count: None,
				layer_start: 0,
				layer_count: None,
			},
		)}.unwrap();

		let sampler = unsafe { context.device.create_sampler(&gfx_hal::image::SamplerDesc::new(gfx_hal::image::Filter::Nearest, gfx_hal::image::WrapMode::Tile)) }.unwrap();

		unsafe {
			context.device.write_descriptor_set(gfx_hal::pso::DescriptorSetWrite {
				set: &mut descriptor_set,
				binding: 0,
				array_offset: 0,
				descriptors: iter![
					gfx_hal::pso::Descriptor::Image(&view, gfx_hal::image::Layout::ShaderReadOnlyOptimal),
					gfx_hal::pso::Descriptor::Sampler(&sampler),
				]
			});
		}

		let mut fence = context.device.create_fence(false).unwrap();
		unsafe {
			let mut command_buffer = context.texture_command_pool.borrow_mut().allocate_one(gfx_hal::command::Level::Primary);
			command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);

			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TOP_OF_PIPE..gfx_hal::pso::PipelineStage::TRANSFER,
				gfx_hal::memory::Dependencies::empty(),
				iter![gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::empty(), gfx_hal::image::Layout::Undefined)
						..
						(gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal),
					target: &image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
			command_buffer.clear_image(
				&image,
				gfx_hal::image::Layout::TransferDstOptimal,
				gfx_hal::command::ClearValue {
					color: gfx_hal::command::ClearColor {
						float32: [
							(color.r as f32).powi(2) / 65_025.0,
							(color.g as f32).powi(2) / 65_025.0,
							(color.b as f32).powi(2) / 65_025.0,
							color.a as f32 / 255.0,
						]
					}
				},
				iter![gfx_hal::image::SubresourceRange {
					aspects: gfx_hal::format::Aspects::COLOR,
					level_start: 0,
					level_count: None,
					layer_start: 0,
					layer_count: None,
				}]
			);
			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TRANSFER..gfx_hal::pso::PipelineStage::FRAGMENT_SHADER,
				gfx_hal::memory::Dependencies::empty(),
				iter![gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal)
						..
						(gfx_hal::image::Access::SHADER_READ, gfx_hal::image::Layout::ShaderReadOnlyOptimal),
					target: &image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);

			command_buffer.finish();

			context.queue_groups.borrow_mut()[0].queues[0].submit(iter![&command_buffer], iter![], iter![], Some(&mut fence));
			context.device.wait_for_fence(&fence, u64::MAX).unwrap();

			context.texture_command_pool.borrow_mut().free(iter![command_buffer]);
		}

		Texture {
			context,
			image: ManuallyDrop::new(image),
			view: ManuallyDrop::new(view),
			sampler: ManuallyDrop::new(sampler),
			descriptor_set: ManuallyDrop::new(descriptor_set),
			memory_block: ManuallyDrop::new(img_block),
			extent: gfx_hal::window::Extent2D { width, height },
			fence: ManuallyDrop::new(RefCell::new(fence)),
		}
	}

	/// Get the dimensions of this texture, in (width, height) order.
	pub fn dimensions(&self) -> (u32, u32) {
		(self.extent.width, self.extent.height)
	}

	pub fn width(&self) -> u32 {
		self.extent.width
	}

	pub fn height(&self) -> u32 {
		self.extent.height
	}

	pub fn get_data(&self) -> Box<[u8]> {
		let memory_device = GfxMemoryDevice::wrap(&self.context.device);

		let mut buffer = unsafe { self.context.device.create_buffer((self.extent.width * self.extent.height) as u64 * 4, gfx_hal::buffer::Usage::TRANSFER_DST) }.unwrap();
		let buf_req = unsafe { self.context.device.get_buffer_requirements(&buffer) };
		let mut buf_block = unsafe { self.context.allocator.borrow_mut().alloc(
			memory_device,
			Request {
				size: buf_req.size,
				align_mask: buf_req.alignment,
				memory_types: buf_req.type_mask,
				usage: UsageFlags::DOWNLOAD | UsageFlags::TRANSIENT,
			}
		)}.unwrap();

		unsafe {
			self.context.device.bind_buffer_memory(buf_block.memory(), buf_block.offset(), &mut buffer).unwrap();

			let mut fence = self.fence.borrow_mut();
			self.context.device.wait_for_fence(&fence, u64::MAX).unwrap();
			self.context.device.reset_fence(&mut fence).unwrap();
			let mut command_buffer = self.context.texture_command_pool.borrow_mut().allocate_one(gfx_hal::command::Level::Primary);

			command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);

			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TOP_OF_PIPE..gfx_hal::pso::PipelineStage::TRANSFER,
				gfx_hal::memory::Dependencies::empty(),
				iter![gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::SHADER_READ, gfx_hal::image::Layout::ShaderReadOnlyOptimal)
						..
						(gfx_hal::image::Access::TRANSFER_READ, gfx_hal::image::Layout::TransferSrcOptimal),
					target: &*self.image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
			command_buffer.copy_image_to_buffer(
				&self.image,
				gfx_hal::image::Layout::TransferSrcOptimal,
				&buffer,
				iter![gfx_hal::command::BufferImageCopy {
					buffer_offset: 0,
					buffer_width: self.extent.width,
					buffer_height: self.extent.height,
					image_layers: gfx_hal::image::SubresourceLayers {
						aspects: gfx_hal::format::Aspects::COLOR,
						level: 0,
						layers: 0..1,
					},
					image_offset: gfx_hal::image::Offset::ZERO,
					image_extent: gfx_hal::image::Extent {
						width: self.extent.width,
						height: self.extent.height,
						depth: 1,
					}
				}]
			);
			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TRANSFER..gfx_hal::pso::PipelineStage::FRAGMENT_SHADER,
				gfx_hal::memory::Dependencies::empty(),
				iter![gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::TRANSFER_WRITE, gfx_hal::image::Layout::TransferDstOptimal)
						..
						(gfx_hal::image::Access::TRANSFER_READ, gfx_hal::image::Layout::TransferSrcOptimal),
					target: &*self.image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
			command_buffer.finish();

			self.context.queue_groups.borrow_mut()[0].queues[0].submit(iter![&command_buffer], iter![], iter![], Some(&mut fence));
			self.context.device.wait_for_fence(&fence, u64::MAX).unwrap(); // To ensure data validity for download
		}

		let size = buf_req.size as usize;
		//TODO: replace with Box::new_uninit_slice when stabilised
		let mut mem = unsafe { Box::from_raw(std::slice::from_raw_parts_mut(std::alloc::alloc(std::alloc::Layout::array::<u8>(size).expect("Array allocation size overflow")), (self.extent.width * self.extent.height) as usize * 4) as *mut [u8]) };

		unsafe {
			buf_block.read_bytes(memory_device, 0, &mut mem).unwrap();
			self.context.device.destroy_buffer(buffer);
			self.context.allocator.borrow_mut().dealloc(memory_device, buf_block);
		}

		mem
	}

	/// Converts pixel coordinates to texture space coordinates
	pub fn pixel(&self, x: i32, y: i32) -> Vector2 {
		Vector2::new(x as f32 / self.extent.width as f32, y as f32 / self.extent.height as f32)
	}

	/// Creates a `PixelTranslator` for this `Texture`, because textures use screen space coords when being rendered to,
	/// not texture space
	pub fn pixel_translator(&self) -> PixelTranslator {
		PixelTranslator::new(Rc::new(self.extent.into()))
	}
}

impl HasRenderer for Texture {
	fn clone_context(&self) -> Rc<Renderer> {
		self.context.clone()
	}
}

impl RenderTarget for Texture {
	fn create_frame(&mut self) -> BaseFrame<'_> {
		let viewport = gfx_hal::pso::Viewport {
			rect: gfx_hal::pso::Rect {
				x: 0,
				y: 0,
				w: self.extent.width as i16,
				h: self.extent.height as i16,
			},
			depth: 0.0..1.0,
		};

		BaseFrame::new(
			&self.context,
			TextureFrame {
				image: &*self.image,
				view: &*self.view
			},
			|drop| {
				FrameResources {
					image: drop.view,
					viewport,
				}
			}
		)
	}
}

impl Drop for Texture {
	fn drop(&mut self) {
		unsafe {
			self.context.texture_descriptor_pool.borrow_mut().free(std::iter::once(ManuallyDrop::take(&mut self.descriptor_set)));
			self.context.device.destroy_sampler(ManuallyDrop::take(&mut self.sampler));
			self.context.device.destroy_image_view(ManuallyDrop::take(&mut self.view));
			self.context.device.destroy_image(ManuallyDrop::take(&mut self.image));
			self.context.allocator.borrow_mut().dealloc(GfxMemoryDevice::wrap(&self.context.device), ManuallyDrop::take(&mut self.memory_block));
		}
	}
}

/// Implementation detail of the `RenderTarget` system
/// 
/// See [`Frame`]
// #[doc(hidden)]
pub struct TextureFrame<'a> {
	image: &'a backend::Image,
	view: &'a backend::ImageView,
}

impl<'a> RenderDrop<'a> for TextureFrame<'a> {
	fn initialize(&mut self, _context: &Renderer, command_buffer: &mut backend::CommandBuffer) {
		unsafe {
			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TOP_OF_PIPE..gfx_hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
				gfx_hal::memory::Dependencies::empty(),
				iter![gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::COLOR_ATTACHMENT_READ, gfx_hal::image::Layout::ShaderReadOnlyOptimal)
						..
						(gfx_hal::image::Access::COLOR_ATTACHMENT_WRITE, gfx_hal::image::Layout::ColorAttachmentOptimal),
					target: &*self.image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
		}
	}

	fn finalize(&mut self, _context: &Renderer, command_buffer: &mut backend::CommandBuffer) {
		unsafe {
			command_buffer.pipeline_barrier(
				gfx_hal::pso::PipelineStage::TRANSFER..gfx_hal::pso::PipelineStage::FRAGMENT_SHADER,
				gfx_hal::memory::Dependencies::empty(),
				iter![gfx_hal::memory::Barrier::Image {
					states:
						(gfx_hal::image::Access::COLOR_ATTACHMENT_WRITE, gfx_hal::image::Layout::ColorAttachmentOptimal)
						..
						(gfx_hal::image::Access::SHADER_READ, gfx_hal::image::Layout::ShaderReadOnlyOptimal),
					target: self.image,
					range: gfx_hal::image::SubresourceRange {
						aspects: gfx_hal::format::Aspects::COLOR,
						level_start: 0,
						level_count: None,
						layer_start: 0,
						layer_count: None,
					},
					families: None,
				}]
			);
		}
	}

	fn cleanup(&mut self, _context: &Renderer, _wait_semaphore: Option<&mut backend::Semaphore>) {
		// Nothing to clean up
	}
}

/// Wrapper for a depth texture, necessary for custom `RenderTarget`s
pub struct DepthTexture {
	context: Rc<Renderer>,
	image: ManuallyDrop<backend::Image>,
	view: ManuallyDrop<backend::ImageView>,
	memory: ManuallyDrop<MemoryBlock<backend::Memory>>,
}

impl DepthTexture {
	pub fn new(context: Rc<Renderer>, size: gfx_hal::window::Extent2D) -> DepthTexture {
		let mut image = unsafe { context.device.create_image(
			gfx_hal::image::Kind::D2(size.width, size.height, 1, 1),
			1,
			gfx_hal::format::Format::D32Sfloat,
			gfx_hal::image::Tiling::Optimal,
			gfx_hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
			gfx_hal::image::ViewCapabilities::empty()
		)}.unwrap();
		let req = unsafe { context.device.get_image_requirements(&image) };
		let memory = unsafe { context.allocator.borrow_mut().alloc(
			GfxMemoryDevice::wrap(&context.device),
			Request {
				size: req.size,
				align_mask: req.alignment,
				memory_types: req.type_mask,
				usage: UsageFlags::FAST_DEVICE_ACCESS,
			}
		)}.unwrap();
		unsafe { context.device.bind_image_memory(&memory.memory(), memory.offset(), &mut image) }.unwrap();
		let view = unsafe { context.device.create_image_view(
			&image,
			gfx_hal::image::ViewKind::D2,
			gfx_hal::format::Format::D32Sfloat,
			gfx_hal::format::Swizzle::NO,
			gfx_hal::image::SubresourceRange {
				aspects: gfx_hal::format::Aspects::DEPTH,
				level_start: 0,
				level_count: None,
				layer_start: 0,
				layer_count: None,
			}
		)}.unwrap();

		DepthTexture {
			context,
			image: ManuallyDrop::new(image),
			view: ManuallyDrop::new(view),
			memory: ManuallyDrop::new(memory),
		}
	}
}

impl HasRenderer for DepthTexture {
	fn clone_context(&self) -> Rc<Renderer> {
		self.context.clone()
	}
}

impl Drop for DepthTexture {
	fn drop(&mut self) {
		unsafe {
			self.context.device.destroy_image_view(ManuallyDrop::take(&mut self.view));
			self.context.device.destroy_image(ManuallyDrop::take(&mut self.image));
			self.context.allocator.borrow_mut().dealloc(GfxMemoryDevice::wrap(&self.context.device), ManuallyDrop::take(&mut self.memory));
		}
	}
}