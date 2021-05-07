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
pub mod pixel;
pub mod vertex;

pub use gpu_alloc;

use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

use arrayvec::ArrayVec;

use gpu_alloc::{GpuAllocator, MemoryBlock, Request, UsageFlags};
use gpu_alloc_gfx::GfxMemoryDevice;

use crate::vertex::*;
use crate::pixel::PixelTranslator;

use raw_window_handle::HasRawWindowHandle;

use gfx_hal::prelude::*;

macro_rules! iter {
	() => {
		std::iter::empty()
	};
	($e:expr) => {
		std::iter::once($e)
	};
	($($e:expr),+ $(,)?) => {
		ArrayVec::from([$($e),+]).into_iter()
	}
}

use align_data::{include_aligned, Align32};
static COLOURED_VERT_SPV: &[u8] = include_aligned!(Align32, "spirv/coloured.vert.spv");
static COLOURED_FRAG_SPV: &[u8] = include_aligned!(Align32, "spirv/coloured.frag.spv");
static TEXTURED_VERT_SPV: &[u8] = include_aligned!(Align32, "spirv/textured.vert.spv");
static TEXTURED_FRAG_SPV: &[u8] = include_aligned!(Align32, "spirv/textured.frag.spv");

pub struct Renderer {
	pub instance: backend::Instance,
	pub device: backend::Device,
	pub queue_groups: RefCell<Vec<gfx_hal::queue::family::QueueGroup<backend::Backend>>>,
	pub adapter: gfx_hal::adapter::Adapter<backend::Backend>,

	stroked_graphics_pipeline: ManuallyDrop<backend::GraphicsPipeline>,
	stroked_graphics_pipeline_layout: ManuallyDrop<backend::PipelineLayout>,
	colour_graphics_pipeline: ManuallyDrop<backend::GraphicsPipeline>,
	colour_graphics_pipeline_layout: ManuallyDrop<backend::PipelineLayout>,
	texture_graphics_pipeline: ManuallyDrop<backend::GraphicsPipeline>,
	texture_graphics_pipeline_layout: ManuallyDrop<backend::PipelineLayout>,
	render_pass: ManuallyDrop<backend::RenderPass>,

	render_command_pool: RefCell<ManuallyDrop<backend::CommandPool>>,
	texture_command_pool: RefCell<ManuallyDrop<backend::CommandPool>>,
	
	frames_in_flight: u32,
	matrix_array_size: u32,

	current_frame: AtomicU32,
	render_command_buffers: ManuallyDrop<Vec<RefCell<backend::CommandBuffer>>>,
	render_semaphores: ManuallyDrop<Vec<RefCell<backend::Semaphore>>>,
	render_fences: ManuallyDrop<Vec<RefCell<backend::Fence>>>,
	render_frame_resources: ManuallyDrop<Vec<ManuallyDrop<RefCell<Vec<RenderResource>>>>>,

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
		let mut render_command_pool = unsafe { gpu.device.create_command_pool(gpu.queue_groups[0].family, gfx_hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL) }.unwrap();
		let render_command_buffers = {
			let mut buffers = Vec::with_capacity(config.frames_in_flight as usize);
			unsafe { render_command_pool.allocate(config.frames_in_flight as usize, gfx_hal::command::Level::Primary, &mut buffers); }
			buffers
		}.into_iter().map(RefCell::new).collect();

		let (render_semaphores, render_fences, render_frame_resources) = {
			let mut semaphores = Vec::with_capacity(config.frames_in_flight as usize);
			let mut fences = Vec::with_capacity(config.frames_in_flight as usize);
			let mut resources = Vec::with_capacity(config.frames_in_flight as usize);
			for _ in 0..config.frames_in_flight {
				semaphores.push(RefCell::new(gpu.device.create_semaphore().unwrap()));
				fences.push(RefCell::new(gpu.device.create_fence(false).unwrap()));
				resources.push(ManuallyDrop::new(RefCell::new(Vec::new())));
			}
			(semaphores, fences, resources)
		};

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
		// The number of object transform matrices we can store in push constants, and therefore how many objects we can draw at once
		// Equal to the number of matrices we can store, minus 1 for the world matrix
		let matrix_array_size = (adapter.physical_device.limits().max_push_constants_size / std::mem::size_of::<Matrix4>() - 1).min((u32::MAX - 1) as usize) as u32;

		let main_pass = unsafe { gpu.device.create_render_pass(
			iter![
				gfx_hal::pass::Attachment {
					format: Some(gfx_hal::format::Format::Bgra8Srgb),
					samples: 1,
					ops: gfx_hal::pass::AttachmentOps {
						load: gfx_hal::pass::AttachmentLoadOp::Load,
						store: gfx_hal::pass::AttachmentStoreOp::Store,
					},
					stencil_ops: gfx_hal::pass::AttachmentOps::DONT_CARE,
					layouts: gfx_hal::image::Layout::Undefined..gfx_hal::image::Layout::Present,
				},
				gfx_hal::pass::Attachment {
					format: Some(gfx_hal::format::Format::D32Sfloat),
					samples: 1,
					ops: gfx_hal::pass::AttachmentOps {
						load: gfx_hal::pass::AttachmentLoadOp::Clear,
						store: gfx_hal::pass::AttachmentStoreOp::DontCare,
					},
					stencil_ops: gfx_hal::pass::AttachmentOps::DONT_CARE,
					layouts: gfx_hal::image::Layout::Undefined..gfx_hal::image::Layout::DepthStencilAttachmentOptimal
				}
			],
			iter![gfx_hal::pass::SubpassDesc {
				colors: &[(0, gfx_hal::image::Layout::ColorAttachmentOptimal)],
				depth_stencil: Some(&(1, gfx_hal::image::Layout::DepthStencilAttachmentOptimal)),
				inputs: &[],
				resolves: &[],
				preserves: &[],
			}],
			iter![]
		)}.unwrap();

		let colour_vs_module = unsafe { gpu.device.create_shader_module(bytemuck::cast_slice(COLOURED_VERT_SPV)) }.unwrap();
		let colour_fs_module = unsafe { gpu.device.create_shader_module(bytemuck::cast_slice(COLOURED_FRAG_SPV)) }.unwrap();
		let texture_vs_module = unsafe { gpu.device.create_shader_module(bytemuck::cast_slice(TEXTURED_VERT_SPV)) }.unwrap();
		let texture_fs_module = unsafe { gpu.device.create_shader_module(bytemuck::cast_slice(TEXTURED_FRAG_SPV)) }.unwrap();

		let stroked_graphics_pipeline_layout = unsafe { gpu.device.create_pipeline_layout(iter![], iter![(gfx_hal::pso::ShaderStageFlags::VERTEX, 0..std::mem::size_of::<Matrix4>() as u32 * (matrix_array_size + 1))]) }.unwrap();
		let stroked_graphics_pipeline = unsafe { gpu.device.create_graphics_pipeline(&gfx_hal::pso::GraphicsPipelineDesc {
			label: Some("polystrip_stroked_pipeline"),
			primitive_assembler: gfx_hal::pso::PrimitiveAssemblerDesc::Vertex {
				buffers: &[gfx_hal::pso::VertexBufferDesc {
					binding: 0,
					stride: std::mem::size_of::<ColorVertex>() as u32,
					rate: gfx_hal::pso::VertexInputRate::Vertex,
				}],
				attributes: ColorVertex::desc(),
				input_assembler: gfx_hal::pso::InputAssemblerDesc {
					primitive: gfx_hal::pso::Primitive::LineList,
					with_adjacency: false,
					restart_index: None,
				},
				vertex: gfx_hal::pso::EntryPoint {
					entry: "main",
					module: &colour_vs_module,
					specialization: gfx_hal::pso::Specialization {
						constants: std::borrow::Cow::Borrowed(&[
							gfx_hal::pso::SpecializationConstant {
								id: 0,
								range: 0..1
							},
							gfx_hal::pso::SpecializationConstant {
								id: 1,
								range: 1..2,
							},
							gfx_hal::pso::SpecializationConstant {
								id: 2,
								range: 4..8,
							},
						]),
						// * Can use the two zeros to store other spec constants when necessary
						data: std::borrow::Cow::Borrowed(bytemuck::cast_slice(&[[cfg!(any(feature = "metal", feature = "dx12")) as u8, config.real_3d as u8, 0, 0], matrix_array_size.to_ne_bytes()])),
					}
				},
				tessellation: None,
				geometry: None,
			},
			rasterizer: gfx_hal::pso::Rasterizer {
				polygon_mode: gfx_hal::pso::PolygonMode::Fill,
				cull_face: gfx_hal::pso::Face::NONE,
				front_face: gfx_hal::pso::FrontFace::CounterClockwise,
				depth_clamping: false,
				depth_bias: None,
				conservative: false,
				line_width: gfx_hal::pso::State::Dynamic,
			},
			fragment: Some(gfx_hal::pso::EntryPoint {
				entry: "main",
				module: &colour_fs_module,
				specialization: gfx_hal::pso::Specialization {
					constants: std::borrow::Cow::Borrowed(&[]),
					data: std::borrow::Cow::Borrowed(&[]),
				}
			}),
			blender: gfx_hal::pso::BlendDesc {
				logic_op: None,
				targets: vec![gfx_hal::pso::ColorBlendDesc {
					mask: gfx_hal::pso::ColorMask::ALL,
					blend: Some(gfx_hal::pso::BlendState {
						color: gfx_hal::pso::BlendOp::Add {
							src: gfx_hal::pso::Factor::SrcAlpha,
							dst: gfx_hal::pso::Factor::OneMinusSrcAlpha,
						},
						alpha: gfx_hal::pso::BlendOp::Add {
							src: gfx_hal::pso::Factor::SrcAlpha,
							dst: gfx_hal::pso::Factor::OneMinusSrcAlpha,
						}
					})
				}]
			},
			depth_stencil: gfx_hal::pso::DepthStencilDesc {
				depth: Some(gfx_hal::pso::DepthTest {
					fun: gfx_hal::pso::Comparison::GreaterEqual,
					write: true,
				}),
				depth_bounds: false,
				stencil: None,
			},
			multisampling: None,
			baked_states: gfx_hal::pso::BakedStates {
				viewport: None,
				scissor: None,
				blend_color: None,
				depth_bounds: None,
			},
			layout: &stroked_graphics_pipeline_layout,
			subpass: gfx_hal::pass::Subpass {
				index: 0,
				main_pass: &main_pass,
			},
			flags: gfx_hal::pso::PipelineCreationFlags::empty(),
			parent: gfx_hal::pso::BasePipeline::None,
		}, None) }.unwrap();

		let colour_graphics_pipeline_layout = unsafe { gpu.device.create_pipeline_layout(iter![], iter![(gfx_hal::pso::ShaderStageFlags::VERTEX, 0..std::mem::size_of::<Matrix4>() as u32 * (matrix_array_size + 1))]) }.unwrap();
		let colour_graphics_pipeline = unsafe { gpu.device.create_graphics_pipeline(&gfx_hal::pso::GraphicsPipelineDesc {
			label: Some("polystrip_colour_pipeline"),
			primitive_assembler: gfx_hal::pso::PrimitiveAssemblerDesc::Vertex {
				buffers: &[gfx_hal::pso::VertexBufferDesc {
					binding: 0,
					stride: std::mem::size_of::<ColorVertex>() as u32,
					rate: gfx_hal::pso::VertexInputRate::Vertex,
				}],
				attributes: ColorVertex::desc(),
				input_assembler: gfx_hal::pso::InputAssemblerDesc {
					primitive: gfx_hal::pso::Primitive::TriangleList,
					with_adjacency: false,
					restart_index: None,
				},
				vertex: gfx_hal::pso::EntryPoint {
					entry: "main",
					module: &colour_vs_module,
					specialization: gfx_hal::pso::Specialization {
						constants: std::borrow::Cow::Borrowed(&[
							gfx_hal::pso::SpecializationConstant {
								id: 0,
								range: 0..1
							},
							gfx_hal::pso::SpecializationConstant {
								id: 1,
								range: 1..2,
							},
							gfx_hal::pso::SpecializationConstant {
								id: 2,
								range: 4..8,
							},
						]),
						data: std::borrow::Cow::Borrowed(bytemuck::cast_slice(&[[cfg!(any(feature = "metal", feature = "dx12")) as u8, config.real_3d as u8, 0, 0], matrix_array_size.to_ne_bytes()])),
					}
				},
				tessellation: None,
				geometry: None,
			},
			rasterizer: gfx_hal::pso::Rasterizer {
				polygon_mode: gfx_hal::pso::PolygonMode::Fill,
				cull_face: gfx_hal::pso::Face::NONE,
				front_face: gfx_hal::pso::FrontFace::CounterClockwise,
				depth_clamping: false,
				depth_bias: None,
				conservative: false,
				line_width: gfx_hal::pso::State::Dynamic,
			},
			fragment: Some(gfx_hal::pso::EntryPoint {
				entry: "main",
				module: &colour_fs_module,
				specialization: gfx_hal::pso::Specialization {
					constants: std::borrow::Cow::Borrowed(&[]),
					data: std::borrow::Cow::Borrowed(&[]),
				}
			}),
			blender: gfx_hal::pso::BlendDesc {
				logic_op: None,
				targets: vec![gfx_hal::pso::ColorBlendDesc {
					mask: gfx_hal::pso::ColorMask::ALL,
					blend: Some(gfx_hal::pso::BlendState {
						color: gfx_hal::pso::BlendOp::Add {
							src: gfx_hal::pso::Factor::SrcAlpha,
							dst: gfx_hal::pso::Factor::OneMinusSrcAlpha,
						},
						alpha: gfx_hal::pso::BlendOp::Add {
							src: gfx_hal::pso::Factor::SrcAlpha,
							dst: gfx_hal::pso::Factor::OneMinusSrcAlpha,
						}
					})
				}]
			},
			depth_stencil: gfx_hal::pso::DepthStencilDesc {
				depth: Some(gfx_hal::pso::DepthTest {
					fun: gfx_hal::pso::Comparison::GreaterEqual,
					write: true,
				}),
				depth_bounds: false,
				stencil: None,
			},
			multisampling: None,
			baked_states: gfx_hal::pso::BakedStates {
				viewport: None,
				scissor: None,
				blend_color: None,
				depth_bounds: None,
			},
			layout: &colour_graphics_pipeline_layout,
			subpass: gfx_hal::pass::Subpass {
				index: 0,
				main_pass: &main_pass,
			},
			flags: gfx_hal::pso::PipelineCreationFlags::empty(),
			parent: gfx_hal::pso::BasePipeline::None,
		}, None) }.unwrap();

		let texture_graphics_pipeline_layout = unsafe { gpu.device.create_pipeline_layout(iter![&texture_descriptor_set_layout], iter![(gfx_hal::pso::ShaderStageFlags::VERTEX, 0..std::mem::size_of::<Matrix4>() as u32 * (matrix_array_size + 1))]) }.unwrap();
		let texture_graphics_pipeline = unsafe { gpu.device.create_graphics_pipeline(&gfx_hal::pso::GraphicsPipelineDesc {
			label: Some("polystrip_texture_pipeline"),
			primitive_assembler: gfx_hal::pso::PrimitiveAssemblerDesc::Vertex {
				buffers: &[gfx_hal::pso::VertexBufferDesc {
					binding: 0,
					stride: std::mem::size_of::<TextureVertex>() as u32,
					rate: gfx_hal::pso::VertexInputRate::Vertex,
				}],
				attributes: TextureVertex::desc(),
				input_assembler: gfx_hal::pso::InputAssemblerDesc {
					primitive: gfx_hal::pso::Primitive::TriangleList,
					with_adjacency: false,
					restart_index: None,
				},
				vertex: gfx_hal::pso::EntryPoint {
					entry: "main",
					module: &texture_vs_module,
					specialization: gfx_hal::pso::Specialization {
						constants: std::borrow::Cow::Borrowed(&[
							gfx_hal::pso::SpecializationConstant {
								id: 0,
								range: 0..1
							},
							gfx_hal::pso::SpecializationConstant {
								id: 1,
								range: 1..2,
							},
							gfx_hal::pso::SpecializationConstant {
								id: 2,
								range: 4..8,
							},
						]),
						data: std::borrow::Cow::Borrowed(bytemuck::cast_slice(&[[cfg!(any(feature = "metal", feature = "dx12")) as u8, config.real_3d as u8, 0, 0], matrix_array_size.to_ne_bytes()])),
					}
				},
				tessellation: None,
				geometry: None,
			},
			rasterizer: gfx_hal::pso::Rasterizer {
				polygon_mode: gfx_hal::pso::PolygonMode::Fill,
				cull_face: gfx_hal::pso::Face::NONE,
				front_face: gfx_hal::pso::FrontFace::CounterClockwise,
				depth_clamping: false,
				depth_bias: None,
				conservative: false,
				line_width: gfx_hal::pso::State::Dynamic,
			},
			fragment: Some(gfx_hal::pso::EntryPoint {
				entry: "main",
				module: &texture_fs_module,
				specialization: gfx_hal::pso::Specialization {
					constants: std::borrow::Cow::Borrowed(&[]),
					data: std::borrow::Cow::Borrowed(&[]),
				}
			}),
			blender: gfx_hal::pso::BlendDesc {
				logic_op: None,
				targets: vec![gfx_hal::pso::ColorBlendDesc {
					mask: gfx_hal::pso::ColorMask::ALL,
					blend: Some(gfx_hal::pso::BlendState {
						color: gfx_hal::pso::BlendOp::Add {
							src: gfx_hal::pso::Factor::SrcAlpha,
							dst: gfx_hal::pso::Factor::OneMinusSrcAlpha,
						},
						alpha: gfx_hal::pso::BlendOp::Add {
							src: gfx_hal::pso::Factor::SrcAlpha,
							dst: gfx_hal::pso::Factor::OneMinusSrcAlpha,
						}
					})
				}]
			},
			depth_stencil: gfx_hal::pso::DepthStencilDesc {
				depth: Some(gfx_hal::pso::DepthTest {
					fun: gfx_hal::pso::Comparison::GreaterEqual,
					write: true,
				}),
				depth_bounds: false,
				stencil: None,
			},
			multisampling: None,
			baked_states: gfx_hal::pso::BakedStates {
				viewport: None,
				scissor: None,
				blend_color: None,
				depth_bounds: None,
			},
			layout: &texture_graphics_pipeline_layout,
			subpass: gfx_hal::pass::Subpass {
				index: 0,
				main_pass: &main_pass,
			},
			flags: gfx_hal::pso::PipelineCreationFlags::empty(),
			parent: gfx_hal::pso::BasePipeline::None,
		}, None) }.unwrap();
	
	// - Final construction
		Renderer {
			instance, adapter,
			device: gpu.device,
			queue_groups: RefCell::new(gpu.queue_groups),

			stroked_graphics_pipeline: ManuallyDrop::new(stroked_graphics_pipeline),
			stroked_graphics_pipeline_layout: ManuallyDrop::new(stroked_graphics_pipeline_layout),
			colour_graphics_pipeline: ManuallyDrop::new(colour_graphics_pipeline),
			colour_graphics_pipeline_layout: ManuallyDrop::new(colour_graphics_pipeline_layout),
			texture_graphics_pipeline: ManuallyDrop::new(texture_graphics_pipeline),
			texture_graphics_pipeline_layout: ManuallyDrop::new(texture_graphics_pipeline_layout),
			render_pass: ManuallyDrop::new(main_pass),

			texture_command_pool: RefCell::new(ManuallyDrop::new(texture_command_pool)),
			render_command_pool: RefCell::new(ManuallyDrop::new(render_command_pool)),
			
			frames_in_flight: config.frames_in_flight,
			matrix_array_size,

			current_frame: AtomicU32::new(0),
			render_command_buffers: ManuallyDrop::new(render_command_buffers),
			render_semaphores: ManuallyDrop::new(render_semaphores),
			render_fences: ManuallyDrop::new(render_fences),
			render_frame_resources: ManuallyDrop::new(render_frame_resources),

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

	/// Waits for the next frame to finish rendering, deallocates its resources, and returns its index.
	/// 
	/// Generally, this won't need to be called in application code, since it is done by [`RenderTarget`]s before creating
	/// a [`Frame`].
	pub fn wait_next_frame(&self) -> usize {
		let frame_idx = self.next_frame_idx();
		unsafe {
			self.device.wait_for_fence(&self.render_fences[frame_idx].borrow(), u64::MAX).unwrap();
		}

		let mut allocator = self.allocator.borrow_mut();
		let mem_device = GfxMemoryDevice::wrap(&self.device);
		for resource in self.render_frame_resources[frame_idx].replace(Vec::new()) {
			match resource {
				RenderResource::Buffer(buffer, memory) => unsafe {
					allocator.dealloc(mem_device, memory);
					self.device.destroy_buffer(buffer);
				}
			}
		}
		frame_idx
	}

	/// Returns the index of the next frame to be rendered, to be used when selecting the command buffer, semaphores
	/// and fences.
	pub fn next_frame_idx(&self) -> usize {
		let frames_in_flight = self.frames_in_flight;
		self.current_frame.fetch_update(Ordering::AcqRel, Ordering::Acquire, |x| Some((x + 1) % frames_in_flight)).unwrap() as usize
	}
}

impl Drop for Renderer {
	fn drop(&mut self) {
		unsafe {
			self.device.wait_idle().unwrap();

			let allocator = self.allocator.get_mut();
			let mem_device = GfxMemoryDevice::wrap(&self.device);

			for resource in ManuallyDrop::take(&mut self.render_frame_resources).into_iter().flat_map(|mut i| ManuallyDrop::take(&mut i).into_inner()) {
				match resource {
					RenderResource::Buffer(buf, block) => {
						self.device.destroy_buffer(buf);
						allocator.dealloc(mem_device, block)
					}
				}
			}

			allocator.cleanup(mem_device);

			let render_command_pool = self.render_command_pool.get_mut();
			render_command_pool.free(ManuallyDrop::take(&mut self.render_command_buffers).into_iter().map(RefCell::into_inner));
			self.device.destroy_command_pool(ManuallyDrop::take(render_command_pool));

			for semaphore in ManuallyDrop::take(&mut self.render_semaphores) {
				self.device.destroy_semaphore(semaphore.into_inner());
			}

			for fence in ManuallyDrop::take(&mut self.render_fences) {
				self.device.destroy_fence(fence.into_inner());
			}

			self.device.destroy_descriptor_set_layout(ManuallyDrop::take(&mut self.texture_descriptor_set_layout));
			self.device.destroy_descriptor_pool(ManuallyDrop::take(self.texture_descriptor_pool.get_mut()));

			self.device.destroy_graphics_pipeline(ManuallyDrop::take(&mut self.stroked_graphics_pipeline));
			self.device.destroy_graphics_pipeline(ManuallyDrop::take(&mut self.colour_graphics_pipeline));
			self.device.destroy_graphics_pipeline(ManuallyDrop::take(&mut self.texture_graphics_pipeline));
			self.device.destroy_pipeline_layout(ManuallyDrop::take(&mut self.stroked_graphics_pipeline_layout));
			self.device.destroy_pipeline_layout(ManuallyDrop::take(&mut self.colour_graphics_pipeline_layout));
			self.device.destroy_pipeline_layout(ManuallyDrop::take(&mut self.texture_graphics_pipeline_layout));
			
			self.device.destroy_render_pass(ManuallyDrop::take(&mut self.render_pass));
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

/// Can be rendered to by a `Frame`
pub trait RenderTarget<'a> {
	type FrameDrop: RenderDrop<'a>;

	fn create_frame(&'a mut self) -> Frame<Self::FrameDrop>;
}


/// Cleanup for a `RenderTarget`.
/// 
/// The `cleanup` function is called upon a `Frame`'s drop, after it has done its own cleanup.
pub trait RenderDrop<'a> {
	fn finalize(&mut self, context: &Renderer, frame_idx: usize);
	fn cleanup(&mut self, context: &Renderer, frame_idx: usize);
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
	real_3d: bool,
	max_textures: usize,
	frames_in_flight: u32,
	alloc_config: Box<dyn FnOnce(&gpu_alloc::DeviceProperties) -> gpu_alloc::Config>,
}

impl RendererBuilder {
	/// Creates a new `RendererBuilder` with default values
	pub fn new() -> RendererBuilder {
		RendererBuilder {
			real_3d: false,
			max_textures: 1024,
			frames_in_flight: 3,
			alloc_config: Box::new(default_memory_config),
		}
	}

	/// If `true`, allows transform matrices to affect sprite depth. This clamps the depth between `0.0` and `1.0`
	/// 
	/// Default: false
	pub fn real_3d(mut self, real_3d: bool) -> RendererBuilder {
		self.real_3d = real_3d;
		self
	}

	/// The allocation size of the texture pool.
	/// 
	/// Default: 1024
	pub fn max_textures(mut self, max_textures: usize) -> RendererBuilder {
		self.max_textures = max_textures;
		self
	}

	/// The number of frames that can be dispatched simultaneously
	/// 
	/// Default: 3
	pub fn frames_in_flight(mut self, frames_in_flight: u32) -> RendererBuilder {
		self.frames_in_flight = frames_in_flight;
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
	framebuffers: Vec<ManuallyDrop<backend::Framebuffer>>,
	swapchain_config: gfx_hal::window::SwapchainConfig,
	extent: Rc<Cell<gfx_hal::window::Extent2D>>,
	depth_texture: DepthTexture,
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
	pub fn new(context: Rc<Renderer>, window: &impl HasRawWindowHandle, (width, height): (u32, u32)) -> WindowTarget {
		let mut surface = unsafe { context.instance.create_surface(window).unwrap() };
		let caps = surface.capabilities(&context.adapter.physical_device);
		let swapchain_config = gfx_hal::window::SwapchainConfig::from_caps(
				&caps,
				gfx_hal::format::Format::Bgra8Srgb,
				gfx_hal::window::Extent2D { width, height }
			)
			.with_image_count(context.frames_in_flight.max(*caps.image_count.start()).min(*caps.image_count.end()));
		unsafe { surface.configure_swapchain(&context.device, swapchain_config.clone()).unwrap(); }

		let depth_texture = DepthTexture::new(context.clone(), swapchain_config.extent);

		let framebuffers = (0..context.frames_in_flight).map(|_| ManuallyDrop::new(unsafe { context.device.create_framebuffer(
			&context.render_pass,
			iter![
				gfx_hal::image::FramebufferAttachment {
					usage: gfx_hal::image::Usage::COLOR_ATTACHMENT,
					view_caps: gfx_hal::image::ViewCapabilities::empty(),
					format: swapchain_config.format,
				},
				gfx_hal::image::FramebufferAttachment {
					usage: gfx_hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
					view_caps: gfx_hal::image::ViewCapabilities::empty(),
					format: gfx_hal::format::Format::D32Sfloat,
				}
			],
			swapchain_config.extent.to_extent()
		)}.unwrap())).collect();

		WindowTarget {
			context,
			surface: ManuallyDrop::new(surface),
			framebuffers,
			swapchain_config,
			extent: Rc::new(Cell::new(gfx_hal::window::Extent2D { width, height })),
			depth_texture,
		}
	}

	/// Returns the next `Frame`, which can be drawn to and will present on drop. The frame will contain the data from the
	/// previous frame. This `Renderer` is borrowed mutably while the `Frame` is alive.
	pub fn next_frame(&mut self) -> Frame<'_, WindowFrame> {
		let image = self.acquire_image();
		self.generate_frame(image, None)
	}

	/// Returns the next `Frame`, which can be drawn to and will present on drop. The frame will be cleared to the specified
	/// `clear_color` (converted from sRGB with a gamma of 2.0). This `Renderer` is borrowed mutably while the `Frame` is alive
	pub fn next_frame_clear(&mut self, clear_color: Color) -> Frame<'_, WindowFrame> {
		let image = self.acquire_image();
		self.generate_frame(image, Some(clear_color))
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

	fn generate_frame(&mut self, image: backend::SwapchainImage, clear_colour: Option<Color>) -> Frame<'_, WindowFrame> {
		use std::borrow::Borrow;

		let viewport = gfx_hal::pso::Viewport {
			rect: gfx_hal::pso::Rect {
				x: 0,
				y: 0,
				w: self.swapchain_config.extent.width as i16,
				h: self.swapchain_config.extent.height as i16,
			},
			depth: 0.0..1.0,
		};
		
		let frame_idx = self.context.wait_next_frame();

		unsafe {
			self.context.device.reset_fence(&mut *self.context.render_fences[frame_idx].borrow_mut()).unwrap();
		}
		let mut command_buffer = self.context.render_command_buffers[frame_idx].borrow_mut();

		unsafe {
			command_buffer.reset(false);

			command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
					
			command_buffer.set_viewports(0, iter![viewport.clone()]);
			command_buffer.set_scissors(0, iter![viewport.rect]);

			command_buffer.begin_render_pass(
				&self.context.render_pass,
				&self.framebuffers[frame_idx],
				viewport.rect,
				iter![
					gfx_hal::command::RenderAttachmentInfo {
						image_view: image.borrow(),
						clear_value: gfx_hal::command::ClearValue {
							color: gfx_hal::command::ClearColor {
								float32:
								if let Some(clear_colour) = clear_colour {[
									(clear_colour.r as f32).powi(2) / 65_025.0,
									(clear_colour.g as f32).powi(2) / 65_025.0,
									(clear_colour.b as f32).powi(2) / 65_025.0,
									clear_colour.a as f32 / 255.0,
								]} else {[
									0.0,
									0.0,
									0.0,
									0.0,
								]}
							}
						}
					},
					gfx_hal::command::RenderAttachmentInfo {
						image_view: &*self.depth_texture.view,
						clear_value: gfx_hal::command::ClearValue {
							depth_stencil: gfx_hal::command::ClearDepthStencil {
								depth: 0.0,
								stencil: 0,
							}
						}
					}
				],
				gfx_hal::command::SubpassContents::Inline
			);
		}

		std::mem::drop(command_buffer);

		Frame::new(
			self.context.clone(),
			frame_idx,
			WindowFrame {
				surface: &mut self.surface,
				swap_chain_frame: ManuallyDrop::new(image),
			},
			viewport,
		)
	}

	fn reconfigure_swapchain(&mut self) {
		self.framebuffers = (0..self.context.frames_in_flight).map(|_| ManuallyDrop::new(unsafe { self.context.device.create_framebuffer(
			&self.context.render_pass,
			iter![
				gfx_hal::image::FramebufferAttachment {
					usage: gfx_hal::image::Usage::COLOR_ATTACHMENT,
					view_caps: gfx_hal::image::ViewCapabilities::empty(),
					format: gfx_hal::format::Format::Bgra8Srgb,
				},
				gfx_hal::image::FramebufferAttachment {
					usage: gfx_hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
					view_caps: gfx_hal::image::ViewCapabilities::empty(),
					format: gfx_hal::format::Format::D32Sfloat,
				}
			],
			self.extent.get().to_extent()
		)}.unwrap())).collect();
		unsafe { self.surface.configure_swapchain(&self.context.device, self.swapchain_config.clone()) }.unwrap();
	}
	
	/// Resizes the internal swapchain and depth texture
	/// 
	/// Call this method in your window's event loop whenever the window gets resized
	/// 
	/// # Arguments
	/// * `size`: The size of the window in pixels, in the order (width, height). For window implementations which
	///           differentiate between physical and logical size, this refers to the logical size
	pub fn resize(&mut self, (width, height): (u32, u32)) {
		self.swapchain_config.extent.width = width;
		self.swapchain_config.extent.height = height;
		self.extent.set(self.swapchain_config.extent);
		self.reconfigure_swapchain();

		self.depth_texture = DepthTexture::new(self.context.clone(), self.swapchain_config.extent);
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

impl Drop for WindowTarget {
	fn drop(&mut self) {
		unsafe {
			let mut surface = ManuallyDrop::take(&mut self.surface);
			surface.unconfigure_swapchain(&self.context.device);
			self.context.instance.destroy_surface(surface);
		}
	}
}

impl<'a> RenderTarget<'a> for WindowTarget {
	type FrameDrop = WindowFrame<'a>;

	fn create_frame(&'a mut self) -> Frame<WindowFrame<'a>> {
		self.next_frame()
	}
}

/// Implementation detail of the `RenderTarget` system.
/// 
/// See [`Frame`]
#[doc(hidden)]
pub struct WindowFrame<'a> {
	surface: &'a mut backend::Surface,
	swap_chain_frame: ManuallyDrop<<backend::Surface as gfx_hal::window::PresentationSurface<backend::Backend>>::SwapchainImage>,
}

impl<'a> RenderDrop<'a> for WindowFrame<'a> {
	fn finalize(&mut self, _context: &Renderer, _frame_idx: usize) {
		// Nothing to finalize
	}

	fn cleanup(&mut self, context: &Renderer, frame_idx: usize) {
		if !std::thread::panicking() {
			unsafe {
				let mut queue_groups = context.queue_groups.borrow_mut();
				queue_groups[0].queues[0].present(&mut self.surface, ManuallyDrop::take(&mut self.swap_chain_frame), Some(&mut *context.render_semaphores[frame_idx].borrow_mut())).unwrap();
			}
		} else {
			unsafe {
				ManuallyDrop::drop(&mut self.swap_chain_frame);
			}
		}
	}
}

enum RenderResource {
	Buffer(backend::Buffer, MemoryBlock<backend::Memory>),
	// Expecting expansion
}

impl RenderResource {
	fn unwrap_buffer_ref(&self) -> (&backend::Buffer, &MemoryBlock<backend::Memory>) {
		match self {
			Self::Buffer(b, m) => (b, m),
		}
	}
}

/// A frame to be drawn to. The frame gets presented on drop.
pub struct Frame<'a, T: RenderDrop<'a>> {
	context: Rc<Renderer>,
	frame_idx: usize,
	resources: T,
	viewport: gfx_hal::pso::Viewport,
	world_transform: Matrix4,
	_marker: std::marker::PhantomData<&'a T>,
}

impl<'a, T: RenderDrop<'a>> Frame<'a, T> {
	pub fn new(context: Rc<Renderer>, frame_idx: usize, resources: T, viewport: gfx_hal::pso::Viewport) -> Frame<'a, T> {
		Frame {
			context,
			frame_idx,
			resources,
			viewport,
			world_transform: Matrix4::identity(),
			_marker: std::marker::PhantomData,
		}
	}

	/// Creates a vertex buffer and an index buffer from the supplied data. The buffers will be placed into the current
	/// frame's resources at the end, vertex buffer first, index buffer second.
	fn create_staging_buffers(&mut self, vertices: &[u8], indices: &[u8]) {
		let mut vertex_buffer = unsafe {
			self.context.device.create_buffer(vertices.len() as u64, gfx_hal::buffer::Usage::VERTEX)
		}.unwrap();
		let mut index_buffer = unsafe {
			self.context.device.create_buffer(indices.len() as u64, gfx_hal::buffer::Usage::INDEX)
		}.unwrap();
		let vertex_mem_req = unsafe { self.context.device.get_buffer_requirements(&vertex_buffer) };
		let index_mem_req = unsafe { self.context.device.get_buffer_requirements(&index_buffer) };

		let memory_device = GfxMemoryDevice::wrap(&self.context.device);
		let mut vertex_block = unsafe {
			self.context.allocator.borrow_mut().alloc(
				memory_device,
				Request {
					size: vertex_mem_req.size,
					align_mask: vertex_mem_req.alignment,
					memory_types: vertex_mem_req.type_mask,
					usage: UsageFlags::UPLOAD | UsageFlags::TRANSIENT, // Implies host-visible
				}
			)
		}.unwrap();
		let mut index_block = unsafe {
			self.context.allocator.borrow_mut().alloc(
				memory_device,
				Request {
					size: index_mem_req.size,
					align_mask: index_mem_req.alignment,
					memory_types: index_mem_req.type_mask,
					usage: UsageFlags::UPLOAD | UsageFlags::TRANSIENT,
				}
			)
		}.unwrap();
		unsafe {
			vertex_block.write_bytes(memory_device, 0, vertices).unwrap();
			index_block.write_bytes(memory_device, 0, indices).unwrap();
			self.context.device.bind_buffer_memory(&vertex_block.memory(), vertex_block.offset(), &mut vertex_buffer).unwrap();
			self.context.device.bind_buffer_memory(&index_block.memory(), index_block.offset(), &mut index_buffer).unwrap();
		}
		let mut frame_res = self.context.render_frame_resources[self.frame_idx].borrow_mut();
		frame_res.push(RenderResource::Buffer(vertex_buffer, vertex_block));
		frame_res.push(RenderResource::Buffer(index_buffer, index_block));
	}

	/// Sets the global transform matrix for draw calls after this method call.
	/// 
	/// If this method is called multiple times, draw calls will use the matrix provided most recently.
	/// 
	/// Draw calls made before this method call use the identity matrix as the global transform matrix.
	pub fn set_global_transform(&mut self, matrix: Matrix4) {
		self.world_transform = matrix;
	}

	/// Draws a [`StrokedShape`](vertex/struct.StrokedShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	pub fn draw_stroked(&mut self, shape: StrokedShape<'_>, obj_transforms: &[Matrix4]) {
		self.create_staging_buffers(bytemuck::cast_slice(&shape.vertices), bytemuck::cast_slice(&shape.indices));
		let resources = self.context.render_frame_resources[self.frame_idx].borrow();
		let (vertex_buffer, _) = resources[resources.len() - 2].unwrap_buffer_ref();
		let (index_buffer, _) = resources[resources.len() - 1].unwrap_buffer_ref();
		let mut command_buffer = self.context.render_command_buffers[self.frame_idx].borrow_mut();

		unsafe {
			command_buffer.bind_vertex_buffers(0, iter![(vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			command_buffer.bind_index_buffer(
				index_buffer,
				gfx_hal::buffer::SubRange::WHOLE,
				gfx_hal::IndexType::U16,
			);

			command_buffer.bind_graphics_pipeline(&self.context.stroked_graphics_pipeline);
			command_buffer.push_graphics_constants(
				&self.context.stroked_graphics_pipeline_layout,
				gfx_hal::pso::ShaderStageFlags::VERTEX,
				0,
				bytemuck::cast_slice::<[[f32; 4]; 4], _>(&[self.world_transform.into()]),
			);
		}

		for chunk in obj_transforms.chunks(self.context.matrix_array_size as usize) {
			unsafe {
				command_buffer.push_graphics_constants(
					&self.context.stroked_graphics_pipeline_layout,
					gfx_hal::pso::ShaderStageFlags::VERTEX,
					std::mem::size_of::<Matrix4>() as u32,
					bytemuck::cast_slice(&chunk.iter().map(|&mat| mat.into()).collect::<Vec<[[f32; 4]; 4]>>()), //TODO: Remove this allocation
				);
				command_buffer.draw_indexed(0..shape.indices.len() as u32 * 3, 0, 0..chunk.len() as u32);
			}
		}
	}

	/// Draws a [`ColoredShape`](vertex/struct.ColoredShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	pub fn draw_colored(&mut self, shape: ColoredShape, obj_transforms: &[Matrix4]) {
		self.create_staging_buffers(bytemuck::cast_slice(&shape.vertices), bytemuck::cast_slice(&shape.indices));
		let resources = self.context.render_frame_resources[self.frame_idx].borrow();
		let (vertex_buffer, _) = resources[resources.len() - 2].unwrap_buffer_ref();
		let (index_buffer, _) = resources[resources.len() - 1].unwrap_buffer_ref();
		let mut command_buffer = self.context.render_command_buffers[self.frame_idx].borrow_mut();

		unsafe {
			command_buffer.bind_vertex_buffers(0, iter![(vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			command_buffer.bind_index_buffer(
				index_buffer,
				gfx_hal::buffer::SubRange::WHOLE,
				gfx_hal::IndexType::U16,
			);

			command_buffer.bind_graphics_pipeline(&self.context.colour_graphics_pipeline);
			command_buffer.push_graphics_constants(
				&self.context.colour_graphics_pipeline_layout,
				gfx_hal::pso::ShaderStageFlags::VERTEX,
				0,
				bytemuck::cast_slice::<[[f32; 4]; 4], _>(&[self.world_transform.into()]),
			);
		}

		for chunk in obj_transforms.chunks(self.context.matrix_array_size as usize) {
			unsafe {
				command_buffer.push_graphics_constants(
					&self.context.colour_graphics_pipeline_layout,
					gfx_hal::pso::ShaderStageFlags::VERTEX,
					std::mem::size_of::<Matrix4>() as u32,
					bytemuck::cast_slice(&chunk),
				);
				command_buffer.draw_indexed(0..shape.indices.len() as u32 * 3, 0, 0..chunk.len() as u32);
			}
		}
	}

	/// Draws a [`TexturedShape`](vertex/struct.TexturedShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	///
	/// `iterations` is a slice of texture references and matrices to draw that texture with.
	pub fn draw_textured(&mut self, shape: TexturedShape, iterations: &[(&'a Texture, &[Matrix4])]) {
		self.create_staging_buffers(bytemuck::cast_slice(&shape.vertices), bytemuck::cast_slice(&shape.indices));
		let resources = self.context.render_frame_resources[self.frame_idx].borrow();
		let (vertex_buffer, _) = resources[resources.len() - 2].unwrap_buffer_ref();
		let (index_buffer, _) = resources[resources.len() - 1].unwrap_buffer_ref();
		let mut command_buffer = self.context.render_command_buffers[self.frame_idx].borrow_mut();

		unsafe {
			command_buffer.bind_vertex_buffers(0, iter![(vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			command_buffer.bind_index_buffer(
				index_buffer,
				gfx_hal::buffer::SubRange::WHOLE,
				gfx_hal::IndexType::U16,
			);
			
			command_buffer.bind_graphics_pipeline(&self.context.texture_graphics_pipeline);
			command_buffer.push_graphics_constants(
				&self.context.texture_graphics_pipeline_layout,
				gfx_hal::pso::ShaderStageFlags::VERTEX,
				0,
				bytemuck::cast_slice::<[[f32; 4]; 4], _>(&[self.world_transform.into()]),
			);
		}

		for (texture, obj_transforms) in iterations {
			if !Rc::ptr_eq(&self.context, &texture.context) {
				panic!("Texture was not made with renderer that made this frame");
			}
			
			unsafe {
				command_buffer.bind_graphics_descriptor_sets(
					&self.context.texture_graphics_pipeline_layout,
					0,
					iter![&*texture.descriptor_set],
					iter![],
				);
			}

			for chunk in obj_transforms.chunks(self.context.matrix_array_size as usize) {
				unsafe {
					command_buffer.push_graphics_constants(
						&self.context.texture_graphics_pipeline_layout,
						gfx_hal::pso::ShaderStageFlags::VERTEX,
						std::mem::size_of::<Matrix4>() as u32,
						bytemuck::cast_slice(&chunk),
					);
					command_buffer.draw_indexed(0..shape.indices.len() as u32 * 3, 0, 0..chunk.len() as u32);
				}
			}
		}
	}

	/// Converts pixel coordinates to Gpu coordinates
	pub fn pixel(&self, x: i32, y: i32) -> Vector2 {
		Vector2::new((x * 2) as f32 / self.viewport.rect.w as f32 - 1.0, -((y * 2) as f32 / self.viewport.rect.h as f32 - 1.0))
	}

	/// Consumes and presents this frame. Equivalent to `std::mem::drop(self)`
	///
	/// Calling this method is **NOT** required for the `Frame` to present. The frame will present on drop.
	/// This method exists to make presentation within a scope more explicit than introducing a sub-scope or
	/// using [`std::mem::drop`]
	pub fn present(self) {
		// Consumes self, so it will drop (and therefore present) at the end of this method
	}
}

impl<'a, T: RenderDrop<'a>> HasRenderer for Frame<'a, T> {
	fn clone_context(&self) -> Rc<Renderer> {
		self.context.clone()
	}
}

impl<'a, T: RenderDrop<'a>> Drop for Frame<'a, T> {
	fn drop(&mut self) {
		self.resources.finalize(&self.context, self.frame_idx);

		unsafe {
			let mut command_buffer = self.context.render_command_buffers[self.frame_idx].borrow_mut();
			command_buffer.end_render_pass();
			command_buffer.finish();

			let mut queue_groups = self.context.queue_groups.borrow_mut();
				queue_groups[0].queues[0].submit(
					iter![&*command_buffer],
					iter![],
					iter![&*self.context.render_semaphores[self.frame_idx].borrow()],
					Some(&mut *self.context.render_fences[self.frame_idx].borrow_mut())
				);
		}

		self.resources.cleanup(&self.context, self.frame_idx);
	}
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
		PixelTranslator::new(Rc::new(Cell::new(self.extent)))
	}
}

impl HasRenderer for Texture {
	fn clone_context(&self) -> Rc<Renderer> {
		self.context.clone()
	}
}

impl<'a> RenderTarget<'a> for Texture {
	type FrameDrop = TextureFrame<'a>;

	fn create_frame(&'a mut self) -> Frame<'a, TextureFrame<'a>> {
		let viewport = gfx_hal::pso::Viewport {
			rect: gfx_hal::pso::Rect {
				x: 0,
				y: 0,
				w: self.extent.width as i16,
				h: self.extent.height as i16,
			},
			depth: 0.0..1.0,
		};

		let depth_texture = DepthTexture::new(self.context.clone(), self.extent);

		let framebuffer = unsafe { self.context.device.create_framebuffer(
			&self.context.render_pass,
			iter![
				gfx_hal::image::FramebufferAttachment {
					usage: gfx_hal::image::Usage::COLOR_ATTACHMENT,
					view_caps: gfx_hal::image::ViewCapabilities::empty(),
					format: gfx_hal::format::Format::Bgra8Srgb,
				},
				gfx_hal::image::FramebufferAttachment {
					usage: gfx_hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
					view_caps: gfx_hal::image::ViewCapabilities::empty(),
					format: gfx_hal::format::Format::D32Sfloat,
				}
			],
			self.extent.to_extent()
		)}.unwrap();

		let frame_idx = self.context.wait_next_frame();

		unsafe {
			let mut fence = self.fence.borrow_mut();
			self.context.device.wait_for_fence(&fence, u64::MAX).unwrap();
			self.context.device.reset_fence(&mut fence).unwrap();
			self.context.device.reset_fence(&mut *self.context.render_fences[frame_idx].borrow_mut()).unwrap();
		}
		let mut command_buffer = self.context.render_command_buffers[frame_idx].borrow_mut();

		unsafe {
			command_buffer.reset(false);

			command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
					
			command_buffer.set_viewports(0, iter![viewport.clone()]);
			command_buffer.set_scissors(0, iter![viewport.rect]);

			command_buffer.begin_render_pass(
				&self.context.render_pass,
				&framebuffer,
				viewport.rect,
				iter![
					gfx_hal::command::RenderAttachmentInfo {
						image_view: &*self.view,
						clear_value: gfx_hal::command::ClearValue {
							color: gfx_hal::command::ClearColor {
								float32: [0.0, 0.0, 0.0, 0.0]
							}
						}
					},
					gfx_hal::command::RenderAttachmentInfo {
						image_view: &*depth_texture.view,
						clear_value: gfx_hal::command::ClearValue {
							depth_stencil: gfx_hal::command::ClearDepthStencil {
								depth: 0.0,
								stencil: 0,
							}
						}
					}
				],
				gfx_hal::command::SubpassContents::Inline
			);

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

		Frame::new(
			self.context.clone(),
			frame_idx,
			TextureFrame {
				framebuffer: ManuallyDrop::new(framebuffer),
				depth_texture,
				image: &self.image,
			},
			viewport
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
#[doc(hidden)]
pub struct TextureFrame<'a> {
	framebuffer: ManuallyDrop<backend::Framebuffer>,
	#[allow(dead_code)] // Must be kept alive while the framebuffer is still using its ImageView.
	depth_texture: DepthTexture,
	image: &'a backend::Image,
}

impl<'a> RenderDrop<'a> for TextureFrame<'a> {
	fn finalize(&mut self, context: &Renderer, frame_idx: usize) {
		unsafe {
			context.render_command_buffers[frame_idx].borrow_mut().pipeline_barrier(
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

	fn cleanup(&mut self, context: &Renderer, _frame_idx: usize) {
		unsafe {
			context.device.destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffer));
		}
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