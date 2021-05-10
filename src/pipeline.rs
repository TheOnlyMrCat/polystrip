use std::cell::RefCell;
use std::mem::ManuallyDrop;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use gfx_hal::prelude::*;

use gpu_alloc::{MemoryBlock, Request, UsageFlags};
use gpu_alloc_gfx::GfxMemoryDevice;

use crate::{BaseFrame, DepthTexture, HasRenderSize, HasRenderer, RenderPipeline, RenderSize, Texture, backend};
use crate::vertex::*;
use crate::Renderer;
use crate::iter;

use align_data::{include_aligned, Align32};
static COLOURED_VERT_SPV: &[u8] = include_aligned!(Align32, "spirv/coloured.vert.spv");
static COLOURED_FRAG_SPV: &[u8] = include_aligned!(Align32, "spirv/coloured.frag.spv");
static TEXTURED_VERT_SPV: &[u8] = include_aligned!(Align32, "spirv/textured.vert.spv");
static TEXTURED_FRAG_SPV: &[u8] = include_aligned!(Align32, "spirv/textured.frag.spv");

pub struct StandardPipeline {
	context: Rc<Renderer>,
	size_handle: RenderSize,

	stroked_graphics_pipeline: ManuallyDrop<backend::GraphicsPipeline>,
	stroked_graphics_pipeline_layout: ManuallyDrop<backend::PipelineLayout>,
	colour_graphics_pipeline: ManuallyDrop<backend::GraphicsPipeline>,
	colour_graphics_pipeline_layout: ManuallyDrop<backend::PipelineLayout>,
	texture_graphics_pipeline: ManuallyDrop<backend::GraphicsPipeline>,
	texture_graphics_pipeline_layout: ManuallyDrop<backend::PipelineLayout>,
	render_pass: ManuallyDrop<backend::RenderPass>,

	frames_in_flight: usize,
	matrix_array_size: u32,

	current_frame: AtomicUsize,
	render_command_buffers: ManuallyDrop<Vec<RefCell<backend::CommandBuffer>>>,

	render_semaphores: ManuallyDrop<Vec<RefCell<backend::Semaphore>>>,
	render_fences: ManuallyDrop<Vec<RefCell<backend::Fence>>>,
	render_frame_resources: ManuallyDrop<Vec<ManuallyDrop<RefCell<Vec<RenderResource>>>>>,

	framebuffers: Vec<ManuallyDrop<backend::Framebuffer>>,
	depth_textures: Vec<ManuallyDrop<DepthTexture>>,
}

impl StandardPipeline {
	pub fn with_config(context: &impl HasRenderer, size: &impl HasRenderSize, config: StandardPipelineBuilder) -> StandardPipeline {
		let context = context.clone_context();

		let render_command_buffers = {
			let mut buffers = Vec::with_capacity(config.frames_in_flight as usize);
			unsafe { context.render_command_pool.borrow_mut().allocate(config.frames_in_flight as usize, gfx_hal::command::Level::Primary, &mut buffers); }
			buffers
		}.into_iter().map(RefCell::new).collect();

		let (render_semaphores, render_fences, render_frame_resources) = {
			let mut semaphores = Vec::with_capacity(config.frames_in_flight as usize);
			let mut fences = Vec::with_capacity(config.frames_in_flight as usize);
			let mut resources = Vec::with_capacity(config.frames_in_flight as usize);
			for _ in 0..config.frames_in_flight {
				semaphores.push(RefCell::new(context.device.create_semaphore().unwrap()));
				fences.push(RefCell::new(context.device.create_fence(false).unwrap()));
				resources.push(ManuallyDrop::new(RefCell::new(Vec::new())));
			}
			(semaphores, fences, resources)
		};

		// The number of object transform matrices we can store in push constants, and therefore how many objects we can draw at once
		// Equal to the number of matrices we can store, minus 1 for the world matrix
		let matrix_array_size = (context.adapter.physical_device.limits().max_push_constants_size / std::mem::size_of::<Matrix4>() - 1).min((u32::MAX - 1) as usize) as u32;

		let main_pass = unsafe { context.device.create_render_pass(
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

		let colour_vs_module = unsafe { context.device.create_shader_module(bytemuck::cast_slice(COLOURED_VERT_SPV)) }.unwrap();
		let colour_fs_module = unsafe { context.device.create_shader_module(bytemuck::cast_slice(COLOURED_FRAG_SPV)) }.unwrap();
		let texture_vs_module = unsafe { context.device.create_shader_module(bytemuck::cast_slice(TEXTURED_VERT_SPV)) }.unwrap();
		let texture_fs_module = unsafe { context.device.create_shader_module(bytemuck::cast_slice(TEXTURED_FRAG_SPV)) }.unwrap();

		let stroked_graphics_pipeline_layout = unsafe { context.device.create_pipeline_layout(iter![], iter![(gfx_hal::pso::ShaderStageFlags::VERTEX, 0..std::mem::size_of::<Matrix4>() as u32 * (matrix_array_size + 1))]) }.unwrap();
		let stroked_graphics_pipeline = unsafe { context.device.create_graphics_pipeline(&gfx_hal::pso::GraphicsPipelineDesc {
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

		let colour_graphics_pipeline_layout = unsafe { context.device.create_pipeline_layout(iter![], iter![(gfx_hal::pso::ShaderStageFlags::VERTEX, 0..std::mem::size_of::<Matrix4>() as u32 * (matrix_array_size + 1))]) }.unwrap();
		let colour_graphics_pipeline = unsafe { context.device.create_graphics_pipeline(&gfx_hal::pso::GraphicsPipelineDesc {
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

		let texture_graphics_pipeline_layout = unsafe { context.device.create_pipeline_layout(iter![&*context.texture_descriptor_set_layout], iter![(gfx_hal::pso::ShaderStageFlags::VERTEX, 0..std::mem::size_of::<Matrix4>() as u32 * (matrix_array_size + 1))]) }.unwrap();
		let texture_graphics_pipeline = unsafe { context.device.create_graphics_pipeline(&gfx_hal::pso::GraphicsPipelineDesc {
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

		let render_size = size.clone_size_handle();

		let framebuffers = (0..config.frames_in_flight).map(|_| ManuallyDrop::new(unsafe { context.device.create_framebuffer(
			&main_pass,
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
			render_size.get().to_extent()
		)}.unwrap())).collect();

		let depth_textures = (0..config.frames_in_flight).map(|_| ManuallyDrop::new(DepthTexture::new(context.clone(), render_size.get()))).collect();

		StandardPipeline {
		    context,
			size_handle: render_size,

			stroked_graphics_pipeline: ManuallyDrop::new(stroked_graphics_pipeline),
			stroked_graphics_pipeline_layout: ManuallyDrop::new(stroked_graphics_pipeline_layout),
			colour_graphics_pipeline: ManuallyDrop::new(colour_graphics_pipeline),
			colour_graphics_pipeline_layout: ManuallyDrop::new(colour_graphics_pipeline_layout),
			texture_graphics_pipeline: ManuallyDrop::new(texture_graphics_pipeline),
			texture_graphics_pipeline_layout: ManuallyDrop::new(texture_graphics_pipeline_layout),
			render_pass: ManuallyDrop::new(main_pass),
			
			frames_in_flight: config.frames_in_flight,
			matrix_array_size,

			current_frame: AtomicUsize::new(0),
			render_command_buffers: ManuallyDrop::new(render_command_buffers),
			render_semaphores: ManuallyDrop::new(render_semaphores),
			render_fences: ManuallyDrop::new(render_fences),
			render_frame_resources: ManuallyDrop::new(render_frame_resources),

		    framebuffers,
		    depth_textures,
		}
	}

	/// Waits for the next frame to finish rendering, deallocates its resources, and returns its index.
	/// 
	/// Generally, this won't need to be called in application code, since it is done by [`RenderTarget`]s before creating
	/// a [`Frame`].
	pub fn wait_next_frame(&self) -> usize {
		let frame_idx = self.next_frame_idx();
		unsafe {
			self.context.device.wait_for_fence(&self.render_fences[frame_idx].borrow(), u64::MAX).unwrap();
		}

		let mut allocator = self.context.allocator.borrow_mut();
		let mem_device = GfxMemoryDevice::wrap(&self.context.device);
		for resource in self.render_frame_resources[frame_idx].replace(Vec::new()) {
			match resource {
				RenderResource::Buffer(buffer, memory) => unsafe {
					allocator.dealloc(mem_device, memory);
					self.context.device.destroy_buffer(buffer);
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

impl<'a> RenderPipeline<'a> for StandardPipeline {
	type Frame = StandardFrame<'a>;

	fn render_to(&'a self, base: BaseFrame<'a>) -> Self::Frame {
		let frame_idx = self.wait_next_frame();

		unsafe {
			self.context.device.reset_fence(&mut *self.render_fences[frame_idx].borrow_mut()).unwrap();
		}
		let mut command_buffer = self.render_command_buffers[frame_idx].borrow_mut();

		unsafe {
			command_buffer.reset(false);

			command_buffer.begin_primary(gfx_hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
					
			command_buffer.set_viewports(0, iter![base.viewport.clone()]);
			command_buffer.set_scissors(0, iter![base.viewport.rect]);

			command_buffer.begin_render_pass(
				&self.render_pass,
				&self.framebuffers[frame_idx],
				base.viewport.rect,
				iter![
					gfx_hal::command::RenderAttachmentInfo {
						image_view: base.image,
						clear_value: gfx_hal::command::ClearValue {
							color: gfx_hal::command::ClearColor {
								float32:[0.0, 0.0, 0.0, 0.0]
							}
						}
					},
					gfx_hal::command::RenderAttachmentInfo {
						image_view: &*self.depth_textures[frame_idx].view,
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

        StandardFrame {
			context: base.context.clone(),
			base,
			pipeline: self,
			frame_idx,
			world_transform: Matrix4::identity(),
		}
    }
}

impl Drop for StandardPipeline {
	fn drop(&mut self) {
		unsafe {
			let fences = ManuallyDrop::take(&mut self.render_fences).into_iter().map(RefCell::into_inner).collect::<Vec<_>>();
			self.context.device.wait_for_fences(fences.iter(), gfx_hal::device::WaitFor::All, 5000000).unwrap();
			for fence in fences {
				self.context.device.destroy_fence(fence);
			}

			for semaphore in ManuallyDrop::take(&mut self.render_semaphores) {
				self.context.device.destroy_semaphore(semaphore.into_inner());
			}

			let mut allocator = self.context.allocator.borrow_mut();
			let mem_device = GfxMemoryDevice::wrap(&self.context.device);

			for resource in ManuallyDrop::take(&mut self.render_frame_resources).into_iter().flat_map(|mut i| ManuallyDrop::take(&mut i).into_inner()) {
				match resource {
					RenderResource::Buffer(buf, block) => {
						self.context.device.destroy_buffer(buf);
						allocator.dealloc(mem_device, block);
					}
				}
			}

			let mut render_command_pool = self.context.render_command_pool.borrow_mut();
			render_command_pool.free(ManuallyDrop::take(&mut self.render_command_buffers).into_iter().map(RefCell::into_inner));

			self.context.device.destroy_graphics_pipeline(ManuallyDrop::take(&mut self.stroked_graphics_pipeline));
			self.context.device.destroy_graphics_pipeline(ManuallyDrop::take(&mut self.colour_graphics_pipeline));
			self.context.device.destroy_graphics_pipeline(ManuallyDrop::take(&mut self.texture_graphics_pipeline));
			self.context.device.destroy_pipeline_layout(ManuallyDrop::take(&mut self.stroked_graphics_pipeline_layout));
			self.context.device.destroy_pipeline_layout(ManuallyDrop::take(&mut self.colour_graphics_pipeline_layout));
			self.context.device.destroy_pipeline_layout(ManuallyDrop::take(&mut self.texture_graphics_pipeline_layout));
			
			self.context.device.destroy_render_pass(ManuallyDrop::take(&mut self.render_pass));
		}
	}
}

pub struct StandardPipelineBuilder {
	real_3d: bool,
	frames_in_flight: usize,
}

impl StandardPipelineBuilder {
	pub fn new() -> StandardPipelineBuilder {
		Self {
			real_3d: false,
			frames_in_flight: 3,
		}
	}

	/// If `true`, allows transform matrices to affect sprite depth. This clamps the depth between `0.0` and `1.0`
	/// 
	/// Default: false
	pub fn real_3d(mut self, real_3d: bool) -> StandardPipelineBuilder {
		self.real_3d = real_3d;
		self
	}

	/// The number of frames that can be dispatched simultaneously
	/// 
	/// Default: 3
	pub fn frames_in_flight(mut self, frames_in_flight: usize) -> StandardPipelineBuilder {
		self.frames_in_flight = frames_in_flight;
		self
	}

	/// Builds the renderer, initialising the `gfx_hal` backend.
	pub fn build(self, context: &impl HasRenderer, size: &impl HasRenderSize) -> StandardPipeline {
		StandardPipeline::with_config(context, size, self)
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
pub struct StandardFrame<'a> {
	context: Rc<Renderer>,
	base: BaseFrame<'a>,
	pipeline: &'a StandardPipeline,
	frame_idx: usize,
	world_transform: Matrix4,
}

impl<'a> StandardFrame<'a> {
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
		let mut frame_res = self.pipeline.render_frame_resources[self.frame_idx].borrow_mut();
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
		let resources = self.pipeline.render_frame_resources[self.frame_idx].borrow();
		let (vertex_buffer, _) = resources[resources.len() - 2].unwrap_buffer_ref();
		let (index_buffer, _) = resources[resources.len() - 1].unwrap_buffer_ref();
		let mut command_buffer = self.pipeline.render_command_buffers[self.frame_idx].borrow_mut();

		unsafe {
			command_buffer.bind_vertex_buffers(0, iter![(vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			command_buffer.bind_index_buffer(
				index_buffer,
				gfx_hal::buffer::SubRange::WHOLE,
				gfx_hal::IndexType::U16,
			);

			command_buffer.bind_graphics_pipeline(&self.pipeline.stroked_graphics_pipeline);
			command_buffer.push_graphics_constants(
				&self.pipeline.stroked_graphics_pipeline_layout,
				gfx_hal::pso::ShaderStageFlags::VERTEX,
				0,
				bytemuck::cast_slice::<[[f32; 4]; 4], _>(&[self.world_transform.into()]),
			);
		}

		for chunk in obj_transforms.chunks(self.pipeline.matrix_array_size as usize) {
			unsafe {
				command_buffer.push_graphics_constants(
					&self.pipeline.stroked_graphics_pipeline_layout,
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
		let resources = self.pipeline.render_frame_resources[self.frame_idx].borrow();
		let (vertex_buffer, _) = resources[resources.len() - 2].unwrap_buffer_ref();
		let (index_buffer, _) = resources[resources.len() - 1].unwrap_buffer_ref();
		let mut command_buffer = self.pipeline.render_command_buffers[self.frame_idx].borrow_mut();

		unsafe {
			command_buffer.bind_vertex_buffers(0, iter![(vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			command_buffer.bind_index_buffer(
				index_buffer,
				gfx_hal::buffer::SubRange::WHOLE,
				gfx_hal::IndexType::U16,
			);

			command_buffer.bind_graphics_pipeline(&self.pipeline.colour_graphics_pipeline);
			command_buffer.push_graphics_constants(
				&self.pipeline.colour_graphics_pipeline_layout,
				gfx_hal::pso::ShaderStageFlags::VERTEX,
				0,
				bytemuck::cast_slice::<[[f32; 4]; 4], _>(&[self.world_transform.into()]),
			);
		}

		for chunk in obj_transforms.chunks(self.pipeline.matrix_array_size as usize) {
			unsafe {
				command_buffer.push_graphics_constants(
					&self.pipeline.colour_graphics_pipeline_layout,
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
		let resources = self.pipeline.render_frame_resources[self.frame_idx].borrow();
		let (vertex_buffer, _) = resources[resources.len() - 2].unwrap_buffer_ref();
		let (index_buffer, _) = resources[resources.len() - 1].unwrap_buffer_ref();
		let mut command_buffer = self.pipeline.render_command_buffers[self.frame_idx].borrow_mut();

		unsafe {
			command_buffer.bind_vertex_buffers(0, iter![(vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)]);
			command_buffer.bind_index_buffer(
				index_buffer,
				gfx_hal::buffer::SubRange::WHOLE,
				gfx_hal::IndexType::U16,
			);
			
			command_buffer.bind_graphics_pipeline(&self.pipeline.texture_graphics_pipeline);
			command_buffer.push_graphics_constants(
				&self.pipeline.texture_graphics_pipeline_layout,
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
					&self.pipeline.texture_graphics_pipeline_layout,
					0,
					iter![&*texture.descriptor_set],
					iter![],
				);
			}

			for chunk in obj_transforms.chunks(self.pipeline.matrix_array_size as usize) {
				unsafe {
					command_buffer.push_graphics_constants(
						&self.pipeline.texture_graphics_pipeline_layout,
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
		Vector2::new((x * 2) as f32 / self.base.viewport.rect.w as f32 - 1.0, -((y * 2) as f32 / self.base.viewport.rect.h as f32 - 1.0))
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

impl Drop for StandardFrame<'_> {
	fn drop(&mut self) {
		self.base.drop_finalize();

		unsafe {
			let mut command_buffer = self.pipeline.render_command_buffers[self.frame_idx].borrow_mut();
			command_buffer.end_render_pass();
			command_buffer.finish();

			let mut queue_groups = self.context.queue_groups.borrow_mut();
				queue_groups[0].queues[0].submit(
					iter![&*command_buffer],
					iter![],
					iter![&*self.pipeline.render_semaphores[self.frame_idx].borrow()],
					Some(&mut *self.pipeline.render_fences[self.frame_idx].borrow_mut())
				);
		}

		self.base.drop_cleanup(Some(&mut *self.pipeline.render_semaphores[self.frame_idx].borrow_mut()));
	}
}