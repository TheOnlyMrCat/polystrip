//! The gon pipeline for 2D rendering

use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::ops::DerefMut;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use wgpu::util::DeviceExt;

use crate::RenderTexture;
use crate::SampledTexture;
use crate::math::*;
use crate::PolystripDevice;
use crate::{BaseFrame, DepthTexture, HasRenderSize, HasRenderer, RenderPipeline, RenderSize, ImageTexture};

/// The `gon` pipeline for 2D rendering.
pub struct GonPipeline {
	context: Rc<PolystripDevice>,
	size_handle: Rc<RenderSize>,

	stroked_graphics_pipeline: wgpu::RenderPipeline,
	colour_graphics_pipeline: wgpu::RenderPipeline,
	texture_graphics_pipeline: wgpu::RenderPipeline,

	frames_in_flight: usize,
	matrix_array_size: u32,
	current_frame: AtomicUsize,

	depth_textures: Vec<ManuallyDrop<DepthTexture>>,
	current_resource_size: (u32, u32),
}

impl GonPipeline {
	/// Creates a `GonPipeline` with default settings
	pub fn new(context: &impl HasRenderer, size: &impl HasRenderSize) -> GonPipeline {
		GonPipelineBuilder::default().build(context, size)
	}

	pub fn with_config(
		context: &impl HasRenderer,
		size: &impl HasRenderSize,
		config: GonPipelineBuilder,
	) -> GonPipeline {
		let context = context.clone_context();

		// The number of object transform matrices we can store in push constants, and therefore how many objects we can draw at once
		// Equal to the number of matrices we can store, minus 1 for the world matrix
		let matrix_array_size =
			(context.adapter.limits().max_push_constant_size as usize / std::mem::size_of::<Matrix4>() - 1).min(127)
				as u32;

		let colour_vs_module = context.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
			label: Some("Polystrip Gon/Colour Vertex shader"),
			source: wgpu::ShaderSource::Glsl {
				shader: include_str!("gon/coloured.vert").into(),
				stage: naga::ShaderStage::Vertex,
				defines: [("MAX_TRANSFORMS".to_owned(), matrix_array_size.to_string())].into_iter().collect(),
			},
		});
		let colour_fs_module = context.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
			label: Some("Polystrip Gon/Colour Fragment shader"),
			source: wgpu::ShaderSource::Glsl {
				shader: include_str!("gon/coloured.frag").into(),
				stage: naga::ShaderStage::Fragment,
				defines: Default::default(),
			},
		});
		let texture_vs_module = context.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
			label: Some("Polystrip Gon/Texture Vertex shader"),
			source: wgpu::ShaderSource::Glsl {
				shader: include_str!("gon/textured.vert").into(),
				stage: naga::ShaderStage::Vertex,
				defines: [("MAX_TRANSFORMS".to_owned(), matrix_array_size.to_string())].into_iter().collect(),
			},
		});
		let texture_fs_module = context.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
			label: Some("Polystrip Gon/Texture Fragment shader"),
			source: wgpu::ShaderSource::Glsl {
				shader: include_str!("gon/textured.frag").into(),
				stage: naga::ShaderStage::Fragment,
				defines: Default::default(),
			},
		});

		let stroked_graphics_pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("Polystrip Gon/Stroked layout"),
			bind_group_layouts: &[],
			push_constant_ranges: &[wgpu::PushConstantRange {
				stages: wgpu::ShaderStages::VERTEX,
				range: 0..std::mem::size_of::<Matrix4>() as u32 * (matrix_array_size + 1),
			}],
		});
		let stroked_graphics_pipeline = context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Polystrip Gon/Stroked"),
			layout: Some(&stroked_graphics_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &colour_vs_module,
				entry_point: "main",
				buffers: &[wgpu::VertexBufferLayout {
					array_stride: std::mem::size_of::<GpuColorVertex>() as u64,
					step_mode: wgpu::VertexStepMode::Vertex,
					attributes: GpuColorVertex::attributes(),
				}],
			},
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::LineList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: None,
				unclipped_depth: false,
				polygon_mode: wgpu::PolygonMode::Fill,
				conservative: false,
			},
			depth_stencil: Some(wgpu::DepthStencilState {
				format: wgpu::TextureFormat::Depth32Float,
				depth_write_enabled: true,
				depth_compare: wgpu::CompareFunction::GreaterEqual,
				stencil: wgpu::StencilState {
					front: wgpu::StencilFaceState::IGNORE,
					back: wgpu::StencilFaceState::IGNORE,
					read_mask: 0,
					write_mask: 0,
				},
				bias: wgpu::DepthBiasState::default(),
			}),
			multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: true },
			fragment: Some(wgpu::FragmentState {
				module: &colour_fs_module,
				entry_point: "main",
				targets: &[wgpu::ColorTargetState {
					format: wgpu::TextureFormat::Bgra8UnormSrgb,
					blend: Some(wgpu::BlendState {
						color: wgpu::BlendComponent {
							src_factor: wgpu::BlendFactor::SrcAlpha,
							dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
							operation: wgpu::BlendOperation::Add,
						},
						alpha: wgpu::BlendComponent {
							src_factor: wgpu::BlendFactor::One,
							dst_factor: wgpu::BlendFactor::One,
							operation: wgpu::BlendOperation::Add,
						},
					}),
					write_mask: wgpu::ColorWrites::ALL,
				}],
			}),
			multiview: None,
		});

		let colour_graphics_pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("Polystrip Gon/Colour layout"),
			bind_group_layouts: &[],
			push_constant_ranges: &[wgpu::PushConstantRange {
				stages: wgpu::ShaderStages::VERTEX,
				range: 0..std::mem::size_of::<Matrix4>() as u32 * (matrix_array_size + 1),
			}],
		});
		let colour_graphics_pipeline = context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Polystrip Gon/Colour"),
			layout: Some(&colour_graphics_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &colour_vs_module,
				entry_point: "main",
				buffers: &[wgpu::VertexBufferLayout {
					array_stride: std::mem::size_of::<GpuColorVertex>() as u64,
					step_mode: wgpu::VertexStepMode::Vertex,
					attributes: GpuColorVertex::attributes(),
				}],
			},
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::TriangleList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: None,
				unclipped_depth: false,
				polygon_mode: wgpu::PolygonMode::Fill,
				conservative: false,
			},
			depth_stencil: Some(wgpu::DepthStencilState {
				format: wgpu::TextureFormat::Depth32Float,
				depth_write_enabled: true,
				depth_compare: wgpu::CompareFunction::GreaterEqual,
				stencil: wgpu::StencilState {
					front: wgpu::StencilFaceState::IGNORE,
					back: wgpu::StencilFaceState::IGNORE,
					read_mask: 0,
					write_mask: 0,
				},
				bias: wgpu::DepthBiasState::default(),
			}),
			multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: true },
			fragment: Some(wgpu::FragmentState {
				module: &colour_fs_module,
				entry_point: "main",
				targets: &[wgpu::ColorTargetState {
					format: wgpu::TextureFormat::Bgra8UnormSrgb,
					blend: Some(wgpu::BlendState {
						color: wgpu::BlendComponent {
							src_factor: wgpu::BlendFactor::SrcAlpha,
							dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
							operation: wgpu::BlendOperation::Add,
						},
						alpha: wgpu::BlendComponent {
							src_factor: wgpu::BlendFactor::One,
							dst_factor: wgpu::BlendFactor::One,
							operation: wgpu::BlendOperation::Add,
						},
					}),
					write_mask: wgpu::ColorWrites::ALL,
				}],
			}),
			multiview: None,
		});

		let texture_graphics_pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("Polystrip Gon/Texture layout"),
			bind_group_layouts: &[&context.texture_bind_group_layout],
			push_constant_ranges: &[wgpu::PushConstantRange {
				stages: wgpu::ShaderStages::VERTEX,
				range: 0..std::mem::size_of::<Matrix4>() as u32 * (matrix_array_size + 1),
			}],
		});
		let texture_graphics_pipeline = context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Polystrip Gon/Texture"),
			layout: Some(&texture_graphics_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &texture_vs_module,
				entry_point: "main",
				buffers: &[wgpu::VertexBufferLayout {
					array_stride: std::mem::size_of::<GpuTextureVertex>() as u64,
					step_mode: wgpu::VertexStepMode::Vertex,
					attributes: GpuTextureVertex::attributes(),
				}],
			},
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::TriangleList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: None,
				unclipped_depth: false,
				polygon_mode: wgpu::PolygonMode::Fill,
				conservative: false,
			},
			depth_stencil: Some(wgpu::DepthStencilState {
				format: wgpu::TextureFormat::Depth32Float,
				depth_write_enabled: true,
				depth_compare: wgpu::CompareFunction::GreaterEqual,
				stencil: wgpu::StencilState {
					front: wgpu::StencilFaceState::IGNORE,
					back: wgpu::StencilFaceState::IGNORE,
					read_mask: 0,
					write_mask: 0,
				},
				bias: wgpu::DepthBiasState::default(),
			}),
			multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: true },
			fragment: Some(wgpu::FragmentState {
				module: &texture_fs_module,
				entry_point: "main",
				targets: &[wgpu::ColorTargetState {
					format: wgpu::TextureFormat::Bgra8UnormSrgb,
					blend: Some(wgpu::BlendState {
						color: wgpu::BlendComponent {
							src_factor: wgpu::BlendFactor::SrcAlpha,
							dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
							operation: wgpu::BlendOperation::Add,
						},
						alpha: wgpu::BlendComponent {
							src_factor: wgpu::BlendFactor::One,
							dst_factor: wgpu::BlendFactor::One,
							operation: wgpu::BlendOperation::Add,
						},
					}),
					write_mask: wgpu::ColorWrites::ALL,
				}],
			}),
			multiview: None,
		});

		let render_size = size.clone_size_handle();

		let (width, height) = render_size.get();

		let depth_textures = Self::create_resizable_resources(&context, config.frames_in_flight, (width, height));

		GonPipeline {
			context,
			size_handle: render_size,

			stroked_graphics_pipeline,
			colour_graphics_pipeline,
			texture_graphics_pipeline,

			frames_in_flight: config.frames_in_flight,
			matrix_array_size,

			current_frame: AtomicUsize::new(0),

			depth_textures,
			current_resource_size: (width, height),
		}
	}

	/// Returns the index of the next frame to be rendered, to be used when selecting the command buffer, semaphores
	/// and fences.
	pub fn next_frame_idx(&self) -> usize {
		let frames_in_flight = self.frames_in_flight;
		self.current_frame
			.fetch_update(Ordering::AcqRel, Ordering::Acquire, |x| Some((x + 1) % frames_in_flight))
			.unwrap() as usize
	}

	fn create_resizable_resources(
		context: &Rc<PolystripDevice>,
		count: usize,
		size: (u32, u32),
	) -> Vec<ManuallyDrop<DepthTexture>> {
		(0..count).map(|_| ManuallyDrop::new(DepthTexture::new(context.clone(), size))).collect()
	}
}

impl HasRenderer for GonPipeline {
	fn context_ref(&self) -> &PolystripDevice {
		&self.context
	}

	fn clone_context(&self) -> Rc<PolystripDevice> {
		self.context.clone()
	}
}

impl HasRenderSize for GonPipeline {
	fn clone_size_handle(&self) -> Rc<RenderSize> {
		self.size_handle.clone()
	}
}

impl<'a> RenderPipeline<'a> for GonPipeline {
	type Frame = GonFrame<'a>;

	fn render_to(&'a mut self, base: &'a mut BaseFrame<'_>) -> Self::Frame {
		if !Rc::ptr_eq(&self.context, &base.context) {
			panic!("frame and pipeline have different Renderers!");
		}

		let recent_size = self.size_handle.get();
		if self.current_resource_size != recent_size {
			self.current_resource_size = recent_size;
			let depth_textures = Self::create_resizable_resources(&self.context, self.frames_in_flight, recent_size);
			let old_depth_textures = std::mem::replace(&mut self.depth_textures, depth_textures);
			for mut texture in old_depth_textures {
				unsafe { ManuallyDrop::drop(&mut texture) };
			}
		}

		let frame_idx = self.next_frame_idx();

		GonFrame {
			context: base.context.clone(),
			render_pass: RenderPass::Uninitialised(
				Some(&mut base.encoder),
				base.resources.image,
				&self.depth_textures[frame_idx].view,
			),
			pipeline: self,
			world_transform: Matrix4::identity(),
		}
	}
}

impl Drop for GonPipeline {
	fn drop(&mut self) {
		for texture in &mut self.depth_textures {
			unsafe { ManuallyDrop::drop(texture) };
		}
	}
}

pub struct GonPipelineBuilder {
	real_3d: bool,
	frames_in_flight: usize,
}

impl GonPipelineBuilder {
	pub fn new() -> GonPipelineBuilder {
		Self { real_3d: false, frames_in_flight: 3 }
	}

	/// If `true`, allows transform matrices to affect sprite depth. This clamps the depth between `0.0` and `1.0`
	///
	/// Default: false
	pub fn real_3d(mut self, real_3d: bool) -> GonPipelineBuilder {
		self.real_3d = real_3d;
		self
	}

	/// The number of frames that can be dispatched simultaneously
	///
	/// Default: 3
	pub fn frames_in_flight(mut self, frames_in_flight: usize) -> GonPipelineBuilder {
		self.frames_in_flight = frames_in_flight;
		self
	}

	/// Builds the renderer, initialising the `gfx_hal` backend.
	pub fn build(self, context: &impl HasRenderer, size: &impl HasRenderSize) -> GonPipeline {
		GonPipeline::with_config(context, size, self)
	}
}

impl Default for GonPipelineBuilder {
	fn default() -> GonPipelineBuilder {
		GonPipelineBuilder::new()
	}
}

#[allow(clippy::large_enum_variant)]
enum RenderPass<'a> {
	Uninitialised(Option<&'a mut wgpu::CommandEncoder>, &'a wgpu::TextureView, &'a wgpu::TextureView),
	Initialised(wgpu::RenderPass<'a>),
}

impl<'a> RenderPass<'a> {
	fn init_clear(&mut self, color: Color) {
		match self {
			RenderPass::Uninitialised(encoder, color_view, depth_view) => {
				*self =
					RenderPass::Initialised(encoder.take().unwrap().begin_render_pass(&wgpu::RenderPassDescriptor {
						label: None,
						color_attachments: &[wgpu::RenderPassColorAttachment {
							view: color_view,
							resolve_target: None,
							ops: wgpu::Operations {
								load: wgpu::LoadOp::Clear(wgpu::Color {
									r: (color.r as f64 / 255.0).powi(2),
									g: (color.g as f64 / 255.0).powi(2),
									b: (color.b as f64 / 255.0).powi(2),
									a: (color.a as f64 / 255.0).powi(2),
								}),
								store: true,
							},
						}],
						depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
							view: depth_view,
							depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: true }),
							stencil_ops: None,
						}),
					}))
			}
			RenderPass::Initialised(_) => {}
		}
	}

	fn init_load(&mut self) {
		match self {
			RenderPass::Uninitialised(encoder, color_view, depth_view) => {
				*self =
					RenderPass::Initialised(encoder.take().unwrap().begin_render_pass(&wgpu::RenderPassDescriptor {
						label: None,
						color_attachments: &[wgpu::RenderPassColorAttachment {
							view: color_view,
							resolve_target: None,
							ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: true },
						}],
						depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
							view: depth_view,
							depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: true }),
							stencil_ops: None,
						}),
					}))
			}
			RenderPass::Initialised(_) => {}
		}
	}

	fn get(&mut self) -> &mut wgpu::RenderPass<'a> {
		match self {
			RenderPass::Uninitialised(_, _, _) => {
				self.init_load();
				self.get()
			}
			RenderPass::Initialised(pass) => pass,
		}
	}
}

/// A frame to be drawn to. The frame gets presented on drop.
pub struct GonFrame<'a> {
	context: Rc<PolystripDevice>,
	render_pass: RenderPass<'a>,
	pipeline: &'a GonPipeline,
	world_transform: Matrix4,
}

impl<'a> GonFrame<'a> {
	/// Clears the entire frame to the passed [`Color`](../vertex/struct.Color.html).
	///
	/// The color is converted from sRGB using a gamma value of 2.0
	///
	/// ## Caveats
	/// This method will only work if it is the first method called on the frame. Otherwise, it will
	/// fail silently.
	pub fn clear(&mut self, color: Color) {
		self.render_pass.init_clear(color);
	}

	/// Sets the global transform matrix for subsequent draw calls.
	///
	/// If this method is called multiple times, draw calls will use the matrix provided most recently.
	/// Draw calls made before this method call use the identity matrix as the global transform matrix.
	pub fn set_global_transform(&mut self, matrix: Matrix4) {
		self.world_transform = matrix;
	}

	/// Draws a [`StrokedShape`](vertex/struct.StrokedShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	pub fn draw_stroked(&mut self, shape: &'a GpuStrokedShape, obj_transforms: &[Matrix4]) {
		let render_pass = self.render_pass.get();

		render_pass.set_vertex_buffer(0, shape.vertex_buffer.slice(..));
		render_pass.set_index_buffer(shape.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

		render_pass.set_pipeline(&self.pipeline.stroked_graphics_pipeline);
		render_pass.set_push_constants(
			wgpu::ShaderStages::VERTEX,
			0,
			bytemuck::cast_slice::<[[f32; 4]; 4], _>(&[self.world_transform.into()]),
		);

		for chunk in obj_transforms.chunks(self.pipeline.matrix_array_size as usize) {
			render_pass.set_push_constants(
				wgpu::ShaderStages::VERTEX,
				std::mem::size_of::<Matrix4>() as u32,
				bytemuck::cast_slice(chunk),
			);
			render_pass.draw_indexed(0..shape.index_count, 0, 0..chunk.len() as u32);
		}
	}

	/// Draws a [`ColoredShape`](vertex/struct.ColoredShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	pub fn draw_colored(&mut self, shape: &'a GpuColoredShape, obj_transforms: &[Matrix4]) {
		let render_pass = self.render_pass.get();

		render_pass.set_vertex_buffer(0, shape.vertex_buffer.slice(..));
		render_pass.set_index_buffer(shape.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

		render_pass.set_pipeline(&self.pipeline.colour_graphics_pipeline);
		render_pass.set_push_constants(
			wgpu::ShaderStages::VERTEX,
			0,
			bytemuck::cast_slice::<[[f32; 4]; 4], _>(&[self.world_transform.into()]),
		);

		for chunk in obj_transforms.chunks(self.pipeline.matrix_array_size as usize) {
			render_pass.set_push_constants(
				wgpu::ShaderStages::VERTEX,
				std::mem::size_of::<Matrix4>() as u32,
				bytemuck::cast_slice(&chunk.iter().map(|&mat| mat.into()).collect::<Vec<[[f32; 4]; 4]>>()), //TODO: Remove this allocation
			);
			render_pass.draw_indexed(0..shape.index_count, 0, 0..chunk.len() as u32);
		}
	}

	/// Draws a [`TexturedShape`](vertex/struct.TexturedShape.html). The shape will be drawn in front of any shapes drawn
	/// before it.
	///
	/// `iterations` is a slice of texture references and matrices to draw that texture with.
	pub fn draw_textured(&mut self, shape: &'a GpuTexturedShape, iterations: &[(SampledTexture<'a>, &[Matrix4])]) {
		let render_pass = self.render_pass.get();

		render_pass.set_vertex_buffer(0, shape.vertex_buffer.slice(..));
		render_pass.set_index_buffer(shape.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

		render_pass.set_pipeline(&self.pipeline.texture_graphics_pipeline);
		render_pass.set_push_constants(
			wgpu::ShaderStages::VERTEX,
			0,
			bytemuck::cast_slice::<[[f32; 4]; 4], _>(&[self.world_transform.into()]),
		);

		for (texture, obj_transforms) in iterations {
			if !Rc::ptr_eq(&self.context, texture.context) {
				panic!("Texture was not made with renderer that made this frame");
			}

			render_pass.set_bind_group(0, texture.bind_group, &[]);

			for chunk in obj_transforms.chunks(self.pipeline.matrix_array_size as usize) {
				render_pass.set_push_constants(
					wgpu::ShaderStages::VERTEX,
					std::mem::size_of::<Matrix4>() as u32,
					bytemuck::cast_slice(&chunk.iter().map(|&mat| mat.into()).collect::<Vec<[[f32; 4]; 4]>>()), //TODO: Remove this allocation
				);
				render_pass.draw_indexed(0..shape.index_count, 0, 0..chunk.len() as u32);
			}
		}
	}

	pub fn draw(&mut self, object: &impl Drawable<'a>) {
		object.draw_to(self);
	}

	/// Converts pixel coordinates to Gpu coordinates
	pub fn pixel(&self, x: i32, y: i32) -> Vector2 {
		Vector2::new(
			(x * 2) as f32 / self.pipeline.current_resource_size.0 as f32 - 1.0,
			-((y * 2) as f32 / self.pipeline.current_resource_size.1 as f32 - 1.0),
		)
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

impl HasRenderer for GonFrame<'_> {
	fn context_ref(&self) -> &PolystripDevice {
		self.context.as_ref()
	}

	fn clone_context(&self) -> Rc<PolystripDevice> {
		self.context.clone()
	}
}

pub trait Drawable<'a> {
	fn draw_to(&self, frame: &mut GonFrame<'a>);
}

#[cfg(feature = "glyph_brush")]
pub struct GlyphBrush<F = glyph_brush::ab_glyph::FontArc, H = glyph_brush::DefaultSectionHasher> {
	brush: glyph_brush::GlyphBrush<[GpuTextureVertex; 4], glyph_brush::Extra, F, H>,
	texture: RenderTexture,
	current_shapes: Vec<GpuTexturedShape>,
}

#[cfg(feature = "glyph_brush")]
impl<F: glyph_brush::ab_glyph::Font + Sync, H: std::hash::BuildHasher> GlyphBrush<F, H> {
	pub fn from_glyph_brush(
		context: &impl HasRenderer,
		brush: glyph_brush::GlyphBrush<[GpuTextureVertex; 4], glyph_brush::Extra, F, H>,
	) -> GlyphBrush<F, H> {
		GlyphBrush {
			texture: RenderTexture::new(context, brush.texture_dimensions()),
			brush,
			current_shapes: Vec::new(),
		}
	}

	pub fn process_queued(&mut self, size: &impl HasRenderSize) {
		let size = size.clone_size_handle().get();

		let texture = &mut self.texture;
		match self.brush.process_queued(
			|size, data| {
				texture.write_section(
					Rect {
						x: size.min[0] as i32,
						y: size.min[1] as i32,
						w: size.width() as i32,
						h: size.height() as i32,
					},
					bytemuck::cast_slice(&data.iter().map(|a| Color::new(255, 255, 255, *a)).collect::<Vec<_>>()),
				)
			},
			|vertex| {
				[
					GpuTextureVertex {
						position: Vector3::new(
							vertex.pixel_coords.min.x * 2. / size.0 as f32 - 1.0,
							-(vertex.pixel_coords.min.y * 2. / size.1 as f32 - 1.0),
							vertex.extra.z,
						),
						tex_coord: Vector2::new(vertex.tex_coords.min.x, vertex.tex_coords.min.y),
					},
					GpuTextureVertex {
						position: Vector3::new(
							vertex.pixel_coords.max.x * 2. / size.0 as f32 - 1.0,
							-(vertex.pixel_coords.min.y * 2. / size.1 as f32 - 1.0),
							vertex.extra.z,
						),
						tex_coord: Vector2::new(vertex.tex_coords.max.x, vertex.tex_coords.min.y),
					},
					GpuTextureVertex {
						position: Vector3::new(
							vertex.pixel_coords.max.x * 2. / size.0 as f32 - 1.0,
							-(vertex.pixel_coords.max.y * 2. / size.1 as f32 - 1.0),
							vertex.extra.z,
						),
						tex_coord: Vector2::new(vertex.tex_coords.max.x, vertex.tex_coords.max.y),
					},
					GpuTextureVertex {
						position: Vector3::new(
							vertex.pixel_coords.min.x * 2. / size.0 as f32 - 1.0,
							-(vertex.pixel_coords.max.y * 2. / size.1 as f32 - 1.0),
							vertex.extra.z,
						),
						tex_coord: Vector2::new(vertex.tex_coords.min.x, vertex.tex_coords.max.y),
					},
				]
			},
		) {
			Ok(glyph_brush::BrushAction::Draw(vertices)) => {
				self.current_shapes = vertices
					.into_iter()
					.map(|vertices| self.texture.create_textured(&vertices, &QUAD_INDICES))
					.collect()
			}
			Ok(glyph_brush::BrushAction::ReDraw) => {}
			Err(glyph_brush::BrushError::TextureTooSmall { suggested }) => {
				self.texture = RenderTexture::new(&self.texture, suggested);
				self.brush.resize_texture(suggested.0, suggested.1);
			}
		}
	}

	pub fn place(&self) -> GlyphBrushDrawable<'_, F, H> {
		GlyphBrushDrawable { brush: self, transform: Matrix4::identity() }
	}
}

#[cfg(feature = "glyph_brush")]
impl<F: glyph_brush::ab_glyph::Font, H: std::hash::BuildHasher> Deref for GlyphBrush<F, H> {
	type Target = glyph_brush::GlyphBrush<[GpuTextureVertex; 4], glyph_brush::Extra, F, H>;

	fn deref(&self) -> &Self::Target {
		&self.brush
	}
}

#[cfg(feature = "glyph_brush")]
impl<F: glyph_brush::ab_glyph::Font, H: std::hash::BuildHasher> DerefMut for GlyphBrush<F, H> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.brush
	}
}

#[cfg(feature = "glyph_brush")]
pub struct GlyphBrushDrawable<'a, F, H> {
	brush: &'a GlyphBrush<F, H>,
	transform: Matrix4,
}

#[cfg(feature = "glyph_brush")]
impl<'a, F: glyph_brush::ab_glyph::Font, H: std::hash::BuildHasher> Drawable<'a> for GlyphBrushDrawable<'a, F, H> {
	fn draw_to(&self, frame: &mut GonFrame<'a>) {
		for shape in &self.brush.current_shapes {
			frame.draw_textured(shape, &[(self.brush.texture.sampled(), &[self.transform])]);
		}
	}
}

pub const QUAD_INDICES: [[u16; 3]; 2] = [[0, 3, 1], [1, 3, 2]];

/// A vertex describing a position and a position on a texture.
///
/// Texture coordinates are interpolated linearly between vertices.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct GpuTextureVertex {
	pub position: Vector3,
	pub tex_coord: Vector2,
}

unsafe impl bytemuck::Pod for GpuTextureVertex {}
unsafe impl bytemuck::Zeroable for GpuTextureVertex {}

impl GpuTextureVertex {
	pub(crate) fn attributes<'a>() -> &'a [wgpu::VertexAttribute] {
		use std::mem::size_of;

		&[
			wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
			wgpu::VertexAttribute {
				format: wgpu::VertexFormat::Float32x2,
				offset: size_of::<[f32; 3]>() as u64,
				shader_location: 1,
			},
		]
	}
}

/// A vertex describing a position and a color.
///
/// Colors are interpolated linearly between vertices.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct GpuColorVertex {
	pub position: Vector3,
	pub color: Color,
}

unsafe impl bytemuck::Pod for GpuColorVertex {}
unsafe impl bytemuck::Zeroable for GpuColorVertex {}

impl GpuColorVertex {
	pub(crate) fn attributes<'a>() -> &'a [wgpu::VertexAttribute] {
		use std::mem::size_of;

		&[
			wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
			wgpu::VertexAttribute {
				format: wgpu::VertexFormat::Unorm8x4,
				offset: size_of::<[f32; 3]>() as u64,
				shader_location: 1,
			},
		]
	}
}

pub trait PolystripShapeExt {
	fn create_stroked(&self, vertices: &[GpuColorVertex], indices: &[[u16; 2]]) -> GpuStrokedShape;
	fn create_colored(&self, vertices: &[GpuColorVertex], indices: &[[u16; 3]]) -> GpuColoredShape;
	fn create_textured(&self, vertices: &[GpuTextureVertex], indices: &[[u16; 3]]) -> GpuTexturedShape;
}

impl PolystripShapeExt for PolystripDevice {
	fn create_stroked(&self, vertices: &[GpuColorVertex], indices: &[[u16; 2]]) -> GpuStrokedShape {
		let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: None,
			contents: bytemuck::cast_slice(vertices),
			usage: wgpu::BufferUsages::VERTEX,
		});
		let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: None,
			contents: bytemuck::cast_slice(indices),
			usage: wgpu::BufferUsages::INDEX,
		});
		GpuStrokedShape { vertex_buffer, index_buffer, index_count: indices.len() as u32 * 2 }
	}

	fn create_colored(&self, vertices: &[GpuColorVertex], indices: &[[u16; 3]]) -> GpuColoredShape {
		let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: None,
			contents: bytemuck::cast_slice(vertices),
			usage: wgpu::BufferUsages::VERTEX,
		});
		let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: None,
			contents: bytemuck::cast_slice(indices),
			usage: wgpu::BufferUsages::INDEX,
		});
		GpuColoredShape { vertex_buffer, index_buffer, index_count: indices.len() as u32 * 3 }
	}

	fn create_textured(&self, vertices: &[GpuTextureVertex], indices: &[[u16; 3]]) -> GpuTexturedShape {
		let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: None,
			contents: bytemuck::cast_slice(vertices),
			usage: wgpu::BufferUsages::VERTEX,
		});
		let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: None,
			contents: bytemuck::cast_slice(indices),
			usage: wgpu::BufferUsages::INDEX,
		});
		GpuTexturedShape { vertex_buffer, index_buffer, index_count: indices.len() as u32 * 3 }
	}
}

impl<T> PolystripShapeExt for T
where
	T: HasRenderer,
{
	fn create_stroked(&self, vertices: &[GpuColorVertex], indices: &[[u16; 2]]) -> GpuStrokedShape {
		self.context_ref().create_stroked(vertices, indices)
	}

	fn create_colored(&self, vertices: &[GpuColorVertex], indices: &[[u16; 3]]) -> GpuColoredShape {
		self.context_ref().create_colored(vertices, indices)
	}

	fn create_textured(&self, vertices: &[GpuTextureVertex], indices: &[[u16; 3]]) -> GpuTexturedShape {
		self.context_ref().create_textured(vertices, indices)
	}
}

/// A set of vertices and indices describing an outlined geometric shape as a set of lines.
///
/// The colors of the lines are determined by interpolating the colors at each
/// [`ColorVertex`](struct.ColorVertex).
#[derive(Debug)]
pub struct GpuStrokedShape {
	vertex_buffer: wgpu::Buffer,
	index_buffer: wgpu::Buffer,
	index_count: u32,
}

impl GpuStrokedShape {
	pub fn with_instances<'a, 'b>(&'a self, instances: &'b [Matrix4]) -> StrokedShapeDrawable<'a, 'b> {
		StrokedShapeDrawable {
			shape: self,
			instances,
		}
	}
}

pub struct StrokedShapeDrawable<'a, 'b> {
	shape: &'a GpuStrokedShape,
	instances: &'b [Matrix4],
}

impl<'a> Drawable<'a> for StrokedShapeDrawable<'a, '_> {
    fn draw_to(&self, frame: &mut GonFrame<'a>) {
        frame.draw_stroked(self.shape, self.instances);
    }
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
///
/// The color of the shape is determined by interpolating the colors at each
/// [`ColorVertex`](struct.ColorVertex).
#[derive(Debug)]
pub struct GpuColoredShape {
	vertex_buffer: wgpu::Buffer,
	index_buffer: wgpu::Buffer,
	index_count: u32,
}

impl GpuColoredShape {
	pub fn with_instances<'a, 'b>(&'a self, instances: &'b [Matrix4]) -> ColoredShapeDrawable<'a, 'b> {
		ColoredShapeDrawable {
			shape: self,
			instances,
		}
	}
}

pub struct ColoredShapeDrawable<'a, 'b> {
	shape: &'a GpuColoredShape,
	instances: &'b [Matrix4],
}

impl<'a> Drawable<'a> for ColoredShapeDrawable<'a, '_> {
	fn draw_to(&self, frame: &mut GonFrame<'a>) {
		frame.draw_colored(self.shape, self.instances);
	}
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
///
/// The color of the shape is determined by interpolating the texture coordinates at each
/// [`TextureVertex`](struct.TextureVertex), and sampling the [`Texture`](../struct.Texture)
/// provided to the [`GonFrame::draw_textured`](struct.GonFrame#method.draw_textured) call this shape
/// is drawn with
pub struct GpuTexturedShape {
	vertex_buffer: wgpu::Buffer,
	index_buffer: wgpu::Buffer,
	index_count: u32,
}

impl GpuTexturedShape {
	pub fn with_instances<'a, 'b, 'c>(&'a self, instances: &'c [(SampledTexture<'a>, &'b [Matrix4])]) -> TexturedShapeDrawable<'a, 'b, 'c> {
		TexturedShapeDrawable {
			shape: self,
			instances,
		}
	}
}

pub struct TexturedShapeDrawable<'a, 'b, 'c> {
	shape: &'a GpuTexturedShape,
	instances: &'c [(SampledTexture<'a>, &'b [Matrix4])],
}

impl<'a> Drawable<'a> for TexturedShapeDrawable<'a, '_, '_> {
	fn draw_to(&self, frame: &mut GonFrame<'a>) {
		frame.draw_textured(self.shape, self.instances);
	}
}

pub struct PixelColorVertex {
	pub position: Vector3,
	pub color: Color,
}

impl PixelColorVertex {
	pub fn convert_for_size(&self, (width, height): (u32, u32)) -> GpuColorVertex {
		GpuColorVertex {
			position: Vector3::new(
				self.position.x * 2.0 / width as f32 - 1.0,
				-(self.position.y * 2.0 / height as f32 - 1.0),
				self.position.z,
			),
			color: self.color,
		}
	}
}

pub struct PixelTextureVertex {
	pub position: Vector3,
	pub tex_coord: Vector2,
}

impl PixelTextureVertex {
	pub fn convert_for_size(&self, (width, height): (u32, u32)) -> GpuTextureVertex {
		GpuTextureVertex {
			position: Vector3::new(
				self.position.x * 2.0 / width as f32 - 1.0,
				-(self.position.y * 2.0 / height as f32 - 1.0),
				self.position.z,
			),
			tex_coord: self.tex_coord,
		}
	}
}

pub struct PixelStrokedShape {
	vertices: Vec<PixelColorVertex>,
	indices: Vec<[u16; 2]>,
	size: Rc<RenderSize>,
	cache: Option<(GpuStrokedShape, (u32, u32))>,
}

impl PixelStrokedShape {
	pub fn new(size: &impl HasRenderSize, vertices: Vec<PixelColorVertex>, indices: Vec<[u16; 2]>) -> Self {
		PixelStrokedShape { vertices, indices, size: size.clone_size_handle(), cache: None }
	}

	pub fn gpu_shape(&mut self, context: &impl HasRenderer) -> &GpuStrokedShape {
		let current_size = self.size.get();
		if !matches!(self.cache, Some((_, size)) if size == current_size) {
			let shape = context.create_stroked(
				&self.vertices.iter().map(|v| v.convert_for_size(current_size)).collect::<Vec<_>>(),
				&self.indices,
			);
			self.cache = Some((shape, current_size));
		}
		&self.cache.as_ref().unwrap().0
	}
}

pub struct PixelColoredShape {
	vertices: Vec<PixelColorVertex>,
	indices: Vec<[u16; 3]>,
	size: Rc<RenderSize>,
	cache: Option<(GpuColoredShape, (u32, u32))>,
}

impl PixelColoredShape {
	pub fn new(size: &impl HasRenderSize, vertices: Vec<PixelColorVertex>, indices: Vec<[u16; 3]>) -> Self {
		PixelColoredShape { vertices, indices, size: size.clone_size_handle(), cache: None }
	}

	pub fn gpu_shape(&mut self, context: &impl HasRenderer) -> &GpuColoredShape {
		let current_size = self.size.get();
		if !matches!(self.cache, Some((_, size)) if size == current_size) {
			let shape = context.create_colored(
				&self.vertices.iter().map(|v| v.convert_for_size(current_size)).collect::<Vec<_>>(),
				&self.indices,
			);
			self.cache = Some((shape, current_size));
		}
		&self.cache.as_ref().unwrap().0
	}
}

pub struct PixelTexturedShape {
	vertices: Vec<PixelTextureVertex>,
	indices: Vec<[u16; 3]>,
	size: Rc<RenderSize>,
	cache: Option<(GpuTexturedShape, (u32, u32))>,
}

impl PixelTexturedShape {
	pub fn new(size: &impl HasRenderSize, vertices: Vec<PixelTextureVertex>, indices: Vec<[u16; 3]>) -> Self {
		PixelTexturedShape { vertices, indices, size: size.clone_size_handle(), cache: None }
	}

	pub fn gpu_shape(&mut self, context: &impl HasRenderer) -> &GpuTexturedShape {
		let current_size = self.size.get();
		if !matches!(self.cache, Some((_, size)) if size == current_size) {
			let shape = context.create_textured(
				&self.vertices.iter().map(|v| v.convert_for_size(current_size)).collect::<Vec<_>>(),
				&self.indices,
			);
			self.cache = Some((shape, current_size));
		}
		&self.cache.as_ref().unwrap().0
	}
}
