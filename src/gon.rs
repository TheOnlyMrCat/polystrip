//! The gon pipeline for 2D rendering

use std::cell::RefCell;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::ops::DerefMut;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use wgpu::util::DeviceExt;

use crate::math::*;
use crate::PolystripDevice;
use crate::{BaseFrame, DepthTexture, HasRenderSize, HasRenderer, RenderPipeline, RenderSize, Texture};

/// The `gon` pipeline for 2D rendering.
pub struct GonPipeline {
	context: Rc<PolystripDevice>,
	size_handle: Rc<RenderSize>,

	stroked_graphics_pipeline: wgpu::RenderPipeline,
	stroked_graphics_pipeline_layout: wgpu::PipelineLayout,
	colour_graphics_pipeline: wgpu::RenderPipeline,
	colour_graphics_pipeline_layout: wgpu::PipelineLayout,
	texture_graphics_pipeline: wgpu::RenderPipeline,
	texture_graphics_pipeline_layout: wgpu::PipelineLayout,

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

		let colour_vs_module =
			context.device.create_shader_module(&wgpu::include_spirv!("../gen/gon/coloured.vert.spv"));
		let colour_fs_module =
			context.device.create_shader_module(&wgpu::include_spirv!("../gen/gon/coloured.frag.spv"));
		let texture_vs_module =
			context.device.create_shader_module(&wgpu::include_spirv!("../gen/gon/textured.vert.spv"));
		let texture_fs_module =
			context.device.create_shader_module(&wgpu::include_spirv!("../gen/gon/textured.frag.spv"));

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
					array_stride: std::mem::size_of::<ColorVertex>() as u64,
					step_mode: wgpu::VertexStepMode::Vertex,
					attributes: ColorVertex::attributes(),
				}],
			},
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::LineList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: None,
				unclipped_depth: false,
				polygon_mode: wgpu::PolygonMode::Line,
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
			multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
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
					array_stride: std::mem::size_of::<ColorVertex>() as u64,
					step_mode: wgpu::VertexStepMode::Vertex,
					attributes: ColorVertex::attributes(),
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
			multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
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
					array_stride: std::mem::size_of::<TextureVertex>() as u64,
					step_mode: wgpu::VertexStepMode::Vertex,
					attributes: TextureVertex::attributes(),
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
			multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
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
			stroked_graphics_pipeline_layout,
			colour_graphics_pipeline,
			colour_graphics_pipeline_layout,
			texture_graphics_pipeline,
			texture_graphics_pipeline_layout,

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
	pub fn draw_stroked(&mut self, shape: &'a StrokedShape, obj_transforms: &[Matrix4]) {
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
	pub fn draw_colored(&mut self, shape: &'a ColoredShape, obj_transforms: &[Matrix4]) {
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
	pub fn draw_textured(&mut self, shape: &'a TexturedShape, iterations: &[(&'a Texture, &[Matrix4])]) {
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
			if !Rc::ptr_eq(&self.context, &texture.context) {
				panic!("Texture was not made with renderer that made this frame");
			}

			render_pass.set_bind_group(0, &texture.bind_group, &[]);

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

	pub fn draw(&mut self, object: &'a impl Drawable) {
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

pub trait Drawable {
	fn draw_to<'a>(&'a self, frame: &mut GonFrame<'a>);
}

#[cfg(feature = "glyph_brush")]
pub struct GlyphBrush<F = glyph_brush::ab_glyph::FontArc, H = glyph_brush::DefaultSectionHasher> {
	brush: glyph_brush::GlyphBrush<[TextureVertex; 4], glyph_brush::Extra, F, H>,
	texture: Texture,
	current_shapes: Vec<TexturedShape>,
}

#[cfg(feature = "glyph_brush")]
impl<F: glyph_brush::ab_glyph::Font + Sync, H: std::hash::BuildHasher> GlyphBrush<F, H> {
	pub fn from_glyph_brush(
		context: &impl HasRenderer,
		brush: glyph_brush::GlyphBrush<[TextureVertex; 4], glyph_brush::Extra, F, H>,
	) -> GlyphBrush<F, H> {
		GlyphBrush {
			texture: Texture::new_solid_color(context, Color::ZERO, brush.texture_dimensions()),
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
					TextureVertex {
						position: Vector3::new(
							vertex.pixel_coords.min.x * 2. / size.0 as f32 - 1.0,
							-(vertex.pixel_coords.min.y * 2. / size.1 as f32 - 1.0),
							vertex.extra.z,
						),
						tex_coords: Vector2::new(vertex.tex_coords.min.x, vertex.tex_coords.min.y),
					},
					TextureVertex {
						position: Vector3::new(
							vertex.pixel_coords.max.x * 2. / size.0 as f32 - 1.0,
							-(vertex.pixel_coords.min.y * 2. / size.1 as f32 - 1.0),
							vertex.extra.z,
						),
						tex_coords: Vector2::new(vertex.tex_coords.max.x, vertex.tex_coords.min.y),
					},
					TextureVertex {
						position: Vector3::new(
							vertex.pixel_coords.max.x * 2. / size.0 as f32 - 1.0,
							-(vertex.pixel_coords.max.y * 2. / size.1 as f32 - 1.0),
							vertex.extra.z,
						),
						tex_coords: Vector2::new(vertex.tex_coords.max.x, vertex.tex_coords.max.y),
					},
					TextureVertex {
						position: Vector3::new(
							vertex.pixel_coords.min.x * 2. / size.0 as f32 - 1.0,
							-(vertex.pixel_coords.max.y * 2. / size.1 as f32 - 1.0),
							vertex.extra.z,
						),
						tex_coords: Vector2::new(vertex.tex_coords.min.x, vertex.tex_coords.max.y),
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
				self.texture = Texture::new_solid_color(&self.texture, Color::ZERO, suggested);
				self.brush.resize_texture(suggested.0, suggested.1);
			}
		}
	}
}

#[cfg(feature = "glyph_brush")]
impl<F: glyph_brush::ab_glyph::Font, H: std::hash::BuildHasher> Drawable for GlyphBrush<F, H> {
	fn draw_to<'a>(&'a self, frame: &mut GonFrame<'a>) {
		for shape in &self.current_shapes {
			frame.draw_textured(shape, &[(&self.texture, &[Matrix4::identity()])]);
		}
	}
}

#[cfg(feature = "glyph_brush")]
impl<F: glyph_brush::ab_glyph::Font, H: std::hash::BuildHasher> Deref for GlyphBrush<F, H> {
	type Target = glyph_brush::GlyphBrush<[TextureVertex; 4], glyph_brush::Extra, F, H>;

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

pub const QUAD_INDICES: [[u16; 3]; 2] = [[0, 3, 1], [1, 3, 2]];

/// A vertex describing a position and a position on a texture.
///
/// Texture coordinates are interpolated linearly between vertices.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TextureVertex {
	pub position: Vector3,
	pub tex_coords: Vector2,
}

unsafe impl bytemuck::Pod for TextureVertex {}
unsafe impl bytemuck::Zeroable for TextureVertex {}

impl TextureVertex {
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
pub struct ColorVertex {
	pub position: Vector3,
	pub color: Color,
}

unsafe impl bytemuck::Pod for ColorVertex {}
unsafe impl bytemuck::Zeroable for ColorVertex {}

impl ColorVertex {
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
	fn create_stroked(&self, vertices: &[ColorVertex], indices: &[[u16; 2]]) -> StrokedShape;
	fn create_colored(&self, vertices: &[ColorVertex], indices: &[[u16; 3]]) -> ColoredShape;
	fn create_textured(&self, vertices: &[TextureVertex], indices: &[[u16; 3]]) -> TexturedShape;
}

impl PolystripShapeExt for PolystripDevice {
	fn create_stroked(&self, vertices: &[ColorVertex], indices: &[[u16; 2]]) -> StrokedShape {
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
		StrokedShape { vertex_buffer, index_buffer, index_count: indices.len() as u32 * 2 }
	}

	fn create_colored(&self, vertices: &[ColorVertex], indices: &[[u16; 3]]) -> ColoredShape {
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
		ColoredShape { vertex_buffer, index_buffer, index_count: indices.len() as u32 * 3 }
	}

	fn create_textured(&self, vertices: &[TextureVertex], indices: &[[u16; 3]]) -> TexturedShape {
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
		TexturedShape { vertex_buffer, index_buffer, index_count: indices.len() as u32 * 3 }
	}
}

impl<T> PolystripShapeExt for T
where
	T: HasRenderer,
{
	fn create_stroked(&self, vertices: &[ColorVertex], indices: &[[u16; 2]]) -> StrokedShape {
		self.context_ref().create_stroked(vertices, indices)
	}

	fn create_colored(&self, vertices: &[ColorVertex], indices: &[[u16; 3]]) -> ColoredShape {
		self.context_ref().create_colored(vertices, indices)
	}

	fn create_textured(&self, vertices: &[TextureVertex], indices: &[[u16; 3]]) -> TexturedShape {
		self.context_ref().create_textured(vertices, indices)
	}
}

/// A set of vertices and indices describing an outlined geometric shape as a set of lines.
///
/// The colors of the lines are determined by interpolating the colors at each
/// [`ColorVertex`](struct.ColorVertex).
#[derive(Debug)]
pub struct StrokedShape {
	vertex_buffer: wgpu::Buffer,
	index_buffer: wgpu::Buffer,
	index_count: u32,
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
///
/// The color of the shape is determined by interpolating the colors at each
/// [`ColorVertex`](struct.ColorVertex).
#[derive(Debug)]
pub struct ColoredShape {
	vertex_buffer: wgpu::Buffer,
	index_buffer: wgpu::Buffer,
	index_count: u32,
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
///
/// The color of the shape is determined by interpolating the texture coordinates at each
/// [`TextureVertex`](struct.TextureVertex), and sampling the [`Texture`](../struct.Texture)
/// provided to the [`GonFrame::draw_textured`](struct.GonFrame#method.draw_textured) call this shape
/// is drawn with
#[derive(Debug)]
pub struct TexturedShape {
	vertex_buffer: wgpu::Buffer,
	index_buffer: wgpu::Buffer,
	index_count: u32,
}

// impl TexturedShape<'_, '_> {
// 	/// A quad rendering the full texture, the top two points being at height 1, the bottom two at height 0
// 	pub const QUAD_FULL_STANDING: TexturedShape<'static, 'static> = TexturedShape {
// 		vertices: Cow::Borrowed(&[
// 			TextureVertex { position: Vector3::new(0., 0., 1.), tex_coords: Vector2::new(0., 0.) },
// 			TextureVertex { position: Vector3::new(1., 0., 1.), tex_coords: Vector2::new(1., 0.) },
// 			TextureVertex { position: Vector3::new(1., 1., 0.), tex_coords: Vector2::new(1., 1.) },
// 			TextureVertex { position: Vector3::new(0., 1., 0.), tex_coords: Vector2::new(0., 1.) },
// 		]),
// 		indices: Cow::Borrowed(&QUAD_INDICES),
// 	};

// 	/// A quad rendering the full texture, all points at height 0
// 	pub const QUAD_FULL_FLAT: TexturedShape<'static, 'static> = TexturedShape {
// 		vertices: Cow::Borrowed(&[
// 			TextureVertex { position: Vector3::new(0., 0., 0.), tex_coords: Vector2::new(0., 0.) },
// 			TextureVertex { position: Vector3::new(1., 0., 0.), tex_coords: Vector2::new(1., 0.) },
// 			TextureVertex { position: Vector3::new(1., 1., 0.), tex_coords: Vector2::new(1., 1.) },
// 			TextureVertex { position: Vector3::new(0., 1., 0.), tex_coords: Vector2::new(0., 1.) },
// 		]),
// 		indices: Cow::Borrowed(&QUAD_INDICES),
// 	};
// }
