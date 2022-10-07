//! A basic render graph implementation.
//!
//! A render graph, implemented by the [`RenderGraph`] type, is a sequence of render nodes,
//! each with their own inputs, outputs, and required state. A render graph collects these
//! nodes, analyzes their dependencies, and optimises the number of discrete render/compute
//! passes required to execute them.
//!
//! In polystrip, nodes are constructed and added to a graph with the various `NodeBuilder` types.
//! These accept a varadic number of resource handles that will be fetched and passed to the
//! execution closure (Note that due to type system limitations, a maximum of 32 elements is
//! imposed on buffers, textures, and bind groups and a maximum of 12 is imposed on passthrough
//! resources).
//!
//! ```no_run
//! # use pollster::FutureExt;
//! # use polystrip::graph::RenderGraph;
//! # let device = polystrip::PolystripDevice::new_from_env().block_on();
//! # let mut renderer = polystrip::Renderer::new(device.device.clone(), device.queue);
//! # let descriptor = wgpu::BufferDescriptor {
//! #     label: None, size: 0, usage: wgpu::BufferUsages::empty(), mapped_at_creation: false
//! # };
//! # let source_handle = renderer.insert_buffer(device.device.create_buffer(&descriptor));
//! # let dest_handle = renderer.insert_buffer(device.device.create_buffer(&descriptor));
//! let mut graph = RenderGraph::new(&mut renderer);
//! graph.add_node()
//!     .with_buffer(source_handle)
//!     .with_buffer_rw(dest_handle)
//!     .build(|encoder, [source, dest], [], [], ()| {
//!         encoder.copy_buffer_to_buffer(source, 0, dest, 256, 256);
//!     });
//! ```

use fxhash::{FxHashMap, FxHashSet};

use crate::{
    ComputePipeline, Handle, IntoBindGroupResources, RenderPassTarget, RenderPipeline, Renderer,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum TrackedResourceHandle {
    Buffer(Handle<wgpu::Buffer>),
    Texture(Handle<wgpu::TextureView>),
}

#[derive(PartialEq, Eq, Hash)]
struct VersionedResourceHandle {
    handle: TrackedResourceHandle,
    version: usize,
}

/// A simple, single-use render graph
///
/// See module-level docs for more details.
pub struct RenderGraph<'r, 'node> {
    renderer: &'r mut Renderer,
    intermediate_buffers: FxHashSet<Handle<wgpu::Buffer>>,
    intermediate_textures: FxHashSet<Handle<wgpu::TextureView>>,
    temporary_textures: FxHashSet<Handle<wgpu::TextureView>>,
    nodes: Vec<Box<dyn GraphNode<'node> + 'node>>,
    node_outputs: FxHashMap<VersionedResourceHandle, usize>,
    node_inputs: FxHashMap<usize, Vec<VersionedResourceHandle>>,
    resource_versions: FxHashMap<TrackedResourceHandle, usize>,
}

impl<'r, 'node> RenderGraph<'r, 'node> {
    /// Create a new render graph
    pub fn new(renderer: &'r mut Renderer) -> RenderGraph<'r, 'node> {
        RenderGraph {
            renderer,
            intermediate_buffers: FxHashSet::default(),
            intermediate_textures: FxHashSet::default(),
            temporary_textures: FxHashSet::default(),
            nodes: Vec::new(),
            node_outputs: FxHashMap::default(),
            node_inputs: FxHashMap::default(),
            resource_versions: FxHashMap::default(),
        }
    }

    /// Get an immutable reference to the [`Renderer`] used to create this graph.
    pub fn renderer(&self) -> &Renderer {
        self.renderer
    }

    /// Get a mutable reference to the [`Renderer`] used to create this graph.
    pub fn renderer_mut(&mut self) -> &mut Renderer {
        self.renderer
    }

    /// Add an intermediate buffer to the graph.
    ///
    /// The same buffers are reused between graphs if they have the same descriptors. Creating
    /// multiple buffers with the same descriptor is possible within the same render graph. Cached
    /// buffers are deallocated if they are not used.
    ///
    /// The returned handle should not be used outside of the context of this graph.
    ///
    /// ```ignore
    /// let buffer = graph.add_intermediate_buffer(wgpu::BufferDescriptor {
    ///     label: None,
    ///     size: dynamic_buffer_size,
    ///     usages: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
    ///     mapped_at_creation: false,
    /// });
    /// graph.renderer().write_buffer(buffer, 0, &buffer_contents);
    /// graph
    ///     .add_render_node(pipeline)
    ///     .with_bind_group((buffer,))
    ///     .build(RenderPassTarget::new(), |pass, [], ()| {
    ///         // --snip--
    ///     });
    /// ```
    pub fn add_intermediate_buffer(
        &mut self,
        descriptor: wgpu::BufferDescriptor,
    ) -> Handle<wgpu::Buffer> {
        let mut hash_descriptor = descriptor.map_label(fxhash::hash64);
        let id = loop {
            match self.renderer.temp_buffers.get(&hash_descriptor) {
                Some(&id) => {
                    if self.intermediate_buffers.insert(Handle::<wgpu::Buffer> {
                        id,
                        _marker: std::marker::PhantomData,
                    }) {
                        break Some(id);
                    }
                }
                None => break None,
            }
            hash_descriptor.label = hash_descriptor.label.wrapping_add(1);
        };
        match id {
            Some(id) => Handle::<wgpu::Buffer> {
                id,
                _marker: std::marker::PhantomData,
            },
            None => {
                let buffer = self.renderer.device.create_buffer(&descriptor);
                let id = self.renderer.next_buffer_id();
                let old = self.renderer.buffers.insert(id, buffer);
                debug_assert!(old.is_none());
                self.intermediate_buffers.insert(Handle::<wgpu::Buffer> {
                    id,
                    _marker: std::marker::PhantomData,
                });
                self.renderer.temp_buffers.insert(hash_descriptor, id);
                Handle::<wgpu::Buffer> {
                    id,
                    _marker: std::marker::PhantomData,
                }
            }
        }
    }

    /// Add an intermediate texture to the graph.
    ///
    /// The same textures are reused between graphs if they have the same descriptors. Creating
    /// multiple textures with the same descriptor is possible within the same render graph. Cached
    /// textures are deallocated if they are not used.
    ///
    /// The returned handle should not be used outside of the context of this graph.
    ///
    /// ```ignore
    /// let resolve_texture = graph.add_intermediate_texture(wgpu::TextureDescriptor {
    ///     label: None,
    ///     size: window_size,
    ///     mip_level_count: 1,
    ///     sample_count: 4,
    ///     dimension: wgpu::TextureDimension::D2,
    ///     format: wgpu::TextureFormat::Bgra8UnormSrgb,
    ///     usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
    /// });
    /// graph
    ///     .add_render_node(pipeline)
    ///     .build(
    ///         RenderPassTarget::new().with_msaa_color(resolve_texture, surface_view, wgpu::Color::BLACK),
    ///         |pass, [], ()| {
    ///             // --snip--
    ///         },
    ///     );
    /// ```
    pub fn add_intermediate_texture(
        &mut self,
        descriptor: wgpu::TextureDescriptor,
    ) -> Handle<wgpu::TextureView> {
        let mut hash_descriptor = descriptor.map_label(fxhash::hash64);
        let id = loop {
            match self.renderer.temp_textures.get(&hash_descriptor) {
                Some(&id) => {
                    if self
                        .intermediate_textures
                        .insert(Handle::<wgpu::TextureView> {
                            id,
                            _marker: std::marker::PhantomData,
                        })
                    {
                        break Some(id);
                    }
                }
                None => break None,
            }
            hash_descriptor.label = hash_descriptor.label.wrapping_add(1);
        };
        match id {
            Some(id) => Handle::<wgpu::TextureView> {
                id,
                _marker: std::marker::PhantomData,
            },
            None => {
                let texture = self.renderer.device.create_texture(&descriptor);
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let id = self.renderer.next_texture_id();
                let old = self.renderer.textures.insert(id, (Some(texture), view));
                debug_assert!(old.is_none());
                self.intermediate_textures
                    .insert(Handle::<wgpu::TextureView> {
                        id,
                        _marker: std::marker::PhantomData,
                    });
                self.renderer.temp_textures.insert(hash_descriptor, id);
                Handle::<wgpu::TextureView> {
                    id,
                    _marker: std::marker::PhantomData,
                }
            }
        }
    }

    /// Add a temporary texture view to the graph.
    ///
    /// The returned handle will be deallocated when this graph is dropped. This method can be
    /// used to render to a [`wgpu::SurfaceTexture`].
    ///
    /// ```ignore
    /// let surface_view = graph.add_temporary_texture_view(
    ///     surface_texture.texture.create_view(Default::default())
    /// );
    /// graph.add_render_node(pipeline).build(
    ///     RenderTarget::new().with_color(surface_view, wgpu::Color::BLACK),
    ///     |pass, [], ()| {
    ///         // --snip--
    ///     },
    /// );
    /// ```
    pub fn add_temporary_texture_view(
        &mut self,
        view: wgpu::TextureView,
    ) -> Handle<wgpu::TextureView> {
        let id = self.renderer.next_texture_id();
        self.renderer.textures.insert(id, (None, view));
        self.temporary_textures.insert(Handle::<wgpu::TextureView> {
            id,
            _marker: std::marker::PhantomData,
        });
        Handle::<wgpu::TextureView> {
            id,
            _marker: std::marker::PhantomData,
        }
    }

    /// Add a command encoder node to the graph.
    ///
    /// A command encoder node takes an [`&wgpu::CommandEncoder`](wgpu::CommandEncoder) as its
    /// operand, and will not be combined with any other nodes.
    ///
    /// See [`EncoderNodeBuilder`] for more details.
    pub fn add_node<'a>(
        &'a mut self,
    ) -> EncoderNodeBuilder<
        'a,
        'r,
        'node,
        [Handle<wgpu::Buffer>; 0],
        [Handle<wgpu::TextureView>; 0],
        [Handle<wgpu::BindGroup>; 0],
        (),
    > {
        EncoderNodeBuilder {
            graph: self,
            dependencies: Default::default(),
            buffers: [],
            textures: [],
            bind_groups: [],
            passthrough: (),
        }
    }

    /// Add a render pass node to the graph.
    ///
    /// A render pass node takes a [`&wgpu::RenderPass`](wgpu::RenderPass) as its operand, and will
    /// be combined with adjacent nodes that share a [`RenderPassTarget`].
    ///
    /// See [`RenderNodeBuilder`] for more details.
    pub fn add_render_node<'a>(
        &'a mut self,
        pipeline: Handle<RenderPipeline>,
    ) -> RenderNodeBuilder<'a, 'r, 'node, [Handle<wgpu::Buffer>; 0], ()> {
        RenderNodeBuilder {
            graph: self,
            dependencies: Default::default(),
            pipeline,
            bind_groups: Vec::new(),
            buffers: [],
            passthrough: (),
        }
    }

    /// Add a compute pass node to the graph.
    ///
    /// A compute pass node takes a [`&wgpu::ComputePass`](wgpu::ComputePass) as its operand, and will
    /// be combined with adjacent compute pass nodes.
    ///
    /// See [`ComputeNodeBuilder`] for more details.
    pub fn add_compute_node<'a>(
        &'a mut self,
        pipeline: Handle<ComputePipeline>,
    ) -> ComputeNodeBuilder<'a, 'r, 'node, [Handle<wgpu::Buffer>; 0], ()> {
        ComputeNodeBuilder {
            graph: self,
            dependencies: Default::default(),
            pipeline,
            bind_groups: Vec::new(),
            buffers: [],
            passthrough: (),
        }
    }

    /// Insert an already-built node into the graph.
    ///
    /// The node will be queried to check which adjacent passes it will be combined with.
    pub fn insert_node(&mut self, node: impl GraphNode<'node> + 'node) {
        let index = self.nodes.len();
        let dependencies = node.dependencies();
        self.node_inputs.insert(
            index,
            dependencies
                .input_buffers
                .into_iter()
                .map(|handle| self.get_versioned_buffer(handle))
                .chain(
                    dependencies
                        .input_textures
                        .into_iter()
                        .map(|handle| self.get_versioned_texture(handle)),
                )
                .collect(),
        );
        let output_resources = dependencies
            .output_buffers
            .into_iter()
            .map(|handle| (self.increment_versioned_buffer(handle), index))
            .collect::<Vec<_>>()
            .into_iter()
            .chain(
                dependencies
                    .output_textures
                    .into_iter()
                    .map(|handle| (self.increment_versioned_texture(handle), index)),
            )
            .collect::<Vec<_>>();
        self.node_outputs.extend(output_resources.into_iter());
        self.nodes.push(Box::new(node));
    }

    /// Execute the render graph with the supplied render target.
    ///
    /// Nodes are executed in the order they were added. Nodes may be culled if their output cannot
    /// be determined to be used.
    ///
    /// Any intermediate resources that were not added to this graph and any temporary resources that
    /// were added to this graph will be deallocated after executing.
    pub fn execute(self) {
        struct CombinedGraphNode<'node> {
            nodes: Vec<Box<dyn GraphNode<'node> + 'node>>,
            compatibility: PassState,
        }

        enum PassState {
            Encoder,
            RenderPass(RenderPassTarget),
            ComputePass,
        }

        let mut combined_nodes = Vec::with_capacity(self.nodes.len());
        for mut node in self.nodes.into_iter().rev() {
            let node_compatibility = match node.requested_state() {
                RequestedState::Encoder => PassState::Encoder,
                RequestedState::RenderPass { target, .. } => PassState::RenderPass(target.unwrap()),
                RequestedState::ComputePass { .. } => PassState::ComputePass,
            };
            match combined_nodes.last_mut() {
                None => combined_nodes.push(CombinedGraphNode {
                    nodes: vec![node],
                    compatibility: node_compatibility,
                }),
                Some(tail) => {
                    let combine = match (&tail.compatibility, &node_compatibility) {
                        (
                            PassState::RenderPass(head_target),
                            PassState::RenderPass(node_target),
                        ) if head_target.is_compatible_with(node_target) => true,
                        (PassState::ComputePass, PassState::ComputePass) => true,
                        _ => false,
                    };
                    if combine {
                        tail.nodes.push(node)
                    } else {
                        combined_nodes.push(CombinedGraphNode {
                            nodes: vec![node],
                            compatibility: node_compatibility,
                        })
                    }
                }
            }
        }

        let mut encoder =
            self.renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Polystrip Command Encoder"),
                });

        let mut cleared_views = FxHashSet::default();
        for group in combined_nodes.into_iter().rev() {
            match group.compatibility {
                PassState::RenderPass(target) => {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &target
                            .color
                            .into_iter()
                            .map(|target| {
                                Some(wgpu::RenderPassColorAttachment {
                                    view: self.renderer.get_texture(target.handle).1,
                                    resolve_target: target
                                        .resolve
                                        .map(|handle| self.renderer.get_texture(handle).1),
                                    ops: wgpu::Operations {
                                        load: if cleared_views.insert(target.handle) {
                                            wgpu::LoadOp::Clear(target.clear)
                                        } else {
                                            wgpu::LoadOp::Load
                                        },
                                        store: true,
                                    },
                                })
                            })
                            .collect::<Vec<_>>(),
                        depth_stencil_attachment: target.depth.map(|target| {
                            wgpu::RenderPassDepthStencilAttachment {
                                view: self.renderer.get_texture(target.handle).1,
                                depth_ops: target.depth_clear.map(|value| wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(value),
                                    store: true,
                                }),
                                stencil_ops: target.stencil_clear.map(|value| wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(value),
                                    store: true,
                                }),
                            }
                        }),
                    });
                    for mut node in group.nodes.into_iter().rev() {
                        let (pipeline_handle, bind_groups) = match node.requested_state() {
                            RequestedState::RenderPass {
                                pipeline,
                                bind_groups,
                                ..
                            } => (pipeline, bind_groups),
                            _ => unreachable!(),
                        };
                        pass.set_pipeline(
                            &self.renderer.get_render_pipeline(pipeline_handle).pipeline,
                        );
                        for (i, handle) in bind_groups.into_iter().enumerate() {
                            pass.set_bind_group(
                                i as u32,
                                self.renderer.get_bind_group(handle),
                                &[],
                            );
                        }
                        node.exec(EncoderOrPass::RenderPass(&mut pass), self.renderer);
                    }
                }
                PassState::ComputePass => {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                    for mut node in group.nodes.into_iter().rev() {
                        let (pipeline_handle, bind_groups) = match node.requested_state() {
                            RequestedState::ComputePass {
                                pipeline,
                                bind_groups,
                            } => (pipeline, bind_groups),
                            _ => unreachable!(),
                        };
                        pass.set_pipeline(
                            &self.renderer.get_compute_pipeline(pipeline_handle).pipeline,
                        );
                        for (i, handle) in bind_groups.into_iter().enumerate() {
                            pass.set_bind_group(
                                i as u32,
                                self.renderer.get_bind_group(handle),
                                &[],
                            );
                        }
                        node.exec(EncoderOrPass::ComputePass(&mut pass), self.renderer);
                    }
                }
                PassState::Encoder => {
                    for mut node in group.nodes.into_iter().rev() {
                        node.exec(EncoderOrPass::Encoder(&mut encoder), self.renderer);
                    }
                }
            }
        }

        self.renderer.queue.submit([encoder.finish()]);

        self.renderer.temp_buffers.retain(|_, &mut id| {
            let pred = self.intermediate_buffers.contains(&Handle::<wgpu::Buffer> {
                id,
                _marker: std::marker::PhantomData,
            });
            if !pred {
                debug_assert!(self.renderer.buffers.remove(&id).is_some());
            }
            pred
        });
        self.renderer.temp_textures.retain(|_, &mut id| {
            let pred = self
                .intermediate_textures
                .contains(&Handle::<wgpu::TextureView> {
                    id,
                    _marker: std::marker::PhantomData,
                });
            if !pred {
                self.renderer.textures.remove(&id);
            }
            pred
        });

        for texture in self.temporary_textures {
            self.renderer.textures.remove(&texture.id);
        }
    }
}

impl RenderGraph<'_, '_> {
    fn get_versioned_buffer(&self, handle: Handle<wgpu::Buffer>) -> VersionedResourceHandle {
        let handle = TrackedResourceHandle::Buffer(handle);
        VersionedResourceHandle {
            handle,
            version: self.resource_versions.get(&handle).copied().unwrap_or(0),
        }
    }

    fn increment_versioned_buffer(
        &mut self,
        handle: Handle<wgpu::Buffer>,
    ) -> VersionedResourceHandle {
        let handle = TrackedResourceHandle::Buffer(handle);
        VersionedResourceHandle {
            handle,
            version: *self
                .resource_versions
                .entry(handle)
                .and_modify(|version| *version += 1)
                .or_insert(1),
        }
    }

    fn get_versioned_texture(&self, handle: Handle<wgpu::TextureView>) -> VersionedResourceHandle {
        let handle = TrackedResourceHandle::Texture(handle);
        VersionedResourceHandle {
            handle,
            version: self.resource_versions.get(&handle).copied().unwrap_or(0),
        }
    }

    fn increment_versioned_texture(
        &mut self,
        handle: Handle<wgpu::TextureView>,
    ) -> VersionedResourceHandle {
        let handle = TrackedResourceHandle::Texture(handle);
        VersionedResourceHandle {
            handle,
            version: *self
                .resource_versions
                .entry(handle)
                .and_modify(|version| *version += 1)
                .or_insert(1),
        }
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct ResourceDependencies {
    pub input_buffers: Vec<Handle<wgpu::Buffer>>,
    pub input_textures: Vec<Handle<wgpu::TextureView>>,
    pub output_buffers: Vec<Handle<wgpu::Buffer>>,
    pub output_textures: Vec<Handle<wgpu::TextureView>>,
}

impl ResourceDependencies {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_input_buffer(mut self, buffer: Handle<wgpu::Buffer>) -> Self {
        self.input_buffers.push(buffer);
        self
    }

    pub fn with_input_texture(mut self, texture: Handle<wgpu::TextureView>) -> Self {
        self.input_textures.push(texture);
        self
    }

    pub fn with_output_buffer(mut self, buffer: Handle<wgpu::Buffer>) -> Self {
        self.output_buffers.push(buffer);
        self
    }

    pub fn with_output_texture(mut self, texture: Handle<wgpu::TextureView>) -> Self {
        self.output_textures.push(texture);
        self
    }
}

/// A node which is given a `&mut wgpu::CommandEncoder`.
///
/// See module docs for more details.
pub struct EncoderNodeBuilder<'a, 'r, 'node, B, T, G, P> {
    graph: &'a mut RenderGraph<'r, 'node>,
    dependencies: ResourceDependencies,
    buffers: B,
    textures: T,
    bind_groups: G,
    passthrough: P,
}

impl<'a, 'r, 'node, B, T, G, P> EncoderNodeBuilder<'a, 'r, 'node, B, T, G, P>
where
    B: ResourceArray<Handle<wgpu::Buffer>> + 'node,
    T: ResourceArray<Handle<wgpu::TextureView>> + 'node,
    G: ResourceArray<Handle<wgpu::BindGroup>> + 'node,
    P: RenderPassthrough + 'node,
{
    /// Add a read-only buffer as a dependency. This buffer will be treated as an input for the
    /// purposes of dependency tracking.
    ///
    /// The buffer will be fetched from the `Renderer` and passed in the second parameter of the closure.
    ///
    /// ```ignore
    /// graph.add_node()
    ///     .with_buffer(handle)
    ///     .build(|encoder, [buffer], [], [], ()| {
    ///         // ...
    ///     })
    /// ```
    pub fn with_buffer(
        self,
        handle: Handle<wgpu::Buffer>,
    ) -> EncoderNodeBuilder<
        'a,
        'r,
        'node,
        <B as ResourceArray<Handle<wgpu::Buffer>>>::ExtendOne,
        T,
        G,
        P,
    > {
        EncoderNodeBuilder {
            graph: self.graph,
            dependencies: self.dependencies.with_input_buffer(handle),
            buffers: self.buffers.extend_one(handle),
            textures: self.textures,
            bind_groups: self.bind_groups,
            passthrough: self.passthrough,
        }
    }

    /// Add a mutable buffer as a dependency. This buffer will be treated as an input and an output
    /// for the purposes of dependency tracking.
    ///
    /// The buffer will be fetched from the `Renderer` and passed in the second parameter of the closure.
    ///
    /// ```ignore
    /// graph.add_node()
    ///     .with_buffer_rw(handle)
    ///     .build(|encoder, [buffer], [], [], ()| {
    ///         // ...
    ///     })
    /// ```
    pub fn with_buffer_rw(
        self,
        handle: Handle<wgpu::Buffer>,
    ) -> EncoderNodeBuilder<
        'a,
        'r,
        'node,
        <B as ResourceArray<Handle<wgpu::Buffer>>>::ExtendOne,
        T,
        G,
        P,
    > {
        EncoderNodeBuilder {
            graph: self.graph,
            dependencies: self
                .dependencies
                .with_input_buffer(handle)
                .with_output_buffer(handle),
            buffers: self.buffers.extend_one(handle),
            textures: self.textures,
            bind_groups: self.bind_groups,
            passthrough: self.passthrough,
        }
    }

    /// Add a read-only texture as a dependency. This texture will be treated as an input for the
    /// purposes of dependency tracking.
    ///
    /// The texture will be fetched from the `Renderer` and passed in the third parameter of the closure.
    ///
    /// ```ignore
    /// graph.add_node()
    ///     .with_texture(handle)
    ///     .build(|encoder, [], [texture], [], ()| {
    ///         // ...
    ///     })
    /// ```
    pub fn with_texture(
        self,
        handle: Handle<wgpu::TextureView>,
    ) -> EncoderNodeBuilder<
        'a,
        'r,
        'node,
        B,
        <T as ResourceArray<Handle<wgpu::TextureView>>>::ExtendOne,
        G,
        P,
    > {
        EncoderNodeBuilder {
            graph: self.graph,
            dependencies: self.dependencies.with_input_texture(handle),
            buffers: self.buffers,
            textures: self.textures.extend_one(handle),
            bind_groups: self.bind_groups,
            passthrough: self.passthrough,
        }
    }

    /// Add a mutable texture as a dependency. This texture will be treated as an input and an
    /// output for the purposes of dependency tracking.
    ///
    /// The texture will be fetched from the `Renderer` and passed in the third parameter of the closure.
    ///
    /// ```ignore
    /// graph.add_node()
    ///     .with_texture_rw(handle)
    ///     .build(|encoder, [], [texture], [], ()| {
    ///         // ...
    ///     })
    /// ```
    pub fn with_texture_rw(
        self,
        handle: Handle<wgpu::TextureView>,
    ) -> EncoderNodeBuilder<
        'a,
        'r,
        'node,
        B,
        <T as ResourceArray<Handle<wgpu::TextureView>>>::ExtendOne,
        G,
        P,
    > {
        EncoderNodeBuilder {
            graph: self.graph,
            dependencies: self
                .dependencies
                .with_input_texture(handle)
                .with_output_texture(handle),
            buffers: self.buffers,
            textures: self.textures.extend_one(handle),
            bind_groups: self.bind_groups,
            passthrough: self.passthrough,
        }
    }

    /// Add a bind group as a dependency. The resources used in this bind group will be treated as
    /// inputs for the purposes of dependency tracking.
    ///
    /// The bind group will be fetched from the `Renderer` and passed in the fourth parameter of
    /// the closure.
    ///
    /// See [`IntoBindGroupResources`] for more details.
    ///
    /// ```ignore
    /// graph.add_node()
    ///     .with_bind_group(layout, (buffer_handle,))
    ///     .build(|encoder, [], [], [bind_group], ()| {
    ///         // ...
    ///     })
    /// ```
    pub fn with_bind_group(
        self,
        layout: Handle<wgpu::BindGroupLayout>,
        resources: impl IntoBindGroupResources,
    ) -> EncoderNodeBuilder<
        'a,
        'r,
        'node,
        B,
        T,
        <G as ResourceArray<Handle<wgpu::BindGroup>>>::ExtendOne,
        P,
    > {
        let resources = resources.into_entries();
        let mut dependencies = self.dependencies;
        for (_, resource) in &resources {
            //TODO: Track output dependencies
            match resource {
                crate::BindGroupResource::Buffer(handle) => {
                    dependencies.input_buffers.push(handle.buffer)
                }
                crate::BindGroupResource::BufferArray(handles) => dependencies
                    .input_buffers
                    .extend(handles.iter().map(|binding| binding.buffer)),
                crate::BindGroupResource::Texture(handle) => {
                    dependencies.input_textures.push(*handle)
                }
                crate::BindGroupResource::TextureArray(handles) => {
                    dependencies.input_textures.extend_from_slice(&handles)
                }
                crate::BindGroupResource::Sampler(_)
                | crate::BindGroupResource::SamplerArray(_) => {}
            }
        }
        let handle = self.graph.renderer.add_bind_group(layout, resources);
        EncoderNodeBuilder {
            graph: self.graph,
            dependencies,
            buffers: self.buffers,
            textures: self.textures,
            bind_groups: self.bind_groups.extend_one(handle),
            passthrough: self.passthrough,
        }
    }

    /// Add passthrough data to the node.
    ///
    /// This data will be reborrowed with lifetime `'pass` and passed to the fifth parameter of the
    /// closure.
    ///
    /// ```ignore
    /// let buffer = device.create_buffer(&wgpu::BufferDescriptor { .. });
    /// graph.add_node()
    ///     .with_passthrough(&buffer)
    ///     .build(|encoder, [], [], [], (buffer,)| {
    ///         // Trying to use `buffer` without the passthrough mechanism would have failed because
    ///         // nothing guarantees 'node outlives 'pass.
    ///     })
    /// ```
    pub fn with_passthrough<A: 'node>(
        self,
        item: &'node A,
    ) -> EncoderNodeBuilder<'a, 'r, 'node, B, T, G, <P as ExtendTuple<*const A>>::ExtendOne>
    where
        P: ExtendTuple<*const A>,
    {
        EncoderNodeBuilder {
            graph: self.graph,
            dependencies: self.dependencies,
            buffers: self.buffers,
            textures: self.textures,
            bind_groups: self.bind_groups,
            passthrough: self.passthrough.extend_one(item as *const A),
        }
    }

    /// Build the node and add it to the graph.
    ///
    /// The closure is given a [`&wgpu::CommandEncoder`](wgpu::CommandEncoder) and references to
    /// the requested resources. The parameters for the closure can be named as follows:
    ///
    /// ```text
    /// fn exec(
    ///     encoder: &'b mut wgpu::CommandEncoder,
    ///     renderer: &'pass Renderer,
    ///     buffers: [&'pass wgpu::Buffer; _],
    ///     textures: [&'pass wgpu::TextureView; _],
    ///     bind_groups: [&'pass wgpu::BindGroup; _],
    ///     passthrough: (T₁, T₂, …, Tₙ),
    /// ) -> ()
    /// ```
    ///
    /// The closure may be skipped if dependency tracking marks this node as unused. To force this
    /// node to be executed, see [`Self::with_external_output`].
    pub fn build<F>(self, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::CommandEncoder,
                &'pass Renderer,
                <B as ResourceArray<Handle<wgpu::Buffer>>>::Fetched<'pass>,
                <T as ResourceArray<Handle<wgpu::TextureView>>>::Fetched<'pass>,
                <G as ResourceArray<Handle<wgpu::BindGroup>>>::Fetched<'pass>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    {
        self.graph.insert_node(EncoderGraphNode {
            inner: Some(EncoderGraphNodeInner {
                dependencies: self.dependencies,
                buffers: self.buffers,
                textures: self.textures,
                passthrough: self.passthrough,
                bind_groups: self.bind_groups,
                exec: Box::new(exec),
            }),
        });
    }
}

struct EncoderGraphNode<
    'node,
    B: ResourceArray<Handle<wgpu::Buffer>>,
    T: ResourceArray<Handle<wgpu::TextureView>>,
    G: ResourceArray<Handle<wgpu::BindGroup>>,
    P: RenderPassthrough,
> {
    inner: Option<EncoderGraphNodeInner<'node, B, T, G, P>>,
}

struct EncoderGraphNodeInner<
    'node,
    B: ResourceArray<Handle<wgpu::Buffer>>,
    T: ResourceArray<Handle<wgpu::TextureView>>,
    G: ResourceArray<Handle<wgpu::BindGroup>>,
    P: RenderPassthrough,
> {
    dependencies: ResourceDependencies,
    buffers: B,
    textures: T,
    bind_groups: G,
    passthrough: P,
    exec: Box<
        dyn for<'b, 'pass> FnOnce(
                &'b mut wgpu::CommandEncoder,
                &'pass Renderer,
                <B as ResourceArray<Handle<wgpu::Buffer>>>::Fetched<'pass>,
                <T as ResourceArray<Handle<wgpu::TextureView>>>::Fetched<'pass>,
                <G as ResourceArray<Handle<wgpu::BindGroup>>>::Fetched<'pass>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    >,
}

/// A node which is given a `&mut wgpu::RenderPass`.
///
/// A `RenderGraphNode` is tied to a single [`RenderPassTarget`], [`wgpu::RenderPipeline`]
/// and set of [`wgpu::BindGroup`]s. Adjacent nodes sharing the same target will execute in the
/// same pass.
pub struct RenderNodeBuilder<'a, 'r, 'node, B, P> {
    graph: &'a mut RenderGraph<'r, 'node>,
    dependencies: ResourceDependencies,
    pipeline: Handle<RenderPipeline>,
    bind_groups: Vec<Handle<wgpu::BindGroup>>,
    buffers: B,
    passthrough: P,
}

impl<'a, 'r, 'node, B, P> RenderNodeBuilder<'a, 'r, 'node, B, P>
where
    B: ResourceArray<Handle<wgpu::Buffer>> + 'node,
    P: RenderPassthrough + 'node,
{
    /// Add a read-only buffer as a dependency. This buffer will be treated as an input for the
    /// purposes of dependency tracking.
    ///
    /// The buffer will be fetched from the `Renderer` and passed in the second parameter of the closure.
    ///
    /// ```ignore
    /// graph.add_render_node(pipeline_handle)
    ///     .with_buffer(handle)
    ///     .build(|render_pass, [buffer], ()| {
    ///         // ...
    ///     })
    /// ```
    pub fn with_buffer(
        self,
        handle: Handle<wgpu::Buffer>,
    ) -> RenderNodeBuilder<'a, 'r, 'node, <B as ResourceArray<Handle<wgpu::Buffer>>>::ExtendOne, P>
    {
        RenderNodeBuilder {
            graph: self.graph,
            dependencies: self.dependencies.with_input_buffer(handle),
            pipeline: self.pipeline,
            bind_groups: self.bind_groups,
            buffers: self.buffers.extend_one(handle),
            passthrough: self.passthrough,
        }
    }

    /// Add a bind group as a dependency. The resources used in this bind group will be treated as
    /// inputs for the purposes of dependency tracking.
    ///
    /// Added bind groups will be set up before the node's execution in the order they were added.
    ///
    /// See [`IntoBindGroupResources`] for more details.
    ///
    /// ```ignore
    /// graph.add_render_node(pipeline_handle)
    ///     .with_bind_group((buffer_handle,))
    ///     .with_bind_group((texture_handle, sampler_handle))
    ///     .build(|render_pass, [], ()| {
    ///         // Bind group set 0 is (buffer_handle)
    ///         // Bind group set 1 is (texture_handle, sampler_handle)
    ///         render_pass.draw(0..3, 0..1);
    ///     })
    /// ```
    pub fn with_bind_group(mut self, resources: impl IntoBindGroupResources) -> Self {
        //TODO: Bind group resource tracking
        self.bind_groups.push(
            self.graph.renderer.add_bind_group(
                self.graph
                    .renderer
                    .get_render_pipeline(self.pipeline)
                    .bind_group_layouts[self.bind_groups.len()],
                resources,
            ),
        );
        self
    }

    /// Add passthrough data to the node.
    ///
    /// This data will be reborrowed with lifetime `'pass` and passed to the fifth parameter of the
    /// closure.
    ///
    /// ```ignore
    /// let buffer = device.create_buffer(&wgpu::BufferDescriptor { .. });
    /// graph.add_render_node()
    ///     .with_passthrough(&buffer)
    ///     .build(|encoder, [], (buffer,)| {
    ///         // Trying to use `buffer` without the passthrough mechanism would have failed because
    ///         // nothing guarantees 'node outlives 'pass.
    ///     })
    /// ```
    pub fn with_passthrough<A: 'node>(
        self,
        item: &'node A,
    ) -> RenderNodeBuilder<'a, 'r, 'node, B, <P as ExtendTuple<*const A>>::ExtendOne>
    where
        P: ExtendTuple<*const A>,
    {
        RenderNodeBuilder {
            graph: self.graph,
            dependencies: self.dependencies,
            pipeline: self.pipeline,
            bind_groups: self.bind_groups,
            buffers: self.buffers,
            passthrough: self.passthrough.extend_one(item as *const A),
        }
    }

    /// Build the node and add it to the graph.
    ///
    /// The closure is given a [`&wgpu::RenderPass`](wgpu::RenderPass) and references to
    /// the requested resources. The parameters for the closure can be named as follows:
    ///
    /// ```text
    /// fn exec(
    ///     encoder: &'b mut wgpu::CommandEncoder,
    ///     buffers: [&'pass wgpu::Buffer; _],
    ///     passthrough: (T₁, T₂, …, Tₙ),
    /// ) -> ()
    /// ```
    ///
    /// Additionally, the render pass will have been set up with the requested render pass
    /// target, pipeline, and bind groups.
    ///
    /// The closure may be skipped if dependency tracking marks this node as unused.
    pub fn build<F>(self, render_pass: RenderPassTarget, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::RenderPass<'pass>,
                <B as ResourceArray<Handle<wgpu::Buffer>>>::Fetched<'pass>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    {
        self.graph.insert_node(RenderGraphNode {
            inner: Some(RenderGraphNodeInner {
                pipeline: self.pipeline,
                dependencies: self.dependencies,
                bind_groups: self.bind_groups,
                buffers: self.buffers,
                passthrough: self.passthrough,
                render_pass: Some(render_pass),
                exec: Box::new(exec),
            }),
        });
    }
}

struct RenderGraphNode<'node, B: ResourceArray<Handle<wgpu::Buffer>>, P: RenderPassthrough> {
    inner: Option<RenderGraphNodeInner<'node, B, P>>,
}

#[allow(clippy::type_complexity)]
struct RenderGraphNodeInner<'node, B: ResourceArray<Handle<wgpu::Buffer>>, P: RenderPassthrough> {
    pipeline: Handle<RenderPipeline>,
    dependencies: ResourceDependencies,
    bind_groups: Vec<Handle<wgpu::BindGroup>>,
    buffers: B,
    passthrough: P,
    render_pass: Option<RenderPassTarget>,
    exec: Box<
        dyn for<'b, 'pass> FnOnce(
                &'b mut wgpu::RenderPass<'pass>,
                <B as ResourceArray<Handle<wgpu::Buffer>>>::Fetched<'pass>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    >,
}

/// A node which is given a `&mut wgpu::ComputePass`.
///
/// A `ComputeGraphNode` is tied to a single [`wgpu::ComputePipeline`] and set of
/// [`wgpu::BindGroup`]s. Adjacent compute passes will execute in the same pass.
pub struct ComputeNodeBuilder<'a, 'r, 'node, B, P> {
    graph: &'a mut RenderGraph<'r, 'node>,
    dependencies: ResourceDependencies,
    pipeline: Handle<ComputePipeline>,
    bind_groups: Vec<Handle<wgpu::BindGroup>>,
    buffers: B,
    passthrough: P,
}

impl<'a, 'r, 'node, B, P> ComputeNodeBuilder<'a, 'r, 'node, B, P>
where
    B: ResourceArray<Handle<wgpu::Buffer>> + 'node,
    P: RenderPassthrough + 'node,
{
    /// Add a read-only buffer as a dependency. This buffer will be treated as an input for the
    /// purposes of dependency tracking.
    ///
    /// The buffer will be fetched from the `Renderer` and passed in the second parameter of the closure.
    ///
    /// ```ignore
    /// graph.add_render_node(pipeline_handle)
    ///     .with_buffer(handle)
    ///     .build(|render_pass, [buffer], ()| {
    ///         // ...
    ///     })
    /// ```
    pub fn with_buffer(
        self,
        handle: Handle<wgpu::Buffer>,
    ) -> ComputeNodeBuilder<'a, 'r, 'node, <B as ResourceArray<Handle<wgpu::Buffer>>>::ExtendOne, P>
    {
        ComputeNodeBuilder {
            graph: self.graph,
            dependencies: self.dependencies.with_input_buffer(handle),
            pipeline: self.pipeline,
            bind_groups: self.bind_groups,
            buffers: self.buffers.extend_one(handle),
            passthrough: self.passthrough,
        }
    }

    /// Add a bind group as a dependency. The resources used in this bind group will be treated as
    /// inputs for the purposes of dependency tracking.
    ///
    /// Added bind groups will be set up before the node's execution in the order they were added.
    ///
    /// See [`IntoBindGroupResources`] for more details.
    ///
    /// ```ignore
    /// graph.add_render_node(pipeline_handle)
    ///     .with_bind_group((buffer_handle,))
    ///     .with_bind_group((texture_handle, sampler_handle))
    ///     .build(|render_pass, [], ()| {
    ///         // Bind group set 0 is (buffer_handle)
    ///         // Bind group set 1 is (texture_handle, sampler_handle)
    ///         render_pass.draw(0..3, 0..1);
    ///     })
    /// ```
    pub fn with_bind_group(mut self, resources: impl IntoBindGroupResources) -> Self {
        //TODO: Bind group dependency tracking
        self.bind_groups.push(
            self.graph.renderer.add_bind_group(
                self.graph
                    .renderer
                    .get_compute_pipeline(self.pipeline)
                    .bind_group_layouts[self.bind_groups.len()],
                resources,
            ),
        );
        self
    }

    /// Add passthrough data to the node.
    ///
    /// This data will be reborrowed with lifetime `'pass` and passed to the fifth parameter of the
    /// closure.
    ///
    /// ```ignore
    /// let buffer = device.create_buffer(&wgpu::BufferDescriptor { .. });
    /// graph.add_render_node()
    ///     .with_passthrough(&buffer)
    ///     .build(|encoder, [], (buffer,)| {
    ///         // Trying to use `buffer` without the passthrough mechanism would have failed because
    ///         // nothing guarantees 'node outlives 'pass.
    ///     })
    /// ```
    pub fn with_passthrough<A: 'node>(
        self,
        item: &'node A,
    ) -> ComputeNodeBuilder<'a, 'r, 'node, B, <P as ExtendTuple<*const A>>::ExtendOne>
    where
        P: ExtendTuple<*const A>,
    {
        ComputeNodeBuilder {
            graph: self.graph,
            dependencies: self.dependencies,
            pipeline: self.pipeline,
            bind_groups: self.bind_groups,
            buffers: self.buffers,
            passthrough: self.passthrough.extend_one(item as *const A),
        }
    }

    /// Build the node and add it to the graph.
    ///
    /// The closure is given a [`&wgpu::RenderPass`](wgpu::RenderPass) and references to
    /// the requested resources. The parameters for the closure can be named as follows:
    ///
    /// ```text
    /// fn exec(
    ///     encoder: &'b mut wgpu::CommandEncoder,
    ///     buffers: [&'pass wgpu::Buffer; _],
    ///     passthrough: (T₁, T₂, …, Tₙ),
    /// ) -> ()
    /// ```
    ///
    /// Additionally, the render pass will have been set up with the requested render pass
    /// target, pipeline, and bind groups.
    ///
    /// The closure may be skipped if dependency tracking marks this node as unused.
    pub fn build<F>(self, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::ComputePass<'pass>,
                <B as ResourceArray<Handle<wgpu::Buffer>>>::Fetched<'pass>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    {
        self.graph.insert_node(ComputeGraphNode {
            inner: Some(ComputeGraphNodeInner {
                pipeline: self.pipeline,
                dependencies: self.dependencies,
                bind_groups: self.bind_groups,
                buffers: self.buffers,
                passthrough: self.passthrough,
                exec: Box::new(exec),
            }),
        });
    }
}

struct ComputeGraphNode<'node, B: ResourceArray<Handle<wgpu::Buffer>>, P: RenderPassthrough> {
    inner: Option<ComputeGraphNodeInner<'node, B, P>>,
}

#[allow(clippy::type_complexity)]
struct ComputeGraphNodeInner<'node, B: ResourceArray<Handle<wgpu::Buffer>>, P: RenderPassthrough> {
    pipeline: Handle<ComputePipeline>,
    dependencies: ResourceDependencies,
    bind_groups: Vec<Handle<wgpu::BindGroup>>,
    buffers: B,
    passthrough: P,
    exec: Box<
        dyn for<'b, 'pass> FnOnce(
                &'b mut wgpu::ComputePass<'pass>,
                <B as ResourceArray<Handle<wgpu::Buffer>>>::Fetched<'pass>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    >,
}

/// The state requested by a [`GraphNode`].
///
/// The requested state will be passed to the node through [`EncoderOrPass`].
pub enum RequestedState {
    /// Node will receive a `&mut wgpu::CommandEncoder` with no assumptions about state.
    Encoder,
    /// Node will receive a `&mut wgpu::RenderPass<'_>` set up with the chosen `RenderPipeline` and `BindGroup`s.
    ///
    /// Adjacent render passes will be combined based on the chosen [`RenderPassTarget`].
    RenderPass {
        target: Option<RenderPassTarget>,
        pipeline: Handle<RenderPipeline>,
        bind_groups: Vec<Handle<wgpu::BindGroup>>,
    },
    /// Node will receive a `&mut wgpu::ComputePass<'_>` set up with the chosen `ComputePipeline` and `BindGroup`s.
    ComputePass {
        pipeline: Handle<ComputePipeline>,
        bind_groups: Vec<Handle<wgpu::BindGroup>>,
    },
}

/// The state passed to a [`GraphNode`].
///
/// Chosen by a `GraphNode` through [`RequestedState`].
pub enum EncoderOrPass<'b, 'pass> {
    Encoder(&'b mut wgpu::CommandEncoder),
    RenderPass(&'b mut wgpu::RenderPass<'pass>),
    ComputePass(&'b mut wgpu::ComputePass<'pass>),
}

impl<'b, 'pass> EncoderOrPass<'b, 'pass> {
    #[inline]
    #[track_caller]
    fn unwrap_encoder(self) -> &'b mut wgpu::CommandEncoder {
        match self {
            Self::Encoder(encoder) => encoder,
            Self::RenderPass(_) => {
                panic!("called `EncoderOrPass::unwrap_encoder()` on a `RenderPass` value")
            }
            Self::ComputePass(_) => {
                panic!("called `EncoderOrPass::unwrap_encoder()` on a `ComputePass` value")
            }
        }
    }

    #[inline]
    #[track_caller]
    fn unwrap_render_pass(self) -> &'b mut wgpu::RenderPass<'pass> {
        match self {
            Self::Encoder(_) => {
                panic!("called `EncoderOrPass::unwrap_render_pass()` on an `Encoder` value")
            }
            Self::RenderPass(pass) => pass,
            Self::ComputePass(_) => {
                panic!("called `EncoderOrPass::unwrap_render_pass()` on an `ComputePass` value")
            }
        }
    }

    #[inline]
    #[track_caller]
    fn unwrap_compute_pass(self) -> &'b mut wgpu::ComputePass<'pass> {
        match self {
            Self::Encoder(_) => {
                panic!("called `EncoderOrPass::unwrap_compute_pass()` on an `Encoder` value")
            }
            Self::RenderPass(_) => {
                panic!("called `EncoderOrPass::unwrap_compute_pass()` on a `RenderPass` value")
            }
            Self::ComputePass(pass) => pass,
        }
    }
}

/// A node to be added to a [`RenderGraph`]
pub trait GraphNode<'node> {
    /// The encoding state expected to be passed to `exec`.
    fn requested_state(&mut self) -> RequestedState;
    fn dependencies(&self) -> ResourceDependencies;
    /// Execute the node, given the requested encoding state
    fn exec<'pass>(&mut self, encoder_or_pass: EncoderOrPass<'_, 'pass>, renderer: &'pass Renderer)
    where
        'node: 'pass;
}

impl<
        'node,
        B: ResourceArray<Handle<wgpu::Buffer>>,
        T: ResourceArray<Handle<wgpu::TextureView>>,
        G: ResourceArray<Handle<wgpu::BindGroup>>,
        P: RenderPassthrough + 'node,
    > GraphNode<'node> for EncoderGraphNode<'node, B, T, G, P>
{
    fn requested_state(&mut self) -> RequestedState {
        RequestedState::Encoder
    }

    fn dependencies(&self) -> ResourceDependencies {
        self.inner.as_ref().unwrap().dependencies.clone()
    }

    fn exec<'pass>(&mut self, encoder_or_pass: EncoderOrPass<'_, 'pass>, renderer: &'pass Renderer)
    where
        'node: 'pass,
    {
        let inner = self.inner.take().unwrap();
        (inner.exec)(
            encoder_or_pass.unwrap_encoder(),
            renderer,
            inner.buffers.fetch_resources(renderer),
            inner.textures.fetch_resources(renderer),
            inner.bind_groups.fetch_resources(renderer),
            unsafe {
                // SAFETY: Not entirely enforced here.
                // * 'node outlives 'pass
                // * Original references are borrowchecked wrt. 'node
                inner.passthrough.reborrow()
            },
        )
    }
}

impl<'node, B: ResourceArray<Handle<wgpu::Buffer>>, P: RenderPassthrough + 'node> GraphNode<'node>
    for RenderGraphNode<'node, B, P>
{
    fn requested_state(&mut self) -> RequestedState {
        let inner = self.inner.as_mut().unwrap();
        RequestedState::RenderPass {
            target: inner.render_pass.take(),
            pipeline: inner.pipeline,
            bind_groups: inner.bind_groups.clone(),
        }
    }

    fn dependencies(&self) -> ResourceDependencies {
        self.inner.as_ref().unwrap().dependencies.clone()
    }

    fn exec<'pass>(&mut self, encoder_or_pass: EncoderOrPass<'_, 'pass>, renderer: &'pass Renderer)
    where
        'node: 'pass,
    {
        let inner = self.inner.take().unwrap();
        let pass = encoder_or_pass.unwrap_render_pass();
        (inner.exec)(pass, inner.buffers.fetch_resources(renderer), unsafe {
            // SAFETY: As above
            inner.passthrough.reborrow()
        })
    }
}

impl<'node, B: ResourceArray<Handle<wgpu::Buffer>>, P: RenderPassthrough + 'node> GraphNode<'node>
    for ComputeGraphNode<'node, B, P>
{
    fn requested_state(&mut self) -> RequestedState {
        let inner = self.inner.as_ref().unwrap();
        RequestedState::ComputePass {
            pipeline: inner.pipeline,
            bind_groups: inner.bind_groups.clone(),
        }
    }

    fn dependencies(&self) -> ResourceDependencies {
        self.inner.as_ref().unwrap().dependencies.clone()
    }

    fn exec<'pass>(&mut self, encoder_or_pass: EncoderOrPass<'_, 'pass>, renderer: &'pass Renderer)
    where
        'node: 'pass,
    {
        let inner = self.inner.take().unwrap();
        let pass = encoder_or_pass.unwrap_compute_pass();
        (inner.exec)(pass, inner.buffers.fetch_resources(renderer), unsafe {
            // SAFETY: As above
            inner.passthrough.reborrow()
        })
    }
}

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

/// (*internal*) A tuple of passthrough resources that can be extended by one element.
///
/// This trait is used in combination with [`RenderPassthrough`] to allow pattern-matching on node
/// passthrough resources. It is not intended to be used in application code, and cannot be implemented
/// on custom types.
pub trait ExtendTuple<T>: Sealed {
    type ExtendOne;

    fn extend_one(self, _: T) -> Self::ExtendOne;
}

/// (*internal*) A tuple of resources that can be borrowed at a shorter lifetime.
///
/// This trait is used in combination with [`ExtendTuple`] to allow pattern-matching on node
/// passthrough resources. It is not intended to be used in application code, and cannot be implemented
/// on custom types.
pub trait RenderPassthrough: Sealed {
    type Reborrowed<'a>
    where
        Self: 'a;

    /// # Safety
    /// It must be valid to create a shared reference from the borrowed data that lives for the bound
    /// lifetime. If you don't know how to verify this off the top of your head, you probably shouldn't
    /// be touching this method.
    unsafe fn reborrow<'a>(self) -> Self::Reborrowed<'a>;
}

impl Sealed for () {}

impl<T> ExtendTuple<T> for () {
    type ExtendOne = (T,);

    fn extend_one(self, element: T) -> Self::ExtendOne {
        (element,)
    }
}

impl RenderPassthrough for () {
    type Reborrowed<'a> = ();

    unsafe fn reborrow<'a>(self) -> Self::Reborrowed<'a> {}
}

impl<T> Sealed for *const T {}
impl<T> Sealed for *mut T {}

impl<T> RenderPassthrough for *const T {
    type Reborrowed<'a> = &'a T where Self: 'a;

    unsafe fn reborrow<'a>(self) -> Self::Reborrowed<'a> {
        &*self
    }
}

impl<T> RenderPassthrough for *mut T {
    type Reborrowed<'a> = &'a mut T where Self: 'a;

    unsafe fn reborrow<'a>(self) -> Self::Reborrowed<'a> {
        &mut *self
    }
}

macro_rules! extend_tuple {
    ($tuple:ident, $element:ident, $($letters:ident,)*) => {{
        let ($($letters,)*) = $tuple;
        ($($letters,)* $element)
    }};
}

macro_rules! reborrow_tuple {
    ($tuple:ident, $($letters:ident,)*) => {{
        let ($($letters,)*) = $tuple;
        ($($letters.reborrow(),)*)
    }}
}

macro_rules! impl_extend_tuple {
    ($head:ident,) => {
        impl<$head> Sealed for ($head,) {}

        impl<$head, T> ExtendTuple<T> for ($head,) {
            type ExtendOne = ($head, T);

            #[allow(non_snake_case)]
            fn extend_one(self, element: T) -> Self::ExtendOne {
                extend_tuple!(self, element, $head,)
            }
        }

        impl<$head: RenderPassthrough> RenderPassthrough for ($head,) {
            type Reborrowed<'a> = (<$head as RenderPassthrough>::Reborrowed<'a>,) where $head: 'a;

            #[allow(non_snake_case)]
            unsafe fn reborrow<'a>(self) -> Self::Reborrowed<'a> {
                reborrow_tuple!(self, $head,)
            }
        }
    };
    ($head:ident, $($tail:ident,)*) => {
        impl<$head, $($tail,)*> Sealed for ($head, $($tail,)*) {}

        impl<$head, $($tail,)* T> ExtendTuple<T> for ($head, $($tail,)*) {
            type ExtendOne = ($head, $($tail,)* T);

            #[allow(non_snake_case)]
            fn extend_one(self, element:T) -> Self::ExtendOne {
                extend_tuple!(self, element, $head, $($tail,)*)
            }
        }

        impl<$head: RenderPassthrough, $($tail: RenderPassthrough,)*> RenderPassthrough for ($head, $($tail,)*) {
            type Reborrowed<'a> = (<$head as RenderPassthrough>::Reborrowed<'a>, $(<$tail as RenderPassthrough>::Reborrowed<'a>,)*) where $head: 'a, $($tail: 'a,)*;

            #[allow(non_snake_case)]
            unsafe fn reborrow<'a>(self) -> Self::Reborrowed<'a> {
                reborrow_tuple!(self, $head, $($tail,)*)
            }
        }
        impl_extend_tuple!($($tail,)*);
    };
}
impl_extend_tuple!(A, B, C, D, E, F, G, H, I, J, K, L,);

/// (*internal*) A [`Handle<T>`] to be resolved upon node execution.
///
/// This trait is used in combination with [`ResourceArray`] to allow pattern-matching on arbitrary
/// numbers of resource handles in node exec closures. It is not intended to be used in application code,
/// and cannot be implemented on custom types.
pub trait RenderResource: Sealed {
    type Resource<'a>;

    fn fetch_resource(self, renderer: &Renderer) -> Self::Resource<'_>;
}

impl Sealed for Handle<wgpu::Buffer> {}
impl Sealed for Handle<wgpu::TextureView> {}
impl Sealed for Handle<wgpu::BindGroup> {}

impl RenderResource for Handle<wgpu::Buffer> {
    type Resource<'a> = &'a wgpu::Buffer;

    fn fetch_resource(self, renderer: &Renderer) -> Self::Resource<'_> {
        renderer.get_buffer(self)
    }
}

impl RenderResource for Handle<wgpu::TextureView> {
    type Resource<'a> = &'a wgpu::TextureView;

    fn fetch_resource(self, renderer: &Renderer) -> Self::Resource<'_> {
        renderer.get_texture(self).1
    }
}

impl RenderResource for Handle<wgpu::BindGroup> {
    type Resource<'a> = &'a wgpu::BindGroup;

    fn fetch_resource(self, renderer: &Renderer) -> Self::Resource<'_> {
        renderer.get_bind_group(self)
    }
}

/// (*internal*) An array of [`Handle<T>`] to be resolved upon node execution.
///
/// This trait is used in combination with [`RenderResource`] to allow pattern-matching on arbitrary
/// numbers of resource handles in node exec closures. It is not intended to be used in application code,
/// and cannot be implemented on custom types.
pub trait ResourceArray<T: RenderResource>: Sealed {
    type Fetched<'a>;
    type ExtendOne;

    fn as_slice(&self) -> &[T];
    fn fetch_resources(self, renderer: &Renderer) -> Self::Fetched<'_>;
    fn extend_one(self, _: T) -> Self::ExtendOne;
}

impl<T> Sealed for [T; 0] {}

impl<T: RenderResource> ResourceArray<T> for [T; 0] {
    type Fetched<'a> = [<T as RenderResource>::Resource<'a>; 0];
    type ExtendOne = [T; 1];

    fn as_slice(&self) -> &[T] {
        self
    }

    fn fetch_resources(self, _renderer: &Renderer) -> Self::Fetched<'_> {
        []
    }

    fn extend_one(self, element: T) -> Self::ExtendOne {
        [element]
    }
}

macro_rules! extend_array {
    ($array:ident, $element:ident, $($letters:ident,)*) => {{
        let [$($letters,)*] = $array;
        [$($letters,)* $element]
    }};
}

macro_rules! impl_resource_array {
    ($size:expr, $head:ident,) => {
        impl<T> Sealed for [T; $size] {}

        impl<T: RenderResource> ResourceArray<T> for [T; $size] {
            type Fetched<'a> = [<T as RenderResource>::Resource<'a>; $size];
            type ExtendOne = [T; {$size + 1}];

            fn as_slice(&self) -> &[T] {
                self
            }

            fn fetch_resources<'a>(self, renderer: &'a Renderer) -> Self::Fetched<'a> {
                self.map(|element| <T as RenderResource>::fetch_resource(element, renderer))
            }

            fn extend_one(self, element: T) -> Self::ExtendOne {
                extend_array!(self, element, $head,)
            }
        }
    };
    ($size:expr, $head:ident, $($letters:ident,)*) => {
        impl<T> Sealed for [T; $size] {}

        impl<T: RenderResource> ResourceArray<T> for [T; $size] {
            type Fetched<'a> = [<T as RenderResource>::Resource<'a>; $size];
            type ExtendOne = [T; {$size + 1}];

            fn as_slice(&self) -> &[T] {
                self
            }

            fn fetch_resources<'a>(self, renderer: &'a Renderer) -> Self::Fetched<'a> {
                self.map(|element| <T as RenderResource>::fetch_resource(element, renderer))
            }

            fn extend_one(self, element: T) -> Self::ExtendOne {
                extend_array!(self, element, $head, $($letters,)*)
            }
        }
        impl_resource_array!(($size - 1), $($letters,)*);
    };
}
impl_resource_array!(
    32, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, ab, ac,
    ad, ae, af,
);
