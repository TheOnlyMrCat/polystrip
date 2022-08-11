use fxhash::FxHashSet;

use crate::{
    BindGroupHandle, BufferHandle, IntoBindGroupResources, RenderPassTarget, RenderTarget,
    Renderer, TextureHandle,
};

pub struct RenderPassResources<'pass> {
    resources: &'pass Renderer,
    output_texture: &'pass wgpu::TextureView,
}

impl<'pass> RenderPassResources<'pass> {
    pub fn get_buffer(&self, handle: BufferHandle) -> &'pass wgpu::Buffer {
        self.resources.buffers.get(&handle.id).unwrap()
    }

    pub fn get_texture(&self, handle: TextureHandle) -> &'pass wgpu::TextureView {
        if handle.id == usize::MAX {
            self.output_texture
        } else {
            let (_, view) = self.resources.textures.get(&handle.id).unwrap();
            view
        }
    }

    pub fn get_bind_group(&self, handle: BindGroupHandle) -> &'pass wgpu::BindGroup {
        &self.resources.bind_groups.get(&handle.id).unwrap().0
    }
}

pub struct RenderGraph<'r, 'node> {
    renderer: &'r mut Renderer,
    intermediate_buffers: FxHashSet<BufferHandle>,
    intermediate_textures: FxHashSet<TextureHandle>,
    nodes: Vec<Box<dyn GraphNodeImpl<'node> + 'node>>,
}

impl<'r, 'node> RenderGraph<'r, 'node> {
    pub fn new(renderer: &'r mut Renderer) -> RenderGraph<'r, 'node> {
        RenderGraph {
            renderer,
            intermediate_buffers: FxHashSet::default(),
            intermediate_textures: FxHashSet::default(),
            nodes: Vec::new(),
        }
    }

    pub fn renderer(&self) -> &Renderer {
        self.renderer
    }

    pub fn renderer_mut(&mut self) -> &mut Renderer {
        self.renderer
    }

    pub fn add_intermediate_buffer(&mut self, descriptor: wgpu::BufferDescriptor) -> BufferHandle {
        let mut hash_descriptor = descriptor.map_label(fxhash::hash64);
        let id = loop {
            match self.renderer.temp_buffers.get(&hash_descriptor) {
                Some(&id) => {
                    if self.intermediate_buffers.insert(BufferHandle { id }) {
                        break Some(id);
                    }
                }
                None => break None,
            }
            hash_descriptor.label = hash_descriptor.label.wrapping_add(1);
        };
        match id {
            Some(id) => BufferHandle { id },
            None => {
                let buffer = self.renderer.device.create_buffer(&descriptor);
                let id = self.renderer.next_buffer_id();
                let old = self.renderer.buffers.insert(id, buffer);
                debug_assert!(old.is_none());
                self.intermediate_buffers.insert(BufferHandle { id });
                self.renderer.temp_buffers.insert(hash_descriptor, id);
                BufferHandle { id }
            }
        }
    }

    pub fn add_intermediate_texture(
        &mut self,
        descriptor: wgpu::TextureDescriptor,
    ) -> TextureHandle {
        let mut hash_descriptor = descriptor.map_label(fxhash::hash64);
        let id = loop {
            match self.renderer.temp_textures.get(&hash_descriptor) {
                Some(&id) => {
                    if self.intermediate_textures.insert(TextureHandle { id }) {
                        break Some(id);
                    }
                }
                None => break None,
            }
            hash_descriptor.label = hash_descriptor.label.wrapping_add(1);
        };
        match id {
            Some(id) => TextureHandle { id },
            None => {
                let texture = self.renderer.device.create_texture(&descriptor);
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let id = self.renderer.next_texture_id();
                let old = self.renderer.textures.insert(id, (texture, view));
                debug_assert!(old.is_none());
                self.intermediate_textures.insert(TextureHandle { id });
                self.renderer.temp_textures.insert(hash_descriptor, id);
                TextureHandle { id }
            }
        }
    }

    pub fn add_node<'a>(
        &'a mut self,
    ) -> NodeBuilder<'a, 'r, 'node, [BufferHandle; 0], [TextureHandle; 0], [BindGroupHandle; 0], ()>
    {
        NodeBuilder {
            graph: self,
            rw_buffers: vec![],
            rw_textures: vec![],
            no_cull: false,
            buffers: [],
            textures: [],
            bind_groups: [],
            passthrough: (),
        }
    }

    pub fn execute(self, target: &mut impl RenderTarget) {
        let output = target.get_current_view();

        let mut encoder =
            self.renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Polystrip Command Encoder"),
                });

        let mut used_buffers = FxHashSet::default();
        let mut used_textures = FxHashSet::default();
        used_textures.insert(TextureHandle::RENDER_TARGET);
        let mut pruned_nodes = Vec::with_capacity(self.nodes.len());
        for node in self.nodes.into_iter().rev() {
            if node.cull(&mut used_buffers, &mut used_textures) {
                pruned_nodes.push(node);
            }
        }

        for mut node in pruned_nodes.into_iter().rev() {
            match node.requested_encoder() {
                RequestedEncoder::RenderPass(pass_targets) => {
                    let resources = RenderPassResources {
                        resources: self.renderer,
                        output_texture: output.view(),
                    };
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &pass_targets
                            .color
                            .into_iter()
                            .map(|target| {
                                Some(wgpu::RenderPassColorAttachment {
                                    view: resources.get_texture(target.handle),
                                    resolve_target: target
                                        .resolve
                                        .map(|handle| resources.get_texture(handle)),
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(target.clear),
                                        store: true,
                                    },
                                })
                            })
                            .collect::<Vec<_>>(),
                        depth_stencil_attachment: pass_targets.depth.map(|target| {
                            wgpu::RenderPassDepthStencilAttachment {
                                view: resources.get_texture(target.handle),
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
                    node.exec(EncoderOrPass::RenderPass(&mut pass), &resources);
                }
                RequestedEncoder::ComputePass => {
                    let resources = RenderPassResources {
                        resources: self.renderer,
                        output_texture: output.view(),
                    };
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None
                    });
                    node.exec(EncoderOrPass::ComputePass(&mut pass), &resources);
                }
                RequestedEncoder::Encoder => {
                    let resources = RenderPassResources {
                        resources: self.renderer,
                        output_texture: output.view(),
                    };
                    node.exec(EncoderOrPass::Encoder(&mut encoder), &resources);
                }
            }
        }

        self.renderer.queue.submit([encoder.finish()]);
        output.present();

        self.renderer.temp_buffers.retain(|_, &mut id| {
            let pred = self.intermediate_buffers.contains(&BufferHandle { id });
            if !pred {
                debug_assert!(self.renderer.buffers.remove(&id).is_some());
            }
            pred
        });
        self.renderer.temp_textures.retain(|_, &mut id| {
            let pred = self.intermediate_textures.contains(&TextureHandle { id });
            if !pred {
                self.renderer.textures.remove(&id);
            }
            pred
        });
    }
}

pub struct NodeBuilder<'a, 'r, 'node, B, T, G, P> {
    graph: &'a mut RenderGraph<'r, 'node>,
    rw_buffers: Vec<BufferHandle>,
    rw_textures: Vec<TextureHandle>,
    no_cull: bool,
    buffers: B,
    textures: T,
    bind_groups: G,
    passthrough: P,
}

impl<'a, 'r, 'node, B, T, G, P> NodeBuilder<'a, 'r, 'node, B, T, G, P>
where
    B: ResourceArray<BufferHandle> + 'node,
    T: ResourceArray<TextureHandle> + 'node,
    G: ResourceArray<BindGroupHandle> + 'node,
    P: RenderPassthrough + 'node,
{
    pub fn with_buffer(
        self,
        handle: BufferHandle,
    ) -> NodeBuilder<'a, 'r, 'node, <B as ResourceArray<BufferHandle>>::ExtendOne, T, G, P> {
        NodeBuilder {
            graph: self.graph,
            rw_buffers: self.rw_buffers,
            rw_textures: self.rw_textures,
            no_cull: self.no_cull,
            buffers: self.buffers.extend_one(handle),
            textures: self.textures,
            bind_groups: self.bind_groups,
            passthrough: self.passthrough,
        }
    }

    pub fn with_buffer_rw(
        mut self,
        handle: BufferHandle,
    ) -> NodeBuilder<'a, 'r, 'node, <B as ResourceArray<BufferHandle>>::ExtendOne, T, G, P> {
        self.rw_buffers.push(handle);
        NodeBuilder {
            graph: self.graph,
            rw_buffers: self.rw_buffers,
            rw_textures: self.rw_textures,
            no_cull: self.no_cull,
            buffers: self.buffers.extend_one(handle),
            textures: self.textures,
            bind_groups: self.bind_groups,
            passthrough: self.passthrough,
        }
    }

    pub fn with_texture(
        self,
        handle: TextureHandle,
    ) -> NodeBuilder<'a, 'r, 'node, B, <T as ResourceArray<TextureHandle>>::ExtendOne, G, P> {
        NodeBuilder {
            graph: self.graph,
            rw_buffers: self.rw_buffers,
            rw_textures: self.rw_textures,
            no_cull: self.no_cull,
            buffers: self.buffers,
            textures: self.textures.extend_one(handle),
            bind_groups: self.bind_groups,
            passthrough: self.passthrough,
        }
    }

    pub fn with_texture_rw(
        mut self,
        handle: TextureHandle,
    ) -> NodeBuilder<'a, 'r, 'node, B, <T as ResourceArray<TextureHandle>>::ExtendOne, G, P> {
        self.rw_textures.push(handle);
        NodeBuilder {
            graph: self.graph,
            rw_buffers: self.rw_buffers,
            rw_textures: self.rw_textures,
            no_cull: self.no_cull,
            buffers: self.buffers,
            textures: self.textures.extend_one(handle),
            bind_groups: self.bind_groups,
            passthrough: self.passthrough,
        }
    }

    pub fn with_bind_group(
        self,
        layout: &wgpu::BindGroupLayout,
        resources: impl IntoBindGroupResources,
    ) -> NodeBuilder<'a, 'r, 'node, B, T, <G as ResourceArray<BindGroupHandle>>::ExtendOne, P> {
        let handle = self.graph.renderer.add_bind_group(layout, resources);
        NodeBuilder {
            graph: self.graph,
            rw_buffers: self.rw_buffers,
            rw_textures: self.rw_textures,
            no_cull: self.no_cull,
            buffers: self.buffers,
            textures: self.textures,
            bind_groups: self.bind_groups.extend_one(handle),
            passthrough: self.passthrough,
        }
    }

    pub fn with_bind_group_rw(
        mut self,
        layout: &wgpu::BindGroupLayout,
        resources: impl IntoBindGroupResources,
    ) -> NodeBuilder<'a, 'r, 'node, B, T, <G as ResourceArray<BindGroupHandle>>::ExtendOne, P> {
        let resources = resources.into_entries();
        for (_, entry) in &resources {
            match entry {
                crate::BindGroupResource::Buffer(binding) => self.rw_buffers.push(binding.buffer),
                crate::BindGroupResource::BufferArray(handles) => {
                    self.rw_buffers
                        .extend(handles.iter().map(|binding| binding.buffer));
                }
                crate::BindGroupResource::Texture(handle) => self.rw_textures.push(*handle),
                crate::BindGroupResource::TextureArray(handles) => {
                    self.rw_textures.extend_from_slice(handles);
                }
                crate::BindGroupResource::Sampler(_)
                | crate::BindGroupResource::SamplerArray(_) => {}
            }
        }
        let handle = self.graph.renderer.add_bind_group(layout, resources);
        NodeBuilder {
            graph: self.graph,
            rw_buffers: self.rw_buffers,
            rw_textures: self.rw_textures,
            no_cull: self.no_cull,
            buffers: self.buffers,
            textures: self.textures,
            bind_groups: self.bind_groups.extend_one(handle),
            passthrough: self.passthrough,
        }
    }

    pub fn with_passthrough<A: 'node>(
        self,
        item: &'node A,
    ) -> NodeBuilder<'a, 'r, 'node, B, T, G, <P as ExtendTuple<*const A>>::ExtendOne>
    where
        P: ExtendTuple<*const A>,
    {
        NodeBuilder {
            graph: self.graph,
            rw_buffers: self.rw_buffers,
            rw_textures: self.rw_textures,
            no_cull: self.no_cull,
            buffers: self.buffers,
            textures: self.textures,
            bind_groups: self.bind_groups,
            passthrough: self.passthrough.extend_one(item as *const A),
        }
    }

    pub fn with_external_output(mut self) -> Self {
        self.no_cull = true;
        self
    }

    pub fn build_with_encoder<F>(self, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::CommandEncoder,
                <B as ResourceArray<BufferHandle>>::Fetched<'pass>,
                <T as ResourceArray<TextureHandle>>::Fetched<'pass>,
                <G as ResourceArray<BindGroupHandle>>::Fetched<'pass>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    {
        self.graph.nodes.push(Box::new(GraphNode {
            inner: Some(GraphNodeInner {
                rw_buffers: self.rw_buffers,
                rw_textures: self.rw_textures,
                no_cull: self.no_cull,
                buffers: self.buffers,
                textures: self.textures,
                passthrough: self.passthrough,
                bind_groups: self.bind_groups,
                requested_encoder: Some(RequestedEncoder::Encoder),
                exec: Box::new(
                    move |encoder, buffers, textures, bind_groups, passthrough| {
                        (exec)(
                            encoder.unwrap_encoder(),
                            buffers,
                            textures,
                            bind_groups,
                            passthrough,
                        )
                    },
                ),
            }),
        }));
    }

    pub fn build_render_pass<F>(mut self, render_pass: RenderPassTarget, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::RenderPass<'pass>,
                <B as ResourceArray<BufferHandle>>::Fetched<'pass>,
                <T as ResourceArray<TextureHandle>>::Fetched<'pass>,
                <G as ResourceArray<BindGroupHandle>>::Fetched<'pass>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    {
        for target in &render_pass.color {
            self.rw_textures.push(target.handle);
            self.rw_textures.extend(&target.resolve);
        }
        if let Some(depth_target) = &render_pass.depth {
            self.rw_textures.push(depth_target.handle);
        }
        self.graph.nodes.push(Box::new(GraphNode {
            inner: Some(GraphNodeInner {
                rw_buffers: self.rw_buffers,
                rw_textures: self.rw_textures,
                no_cull: self.no_cull,
                buffers: self.buffers,
                textures: self.textures,
                passthrough: self.passthrough,
                bind_groups: self.bind_groups,
                requested_encoder: Some(RequestedEncoder::RenderPass(render_pass)),
                exec: Box::new(move |pass, buffers, textures, bind_groups, passthrough| {
                    (exec)(
                        pass.unwrap_render_pass(),
                        buffers,
                        textures,
                        bind_groups,
                        passthrough,
                    )
                }),
            }),
        }))
    }

    pub fn build_compute_pass<F>(self, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::ComputePass<'pass>,
                <B as ResourceArray<BufferHandle>>::Fetched<'pass>,
                <T as ResourceArray<TextureHandle>>::Fetched<'pass>,
                <G as ResourceArray<BindGroupHandle>>::Fetched<'pass>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    {
        self.graph.nodes.push(Box::new(GraphNode {
            inner: Some(GraphNodeInner {
                rw_buffers: self.rw_buffers,
                rw_textures: self.rw_textures,
                no_cull: self.no_cull,
                buffers: self.buffers,
                textures: self.textures,
                passthrough: self.passthrough,
                bind_groups: self.bind_groups,
                requested_encoder: Some(RequestedEncoder::ComputePass),
                exec: Box::new(move |pass, buffers, textures, bind_groups, passthrough| {
                    (exec)(
                        pass.unwrap_compute_pass(),
                        buffers,
                        textures,
                        bind_groups,
                        passthrough,
                    )
                }),
            }),
        }))
    }
}

enum RequestedEncoder {
    Encoder,
    RenderPass(RenderPassTarget),
    ComputePass,
}

enum EncoderOrPass<'b, 'pass> {
    Encoder(&'b mut wgpu::CommandEncoder),
    RenderPass(&'b mut wgpu::RenderPass<'pass>),
    ComputePass(&'b mut wgpu::ComputePass<'pass>),
}

impl<'b, 'pass> EncoderOrPass<'b, 'pass> {
    fn unwrap_encoder(self) -> &'b mut wgpu::CommandEncoder {
        match self {
            Self::Encoder(encoder) => encoder,
            _ => unreachable!(),
        }
    }

    fn unwrap_render_pass(self) -> &'b mut wgpu::RenderPass<'pass> {
        match self {
            Self::RenderPass(pass) => pass,
            _ => unreachable!(),
        }
    }

    fn unwrap_compute_pass(self) -> &'b mut wgpu::ComputePass<'pass> {
        match self {
            Self::ComputePass(pass) => pass,
            _ => unreachable!(),
        }
    }
}

struct GraphNode<
    'node,
    B: ResourceArray<BufferHandle>,
    T: ResourceArray<TextureHandle>,
    G: ResourceArray<BindGroupHandle>,
    P: RenderPassthrough,
> {
    //TODO: This could be made a MaybeUninit, if absolutely necessary. It probably isn't necessary.
    inner: Option<GraphNodeInner<'node, B, T, G, P>>,
}

#[allow(clippy::type_complexity)]
struct GraphNodeInner<
    'node,
    B: ResourceArray<BufferHandle>,
    T: ResourceArray<TextureHandle>,
    G: ResourceArray<BindGroupHandle>,
    P: RenderPassthrough,
> {
    rw_buffers: Vec<BufferHandle>,
    rw_textures: Vec<TextureHandle>,
    no_cull: bool,
    buffers: B,
    textures: T,
    bind_groups: G,
    passthrough: P,
    requested_encoder: Option<RequestedEncoder>,
    exec: Box<
        dyn for<'b, 'pass> FnOnce(
                EncoderOrPass<'b, 'pass>,
                <B as ResourceArray<BufferHandle>>::Fetched<'pass>,
                <T as ResourceArray<TextureHandle>>::Fetched<'pass>,
                <G as ResourceArray<BindGroupHandle>>::Fetched<'pass>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    >,
}

trait GraphNodeImpl<'node> {
    fn requested_encoder(&mut self) -> RequestedEncoder;
    fn cull(
        &self,
        used_buffers: &mut FxHashSet<BufferHandle>,
        used_textures: &mut FxHashSet<TextureHandle>,
    ) -> bool;
    fn exec<'pass>(
        &mut self,
        encoder_or_pass: EncoderOrPass<'_, 'pass>,
        resources: &'pass RenderPassResources,
    ) where
        'node: 'pass;
}

impl<
        'node,
        B: ResourceArray<BufferHandle>,
        T: ResourceArray<TextureHandle>,
        G: ResourceArray<BindGroupHandle>,
        P: RenderPassthrough + 'node,
    > GraphNodeImpl<'node> for GraphNode<'node, B, T, G, P>
{
    fn requested_encoder(&mut self) -> RequestedEncoder {
        self.inner.as_mut().unwrap().requested_encoder.take().unwrap()
    }

    fn cull(
        &self,
        used_buffers: &mut FxHashSet<BufferHandle>,
        used_textures: &mut FxHashSet<TextureHandle>,
    ) -> bool {
        let inner = self.inner.as_ref().unwrap();
        if inner.no_cull
            || inner
                .rw_buffers
                .iter()
                .any(|buf| used_buffers.contains(buf))
            || inner.rw_textures.iter().any(|t| used_textures.contains(t))
        {
            used_buffers.extend(inner.buffers.as_slice().iter().cloned());
            used_textures.extend(inner.textures.as_slice().iter().cloned());
            true
        } else {
            false
        }
    }

    fn exec<'pass>(
        &mut self,
        encoder_or_pass: EncoderOrPass<'_, 'pass>,
        renderer: &'pass RenderPassResources,
    ) where
        'node: 'pass,
    {
        let inner = self.inner.take().unwrap();
        (inner.exec)(
            encoder_or_pass,
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

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

pub trait ExtendTuple<T>: Sealed {
    type ExtendOne;

    fn extend_one(self, _: T) -> Self::ExtendOne;
}

pub trait RenderPassthrough: Sealed {
    type Reborrowed<'a>
    where
        Self: 'a;

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

pub trait RenderResource {
    type Resource<'a>;

    fn fetch_resource<'a>(self, resources: &'a RenderPassResources) -> Self::Resource<'a>;
}

impl RenderResource for BufferHandle {
    type Resource<'a> = &'a wgpu::Buffer;

    fn fetch_resource<'a>(self, resources: &'a RenderPassResources) -> Self::Resource<'a> {
        resources.get_buffer(self)
    }
}

impl RenderResource for TextureHandle {
    type Resource<'a> = &'a wgpu::TextureView;

    fn fetch_resource<'a>(self, resources: &'a RenderPassResources) -> Self::Resource<'a> {
        resources.get_texture(self)
    }
}

impl RenderResource for BindGroupHandle {
    type Resource<'a> = &'a wgpu::BindGroup;

    fn fetch_resource<'a>(self, resources: &'a RenderPassResources) -> Self::Resource<'a> {
        resources.get_bind_group(self)
    }
}

pub trait ResourceArray<T: RenderResource>: Sealed {
    type Fetched<'a>;
    type ExtendOne;

    fn as_slice(&self) -> &[T];
    fn fetch_resources<'a>(self, resources: &'a RenderPassResources) -> Self::Fetched<'a>;
    fn extend_one(self, _: T) -> Self::ExtendOne;
}

impl<T> Sealed for [T; 0] {}

impl<T: RenderResource> ResourceArray<T> for [T; 0] {
    type Fetched<'a> = [<T as RenderResource>::Resource<'a>; 0];
    type ExtendOne = [T; 1];

    fn as_slice(&self) -> &[T] {
        self
    }

    fn fetch_resources<'a>(self, _resources: &'a RenderPassResources) -> Self::Fetched<'a> {
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

            fn fetch_resources<'a>(self, resources: &'a RenderPassResources) -> Self::Fetched<'a> {
                self.map(|element| <T as RenderResource>::fetch_resource(element, resources))
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

            fn fetch_resources<'a>(self, resources: &'a RenderPassResources) -> Self::Fetched<'a> {
                self.map(|element| <T as RenderResource>::fetch_resource(element, resources))
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
