use either::{Either, Left, Right};
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
    current_buffers: FxHashSet<usize>,
    current_textures: FxHashSet<usize>,
    nodes: Vec<Box<dyn GraphNodeImpl<'node> + 'node>>,
}

impl<'r, 'node> RenderGraph<'r, 'node> {
    pub fn new(renderer: &'r mut Renderer) -> RenderGraph<'r, 'node> {
        RenderGraph {
            renderer,
            current_buffers: FxHashSet::default(),
            current_textures: FxHashSet::default(),
            nodes: Vec::new(),
        }
    }

    pub fn add_intermediate_buffer(&mut self, descriptor: wgpu::BufferDescriptor) -> BufferHandle {
        let mut hash_descriptor = descriptor.map_label(fxhash::hash64);
        let id = loop {
            match self.renderer.temp_buffers.get(&hash_descriptor) {
                Some(&id) => {
                    if self.current_buffers.insert(id) {
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
                    if self.current_textures.insert(id) {
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
                self.renderer.temp_textures.insert(hash_descriptor, id);
                TextureHandle { id }
            }
        }
    }

    pub fn add_bind_group(
        &mut self,
        layout: &wgpu::BindGroupLayout,
        resources: impl IntoBindGroupResources,
    ) -> BindGroupHandle {
        self.renderer.add_bind_group(layout, resources)
    }

    pub fn add_node<'a>(
        &'a mut self,
    ) -> NodeBuilder<'a, 'r, 'node, [BufferHandle; 0], [TextureHandle; 0], [BindGroupHandle; 0], ()> {
        NodeBuilder {
            graph: self,
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

        for mut node in self.nodes.into_iter() {
            match node.render_pass() {
                Some(pass_targets) => {
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
                    node.exec(Right(&mut pass), &resources);
                }
                None => {
                    let resources = RenderPassResources {
                        resources: self.renderer,
                        output_texture: output.view(),
                    };
                    node.exec(Left(&mut encoder), &resources);
                }
            }
        }

        self.renderer.queue.submit([encoder.finish()]);
        output.present();
    }
}

pub struct NodeBuilder<'a, 'r, 'node, B, T, G, P> {
    graph: &'a mut RenderGraph<'r, 'node>,
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
            buffers: self.buffers,
            textures: self.textures.extend_one(handle),
            bind_groups: self.bind_groups,
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
            buffers: self.buffers,
            textures: self.textures,
            bind_groups: self.bind_groups,
            passthrough: self.passthrough.extend_one(item as *const A),
        }
    }

    pub fn build_with_encoder<F>(self, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::CommandEncoder,
                <B as ResourceArray<BufferHandle>>::Fetched<'b>,
                <T as ResourceArray<TextureHandle>>::Fetched<'b>,
                <G as ResourceArray<BindGroupHandle>>::Fetched<'b>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    {
        self.graph.nodes.push(Box::new(GraphNode {
            inner: Some(GraphNodeInner {
                buffers: self.buffers,
                textures: self.textures,
                passthrough: self.passthrough,
                bind_groups: self.bind_groups,
                render_pass: None,
                exec: Box::new(move |encoder, buffers, textures, bind_groups, passthrough| {
                    (exec)(encoder.unwrap_left(), buffers, textures, bind_groups, passthrough)
                }),
            }),
        }));
    }

    pub fn build_renderpass<F>(self, render_pass: RenderPassTarget, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::RenderPass<'pass>,
                <B as ResourceArray<BufferHandle>>::Fetched<'b>,
                <T as ResourceArray<TextureHandle>>::Fetched<'b>,
                <G as ResourceArray<BindGroupHandle>>::Fetched<'b>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    {
        self.graph.nodes.push(Box::new(GraphNode {
            inner: Some(GraphNodeInner {
                buffers: self.buffers,
                textures: self.textures,
                passthrough: self.passthrough,
                bind_groups: self.bind_groups,
                render_pass: Some(render_pass),
                exec: Box::new(move |pass, buffers, textures, bind_groups, passthrough| {
                    (exec)(pass.unwrap_right(), buffers, textures, bind_groups, passthrough)
                }),
            }),
        }))
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
    buffers: B,
    textures: T,
    bind_groups: G,
    passthrough: P,
    render_pass: Option<RenderPassTarget>,
    exec: Box<
        dyn for<'b, 'pass> FnOnce(
                Either<&'b mut wgpu::CommandEncoder, &'b mut wgpu::RenderPass<'pass>>,
                <B as ResourceArray<BufferHandle>>::Fetched<'b>,
                <T as ResourceArray<TextureHandle>>::Fetched<'b>,
                <G as ResourceArray<BindGroupHandle>>::Fetched<'b>,
                <P as RenderPassthrough>::Reborrowed<'pass>,
            ) + 'node,
    >,
}

trait GraphNodeImpl<'node> {
    fn render_pass(&mut self) -> Option<RenderPassTarget>;
    fn exec<'pass>(
        &mut self,
        encoder_or_pass: Either<&mut wgpu::CommandEncoder, &mut wgpu::RenderPass<'pass>>,
        resources: &RenderPassResources,
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
    fn render_pass(&mut self) -> Option<RenderPassTarget> {
        self.inner.as_mut().unwrap().render_pass.take()
    }

    fn exec<'pass>(
        &mut self,
        encoder_or_pass: Either<&mut wgpu::CommandEncoder, &mut wgpu::RenderPass<'pass>>,
        renderer: &RenderPassResources,
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
                // * 'pass is shorter than 'node
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

    fn fetch_resources<'a>(self, resources: &'a RenderPassResources) -> Self::Fetched<'a>;
    fn extend_one(self, _: T) -> Self::ExtendOne;
}

impl<T> Sealed for [T; 0] {}

impl<T: RenderResource> ResourceArray<T> for [T; 0] {
    type Fetched<'a> = [<T as RenderResource>::Resource<'a>; 0];
    type ExtendOne = [T; 1];

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
