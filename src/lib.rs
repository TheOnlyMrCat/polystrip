#![feature(generic_associated_types)]
// Can also experiment with #![feature(generic_const_exprs)]

pub mod math;

pub use wgpu;

use std::{hash::Hash, marker::PhantomData};
use std::num::NonZeroU64;
use std::sync::Arc;

use either::{Either, Left, Right};
use fxhash::{FxHashMap, FxHashSet};

pub struct PolystripDevice {
    pub instance: Arc<wgpu::Instance>,
    pub adapter: Arc<wgpu::Adapter>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl PolystripDevice {
    /// Create a `PolystripDevice` with default settings read from the environment.
    ///
    /// This method uses the initialisation methods in [`wgpu::util`], and therefore reads the same environment variables they
    /// do. In particular:
    ///
    /// - `WGPU_BACKEND`: A comma-separated list of backends for wgpu to use
    /// - `WGPU_ADAPTER_NAME`: A substring of the name of an adapter for wgpu to use.
    /// - `WGPU_POWER_PREF`: If `WGPU_ADAPTER_NAME` is unset, "low" or "high" power preference for a default adapter
    ///
    /// If these are unset, [`wgpu::Backends::PRIMARY`] and [`wgpu::PowerPreference::LowPower`] will be chosen.
    ///
    /// No extra features or limits are requested for the `wgpu::Device`.
    ///
    /// # Panics
    /// Panics if any part of initialisation fails, which can happen for any of the following
    /// reasons:
    ///
    /// - Environment variables are set incorrectly, or `WGPU_ADAPTER_NAME` doesn't refer to a
    /// valid adapter
    /// - Adapter does not support all features required for wgpu or polystrip to operate
    pub async fn new_from_env() -> PolystripDevice {
        let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let instance = wgpu::Instance::new(backends);
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, backends, None)
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Polystrip"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();
        PolystripDevice {
            instance: instance.into(),
            adapter: adapter.into(),
            device: device.into(),
            queue: queue.into(),
        }
    }

    /// Create a [`WindowTarget`] and a `PolystripDevice` compatible with that target.
    ///
    /// See [`Self::new_from_env`] and [`WindowTarget::new`] for more details
    ///
    /// # Panics
    /// Panics on macOS if called from a thread other than the main thread
    ///
    /// # Safety
    /// `handle` must be a valid object to create a [`wgpu::Surface`] upon and must remain valid for the lifetime of the
    /// returned `WindowTarget`
    pub async unsafe fn new_from_env_with_window(
        handle: &impl raw_window_handle::HasRawWindowHandle,
        (width, height): (u32, u32),
        present_mode: wgpu::PresentMode,
    ) -> (PolystripDevice, WindowTarget) {
        let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let instance = wgpu::Instance::new(backends);
        let surface = instance.create_surface(&handle);
        let adapter =
            wgpu::util::initialize_adapter_from_env_or_default(&instance, backends, Some(&surface))
                .await
                .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Polystrip"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();
        let device = Arc::new(device);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width,
            height,
            present_mode,
        };
        surface.configure(&device, &config);
        let target = WindowTarget::from_wgpu(device.clone(), surface, config);
        (
            PolystripDevice {
                instance: instance.into(),
                adapter: adapter.into(),
                device,
                queue: queue.into(),
            },
            target,
        )
    }

    pub fn create_renderer(&self) -> Renderer {
        Renderer::new(self.device.clone(), self.queue.clone())
    }
}

pub enum OutputTexture {
    Surface {
        texture: wgpu::SurfaceTexture,
        view: wgpu::TextureView,
    },
    View(wgpu::TextureView),
}

impl OutputTexture {
    pub fn view(&self) -> &wgpu::TextureView {
        match self {
            Self::Surface { view, .. } => view,
            Self::View(view) => view,
        }
    }

    pub fn present(self) {
        if let Self::Surface { texture, .. } = self {
            texture.present()
        }
    }
}

pub trait RenderTarget {
    fn get_current_view(&mut self) -> OutputTexture;
}

pub struct WindowTarget {
    device: Arc<wgpu::Device>,
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
}

impl WindowTarget {
    /// Create a `WindowTarget` from a window, given its size.
    ///
    /// # Safety
    /// `handle` must be a valid object to create a [`wgpu::Surface`] upon and must remain valid for the lifetime of the
    /// returned `WindowTarget`
    pub unsafe fn new(
        device: &PolystripDevice,
        handle: &impl raw_window_handle::HasRawWindowHandle,
        (width, height): (u32, u32),
        present_mode: wgpu::PresentMode,
    ) -> WindowTarget {
        let surface = device.instance.create_surface(handle);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width,
            height,
            present_mode,
        };
        Self::from_wgpu(device.device.clone(), surface, config)
    }

    /// Create a `WindowTarget` from a pre-initialised [`wgpu::Surface`].
    ///
    /// `surface` should already be configured with the passed `config`.
    pub fn from_wgpu(
        device: Arc<wgpu::Device>,
        surface: wgpu::Surface,
        config: wgpu::SurfaceConfiguration,
    ) -> WindowTarget {
        WindowTarget {
            device,
            surface,
            config,
        }
    }

    /// Resize and reconfigure the surface.
    ///
    /// This should be called in your window's event loop every time the window is resized.
    pub fn resize(&mut self, (width, height): (u32, u32)) {
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Change the present mode and reconfigure the surface.
    ///
    /// This can be used to implement an interactive VSync option.
    pub fn set_present_mode(&mut self, present_mode: wgpu::PresentMode) {
        self.config.present_mode = present_mode;
        self.surface.configure(&self.device, &self.config);
    }
}

impl RenderTarget for WindowTarget {
    fn get_current_view(&mut self) -> OutputTexture {
        let texture = self.surface.get_current_texture().unwrap();
        let view = texture.texture.create_view(&Default::default());
        OutputTexture::Surface { texture, view }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle {
    id: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle {
    id: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerHandle {
    id: usize,
}

pub struct BindGroupHandle {
    id: usize,
}

impl TextureHandle {
    pub const RENDER_TARGET: TextureHandle = TextureHandle { id: usize::MAX };
}

#[derive(Default)]
pub struct RenderPassTarget {
    pub color: Vec<RenderPassColorTarget>,
    pub depth: Option<RenderPassDepthTarget>,
}

pub struct RenderPassColorTarget {
    pub handle: TextureHandle,
    pub resolve: Option<TextureHandle>,
    pub clear: wgpu::Color,
}

pub struct RenderPassDepthTarget {
    pub handle: TextureHandle,
    pub depth_clear: Option<f32>,
    pub stencil_clear: Option<u32>,
}

impl RenderPassTarget {
    pub fn new() -> RenderPassTarget {
        Self {
            color: vec![],
            depth: None,
        }
    }

    pub fn with_color(mut self, handle: TextureHandle, clear: wgpu::Color) -> Self {
        self.color.push(RenderPassColorTarget {
            handle,
            resolve: None,
            clear,
        });
        self
    }

    pub fn with_msaa_color(
        mut self,
        handle: TextureHandle,
        resolve: TextureHandle,
        clear: wgpu::Color,
    ) -> Self {
        self.color.push(RenderPassColorTarget {
            handle,
            resolve: Some(resolve),
            clear,
        });
        self
    }

    pub fn with_depth(mut self, handle: TextureHandle, clear: f32) -> Self {
        self.depth = Some(RenderPassDepthTarget {
            handle,
            depth_clear: Some(clear),
            stencil_clear: None,
        });
        self
    }
}

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: FxHashMap<usize, wgpu::Buffer>,
    temp_buffers: FxHashMap<wgpu_types::BufferDescriptor<u64>, usize>,
    textures: FxHashMap<usize, (wgpu::Texture, wgpu::TextureView)>,
    temp_textures: FxHashMap<wgpu_types::TextureDescriptor<u64>, usize>,
    samplers: FxHashMap<usize, wgpu::Sampler>,
    bind_group_descriptors: FxHashMap<Vec<(u32, BindGroupResource)>, usize>,
    bind_groups: FxHashMap<usize, (wgpu::BindGroup, Vec<(u32, BindGroupResource)>)>,
    next_buffer: usize,
    next_texture: usize,
    next_sampler: usize,
    next_bind_group: usize,
}

impl Renderer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Renderer {
        Self {
            device,
            queue,
            buffers: FxHashMap::default(),
            temp_buffers: FxHashMap::default(),
            textures: FxHashMap::default(),
            temp_textures: FxHashMap::default(),
            samplers: FxHashMap::default(),
            bind_group_descriptors: FxHashMap::default(),
            bind_groups: FxHashMap::default(),
            next_buffer: 0,
            next_texture: 0,
            next_sampler: 0,
            next_bind_group: 0,
        }
    }

    pub fn insert_buffer(&mut self, buffer: wgpu::Buffer) -> BufferHandle {
        let id = self.next_buffer_id();
        self.buffers.insert(id, buffer);
        BufferHandle { id }
    }

    pub fn insert_texture(
        &mut self,
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    ) -> TextureHandle {
        let id = self.next_texture_id();
        self.textures.insert(id, (texture, view));
        TextureHandle { id }
    }

    pub fn insert_sampler(&mut self, sampler: wgpu::Sampler) -> SamplerHandle {
        let id = self.next_sampler_id();
        self.samplers.insert(id, sampler);
        SamplerHandle { id }
    }

    pub fn add_bind_group(
        &mut self,
        layout: &wgpu::BindGroupLayout,
        resources: impl IntoBindGroupResources,
    ) -> BindGroupHandle {
        let key = resources.into_entries();
        match self.bind_group_descriptors.get(&key) {
            Some(&id) => BindGroupHandle { id },
            None => {
                let id = self.insert_bind_group(layout, key.clone());
                self.bind_group_descriptors.insert(key, id);
                BindGroupHandle { id }
            }
        }
    }

    pub fn get_buffer(&self, handle: BufferHandle) -> &wgpu::Buffer {
        self.buffers.get(&handle.id).unwrap()
    }

    pub fn get_texture(&self, handle: TextureHandle) -> (&wgpu::Texture, &wgpu::TextureView) {
        let (texture, view) = self.textures.get(&handle.id).unwrap();
        (texture, view)
    }

    pub fn get_sampler(&self, handle: SamplerHandle) -> &wgpu::Sampler {
        self.samplers.get(&handle.id).unwrap()
    }

    pub fn get_bind_group(&self, handle: BindGroupHandle) -> &wgpu::BindGroup {
        let (bind_group, _) = self.bind_groups.get(&handle.id).unwrap();
        bind_group
    }
}

impl Renderer {
    fn next_buffer_id(&mut self) -> usize {
        let id = self.next_buffer;
        self.next_buffer += 1;
        id
    }

    fn next_texture_id(&mut self) -> usize {
        let id = self.next_texture;
        self.next_texture += 1;
        id
    }

    fn next_sampler_id(&mut self) -> usize {
        let id = self.next_sampler;
        self.next_sampler += 1;
        id
    }

    fn insert_bind_group(
        &mut self,
        layout: &wgpu::BindGroupLayout,
        resources: BindGroupResources,
    ) -> usize {
        enum Mapped<'a> {
            Buffer(wgpu::BufferBinding<'a>),
            BufferArray(usize),
            Texture(&'a wgpu::TextureView),
            TextureArray(usize),
            Sampler(&'a wgpu::Sampler),
            SamplerArray(usize),
        }

        let mut buffer_arrays = vec![];
        let mut texture_arrays = vec![];
        let mut sampler_arrays = vec![];

        // This must be done in two passes: The first pass allocates all the resources
        // and the second pass gathers the references to all of the resources.
        //
        // This is only necessary to handle array resources. Perhaps a fast-track option could be
        // used when we're guaranteed not to have arrays?
        #[allow(clippy::needless_collect)]
        let mapped_resources = resources
            .iter()
            .map(|(binding, resource)| {
                (
                    binding,
                    match resource {
                        BindGroupResource::Buffer(binding) => Mapped::Buffer(wgpu::BufferBinding {
                            buffer: self.get_buffer(binding.buffer),
                            offset: binding.offset,
                            size: binding.size,
                        }),
                        BindGroupResource::BufferArray(bindings) => {
                            let array = bindings
                                .iter()
                                .map(|binding| wgpu::BufferBinding {
                                    buffer: self.get_buffer(binding.buffer),
                                    offset: binding.offset,
                                    size: binding.size,
                                })
                                .collect::<Vec<_>>();
                            buffer_arrays.push(array);
                            Mapped::BufferArray(buffer_arrays.len() - 1)
                        }
                        BindGroupResource::Texture(handle) => {
                            Mapped::Texture(self.get_texture(*handle).1)
                        }
                        BindGroupResource::TextureArray(handles) => {
                            let array = handles
                                .iter()
                                .map(|&handle| self.get_texture(handle).1)
                                .collect::<Vec<_>>();
                            texture_arrays.push(array);
                            Mapped::TextureArray(texture_arrays.len() - 1)
                        }
                        BindGroupResource::Sampler(handle) => {
                            Mapped::Sampler(self.get_sampler(*handle))
                        }
                        BindGroupResource::SamplerArray(handles) => {
                            let array = handles
                                .iter()
                                .map(|&handle| self.get_sampler(handle))
                                .collect::<Vec<_>>();
                            sampler_arrays.push(array);
                            Mapped::SamplerArray(sampler_arrays.len() - 1)
                        }
                    },
                )
            })
            .collect::<Vec<_>>();

        let bind_group_entries = mapped_resources
            .into_iter()
            .map(|(&binding, resource)| wgpu::BindGroupEntry {
                binding,
                resource: match resource {
                    Mapped::Buffer(binding) => wgpu::BindingResource::Buffer(binding),
                    Mapped::Texture(view) => wgpu::BindingResource::TextureView(view),
                    Mapped::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                    Mapped::BufferArray(index) => {
                        wgpu::BindingResource::BufferArray(&buffer_arrays[index])
                    }
                    Mapped::TextureArray(index) => {
                        wgpu::BindingResource::TextureViewArray(&texture_arrays[index])
                    }
                    Mapped::SamplerArray(index) => {
                        wgpu::BindingResource::SamplerArray(&sampler_arrays[index])
                    }
                },
            })
            .collect::<Vec<_>>();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &bind_group_entries,
        });

        let id = self.next_bind_group;
        self.next_bind_group += 1;
        self.bind_groups.insert(id, (bind_group, resources));
        id
    }
}

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

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BufferBinding {
    pub buffer: BufferHandle,
    pub offset: u64,
    pub size: Option<NonZeroU64>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum BindGroupResource {
    Buffer(BufferBinding),
    BufferArray(Vec<BufferBinding>),
    Texture(TextureHandle),
    TextureArray(Vec<TextureHandle>),
    Sampler(SamplerHandle),
    SamplerArray(Vec<SamplerHandle>),
}

type BindGroupResources = Vec<(u32, BindGroupResource)>;

pub trait IntoBindGroupResource {
    fn into_bind_group_resource(self) -> BindGroupResource;
}

impl IntoBindGroupResource for BindGroupResource {
    fn into_bind_group_resource(self) -> BindGroupResource {
        self
    }
}

impl IntoBindGroupResource for BufferHandle {
    fn into_bind_group_resource(self) -> BindGroupResource {
        BindGroupResource::Buffer(BufferBinding {
            buffer: self,
            offset: 0,
            size: None,
        })
    }
}

impl IntoBindGroupResource for TextureHandle {
    fn into_bind_group_resource(self) -> BindGroupResource {
        BindGroupResource::Texture(self)
    }
}

impl IntoBindGroupResource for SamplerHandle {
    fn into_bind_group_resource(self) -> BindGroupResource {
        BindGroupResource::Sampler(self)
    }
}

pub trait IntoBindGroupResources {
    fn into_entries(self) -> BindGroupResources;
}

impl IntoBindGroupResources for BindGroupResources {
    fn into_entries(self) -> BindGroupResources {
        self
    }
}

macro_rules! create_indexed_tuples {
    ($base:expr, $head:ident,) => {
        let $head = ($base, <$head as IntoBindGroupResource>::into_bind_group_resource($head));
    };
    ($base:expr, $head:ident, $($tail:ident,)*) => {
        create_indexed_tuples!($base, $head,);
        create_indexed_tuples!($base + 1, $($tail,)*);
    };
}

macro_rules! bind_group_resources_tuple {
    ($head:ident,) => {}; // Stop recursion
    ($head:ident, $($tail:ident,)*) => {
        #[allow(non_snake_case)]
        impl<$($tail,)*> IntoBindGroupResources for ($($tail,)*)
            where $($tail: IntoBindGroupResource,)*
            {
                fn into_entries(self) -> BindGroupResources {
                    let ($($tail,)*) = self;
                    create_indexed_tuples!(0, $($tail,)*);
                    vec![$($tail,)*]
                }
            }
        bind_group_resources_tuple!($($tail,)*);
    }
}
bind_group_resources_tuple!(recursion_dummy, A, B, C, D, E, F, G, H, I, J, K, L,);

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
    ) -> NodeBuilder<'a, 'r, 'node, [BufferHandle; 0], [TextureHandle; 0], ()> {
        NodeBuilder {
            graph: self,
            buffers: [],
            textures: [],
            passthrough: (),
            passthrough_container: PassthroughContainer::new(),
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

pub struct NodeBuilder<'a, 'r, 'node, B, T, P> {
    graph: &'a mut RenderGraph<'r, 'node>,
    buffers: B,
    textures: T,
    passthrough: P,
    passthrough_container: PassthroughContainer<'node>,
}

impl<'a, 'r, 'node, B, T, P> NodeBuilder<'a, 'r, 'node, B, T, P>
where
    B: ResourceArray<BufferHandle> + 'node,
    T: ResourceArray<TextureHandle> + 'node,
    P: 'node,
{
    pub fn with_buffer(
        self,
        handle: BufferHandle,
    ) -> NodeBuilder<'a, 'r, 'node, <B as ResourceArray<BufferHandle>>::ExtendOne, T, P> {
        NodeBuilder {
            graph: self.graph,
            buffers: self.buffers.extend_one(handle),
            textures: self.textures,
            passthrough: self.passthrough,
            passthrough_container: self.passthrough_container,
        }
    }

    pub fn with_texture(
        self,
        handle: TextureHandle,
    ) -> NodeBuilder<'a, 'r, 'node, B, <T as ResourceArray<TextureHandle>>::ExtendOne, P> {
        NodeBuilder {
            graph: self.graph,
            buffers: self.buffers,
            textures: self.textures.extend_one(handle),
            passthrough: self.passthrough,
            passthrough_container: self.passthrough_container,
        }
    }

    pub fn with_passthrough<A: 'node>(
        self,
        item: &'node A,
    ) -> NodeBuilder<'a, 'r, 'node, B, T, <P as ExtendTuple<PassthroughRef<A>>>::ExtendOne>
    where P: ExtendTuple<PassthroughRef<A>>
    {
        let mut container = self.passthrough_container;
        let item = container.insert(item);
        NodeBuilder {
            graph: self.graph,
            buffers: self.buffers,
            textures: self.textures,
            passthrough: self.passthrough.extend_one(item),
            passthrough_container: container,
        }
    }

    pub fn build_with_encoder<F>(self, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::CommandEncoder,
                <B as ResourceArray<BufferHandle>>::Fetched<'b>,
                <T as ResourceArray<TextureHandle>>::Fetched<'b>,
                PassthroughContainer<'pass>,
                P,
            ) + 'node,
    {
        self.graph.nodes.push(Box::new(GraphNode {
            inner: Some(GraphNodeInner {
                buffers: self.buffers,
                textures: self.textures,
                passthrough: self.passthrough,
                passthrough_container: self.passthrough_container,
                render_pass: None,
                exec: Box::new(move |encoder, buffers, textures, container, passthrough| {
                    (exec)(encoder.unwrap_left(), buffers, textures, container, passthrough)
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
                PassthroughContainer<'pass>,
                P,
            ) + 'node,
    {
        self.graph.nodes.push(Box::new(GraphNode {
            inner: Some(GraphNodeInner {
                buffers: self.buffers,
                textures: self.textures,
                passthrough: self.passthrough,
                passthrough_container: self.passthrough_container,
                render_pass: Some(render_pass),
                exec: Box::new(move |pass, buffers, textures, container, passthrough| {
                    (exec)(pass.unwrap_right(), buffers, textures, container, passthrough)
                }),
            }),
        }))
    }
}

struct GraphNode<'node, B: ResourceArray<BufferHandle>, T: ResourceArray<TextureHandle>, P> {
    //TODO: This could be made a MaybeUninit, if absolutely necessary. It probably isn't necessary.
    inner: Option<GraphNodeInner<'node, B, T, P>>,
}

#[allow(clippy::type_complexity)]
struct GraphNodeInner<'node, B: ResourceArray<BufferHandle>, T: ResourceArray<TextureHandle>, P> {
    buffers: B,
    textures: T,
    passthrough: P,
    passthrough_container: PassthroughContainer<'node>,
    render_pass: Option<RenderPassTarget>,
    exec: Box<
        dyn for<'b, 'pass> FnOnce(
                Either<&'b mut wgpu::CommandEncoder, &'b mut wgpu::RenderPass<'pass>>,
                <B as ResourceArray<BufferHandle>>::Fetched<'b>,
                <T as ResourceArray<TextureHandle>>::Fetched<'b>,
                PassthroughContainer<'pass>,
                P,
            ) + 'node,
    >,
}

trait GraphNodeImpl<'node> {
    fn render_pass(&mut self) -> Option<RenderPassTarget>;
    fn exec<'pass>(
        &mut self,
        encoder_or_pass: Either<&mut wgpu::CommandEncoder, &mut wgpu::RenderPass<'pass>>,
        resources: &RenderPassResources,
    ) where 'node: 'pass;
}

impl<'node, B: ResourceArray<BufferHandle>, T: ResourceArray<TextureHandle>, P> GraphNodeImpl<'node>
    for GraphNode<'node, B, T, P>
{
    fn render_pass(&mut self) -> Option<RenderPassTarget> {
        self.inner.as_mut().unwrap().render_pass.take()
    }

    fn exec<'pass>(
        &mut self,
        encoder_or_pass: Either<&mut wgpu::CommandEncoder, &mut wgpu::RenderPass<'pass>>,
        renderer: &RenderPassResources,
    ) where 'node: 'pass {
        let inner = self.inner.take().unwrap();
        (inner.exec)(
            encoder_or_pass,
            inner.buffers.fetch_resources(renderer),
            inner.textures.fetch_resources(renderer),
            inner.passthrough_container,
            inner.passthrough,
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

impl Sealed for () {}

impl<T> ExtendTuple<T> for () {
    type ExtendOne = (T,);

    fn extend_one(self, element: T) -> Self::ExtendOne {
        (element,)
    }
}

macro_rules! extend_tuple {
    ($tuple:ident, $element:ident, $($letters:ident,)*) => {{
        let ($($letters,)*) = $tuple;
        ($($letters,)* $element)
    }};
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

pub struct PassthroughContainer<'a> {
    data: Vec<*const ()>,
    _marker: PhantomData<&'a ()>,
}

pub struct PassthroughRef<T> {
    data_idx: usize,
    _marker: PhantomData<*const T>,
}

impl<'a> PassthroughContainer<'a> {
    fn new() -> Self {
        Self {
            data: vec![],
            _marker: PhantomData,
        }
    }

    fn insert<T>(&mut self, data: &'a T) -> PassthroughRef<T> {
        self.data.push(data as *const T as *const ());
        PassthroughRef {
            data_idx: self.data.len() - 1,
            _marker: PhantomData,
        }
    }
}

impl<'pass> PassthroughContainer<'pass> {
    pub fn get<T>(&self, handle: PassthroughRef<T>) -> &'pass T {
        unsafe {
            // SAFETY: `handle` cannot have been created except by us, and is type-tagged to be
            // something covariant to the original type stored.
            &*(self.data[handle.data_idx] as *const T)
        }
    }
}
