pub mod math;

pub use wgpu;

use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
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

pub struct Dependency<T>(T);

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
    pub handle: Dependency<TextureHandle>,
    pub resolve: Option<Dependency<TextureHandle>>,
    pub clear: wgpu::Color,
}

pub struct RenderPassDepthTarget {
    pub handle: Dependency<TextureHandle>,
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

    pub fn with_color(mut self, handle: Dependency<TextureHandle>, clear: wgpu::Color) -> Self {
        self.color.push(RenderPassColorTarget {
            handle,
            resolve: None,
            clear,
        });
        self
    }

    pub fn with_msaa_color(
        mut self,
        handle: Dependency<TextureHandle>,
        resolve: Dependency<TextureHandle>,
        clear: wgpu::Color,
    ) -> Self {
        self.color.push(RenderPassColorTarget {
            handle,
            resolve: Some(resolve),
            clear,
        });
        self
    }

    pub fn with_depth(mut self, handle: Dependency<TextureHandle>, clear: f32) -> Self {
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
    pub fn get_buffer(&self, handle: Dependency<BufferHandle>) -> &'pass wgpu::Buffer {
        self.resources.buffers.get(&handle.0.id).unwrap()
    }

    pub fn get_texture(&self, handle: Dependency<TextureHandle>) -> &'pass wgpu::TextureView {
        if handle.0.id == usize::MAX {
            self.output_texture
        } else {
            let (_, view) = self.resources.textures.get(&handle.0.id).unwrap();
            view
        }
    }

    pub fn get_bind_group(&self, handle: Dependency<BindGroupHandle>) -> &'pass wgpu::BindGroup {
        &self.resources.bind_groups.get(&handle.0.id).unwrap().0
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
    nodes: Vec<GraphNode<'node>>,
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

    pub fn add_node<'a>(&'a mut self) -> NodeBuilder<'a, 'r, 'node> {
        NodeBuilder {
            graph: self,
            input_buffers: Vec::new(),
            output_buffers: Vec::new(),
            input_textures: Vec::new(),
            output_textures: Vec::new(),
            external_output: false,
            passthrough: PassthroughContainer::new(),
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
            let outputs_used = node
                .output_buffers
                .iter()
                .any(|output| used_buffers.contains(output))
                || node
                    .output_textures
                    .iter()
                    .any(|output| used_textures.contains(output));
            if outputs_used || node.external_output {
                used_buffers.extend(node.input_buffers.iter().cloned());
                used_textures.extend(node.input_textures.iter().cloned());
                pruned_nodes.push(node);
            }
        }

        for node in pruned_nodes.into_iter().rev() {
            match node.render_pass {
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
                    (node.exec)(Right(&mut pass), node.passthrough, &resources);
                }
                None => {
                    let resources = RenderPassResources {
                        resources: self.renderer,
                        output_texture: output.view(),
                    };
                    (node.exec)(Left(&mut encoder), node.passthrough, &resources);
                }
            }
        }

        self.renderer.queue.submit([encoder.finish()]);
        output.present();

        self.renderer.temp_buffers.retain(|_, &mut id| {
            let pred = used_buffers.contains(&BufferHandle { id });
            if !pred {
                self.renderer.buffers.remove(&id);
            }
            pred
        });

        self.renderer.temp_textures.retain(|_, &mut id| {
            let pred = used_textures.contains(&TextureHandle { id });
            if !pred {
                self.renderer.textures.remove(&id);
            }
            pred
        });
    }
}

pub struct NodeBuilder<'a, 'r, 'node> {
    graph: &'a mut RenderGraph<'r, 'node>,
    input_buffers: Vec<BufferHandle>,
    output_buffers: Vec<BufferHandle>,
    input_textures: Vec<TextureHandle>,
    output_textures: Vec<TextureHandle>,
    external_output: bool,
    passthrough: PassthroughContainer<'node>,
}

impl<'node> NodeBuilder<'_, '_, 'node> {
    pub fn add_input_buffer(&mut self, handle: BufferHandle) -> Dependency<BufferHandle> {
        self.input_buffers.push(handle);
        Dependency(handle)
    }

    pub fn add_output_buffer(&mut self, handle: BufferHandle) -> Dependency<BufferHandle> {
        self.input_buffers.push(handle);
        self.output_buffers.push(handle);
        Dependency(handle)
    }

    pub fn add_input_texture(&mut self, handle: TextureHandle) -> Dependency<TextureHandle> {
        self.input_textures.push(handle);
        Dependency(handle)
    }

    pub fn add_output_texture(&mut self, handle: TextureHandle) -> Dependency<TextureHandle> {
        self.input_textures.push(handle);
        self.output_textures.push(handle);
        Dependency(handle)
    }

    pub fn add_input_bind_group(&mut self, handle: BindGroupHandle) -> Dependency<BindGroupHandle> {
        let (_, resources) = self.graph.renderer.bind_groups.get(&handle.id).unwrap();
        for (_, resource) in resources {
            match resource {
                BindGroupResource::Buffer(BufferBinding { buffer, .. }) => {
                    self.input_buffers.push(*buffer)
                }
                BindGroupResource::Texture(handle) => self.input_textures.push(*handle),
                BindGroupResource::BufferArray(bindings) => {
                    for binding in bindings {
                        self.input_buffers.push(binding.buffer);
                    }
                }
                BindGroupResource::TextureArray(textures) => {
                    for handle in textures {
                        self.input_textures.push(*handle);
                    }
                }
                BindGroupResource::Sampler(_) | BindGroupResource::SamplerArray(_) => {}
            }
        }
        Dependency(handle)
    }

    pub fn add_external_output(&mut self) {
        self.external_output = true;
    }

    pub fn passthrough_ref<T: 'node>(&mut self, data: &'node T) -> PassthroughRef<T> {
        self.passthrough.insert(data)
    }

    pub fn passthrough_mut<T: 'node>(&mut self, data: &'node mut T) -> PassthroughMut<T> {
        self.passthrough.insert_mut(data)
    }

    pub fn build_with_encoder<F>(self, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::CommandEncoder,
                PassthroughContainer<'pass>,
                &'b RenderPassResources<'pass>,
            ) + 'node,
    {
        self.graph.nodes.push(GraphNode {
            input_buffers: self.input_buffers,
            output_buffers: self.output_buffers,
            input_textures: self.input_textures,
            output_textures: self.output_textures,
            external_output: self.external_output,
            passthrough: self.passthrough,
            render_pass: None,
            exec: Box::new(move |encoder, passthrough, res| {
                exec(encoder.unwrap_left(), passthrough, res)
            }),
        })
    }

    pub fn build_renderpass<F>(self, pass: RenderPassTarget, exec: F)
    where
        F: for<'b, 'pass> FnOnce(
                &'b mut wgpu::RenderPass<'pass>,
                PassthroughContainer<'pass>,
                &'b RenderPassResources<'pass>,
            ) + 'node,
    {
        self.graph.nodes.push(GraphNode {
            input_buffers: self.input_buffers,
            output_buffers: self.output_buffers,
            input_textures: self.input_textures,
            output_textures: self.output_textures,
            external_output: self.external_output,
            passthrough: self.passthrough,
            render_pass: Some(pass),
            exec: Box::new(move |encoder, passthrough, res| {
                exec(encoder.unwrap_right(), passthrough, res)
            }),
        })
    }
}

#[allow(clippy::type_complexity)]
struct GraphNode<'node> {
    input_buffers: Vec<BufferHandle>,
    output_buffers: Vec<BufferHandle>,
    input_textures: Vec<TextureHandle>,
    output_textures: Vec<TextureHandle>,
    external_output: bool,
    passthrough: PassthroughContainer<'node>,
    render_pass: Option<RenderPassTarget>,
    exec: Box<
        dyn for<'b, 'pass> FnOnce(
                Either<&'b mut wgpu::CommandEncoder, &'b mut wgpu::RenderPass<'pass>>,
                PassthroughContainer<'pass>,
                &'b RenderPassResources<'pass>,
            ) + 'node,
    >,
}

pub struct PassthroughContainer<'a> {
    id: usize,
    data: Vec<*const ()>,
    _marker: PhantomData<&'a ()>,
}

pub struct PassthroughRef<T> {
    container_id: usize,
    data_idx: usize,
    _marker: PhantomData<*const T>,
}

pub struct PassthroughMut<T> {
    container_id: usize,
    data_idx: usize,
    _marker: PhantomData<*mut T>,
}

impl<'a> PassthroughContainer<'a> {
    fn new() -> Self {
        static ID: AtomicUsize = AtomicUsize::new(0);
        Self {
            id: ID.fetch_add(1, Ordering::Relaxed),
            data: vec![],
            _marker: PhantomData,
        }
    }

    fn insert<T>(&mut self, data: &'a T) -> PassthroughRef<T> {
        self.data.push(data as *const T as *const ());
        PassthroughRef {
            container_id: self.id,
            data_idx: self.data.len() - 1,
            _marker: PhantomData,
        }
    }

    fn insert_mut<T>(&mut self, data: &'a mut T) -> PassthroughMut<T> {
        self.data.push(data as *const T as *const ());
        PassthroughMut {
            container_id: self.id,
            data_idx: self.data.len() - 1,
            _marker: PhantomData,
        }
    }
}

impl<'pass> PassthroughContainer<'pass> {
    pub fn get<T>(&self, handle: PassthroughRef<T>) -> &'pass T {
        assert_eq!(self.id, handle.container_id);
        unsafe {
            // SAFETY: `handle` cannot have been created except by us, and is type-tagged to be
            // something covariant to the original type stored.
            &*(self.data[handle.data_idx] as *const T)
        }
    }

    pub fn get_mut<T>(&self, handle: PassthroughMut<T>) -> &'pass mut T {
        assert_eq!(self.id, handle.container_id);
        unsafe {
            // SAFETY: `handle` cannot have been created except by us, cannot have been cloned, and
            // is type-tagged to be invariant to the original type stored
            &mut *(self.data[handle.data_idx] as *mut T)
        }
    }
}
