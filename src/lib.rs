pub mod math;

pub use wgpu;

use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ahash::{AHashSet, AHasher};
use either::{Either, Left, Right};
use indexmap::IndexMap;

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
pub struct TextureHandle {
    id: u32,
}

impl TextureHandle {
    pub const RENDER_TARGET: TextureHandle = TextureHandle { id: u32::MAX };
}

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
    pub fn color(handle: Dependency<TextureHandle>, clear: wgpu::Color) -> RenderPassTarget {
        RenderPassTarget {
            color: vec![RenderPassColorTarget {
                handle,
                resolve: None,
                clear,
            }],
            depth: None,
        }
    }

    pub fn color_with_resolve(
        handle: Dependency<TextureHandle>,
        resolve: Dependency<TextureHandle>,
        clear: wgpu::Color,
    ) -> RenderPassTarget {
        RenderPassTarget {
            color: vec![RenderPassColorTarget {
                handle,
                resolve: Some(resolve),
                clear,
            }],
            depth: None,
        }
    }
}

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    textures: IndexMap<wgpu_types::TextureDescriptor<u64>, (wgpu::Texture, wgpu::TextureView)>,
}

impl Renderer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Renderer {
        Self {
            device,
            queue,
            textures: IndexMap::new(),
        }
    }
}

pub struct RenderPassResources<'pass> {
    resources: &'pass Renderer,
    output_texture: &'pass wgpu::TextureView,
}

impl<'pass> RenderPassResources<'pass> {
    pub fn get_texture(&self, handle: Dependency<TextureHandle>) -> &'pass wgpu::TextureView {
        if handle.0.id == u32::MAX {
            self.output_texture
        } else {
            let (_, (_, view)) = self
                .resources
                .textures
                .get_index(handle.0.id as usize)
                .unwrap();
            view
        }
    }
}

pub struct RenderGraph<'r, 'node> {
    renderer: &'r mut Renderer,
    current_textures: AHashSet<u32>,
    nodes: Vec<GraphNode<'node>>,
}

impl<'r, 'node> RenderGraph<'r, 'node> {
    pub fn new(renderer: &'r mut Renderer) -> RenderGraph<'r, 'node> {
        RenderGraph {
            renderer,
            current_textures: AHashSet::new(),
            nodes: Vec::new(),
        }
    }

    pub fn add_intermediate_texture(
        &mut self,
        descriptor: wgpu::TextureDescriptor,
    ) -> TextureHandle {
        let mut hash_descriptor = descriptor.map_label(|s| {
            let mut hasher = AHasher::default();
            if let Some(s) = s {
                s.hash(&mut hasher)
            }
            hasher.finish()
        });
        let id = loop {
            match self.renderer.textures.get_index_of(&hash_descriptor) {
                Some(id) => {
                    if self.current_textures.insert(id as u32) {
                        break Some(id as u32);
                    }
                }
                None => break None,
            }
            hash_descriptor.label += 1;
        };
        match id {
            Some(id) => TextureHandle { id },
            None => {
                let texture = self.renderer.device.create_texture(&descriptor);
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let (id, old) = self
                    .renderer
                    .textures
                    .insert_full(hash_descriptor, (texture, view));
                debug_assert!(old.is_none());
                TextureHandle { id: id as u32 }
            }
        }
    }

    pub fn add_node<'a>(&'a mut self) -> NodeBuilder<'a, 'r, 'node> {
        NodeBuilder {
            graph: self,
            inputs: Vec::new(),
            outputs: Vec::new(),
            external_output: false,
            passthrough: PassthroughContainer::new(),
        }
    }

    pub fn execute(self, target: &mut impl RenderTarget) {
        let output = target.get_current_view();
        let resources = RenderPassResources {
            resources: self.renderer,
            output_texture: output.view(),
        };

        let mut encoder =
            self.renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Polystrip Command Encoder"),
                });

        let mut used_resources = AHashSet::new();
        used_resources.insert(TextureHandle::RENDER_TARGET);
        let mut pruned_nodes = Vec::with_capacity(self.nodes.len());
        for node in self.nodes.into_iter().rev() {
            let outputs_used = node
                .outputs
                .iter()
                .any(|output| used_resources.contains(output));
            if outputs_used || node.external_output {
                used_resources.extend(node.inputs.iter().cloned());
                pruned_nodes.push(node);
            }
        }

        for node in pruned_nodes.into_iter().rev() {
            match node.render_pass {
                Some(pass_targets) => {
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
                    (node.exec)(Left(&mut encoder), node.passthrough, &resources);
                }
            }
        }

        self.renderer.queue.submit([encoder.finish()]);
        output.present();

        for texture_idx in (0..self.renderer.textures.len()).rev() {
            if !used_resources.contains(&TextureHandle { id: texture_idx as u32 }) {
                self.renderer.textures.swap_remove_index(texture_idx);
            }
        }
    }
}

pub struct NodeBuilder<'a, 'r, 'node> {
    graph: &'a mut RenderGraph<'r, 'node>,
    inputs: Vec<TextureHandle>,
    outputs: Vec<TextureHandle>,
    external_output: bool,
    passthrough: PassthroughContainer<'node>,
}

impl<'node> NodeBuilder<'_, '_, 'node> {
    pub fn add_input_texture(&mut self, handle: TextureHandle) -> Dependency<TextureHandle> {
        self.inputs.push(handle);
        Dependency(handle)
    }

    pub fn add_output_texture(&mut self, handle: TextureHandle) -> Dependency<TextureHandle> {
        self.inputs.push(handle);
        self.outputs.push(handle);
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
            inputs: self.inputs,
            outputs: self.outputs,
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
            inputs: self.inputs,
            outputs: self.outputs,
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
    inputs: Vec<TextureHandle>,
    outputs: Vec<TextureHandle>,
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

    pub fn get_mut<T>(&self, handle: PassthroughMut<T>) -> &'pass T {
        assert_eq!(self.id, handle.container_id);
        unsafe {
            // SAFETY: `handl` cannot have been created except by us, cannot have been cloned, and
            // is type-tagged to be invariant to the original type stored
            &*(self.data[handle.data_idx] as *mut T)
        }
    }
}
