#![feature(generic_associated_types)] // Can also experiment with #![feature(generic_const_exprs)]
#![allow(clippy::type_complexity)]

pub mod graph;

pub use wgpu;

use std::borrow::Cow;
use std::hash::Hash;
use std::num::NonZeroU64;
use std::sync::Arc;

use fxhash::FxHashMap;

/// Thin wrapper around an `Instance`, `Adapter`, `Device` and `Queue`.
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

    /// Create a `Renderer` from this `PolystripDevice`.
    ///
    /// Convenience method for [`Renderer::new`]
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

/// A `wgpu::Surface` configured for rendering.
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

/// A non-owning handle to a render resource.
///
/// Handles are created by adding or inserting a resource into a [`Renderer`]. The referenced
/// resource can be retrieved through one of the `Renderer::get_*` methods.
pub struct Handle<T> {
    id: usize,
    _marker: std::marker::PhantomData<*mut T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle {
            id: self.id,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.id);
    }
}

impl Handle<wgpu::TextureView> {
    pub const RENDER_TARGET: Handle<wgpu::TextureView> = Handle { id: usize::MAX, _marker: std::marker::PhantomData };
}

/// Core resource manager for render resources.
///
/// Most interactions with `polystrip` will involve a `Renderer`, which enables the library to keep
/// track of buffers, textures, samplers, bind groups, bind group layouts, and render/compute
/// pipelines. Various methods on this type facilitate the construction of these resources.
pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: FxHashMap<usize, wgpu::Buffer>,
    temp_buffers: FxHashMap<wgpu_types::BufferDescriptor<u64>, usize>,
    textures: FxHashMap<usize, (wgpu::Texture, wgpu::TextureView)>,
    temp_textures: FxHashMap<wgpu_types::TextureDescriptor<u64>, usize>,
    samplers: FxHashMap<usize, wgpu::Sampler>,
    bind_group_layout_descriptors: FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, usize>,
    bind_group_layouts: FxHashMap<usize, wgpu::BindGroupLayout>,
    bind_group_descriptors: FxHashMap<Vec<(u32, BindGroupResource)>, usize>,
    bind_groups: FxHashMap<usize, (wgpu::BindGroup, Vec<(u32, BindGroupResource)>)>,
    render_pipelines: FxHashMap<usize, RenderPipeline>,
    compute_pipelines: FxHashMap<usize, ComputePipeline>,
    next_buffer: usize,
    next_texture: usize,
    next_sampler: usize,
    next_bind_group_layout: usize,
    next_bind_group: usize,
    next_render_pipeline: usize,
    next_compute_pipeline: usize,
}

impl Renderer {
    /// Create a new, empty renderer.
    ///
    /// The [`Device`][wgpu::Device] and [`Queue`][wgpu::Queue] must be from the same device. This
    /// operation never fails.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Renderer {
        Self {
            device,
            queue,
            buffers: FxHashMap::default(),
            temp_buffers: FxHashMap::default(),
            textures: FxHashMap::default(),
            temp_textures: FxHashMap::default(),
            samplers: FxHashMap::default(),
            bind_group_layout_descriptors: FxHashMap::default(),
            bind_group_layouts: FxHashMap::default(),
            bind_group_descriptors: FxHashMap::default(),
            bind_groups: FxHashMap::default(),
            render_pipelines: FxHashMap::default(),
            compute_pipelines: FxHashMap::default(),
            next_buffer: 0,
            next_texture: 0,
            next_sampler: 0,
            next_bind_group_layout: 0,
            next_bind_group: 0,
            next_render_pipeline: 0,
            next_compute_pipeline: 0,
        }
    }

    /// Insert an existing `Buffer` into the renderer.
    ///
    /// Buffer must have been allocated from the same device as this `Renderer`.
    pub fn insert_buffer(&mut self, buffer: wgpu::Buffer) -> Handle<wgpu::Buffer> {
        let id = self.next_buffer_id();
        self.buffers.insert(id, buffer);
        Handle::<wgpu::Buffer> { id, _marker: std::marker::PhantomData }
    }

    /// Insert an existing `Texture` and corresponding `TextureView` into the renderer.
    ///
    /// Texture must have been allocated from the same device as this `Renderer`.
    pub fn insert_texture(
        &mut self,
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    ) -> Handle<wgpu::TextureView> {
        let id = self.next_texture_id();
        self.textures.insert(id, (texture, view));
        Handle::<wgpu::TextureView> { id, _marker: std::marker::PhantomData }
    }

    /// Insert an existing `Sampler` into the renderer.
    ///
    /// Sampler must have been allocated from the same device this `Renderer` was created from.
    pub fn insert_sampler(&mut self, sampler: wgpu::Sampler) -> Handle<wgpu::Sampler> {
        let id = self.next_sampler_id();
        self.samplers.insert(id, sampler);
        Handle::<wgpu::Sampler> { id, _marker: std::marker::PhantomData }
    }

    /// Insert an existing `BindGroupLayout` into the renderer.
    ///
    /// Layout must have been allocated from the same device as this `Renderer`.
    pub fn insert_bind_group_layout(
        &mut self,
        layout: wgpu::BindGroupLayout,
    ) -> Handle<wgpu::BindGroupLayout> {
        let id = self.next_bind_group_layout_id();
        self.bind_group_layouts.insert(id, layout);
        Handle::<wgpu::BindGroupLayout> { id, _marker: std::marker::PhantomData }
    }

    /// Insert an existing `RenderPipeline` into the renderer.
    ///
    /// Pipeline must have been created from the same device as this `Renderer`, and the contained
    /// BindGroupLayouts must correspond with the PipelineLayout from which the pipeline was
    /// created.
    pub fn insert_render_pipeline(&mut self, pipeline: RenderPipeline) -> Handle<RenderPipeline> {
        let id = self.next_render_pipeline_id();
        self.render_pipelines.insert(id, pipeline);
        Handle::<RenderPipeline> { id, _marker: std::marker::PhantomData }
    }

    /// Insert an existing `ComputePipeline` into the renderer.
    ///
    /// Pipeline must have been created from the same device as this `Renderer`, and the contained
    /// BindGroupLayouts must correspond with the PipelineLayout from which the pipeline was
    /// created.
    pub fn insert_compute_pipeline(&mut self, pipeline: ComputePipeline) -> Handle<ComputePipeline> {
        let id = self.next_compute_pipeline_id();
        self.compute_pipelines.insert(id, pipeline);
        Handle::<ComputePipeline> { id, _marker: std::marker::PhantomData }
    }

    /// Retrieve a `Buffer` from this renderer.
    pub fn get_buffer(&self, handle: Handle<wgpu::Buffer>) -> &wgpu::Buffer {
        self.buffers.get(&handle.id).unwrap()
    }

    /// Retrieve a `Texture` and its corresponding `TextureView` from this renderer.
    pub fn get_texture(&self, handle: Handle<wgpu::TextureView>) -> (&wgpu::Texture, &wgpu::TextureView) {
        let (texture, view) = self.textures.get(&handle.id).unwrap();
        (texture, view)
    }

    /// Retrieve a `Sampler` from this renderer.
    pub fn get_sampler(&self, handle: Handle<wgpu::Sampler>) -> &wgpu::Sampler {
        self.samplers.get(&handle.id).unwrap()
    }

    /// Retrieve a `BindGroupLayout` from this renderer.
    pub fn get_bind_group_layout(&self, handle: Handle<wgpu::BindGroupLayout>) -> &wgpu::BindGroupLayout {
        self.bind_group_layouts.get(&handle.id).unwrap()
    }

    /// Retrieve a `BindGroup` from this renderer.
    pub fn get_bind_group(&self, handle: Handle<wgpu::BindGroup>) -> &wgpu::BindGroup {
        let (bind_group, _) = self.bind_groups.get(&handle.id).unwrap();
        bind_group
    }

    /// Retrieve a `RenderPipeline` from this renderer.
    pub fn get_render_pipeline(&self, handle: Handle<RenderPipeline>) -> &RenderPipeline {
        self.render_pipelines.get(&handle.id).unwrap()
    }

    /// Retrieve a `ComputePipeline` from this renderer.
    pub fn get_compute_pipeline(&self, handle: Handle<ComputePipeline>) -> &ComputePipeline {
        self.compute_pipelines.get(&handle.id).unwrap()
    }
}

impl Renderer {
    /// Create a new `BindGroupLayout` from a collection of
    /// [`BindGroupLayoutEntry`][wgpu::BindGroupLayoutEntry], and insert it into the renderer
    /// immediately.
    pub fn add_bind_group_layout<'a>(
        &mut self,
        layout: impl Into<Cow<'a, [wgpu::BindGroupLayoutEntry]>>,
    ) -> Handle<wgpu::BindGroupLayout> {
        let layout = layout.into();
        if let Some(index) = self.bind_group_layout_descriptors.get(layout.as_ref()) {
            return Handle::<wgpu::BindGroupLayout> { id: *index, _marker: std::marker::PhantomData };
        }
        let id = self.next_bind_group_layout_id();
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &layout,
                });
        self.bind_group_layouts.insert(id, bind_group_layout);
        self.bind_group_layout_descriptors
            .insert(layout.into_owned(), id);
        Handle::<wgpu::BindGroupLayout> { id, _marker: std::marker::PhantomData }
    }

    //TODO: Document exactly what `resources` can be
    /// Create a new `BindGroup` from the specified set of resources.
    ///
    /// If a bind group has already been created from this set of resources, the cached one will be
    /// returned. Otherwise a new one will be created.
    pub fn add_bind_group(
        &mut self,
        layout: Handle<wgpu::BindGroupLayout>,
        resources: impl IntoBindGroupResources,
    ) -> Handle<wgpu::BindGroup> {
        let key = resources.into_entries();
        match self.bind_group_descriptors.get(&key) {
            Some(&id) => Handle::<wgpu::BindGroup> { id, _marker: std::marker::PhantomData },
            None => {
                let id = self.insert_bind_group(layout, key.clone());
                self.bind_group_descriptors.insert(key, id);
                Handle::<wgpu::BindGroup> { id, _marker: std::marker::PhantomData }
            }
        }
    }
    
    /// Create a new `RenderPipeline` from wgsl source code.
    ///
    /// Bind group layouts and vertex attributes will be inferred from the shader source code. See
    /// [`RenderPipelineBuilder`] for more details.
    pub fn add_render_pipeline_from_wgsl(&mut self, shader_source: &str) -> RenderPipelineBuilder<'_> {
        RenderPipelineBuilder::from_wgsl(self, shader_source)
    }
    
    /// Create a new `ComputePipeline` from wgsl source code.
    ///
    /// Bind group layouts will be inferred from the shader source code. See
    /// [`ComputePipelineBuilder`] for more details.
    pub fn add_compute_pipeline_from_wgsl(&mut self, shader_source: &str) -> ComputePipelineBuilder<'_> {
        ComputePipelineBuilder::from_wgsl(self, shader_source)
    }
}

impl Renderer {
    /// Schedule a data write into `buffer` starting at `offset`.
    ///
    /// See [`wgpu::Queue::write_buffer`] for more details.
    pub fn write_buffer(&self, handle: Handle<wgpu::Buffer>, offset: wgpu::BufferAddress, data: &[u8]) {
        self.queue
            .write_buffer(self.get_buffer(handle), offset, data)
    }

    /// Schedule a data write into `buffer` starting at `offset` via the returned [`QueueWriteBufferView`][wgpu::QueueWriteBufferView].
    /// 
    /// See [`wgpu::Queue::write_buffer_with`] for more details.
    pub fn write_buffer_with(
        &self,
        handle: Handle<wgpu::Buffer>,
        offset: wgpu::BufferAddress,
        size: wgpu::BufferSize,
    ) -> wgpu::QueueWriteBufferView<'_> {
        self.queue
            .write_buffer_with(self.get_buffer(handle), offset, size)
    }

    /// Schedule a data write into `texture`.
    ///
    /// See [`wgpu::Queue::write_texture`] for more details.
    pub fn write_texture(
        &self,
        texture: wgpu::ImageCopyTexture<'_>,
        data: &[u8],
        data_layout: wgpu::ImageDataLayout,
        size: wgpu::Extent3d,
    ) {
        self.queue.write_texture(texture, data, data_layout, size)
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

    fn next_bind_group_layout_id(&mut self) -> usize {
        let id = self.next_bind_group_layout;
        self.next_bind_group_layout += 1;
        id
    }

    fn next_render_pipeline_id(&mut self) -> usize {
        let id = self.next_render_pipeline;
        self.next_render_pipeline += 1;
        id
    }
    
    fn next_compute_pipeline_id(&mut self) -> usize {
        let id = self.next_compute_pipeline;
        self.next_compute_pipeline += 1;
        id
    }

    fn insert_bind_group(
        &mut self,
        layout: Handle<wgpu::BindGroupLayout>,
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
            layout: self.get_bind_group_layout(layout),
            entries: &bind_group_entries,
        });

        let id = self.next_bind_group;
        self.next_bind_group += 1;
        self.bind_groups.insert(id, (bind_group, resources));
        id
    }
}

#[derive(Default)]
pub struct RenderPassTarget {
    pub color: Vec<RenderPassColorTarget>,
    pub depth: Option<RenderPassDepthTarget>,
}

pub struct RenderPassColorTarget {
    pub handle: Handle<wgpu::TextureView>,
    pub resolve: Option<Handle<wgpu::TextureView>>,
    pub clear: wgpu::Color,
}

pub struct RenderPassDepthTarget {
    pub handle: Handle<wgpu::TextureView>,
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

    pub fn with_color(mut self, handle: Handle<wgpu::TextureView>, clear: wgpu::Color) -> Self {
        self.color.push(RenderPassColorTarget {
            handle,
            resolve: None,
            clear,
        });
        self
    }

    pub fn with_msaa_color(
        mut self,
        handle: Handle<wgpu::TextureView>,
        resolve: Handle<wgpu::TextureView>,
        clear: wgpu::Color,
    ) -> Self {
        self.color.push(RenderPassColorTarget {
            handle,
            resolve: Some(resolve),
            clear,
        });
        self
    }

    pub fn with_depth(mut self, handle: Handle<wgpu::TextureView>, clear: f32) -> Self {
        self.depth = Some(RenderPassDepthTarget {
            handle,
            depth_clear: Some(clear),
            stencil_clear: None,
        });
        self
    }
}

impl RenderPassTarget {
    pub fn is_compatible_with(&self, other: &RenderPassTarget) -> bool {
        for (left, right) in self.color.iter().zip(other.color.iter()) {
            if (left.handle, left.resolve) != (right.handle, right.resolve) {
                return false;
            }
        }

        self.depth.as_ref().map(|target| target.handle) == other.depth.as_ref().map(|target| target.handle)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BufferBinding {
    pub buffer: Handle<wgpu::Buffer>,
    pub offset: u64,
    pub size: Option<NonZeroU64>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum BindGroupResource {
    Buffer(BufferBinding),
    BufferArray(Vec<BufferBinding>),
    Texture(Handle<wgpu::TextureView>),
    TextureArray(Vec<Handle<wgpu::TextureView>>),
    Sampler(Handle<wgpu::Sampler>),
    SamplerArray(Vec<Handle<wgpu::Sampler>>),
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

impl IntoBindGroupResource for Handle<wgpu::Buffer> {
    fn into_bind_group_resource(self) -> BindGroupResource {
        BindGroupResource::Buffer(BufferBinding {
            buffer: self,
            offset: 0,
            size: None,
        })
    }
}

impl IntoBindGroupResource for Handle<wgpu::TextureView> {
    fn into_bind_group_resource(self) -> BindGroupResource {
        BindGroupResource::Texture(self)
    }
}

impl IntoBindGroupResource for Handle<wgpu::Sampler> {
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

/// A render pipeline and its associated bind group layouts
pub struct RenderPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layouts: Vec<Handle<wgpu::BindGroupLayout>>,
}

enum BindGroupLayout {
    Descriptor(Vec<wgpu::BindGroupLayoutEntry>),
    Handle(Handle<wgpu::BindGroupLayout>),
}

/// Builder for a `wgpu::RenderPipeline`
///
/// The types and sizes of bind group layouts can be inferred by reading the code of a
/// shader. Naturally, since this is exactly what [`naga`] does when translating shaders, the
/// intermediate data can be used to construct the pipeline as well. This obviates the need to
/// specify many of the parameters of a [`wgpu::RenderPipelineDescriptor`], and the rest are
/// overrideable defaults.
///
/// This builder assumes there is only one vertex buffer supplying vertex attributes. If there is
/// more than one, the pipeline must be created manually for now. In addition, many options
/// available to a `wgpu::RenderPipelineDescriptor` have not been implemented yet.
///
/// This type cannot be created directly, and must be constructed from a `Renderer` with the
/// [`Renderer::add_render_pipeline_from_wgsl`] method.
pub struct RenderPipelineBuilder<'a> {
    renderer: &'a mut Renderer,
    shader: naga::Module,
    vertex_entry: usize,
    fragment_entry: usize,
    vertex_array_stride: wgpu::BufferAddress,
    vertex_step_mode: wgpu::VertexStepMode,
    vertex_attributes: Vec<wgpu::VertexAttribute>,
    bind_group_layouts: Vec<Option<BindGroupLayout>>,
    primitive_topology: wgpu::PrimitiveTopology,
    depth_stencil: Option<wgpu::DepthStencilState>,
    sample_count: u32,
}

impl<'a> RenderPipelineBuilder<'a> {
    fn from_wgsl(renderer: &'a mut Renderer, shader_source: &str) -> RenderPipelineBuilder<'a> {
        let shader = naga::front::wgsl::parse_str(shader_source).unwrap();

        let (vertex_entry, fragment_entry, global_stages) = {
            let mut vertex_entry = None;
            let mut fragment_entry = None;
            let mut global_stages = FxHashMap::default();
            for (i, entry) in shader.entry_points.iter().enumerate() {
                match entry.stage {
                    naga::ShaderStage::Vertex if vertex_entry.is_none() => vertex_entry = Some(i),
                    naga::ShaderStage::Fragment if fragment_entry.is_none() => {
                        fragment_entry = Some(i)
                    }
                    _ => {}
                }
                for (_, expression) in entry.function.expressions.iter() {
                    scan_expression(
                        expression,
                        &mut global_stages,
                        match entry.stage {
                            naga::ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
                            naga::ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
                            naga::ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
                        },
                        &entry.function,
                    );
                }
            }
            (
                vertex_entry.unwrap(),
                fragment_entry.unwrap(),
                global_stages,
            )
        };

        let (vertex_attributes, vertex_array_stride) = {
            let mut vertex_attributes = shader.entry_points[vertex_entry]
                .function
                .arguments
                .iter()
                .flat_map(|argument| {
                    fn check_binding(
                        binding: Option<&naga::Binding>,
                        ty: naga::Handle<naga::Type>,
                        shader: &naga::Module,
                    ) -> Vec<(u32, wgpu::VertexFormat)> {
                        match binding {
                            Some(naga::Binding::Location { location, .. }) => {
                                let var_type = shader.types.get_handle(ty).unwrap();
                                let format = vertex_format_from_type(&var_type.inner);
                                vec![(*location, format)]
                            }
                            Some(naga::Binding::BuiltIn(_)) => vec![],
                            None => {
                                if let naga::TypeInner::Struct { members, .. } =
                                    &shader.types.get_handle(ty).unwrap().inner
                                {
                                    members
                                        .iter()
                                        .flat_map(|member| {
                                            check_binding(
                                                member.binding.as_ref(),
                                                member.ty,
                                                shader,
                                            )
                                        })
                                        .collect()
                                } else {
                                    unimplemented!()
                                }
                            }
                        }
                    }

                    check_binding(argument.binding.as_ref(), argument.ty, &shader)
                })
                .collect::<Vec<_>>();
            vertex_attributes.sort_unstable_by_key(|(location, _)| *location);
            let mut offset = 0;
            (
                vertex_attributes
                    .into_iter()
                    .map(|(location, format)| wgpu::VertexAttribute {
                        format,
                        offset: {
                            let this_offset = offset;
                            offset += format.size();
                            this_offset
                        },
                        shader_location: location,
                    })
                    .collect::<Vec<_>>(),
                offset,
            )
        };

        let mut bind_group_layouts = Vec::<Option<Vec<_>>>::default();
        for (handle, global) in shader.global_variables.iter() {
            if let Some(binding) = &global.binding {
                let shader_type = shader.types.get_handle(global.ty).unwrap();
                let size = shader_type.inner.size(&shader.constants);

                let visibility = global_stages
                    .get(&handle)
                    .copied()
                    .unwrap_or(wgpu::ShaderStages::empty());
                let ty = wgpu::BindingType::Buffer {
                    ty: match global.space {
                        naga::AddressSpace::Uniform => wgpu::BufferBindingType::Uniform,
                        naga::AddressSpace::Storage { access } => {
                            wgpu::BufferBindingType::Storage {
                                read_only: !access.contains(naga::StorageAccess::STORE),
                            }
                        }
                        _ => todo!(),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size as u64),
                };

                if bind_group_layouts.len() <= binding.group as usize {
                    bind_group_layouts.resize(binding.group as usize + 1, None);
                }
                bind_group_layouts[binding.group as usize]
                    .get_or_insert_with(Vec::new)
                    .push(wgpu::BindGroupLayoutEntry {
                        binding: binding.binding,
                        visibility,
                        ty,
                        count: None,
                    })
            }
        }

        RenderPipelineBuilder {
            renderer,
            shader,
            vertex_entry,
            fragment_entry,
            vertex_array_stride,
            vertex_step_mode: wgpu::VertexStepMode::Vertex,
            vertex_attributes,
            bind_group_layouts: bind_group_layouts
                .into_iter()
                .map(|v| v.map(BindGroupLayout::Descriptor))
                .collect(),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            depth_stencil: None,
            sample_count: 1,
        }
    }

    /// Override the automatically-derived bind group layout specified by `index`.
    pub fn with_bind_group_layout(mut self, index: usize, handle: Handle<wgpu::BindGroupLayout>) -> Self {
        if self.bind_group_layouts.len() <= index {
            self.bind_group_layouts.resize_with(index + 1, || None);
        }
        self.bind_group_layouts[index] = Some(BindGroupLayout::Handle(handle));
        self
    }

    /// Set the step mode of the vertex buffer.
    pub fn with_vertex_step_mode(mut self, step_mode: wgpu::VertexStepMode) -> Self {
        self.vertex_step_mode = step_mode;
        self
    }

    /// Set the primitive topology of the pipeline.
    pub fn with_primitive_topology(mut self, topology: wgpu::PrimitiveTopology) -> Self {
        self.primitive_topology = topology;
        self
    }

    /// Set the depth stencil state of the pipeline.
    pub fn with_depth_stencil(mut self, depth_stencil: wgpu::DepthStencilState) -> Self {
        self.depth_stencil = Some(depth_stencil);
        self
    }

    /// Set the sample count that the pipeline can render to.
    ///
    /// In future, perhaps this can be automatically derived from the textures being rendered to.
    pub fn with_msaa(mut self, sample_count: u32) -> Self {
        self.sample_count = sample_count;
        self
    }

    /// Build the `RenderPipeline` and return a handle to its resource in the [`Renderer`].
    pub fn build(self) -> Handle<RenderPipeline> {
        let vertex_entry_name = self.shader.entry_points[self.vertex_entry].name.clone();
        let fragment_entry_name = self.shader.entry_points[self.fragment_entry].name.clone();

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: self.vertex_array_stride,
            step_mode: self.vertex_step_mode,
            attributes: &self.vertex_attributes,
        };

        let module = self
            .renderer
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Naga(self.shader),
            });

        let bind_group_layouts = self
            .bind_group_layouts
            .into_iter()
            .map(|layout| match layout.unwrap() {
                BindGroupLayout::Handle(handle) => handle,
                BindGroupLayout::Descriptor(descriptor) => {
                    self.renderer.add_bind_group_layout(descriptor)
                }
            })
            .collect::<Vec<_>>();

        let resolved_layouts = bind_group_layouts
            .iter()
            .cloned()
            .map(|handle| self.renderer.get_bind_group_layout(handle))
            .collect::<Vec<_>>();
        let pipeline_layout =
            self.renderer
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &resolved_layouts,
                    push_constant_ranges: &[],
                });

        let pipeline =
            self.renderer
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: &vertex_entry_name,
                        buffers: if self.vertex_attributes.is_empty() {
                            &[]
                        } else {
                            std::slice::from_ref(&vertex_buffer_layout)
                        },
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: self.primitive_topology,
                        ..Default::default()
                    },
                    depth_stencil: self.depth_stencil,
                    multisample: wgpu::MultisampleState {
                        count: self.sample_count,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: &fragment_entry_name,
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Bgra8UnormSrgb,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview: None,
                });

        self.renderer.insert_render_pipeline(RenderPipeline {
            pipeline,
            bind_group_layouts,
        })
    }
}

/// A compute pipeline and its associated bind group layouts
pub struct ComputePipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layouts: Vec<Handle<wgpu::BindGroupLayout>>,
}

/// Builder for a `wgpu::ComputePipeline`
///
/// The types and sizes of bind group layouts can be inferred by reading the code of a
/// shader. Naturally, since this is exactly what [`naga`] does when translating shaders, the
/// intermediate data can be used to construct the pipeline as well. This obviates the need to
/// specify many of the parameters of a [`wgpu::ComputePipelineDescriptor`], and the rest are
/// overrideable defaults.
///
/// This type cannot be created directly, and must be constructed from a `Renderer` with the
/// [`Renderer::add_compute_pipeline_from_wgsl`] method.
pub struct ComputePipelineBuilder<'a> {
    renderer: &'a mut Renderer,
    shader: naga::Module,
    entry: usize,
    bind_group_layouts: Vec<Option<BindGroupLayout>>,
}

impl<'a> ComputePipelineBuilder<'a> {
    fn from_wgsl(renderer: &'a mut Renderer, shader_source: &str) -> ComputePipelineBuilder<'a> {
        let shader = naga::front::wgsl::parse_str(shader_source).unwrap();

        let entry = {
            let mut entry_index = None;
            for (i, entry) in shader.entry_points.iter().enumerate() {
                if let naga::ShaderStage::Compute = entry.stage {
                    entry_index = Some(i)
                }
            }
            entry_index.unwrap()
        };

        let mut bind_group_layouts = Vec::<Option<Vec<_>>>::default();
        for (_, global) in shader.global_variables.iter() {
            if let Some(binding) = &global.binding {
                let shader_type = shader.types.get_handle(global.ty).unwrap();
                let size = shader_type.inner.size(&shader.constants);

                let ty = wgpu::BindingType::Buffer {
                    ty: match global.space {
                        naga::AddressSpace::Uniform => wgpu::BufferBindingType::Uniform,
                        naga::AddressSpace::Storage { access } => {
                            wgpu::BufferBindingType::Storage {
                                read_only: !access.contains(naga::StorageAccess::STORE),
                            }
                        }
                        _ => todo!(),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size as u64),
                };

                if bind_group_layouts.len() <= binding.group as usize {
                    bind_group_layouts.resize(binding.group as usize + 1, None);
                }
                bind_group_layouts[binding.group as usize]
                    .get_or_insert_with(Vec::new)
                    .push(wgpu::BindGroupLayoutEntry {
                        binding: binding.binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty,
                        count: None,
                    })
            }
        }

        ComputePipelineBuilder {
            renderer,
            entry,
            shader,
            bind_group_layouts: bind_group_layouts
                .into_iter()
                .map(|v| v.map(BindGroupLayout::Descriptor))
                .collect(),
        }
    }

    /// Override the automatically-derived bind group layout specified by `index`.
    pub fn with_bind_group_layout(mut self, index: usize, handle: Handle<wgpu::BindGroupLayout>) -> Self {
        if self.bind_group_layouts.len() <= index {
            self.bind_group_layouts.resize_with(index + 1, || None);
        }
        self.bind_group_layouts[index] = Some(BindGroupLayout::Handle(handle));
        self
    }

    /// Build the `ComputePipeline` and return a handle to its resource in the [`Renderer`].
    pub fn build(self) -> Handle<ComputePipeline> {
        let entry = self.shader.entry_points[self.entry].name.clone();

        let module = self.renderer.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Naga(self.shader),
        });
        
        let bind_group_layouts = self
            .bind_group_layouts
            .into_iter()
            .map(|layout| match layout.unwrap() {
                BindGroupLayout::Handle(handle) => handle,
                BindGroupLayout::Descriptor(descriptor) => {
                    self.renderer.add_bind_group_layout(descriptor)
                }
            })
            .collect::<Vec<_>>();

        let resolved_layouts = bind_group_layouts
            .iter()
            .cloned()
            .map(|handle| self.renderer.get_bind_group_layout(handle))
            .collect::<Vec<_>>();
        let pipeline_layout =
            self.renderer
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &resolved_layouts,
                    push_constant_ranges: &[],
                });

        let pipeline = self.renderer.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: &entry,
        });

        self.renderer.insert_compute_pipeline(ComputePipeline {
            pipeline,
            bind_group_layouts,
        })
    }
}

fn vertex_format_from_type(inner: &naga::TypeInner) -> wgpu::VertexFormat {
    match inner {
        naga::TypeInner::Scalar {
            kind: naga::ScalarKind::Uint,
            width: 4,
        } => wgpu::VertexFormat::Uint32,
        naga::TypeInner::Scalar {
            kind: naga::ScalarKind::Sint,
            width: 4,
        } => wgpu::VertexFormat::Sint32,
        naga::TypeInner::Scalar {
            kind: naga::ScalarKind::Float,
            width: 4,
        } => wgpu::VertexFormat::Float32,
        naga::TypeInner::Scalar {
            kind: naga::ScalarKind::Float,
            width: 8,
        } => wgpu::VertexFormat::Float64,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Uint,
            width: 1,
        } => wgpu::VertexFormat::Uint8x2,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Uint,
            width: 1,
        } => wgpu::VertexFormat::Uint8x4,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Sint,
            width: 1,
        } => wgpu::VertexFormat::Sint8x4,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Sint,
            width: 1,
        } => wgpu::VertexFormat::Sint8x4,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Uint,
            width: 2,
        } => wgpu::VertexFormat::Uint16x2,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Uint,
            width: 2,
        } => wgpu::VertexFormat::Uint16x4,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Sint,
            width: 2,
        } => wgpu::VertexFormat::Sint16x4,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Sint,
            width: 2,
        } => wgpu::VertexFormat::Sint16x4,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width: 2,
        } => wgpu::VertexFormat::Float16x2,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width: 2,
        } => wgpu::VertexFormat::Float16x4,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Uint,
            width: 4,
        } => wgpu::VertexFormat::Uint32x2,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Tri,
            kind: naga::ScalarKind::Uint,
            width: 4,
        } => wgpu::VertexFormat::Uint32x3,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Uint,
            width: 4,
        } => wgpu::VertexFormat::Uint32x4,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Sint,
            width: 4,
        } => wgpu::VertexFormat::Sint32x2,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Tri,
            kind: naga::ScalarKind::Sint,
            width: 4,
        } => wgpu::VertexFormat::Sint32x3,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Sint,
            width: 4,
        } => wgpu::VertexFormat::Sint32x4,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width: 4,
        } => wgpu::VertexFormat::Float32x2,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Tri,
            kind: naga::ScalarKind::Float,
            width: 4,
        } => wgpu::VertexFormat::Float32x3,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width: 4,
        } => wgpu::VertexFormat::Float32x4,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width: 8,
        } => wgpu::VertexFormat::Float64x2,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Tri,
            kind: naga::ScalarKind::Float,
            width: 8,
        } => wgpu::VertexFormat::Float64x3,
        naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width: 8,
        } => wgpu::VertexFormat::Float64x4,
        _ => panic!("Unsupported vertex attribute type!"),
    }
}

fn scan_expression(
    expression: &naga::Expression,
    global_stages: &mut FxHashMap<naga::Handle<naga::GlobalVariable>, wgpu::ShaderStages>,
    current_stage: wgpu::ShaderStages,
    function: &naga::Function,
) {
    macro_rules! scan {
        ($($expr:expr),*) => {{ $(scan_expression(&function.expressions[$expr.clone()], global_stages, current_stage, function));* }}
    }
    match expression {
        naga::Expression::GlobalVariable(handle) => {
            (*global_stages
                .entry(*handle)
                .or_insert(wgpu::ShaderStages::empty())) |= current_stage
        }
        naga::Expression::Constant(_)
        | naga::Expression::FunctionArgument(_)
        | naga::Expression::LocalVariable(_)
        | naga::Expression::AtomicResult { .. } => {}
        naga::Expression::Access { base, index } => scan!(base, index),
        naga::Expression::AccessIndex { base, .. } => scan!(base),
        naga::Expression::Splat { value, .. } => scan!(value),
        naga::Expression::Swizzle { vector, .. } => scan!(vector),
        naga::Expression::Compose { components, .. } => components.iter().for_each(|c| scan!(c)),
        naga::Expression::Load { pointer } => scan!(pointer),
        naga::Expression::ImageQuery { image, .. } => scan!(image),
        naga::Expression::Unary { expr, .. } => scan!(expr),
        naga::Expression::Binary { left, right, .. } => scan!(left, right),
        naga::Expression::Derivative { expr, .. } => scan!(expr),
        naga::Expression::Relational { argument, .. } => scan!(argument),
        naga::Expression::As { expr, .. } => scan!(expr),
        naga::Expression::ArrayLength(expr) => scan!(expr),
        naga::Expression::ImageSample {
            image,
            sampler,
            coordinate,
            array_index,
            depth_ref,
            ..
        } => {
            scan!(image, sampler, coordinate);
            array_index.map(|idx| scan!(idx));
            depth_ref.map(|idx| scan!(idx));
        }
        naga::Expression::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            scan!(image, coordinate);
            array_index.map(|idx| scan!(idx));
            sample.map(|idx| scan!(idx));
            level.map(|idx| scan!(idx));
        }
        naga::Expression::Select {
            condition,
            accept,
            reject,
        } => scan!(condition, accept, reject),
        naga::Expression::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            scan!(arg);
            arg1.map(|idx| scan!(idx));
            arg2.map(|idx| scan!(idx));
            arg3.map(|idx| scan!(idx));
        }
        naga::Expression::CallResult(_) => todo!(),
    }
}
