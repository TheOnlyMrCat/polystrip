#![allow(dead_code)]

#[cfg(not(any(
	feature = "gfx-backend-vulkan",
	feature = "gfx-backend-metal",
	feature = "gfx-backend-dx12",
	feature = "gfx-backend-empty"
)))]
compile_error!("Need to select a backend! Enable one of the [\"vulkan\", \"metal\", \"dx12\"] features");

#[cfg(feature = "gfx-backend-vulkan")]
pub type Backend = gfx_backend_vulkan::Backend;

#[cfg(feature = "gfx-backend-metal")]
pub type Backend = gfx_backend_metal::Backend;

#[cfg(feature = "gfx-backend-dx12")]
pub type Backend = gfx_backend_dx12::Backend;

#[cfg(feature = "gfx-backend-empty")]
pub type Backend = gfx_backend_empty::Backend;

pub type Instance = <Backend as gfx_hal::Backend>::Instance;
pub type PhysicalDevice = <Backend as gfx_hal::Backend>::PhysicalDevice;
pub type Device = <Backend as gfx_hal::Backend>::Device;
pub type Surface = <Backend as gfx_hal::Backend>::Surface;

pub type QueueFamily = <Backend as gfx_hal::Backend>::QueueFamily;
pub type CommandQueue = <Backend as gfx_hal::Backend>::CommandQueue;
pub type CommandBuffer = <Backend as gfx_hal::Backend>::CommandBuffer;

pub type ShaderModule = <Backend as gfx_hal::Backend>::ShaderModule;
pub type RenderPass = <Backend as gfx_hal::Backend>::RenderPass;
pub type Framebuffer = <Backend as gfx_hal::Backend>::Framebuffer;

pub type Memory = <Backend as gfx_hal::Backend>::Memory;
pub type CommandPool = <Backend as gfx_hal::Backend>::CommandPool;

pub type Buffer = <Backend as gfx_hal::Backend>::Buffer;
pub type BufferView = <Backend as gfx_hal::Backend>::BufferView;
pub type Image = <Backend as gfx_hal::Backend>::Image;
pub type ImageView = <Backend as gfx_hal::Backend>::ImageView;
pub type Sampler = <Backend as gfx_hal::Backend>::Sampler;

pub type ComputePipeline = <Backend as gfx_hal::Backend>::ComputePipeline;
pub type GraphicsPipeline = <Backend as gfx_hal::Backend>::GraphicsPipeline;
pub type PipelineCache = <Backend as gfx_hal::Backend>::PipelineCache;
pub type PipelineLayout = <Backend as gfx_hal::Backend>::PipelineLayout;
pub type DescriptorPool = <Backend as gfx_hal::Backend>::DescriptorPool;
pub type DescriptorSet = <Backend as gfx_hal::Backend>::DescriptorSet;
pub type DescriptorSetLayout = <Backend as gfx_hal::Backend>::DescriptorSetLayout;

pub type Fence = <Backend as gfx_hal::Backend>::Fence;
pub type Semaphore = <Backend as gfx_hal::Backend>::Semaphore;
pub type Event = <Backend as gfx_hal::Backend>::Event;
pub type QueryPool = <Backend as gfx_hal::Backend>::QueryPool;

pub type SwapchainImage = <Surface as gfx_hal::window::PresentationSurface<Backend>>::SwapchainImage;
