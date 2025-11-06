use anyhow::format_err;
use encase::internal::WriteInto;
use encase::{ShaderType, UniformBuffer};
use wgpu::util::DeviceExt;

pub(crate) enum BufferType {
    Vertex,
    Index,
    Uniform,
    Storage,
}

pub(crate) struct Buffer<T> {
    pub(crate) name: String,
    pub data: T,
    buffer: wgpu::Buffer,
}

impl<T> Buffer<T> {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl<T> Buffer<T>
where
    T: ShaderType + WriteInto,
{
    pub fn new(
        device: &wgpu::Device,
        name: impl Into<String>,
        usage: BufferType,
        data: T,
    ) -> anyhow::Result<Self> {
        let usage = match usage {
            BufferType::Vertex => wgpu::BufferUsages::VERTEX,
            BufferType::Index => wgpu::BufferUsages::INDEX,
            BufferType::Uniform => wgpu::BufferUsages::UNIFORM,
            BufferType::Storage => wgpu::BufferUsages::STORAGE,
        };

        let name = name.into();

        let uniform_buffer: Vec<u8> = UniformBuffer::<T>::content_of(&data)
            .map_err(|e| format_err!("Failed to create uniform buffer: {}", e))?;

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{name} Buffer")),
            usage: usage | wgpu::BufferUsages::COPY_DST,
            contents: &uniform_buffer,
        });

        Ok(Self { name, data, buffer })
    }

    pub fn write_buffer(&self, queue: &wgpu::Queue) -> anyhow::Result<()> {
        let uniform_buffer: Vec<u8> = UniformBuffer::<T>::content_of(&self.data)
            .map_err(|e| format_err!("Failed to create uniform buffer: {}", e))?;

        queue.write_buffer(&self.buffer, 0, &uniform_buffer);

        Ok(())
    }
}
