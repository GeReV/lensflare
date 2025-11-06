use crate::buffer::Buffer;

pub struct BindGroup<T> {
    buffer: Buffer<T>,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl<T> BindGroup<T> {
    pub fn new(device: &wgpu::Device, visibility: wgpu::ShaderStages, buffer: Buffer<T>) -> Self {
        let buffer_usage = &buffer.buffer().usage();
        let ty = if buffer_usage.contains(wgpu::BufferUsages::STORAGE) {
            wgpu::BufferBindingType::Storage {
                // TODO: Also wgpu::BufferUsages::MAP_WRITE?
                read_only: buffer_usage.contains(wgpu::BufferUsages::COPY_DST),
            }
        } else {
            wgpu::BufferBindingType::Uniform
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} Bind Group Layout", buffer.name)),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", buffer.name)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.buffer().as_entire_binding(),
                }
            ],
        });

        Self {
            buffer,
            bind_group,
            bind_group_layout,
        }
    }

    pub fn buffer(&self) -> &Buffer<T> {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut Buffer<T> {
        &mut self.buffer
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}