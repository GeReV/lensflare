use crate::shaders::create_shader;
use anyhow::anyhow;
use wesl::{Feature, StandardResolver, Wesl};
use wgpu::wgt::{TextureDescriptor, TextureViewDescriptor};
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType,
    Buffer, BufferAddress, BufferBinding, BufferBindingType, BufferSlice, CommandEncoder,
    ComputePass, ComputePipeline, ComputePipelineDescriptor, Device, Extent3d,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderStages, StorageTextureAccess,
    Texture, TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDimension,
};
use crate::texture::TextureExt;

const SRC_DST_STORAGE_TEXTURES_BIND_GROUP_LAYOUT_ENTRIES: [BindGroupLayoutEntry; 2] = [
    BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            format: TextureFormat::Rg32Float,
            access: StorageTextureAccess::ReadOnly,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    },
    BindGroupLayoutEntry {
        binding: 1,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            format: TextureFormat::Rg32Float,
            view_dimension: TextureViewDimension::D2,
            access: StorageTextureAccess::WriteOnly,
        },
        count: None,
    },
];

pub(crate) struct ComputeFftPipeline {
    fft_rows_pipeline: ComputePipeline,
    fft_cols_pipeline: ComputePipeline,

    ifft_rows_pipeline: ComputePipeline,
    ifft_cols_pipeline: ComputePipeline,

    fftshift_rows_pipeline: ComputePipeline,
    fftshift_cols_pipeline: ComputePipeline,

    texture_multiply_const_pipeline: TextureMultiplyConstPipeline,

    bind_group_layout: BindGroupLayout,

    staging_texture: Texture,
    staging_texture_view: TextureView,

    fft_size: usize,
}

impl ComputeFftPipeline {
    pub fn new(
        device: &Device,
        compiler: &mut Wesl<StandardResolver>,
        fft_size: usize,
    ) -> anyhow::Result<Self> {
        assert!(fft_size.is_power_of_two());

        {
            let limits = device.limits();

            debug_assert!(
                fft_size <= limits.max_compute_workgroup_size_x as usize,
                "FFT size is too large for running in a single compute shader"
            );
        }

        compiler.add_constants([
            ("workgroup_size_x", fft_size as f64),
            ("workgroup_size_y", 1.0),
            ("workgroup_size_z", 1.0),
            ("size", fft_size as f64),
        ]);

        compiler.set_feature("fft_cols", Feature::Disable);

        let module_rows = create_shader(
            device,
            compiler,
            Some("FFT Rows Compute Shader"),
            "package::fft",
        )?;

        compiler.set_feature("fft_cols", Feature::Enable);

        let module_cols = create_shader(
            device,
            compiler,
            Some("FFT Cols Compute Shader"),
            "package::fft",
        )?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &SRC_DST_STORAGE_TEXTURES_BIND_GROUP_LAYOUT_ENTRIES,
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("FFT Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let fft_rows_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("FFT Rows Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_rows,
            entry_point: Some("fft"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let fft_cols_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("FFT Cols Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_cols,
            entry_point: Some("fft"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let ifft_rows_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("IFFT Rows Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_rows,
            entry_point: Some("ifft"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let ifft_cols_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("IFFT Cols Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_cols,
            entry_point: Some("ifft"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let fftshift_rows_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("FFT Shift Rows Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_rows,
            entry_point: Some("fftshift"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let fftshift_cols_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("FFT Shift Cols Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_cols,
            entry_point: Some("fftshift"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let staging_texture = device.create_texture(&TextureDescriptor {
            label: Some("FFT Staging Texture"),
            size: Extent3d {
                width: fft_size as u32,
                height: fft_size as u32,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::Rg32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            dimension: TextureDimension::D2,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        let staging_texture_view = staging_texture.create_view_default();

        let texture_multiply_const_pipeline = TextureMultiplyConstPipeline::new(
            device,
            compiler,
            staging_texture.format(),
            1.0 / (fft_size * fft_size) as f32,
        )?;

        Ok(Self {
            fft_rows_pipeline,
            fft_cols_pipeline,
            ifft_rows_pipeline,
            ifft_cols_pipeline,
            fftshift_rows_pipeline,
            fftshift_cols_pipeline,
            texture_multiply_const_pipeline,
            bind_group_layout,
            staging_texture,
            staging_texture_view,
            fft_size,
        })
    }

    pub fn process_fft_rows(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
    ) {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FFT Rows Compute Pass"),
            timestamp_writes: None,
        });

        self.process(device, &self.fft_rows_pipeline, &mut compute_pass, src, dst);
    }

    fn process_fft_cols(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
    ) {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FFT Cols Compute Pass"),
            timestamp_writes: None,
        });

        self.process(device, &self.fft_cols_pipeline, &mut compute_pass, src, dst);
    }

    pub fn process_fft2d(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
    ) {
        self.process_fft_rows(device, command_encoder, src, dst);
        self.process_fft_cols(device, command_encoder, dst, &self.staging_texture_view);

        self.process_fftshift_rows(device, command_encoder, &self.staging_texture_view, dst);
        self.process_fftshift_cols(device, command_encoder, dst, &self.staging_texture_view);

        self.texture_multiply_const_pipeline.process(
            device,
            command_encoder,
            &self.staging_texture_view,
            dst,
        );
    }

    pub fn process_ifft_rows(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src_texture: &TextureView,
        dst_texture: &TextureView,
    ) {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("IFFT Rows Compute Pass"),
            timestamp_writes: None,
        });

        self.process(
            device,
            &self.ifft_rows_pipeline,
            &mut compute_pass,
            src_texture,
            dst_texture,
        );
    }

    fn process_ifft_cols(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
    ) {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("IFFT Cols Compute Pass"),
            timestamp_writes: None,
        });

        self.process(
            device,
            &self.ifft_cols_pipeline,
            &mut compute_pass,
            src,
            dst,
        );
    }

    pub fn process_ifft2d(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
    ) {
        self.process_ifft_rows(device, command_encoder, src, &self.staging_texture_view);
        self.process_ifft_cols(device, command_encoder, &self.staging_texture_view, dst);
    }

    pub fn process_fftshift_rows(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
    ) {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FFT Shift Rows Compute Pass"),
            timestamp_writes: None,
        });

        self.process(
            device,
            &self.fftshift_rows_pipeline,
            &mut compute_pass,
            src,
            dst,
        );
    }

    fn process_fftshift_cols(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
    ) {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FFT Shift Cols Compute Pass"),
            timestamp_writes: None,
        });

        self.process(
            device,
            &self.fftshift_cols_pipeline,
            &mut compute_pass,
            src,
            dst,
        );
    }

    fn process(
        &self,
        device: &Device,
        pipeline: &ComputePipeline,
        compute_pass: &mut ComputePass,
        src: &TextureView,
        dst: &TextureView,
    ) {
        debug_assert!(
            src.texture()
                .usage()
                .contains(TextureUsages::TEXTURE_BINDING),
            "Source texture must be TEXTURE_BINDING usage"
        );

        debug_assert!(
            dst.texture()
                .usage()
                .contains(TextureUsages::STORAGE_BINDING),
            "Destination texture must be STORAGE_BINDING usage"
        );

        compute_pass.set_pipeline(pipeline);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(dst),
                },
            ],
        });

        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(1, self.fft_size.max(1) as u32, 1);
    }
}

pub struct CopyToComplexPipeline {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,

    size: usize,
}

impl CopyToComplexPipeline {
    const WORKGROUP_SIZE: usize = 32;

    pub fn new(
        device: &Device,
        compiler: &mut Wesl<StandardResolver>,
        size: usize,
    ) -> anyhow::Result<Self> {
        compiler.add_constants([
            ("workgroup_size_x", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_y", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_z", 1.0),
            ("size", size as f64),
        ]);

        let module = create_shader(
            device,
            compiler,
            Some("Copy To Complex Compute Shader"),
            "package::copy_to_complex",
        )?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Copy To Complex Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                SRC_DST_STORAGE_TEXTURES_BIND_GROUP_LAYOUT_ENTRIES[1],
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Copy To Complex Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Copy To Complex Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("copy"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            size,
        })
    }

    pub fn copy(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
    ) {
        debug_assert!(
            src.texture()
                .usage()
                .contains(TextureUsages::TEXTURE_BINDING),
            "Source texture must be TEXTURE_BINDING usage"
        );

        debug_assert!(
            dst.texture()
                .usage()
                .contains(TextureUsages::STORAGE_BINDING),
            "Destination texture must be STORAGE_BINDING usage"
        );

        let invocation_count = (self.size / Self::WORKGROUP_SIZE) as u32;

        debug_assert!(
            invocation_count <= device.limits().max_compute_invocations_per_workgroup,
            "Too many invocations to run in a single compute shader"
        );

        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Copy To Complex Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Copy To Complex Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(dst),
                },
            ],
        });

        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(invocation_count, invocation_count, 1);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GenerateFrequenciesParameters {
    pub delta: f32,
    pub z: f32,
    pub wavelength_meters: f32,
}

pub struct GenerateFrequenciesPipeline {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    pub size: usize,
}

impl GenerateFrequenciesPipeline {
    const WORKGROUP_SIZE: usize = 32;

    pub fn new(
        device: &Device,
        compiler: &mut Wesl<StandardResolver>,
        size: usize,
    ) -> anyhow::Result<Self> {
        compiler.add_constants([
            ("workgroup_size_x", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_y", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_z", 1.0),
            ("size", size as f64),
        ]);

        let module = create_shader(
            device,
            compiler,
            Some("Generate Frequencies Compute Shader"),
            "package::generate_frequencies",
        )?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Generate Frequencies Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::Rg32Float,
                        access: StorageTextureAccess::WriteOnly,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Generate Frequencies Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Generate Frequencies Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("generate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            size,
        })
    }

    pub fn generate(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        dst: &TextureView,
        parameters: &Buffer,
        parameters_offset: u32,
    ) {
        debug_assert!(
            dst.texture()
                .usage()
                .contains(TextureUsages::STORAGE_BINDING),
            "Destination texture must be STORAGE_BINDING usage"
        );

        let invocation_count = (self.size / Self::WORKGROUP_SIZE) as u32;

        debug_assert!(
            invocation_count <= device.limits().max_compute_invocations_per_workgroup,
            "Too many invocations to run in a single compute shader"
        );

        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Generate Frequencies Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Generate Frequencies Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(dst),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: parameters,
                        offset: parameters_offset as BufferAddress,
                        size: None,
                    }),
                },
            ],
        });

        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(invocation_count, invocation_count, 1);
    }
}

pub struct MultiplyComplexPipeline {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    pub size: usize,
}

impl MultiplyComplexPipeline {
    const WORKGROUP_SIZE: usize = 32;

    pub fn new(
        device: &Device,
        compiler: &mut Wesl<StandardResolver>,
        size: usize,
    ) -> anyhow::Result<Self> {
        compiler.add_constants([
            ("workgroup_size_x", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_y", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_z", 1.0),
            ("size", size as f64),
        ]);

        let module = create_shader(
            device,
            compiler,
            Some("Multiply Complex Compute Shader"),
            "package::multiply_complex",
        )?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Multiply Complex Bind Group Layout"),
            entries: &[
                SRC_DST_STORAGE_TEXTURES_BIND_GROUP_LAYOUT_ENTRIES[0],
                SRC_DST_STORAGE_TEXTURES_BIND_GROUP_LAYOUT_ENTRIES[1],
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::Rg32Float,
                        access: StorageTextureAccess::ReadOnly,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Multiply Complex Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Multiply Complex Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("multiply"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            size,
        })
    }

    pub fn multiply(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        coefficients: &TextureView,
        dst: &TextureView,
    ) {
        let src_tex = src.texture();

        debug_assert!(
            src_tex.usage().contains(TextureUsages::STORAGE_BINDING),
            "Source texture must be STORAGE_BINDING usage"
        );

        debug_assert!(
            dst.texture()
                .usage()
                .contains(TextureUsages::STORAGE_BINDING),
            "Destination texture must be STORAGE_BINDING usage"
        );

        let coefficients_tex = coefficients.texture();

        debug_assert!(
            coefficients_tex
                .usage()
                .contains(TextureUsages::STORAGE_BINDING),
            "Coefficients buffer must be STORAGE usage"
        );

        debug_assert!(
            (coefficients_tex.width() * coefficients_tex.height())
                == (src_tex.width() * src_tex.height()),
            "Coefficients texture must be the same size as the source texture"
        );

        let invocation_count = (self.size / Self::WORKGROUP_SIZE) as u32;

        debug_assert!(
            invocation_count <= device.limits().max_compute_invocations_per_workgroup,
            "Too many invocations to run in a single compute shader"
        );

        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Multiply Complex Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Multiply Complex Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(dst),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(coefficients),
                },
            ],
        });

        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(invocation_count, invocation_count, 1);
    }
}

pub struct ComplexNormalizePipeline {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl ComplexNormalizePipeline {
    const WORKGROUP_SIZE: usize = 32;

    pub fn new(device: &Device, compiler: &mut Wesl<StandardResolver>) -> anyhow::Result<Self> {
        compiler.add_constants([
            ("workgroup_size_x", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_y", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_z", 1.0),
        ]);

        let module = create_shader(
            device,
            compiler,
            Some("Complex Normalize Compute Shader"),
            "package::complex_normalize",
        )?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Complex Normalize Bind Group Layout"),
            entries: &[
                SRC_DST_STORAGE_TEXTURES_BIND_GROUP_LAYOUT_ENTRIES[0],
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::R32Float,
                        access: StorageTextureAccess::WriteOnly,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Complex Normalize Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Complex Normalize Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("normalize"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    pub fn normalize(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
    ) {
        debug_assert!(
            src.texture()
                .usage()
                .contains(TextureUsages::STORAGE_BINDING),
            "Source texture must be STORAGE_BINDING usage"
        );

        debug_assert!(
            dst.texture()
                .usage()
                .contains(TextureUsages::STORAGE_BINDING),
            "Destination texture must be STORAGE_BINDING usage"
        );

        let invocation_count = (src.texture().width() as usize / Self::WORKGROUP_SIZE) as u32;

        debug_assert!(
            invocation_count <= device.limits().max_compute_invocations_per_workgroup,
            "Too many invocations to run in a single compute shader"
        );

        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Complex Normalize Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Complex Normalize Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(dst),
                },
            ],
        });

        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(invocation_count, invocation_count, 1);
    }
}

pub struct TextureMultiplyConstPipeline {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl TextureMultiplyConstPipeline {
    const WORKGROUP_SIZE: usize = 32;

    pub fn new(
        device: &Device,
        compiler: &mut Wesl<StandardResolver>,
        texture_format: TextureFormat,
        constant: f32,
    ) -> anyhow::Result<Self> {
        compiler.add_constants([
            ("workgroup_size_x", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_y", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_z", 1.0),
            ("C", constant as f64),
        ]);

        // TODO: Support other texture formats?
        match texture_format {
            TextureFormat::Rg32Float => {
                compiler.set_feature("rg32", Feature::Enable);
            }
            TextureFormat::Rgba32Float => {
                compiler.set_feature("rg32", Feature::Disable);
            }
            _ => {
                return Err(anyhow!("Unsupported texture format: {:?}", texture_format));
            }
        }

        let module = create_shader(
            device,
            compiler,
            Some("Texture Multiply Const Compute Shader"),
            "package::texture_multiply_const",
        )?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Multiply Const Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        format: texture_format,
                        view_dimension: TextureViewDimension::D2,
                        access: StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        format: texture_format,
                        view_dimension: TextureViewDimension::D2,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Texture Multiply Const Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Texture Multiply Const Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("process"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    pub fn process(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
    ) {
        debug_assert!(
            src.texture()
                .usage()
                .contains(TextureUsages::STORAGE_BINDING),
            "Texture A must be STORAGE_BINDING usage"
        );

        debug_assert!(
            dst.texture()
                .usage()
                .contains(TextureUsages::STORAGE_BINDING),
            "Destination texture must be STORAGE_BINDING usage"
        );

        let invocation_count = (src.texture().width() as usize / Self::WORKGROUP_SIZE) as u32;

        debug_assert!(
            invocation_count <= device.limits().max_compute_invocations_per_workgroup,
            "Too many invocations to run in a single compute shader"
        );

        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Texture Multiply Const Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Texture Multiply Const Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(dst),
                },
            ],
        });

        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(invocation_count, invocation_count, 1);
    }
}

pub struct TextureMultiplyAddPipeline {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl TextureMultiplyAddPipeline {
    const WORKGROUP_SIZE: usize = 32;

    pub fn new(device: &Device, compiler: &mut Wesl<StandardResolver>) -> anyhow::Result<Self> {
        compiler.add_constants([
            ("workgroup_size_x", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_y", Self::WORKGROUP_SIZE as f64),
            ("workgroup_size_z", 1.0),
        ]);

        let module = create_shader(
            device,
            compiler,
            Some("Texture Multiply-Add Compute Shader"),
            "package::texture_multiply_add",
        )?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Multiply-Add Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                        access: StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                        access: StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                        access: StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Texture Multiply-Add Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Texture Multiply-Add Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("process"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    pub fn process(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        a: &TextureView,
        b: &TextureView,
        dst: &TextureView,
        colors: &BufferSlice,
    ) {
        debug_assert!(
            a.texture().usage().contains(TextureUsages::STORAGE_BINDING),
            "Texture A must be STORAGE_BINDING usage"
        );

        debug_assert!(
            b.texture().usage().contains(TextureUsages::STORAGE_BINDING),
            "Texture B must be STORAGE_BINDING usage"
        );

        debug_assert!(
            dst.texture()
                .usage()
                .contains(TextureUsages::STORAGE_BINDING),
            "Destination texture must be STORAGE_BINDING usage"
        );

        let invocation_count = (a.texture().width() as usize / Self::WORKGROUP_SIZE) as u32;

        debug_assert!(
            invocation_count <= device.limits().max_compute_invocations_per_workgroup,
            "Too many invocations to run in a single compute shader"
        );

        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Texture Multiply-Add Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Texture Multiply-Add Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(b),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(dst),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: colors.buffer(),
                        offset: colors.offset(),
                        size: None,
                    }),
                },
            ],
        });

        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(invocation_count, invocation_count, 1);
    }
}