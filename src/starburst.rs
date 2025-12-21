use crate::fft::wgpu::{ComputeFftPipeline, CopyToComplexPipeline, TextureMultiplyConstPipeline};
use crate::texture::TextureExt;
use crate::utils::wavelengths_to_colors;
use itertools::Itertools;
use wesl::{StandardResolver, Wesl};
use wgpu::util::{BufferInitDescriptor, DeviceExt, TextureBlitter};
use wgpu::wgt::TextureDescriptor;
use wgpu::{
    BufferUsages
    , CommandEncoderDescriptor, Device, Extent3d


    , Queue
    , Texture, TextureDimension,
    TextureFormat, TextureUsages
    ,
};
use crate::starburst_pipeline::StarburstPipeline;

pub fn create_starburst_gpu(
    device: &Device,
    queue: &Queue,
    compiler: &mut Wesl<StandardResolver>,
    aperture_texture: &Texture,
    dust_texture: &Texture,
    dst_texture: &Texture,
    wavelengths: &[f32],
) -> anyhow::Result<()> {
    let width = aperture_texture.width();
    let height = aperture_texture.height();

    let size = width as usize;

    let texture_view = dst_texture.create_view_default();

    let min_storage_buffer_offset_alignment =
        device.limits().min_storage_buffer_offset_alignment as usize;

    let colors = wavelengths_to_colors(wavelengths)
        .iter()
        .flat_map(|&color| {
            let mut bytes = vec![0u8; min_storage_buffer_offset_alignment];

            bytes[..size_of_val(&color)].copy_from_slice(bytemuck::cast_slice(&[color]));

            bytes
        })
        .collect::<Vec<_>>();

    let colors_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Starburst Colors Buffer"),
        contents: &colors,
        usage: BufferUsages::STORAGE,
    });

    let temp_textures_rg32float = (0..2)
        .map(|i| {
            device.create_texture(&TextureDescriptor {
                label: Some(&format!("Starburst Temporary Rg32Float Texture {i}")),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                format: TextureFormat::Rg32Float,
                dimension: TextureDimension::D2,
                sample_count: 1,
                mip_level_count: 1,
                view_formats: &[],
            })
        })
        .collect_vec()
        .into_boxed_slice();

    let temp_texture_rg32float_views = temp_textures_rg32float
        .iter()
        .map(|tex| tex.create_view_default())
        .collect_vec()
        .into_boxed_slice();

    let temp_texture_rg16float = (0..2)
        .map(|i| {
            device.create_texture(&TextureDescriptor {
                label: Some("Starburst FFT Temporary R16Float Texture"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rg16Float,
                sample_count: 1,
                mip_level_count: 1,
                view_formats: &[],
            })
        })
        .collect_vec()
        .into_boxed_slice();

    let temp_texture_rg16float_view = temp_texture_rg16float
        .iter()
        .map(|tex| tex.create_view_default())
        .collect_vec()
        .into_boxed_slice();

    let temp_texture = device.create_texture(&TextureDescriptor {
        label: Some("Starburst Prefilter Texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        usage: dst_texture.usage(),
        format: dst_texture.format(),
        dimension: TextureDimension::D2,
        sample_count: 1,
        mip_level_count: 1,
        view_formats: &[],
    });
    let temp_texture_view = temp_texture.create_view_default();

    let copy_to_complex_pipeline = CopyToComplexPipeline::new(device, compiler, size)?;

    let fft_pipeline = ComputeFftPipeline::new(device, compiler, size)?;

    let texture_multiply_const_pipeline = TextureMultiplyConstPipeline::new(
        device,
        compiler,
        TextureFormat::Rg32Float,
        (size * size) as f32,
    )?;

    let dust_texture_multiply_const_pipeline = TextureMultiplyConstPipeline::new(
        device,
        compiler,
        TextureFormat::Rg32Float,
        (dust_texture.width() * dust_texture.height()) as f32,
    )?;

    let starburst_pipeline = StarburstPipeline::new(device, compiler)?;

    let texture_blitter = TextureBlitter::new(device, TextureFormat::Rg16Float);

    let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Starburst Command Encoder"),
    });

    // Aperture texture
    {
        // Copy the aperture texture to the source buffer as complex numbers.
        copy_to_complex_pipeline.copy(
            device,
            &mut command_encoder,
            &aperture_texture.create_view_default(),
            &temp_texture_rg32float_views[0],
        );

        fft_pipeline.process_fft2d(
            device,
            &mut command_encoder,
            &temp_texture_rg32float_views[0],
            &temp_texture_rg32float_views[1],
        );

        texture_multiply_const_pipeline.process(
            device,
            &mut command_encoder,
            &temp_texture_rg32float_views[1],
            &temp_texture_rg32float_views[0],
        );

        texture_blitter.copy(
            device,
            &mut command_encoder,
            &temp_texture_rg32float_views[0],
            &temp_texture_rg16float_view[0],
        );
    }

    // Dust texture
    {
        copy_to_complex_pipeline.copy(
            device,
            &mut command_encoder,
            &dust_texture.create_view_default(),
            &temp_texture_rg32float_views[0],
        );

        fft_pipeline.process_fft2d(
            device,
            &mut command_encoder,
            &temp_texture_rg32float_views[0],
            &temp_texture_rg32float_views[1],
        );

        texture_multiply_const_pipeline.process(
            device,
            &mut command_encoder,
            &temp_texture_rg32float_views[1],
            &temp_texture_rg32float_views[0],
        );

        texture_blitter.copy(
            device,
            &mut command_encoder,
            &temp_texture_rg32float_views[0],
            &temp_texture_rg16float_view[1],
        );
    }

    starburst_pipeline.render(
        device,
        &mut command_encoder,
        &temp_texture_rg16float_view[0],
        &temp_texture_rg16float_view[1],
        &temp_texture_view,
        &colors_buffer,
    );

    starburst_pipeline.filter(
        device,
        &mut command_encoder,
        &temp_texture_view,
        &texture_view,
        &colors_buffer,
    );

    queue.submit(Some(command_encoder.finish()));

    Ok(())
}

