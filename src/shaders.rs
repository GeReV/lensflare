use std::borrow::Cow;
use anyhow::Error;
use wesl::{StandardResolver, Wesl};
use wgpu::{Device, Label, ShaderModule, ShaderModuleDescriptor, ShaderSource};

pub fn create_shader(
    device: &Device,
    compiler: &mut Wesl<StandardResolver>,
    label: Label,
    name: &str,
) -> anyhow::Result<ShaderModule> {
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let label = label.unwrap_or("Default Render Pipeline");

    let compile_result = compiler.compile(&name.parse()?)?;

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(&format!("{label} Shader")),
        source: ShaderSource::Wgsl(Cow::Owned(compile_result.to_string())),
    });

    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        return Err(Error::new(error));
    }

    anyhow::Ok(shader)
}