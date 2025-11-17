use std::borrow::Cow;
use std::path::Path;
use anyhow::Error;
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor};
use wgpu::{Device, Label, ShaderModule, ShaderModuleDescriptor, ShaderSource};

pub fn init_composer() -> anyhow::Result<Composer> {
    let mut composer = Composer::default();

    let mut load_composable = |file_path: &str| -> anyhow::Result<()> {
        let source = std::fs::read_to_string(file_path)?;

        match composer.add_composable_module(ComposableModuleDescriptor {
            source: &source,
            file_path,
            ..Default::default()
        }) {
            Ok(_) => Ok(()),
            Err(err) => {
                return Err(err.into());
            }
        }
    };

    load_composable("src/shaders/utils.wgsl")?;
    load_composable("src/shaders/colors.wgsl")?;
    load_composable("src/shaders/camera.wgsl")?;
    load_composable("src/shaders/lenses.wgsl")?;

    Ok(composer)
}

pub fn create_shader<'a>(
    device: &Device,
    composer: &mut Composer,
    label: Label,
    path: &Path,
) -> anyhow::Result<ShaderModule> {
    let shader_source = std::fs::read_to_string(path)?;

    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let label = label.unwrap_or("Default Render Pipeline");

    let module = composer.make_naga_module(NagaModuleDescriptor {
        source: &shader_source,
        file_path: path.to_str().unwrap(),
        ..Default::default()
    }).map_err(|err| {
        println!("{err:?}");
        Error::new(err)
    })?;

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(&format!("{label} Shader")),
        source: ShaderSource::Naga(Cow::Owned(module)),
    });

    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        return Err(Error::new(error));
    }

    anyhow::Ok(shader)
}