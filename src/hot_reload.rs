use crate::shaders::create_shader;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use wesl::{StandardResolver, Wesl};
use wgpu::{Device, ShaderModule};

pub enum HotReloadResult {
    Updated(ShaderModule),
    Unchanged,
}

pub struct HotReloadShader {
    pub path: PathBuf,
    pub shader_last_modification: SystemTime,
    pub shader_last_compilation_check: SystemTime,
    pub shader_last_error: Option<String>,
}

impl HotReloadShader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            shader_last_modification: SystemTime::UNIX_EPOCH,
            shader_last_compilation_check: SystemTime::UNIX_EPOCH,
            shader_last_error: None,
        }
    }

    pub fn try_hot_reload(&mut self, device: &Device, compiler: &mut Wesl<StandardResolver>) -> anyhow::Result<HotReloadResult> {
        if self.shader_last_compilation_check.elapsed()? <= Duration::from_secs(1) {
            return Ok(HotReloadResult::Unchanged);
        }

        self.shader_last_compilation_check = SystemTime::now();

        let shader_metadata = std::fs::metadata(&self.path)?;
        let modified = shader_metadata.modified()?;

        if modified > self.shader_last_modification {
            self.shader_last_modification = SystemTime::now();

            let name = self.path.file_prefix().unwrap().to_str().unwrap();

            return match create_shader(device, compiler, Some(&format!("Shader {}", self.path.display())), &format!("package::{name}")) {
                Ok(shader) => {
                    self.shader_last_error = None;

                    Ok(HotReloadResult::Updated(shader))
                },
                Err(err) => {
                    self.shader_last_error = Some(err.to_string());

                    Err(err)
                }
            }

        }

        Ok(HotReloadResult::Unchanged)
    }
}