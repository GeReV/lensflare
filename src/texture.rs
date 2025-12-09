use wgpu::{Texture, TextureView, TextureViewDescriptor};

pub trait TextureExt {
    fn create_view_default(&self) -> TextureView;
}

impl TextureExt for Texture {
    fn create_view_default(&self) -> TextureView {
        self.create_view(&TextureViewDescriptor::default())
    }
}