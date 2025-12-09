mod arc;
mod camera;
mod grids;
mod hot_reload;
mod lenses;
mod registry;
mod shaders;
mod software;
mod uniforms;
mod vertex;
mod state;
mod app;
mod utils;
mod ghost;
mod fft;
mod colors;
mod texture;

use anyhow::*;
use winit::event_loop::EventLoop;
use app::App;

pub fn run() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::with_user_event().build()?;

    let mut app = App::new();

    event_loop.run_app(&mut app)?;

    Ok(())
}
