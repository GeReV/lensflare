use std::f32::consts::PI;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, Event, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use wgpu::SurfaceError;
use glam::Vec2;
use crate::state::State;

pub struct App {
    state: Option<State>,
}

impl App {
    pub(crate) fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes =
            Window::default_attributes().with_inner_size(LogicalSize::new(1920, 1080));

        let window = std::sync::Arc::new(event_loop.create_window(window_attributes).unwrap());

        self.state = Some(pollster::block_on(State::new(window)).unwrap());
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        let imgui = &mut event.imgui;

        imgui.platform.handle_event::<()>(
            imgui.context.io_mut(),
            &event.window,
            &Event::UserEvent(()),
        );

        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match &event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now - state.last_frame;

                state.last_frame = now;

                state.update(dt);

                if let Err(err) = state.render() {
                    match err.downcast_ref::<SurfaceError>() {
                        Some(SurfaceError::Lost | SurfaceError::Outdated) => {
                            let size = state.window.inner_size();
                            state.resize(size.width, size.height);
                        }
                        _ => {
                            log::error!("Unable to render: {:?}", err);
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                if !state.imgui.context.io().want_capture_mouse {
                    state.handle_mouse_button(*button, btn_state.is_pressed())
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !state.imgui.context.io().want_capture_mouse {
                    state.handle_mouse_scroll(&delta);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Mouse cursor position handling - not needed for camera control
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                if !state.imgui.context.io().want_capture_keyboard {
                    state.handle_key(event_loop, *code, key_state.is_pressed())
                }
            }
            _ => {}
        }

        let imgui = &mut state.imgui;

        imgui.platform.handle_event::<State>(
            imgui.context.io_mut(),
            &state.window,
            &Event::WindowEvent { window_id, event },
        );
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let state = if let Some(state) = &mut self.state {
            state
        } else {
            return;
        };

        match event {
            DeviceEvent::MouseMotion { delta } => {
                if !state.imgui.context.io().want_capture_mouse {
                    if state.mouse_left_pressed {
                        state.camera_controller.handle_mouse(delta.0, -delta.1);
                    }

                    if state.mouse_right_pressed {
                        let delta = Vec2::new(-delta.1 as f32, delta.0 as f32) * 0.001;

                        let mut l = state.light_angles;

                        l.x += delta.x;
                        l.y += delta.y;
                        l.x = l.x.clamp(-PI * 0.25, PI * 0.25);

                        state.light_angles = l;
                    }
                }
            }
            _ => {}
        }

        let imgui = &mut state.imgui;

        imgui.platform.handle_event::<State>(
            imgui.context.io_mut(),
            &state.window,
            &Event::DeviceEvent { device_id, event },
        );
    }
}