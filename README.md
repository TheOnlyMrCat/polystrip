# Polystrip

[![Build Status](https://img.shields.io/travis/TheOnlyMrCat/polystrip)](https://travis-ci.org/github/TheOnlyMrCat/polystrip)
[![Crates.io](https://img.shields.io/crates/v/polystrip)](https://crates.io/crates/polystrip)

Polystrip is a 2D hardware-accelerated rendering library, primarily targeted at game development, but which could
probably be used in other applications as well. Polystrip intends to be a pure-rust replacement for the graphics
of the `sdl2` crate.

**This crate should not be considered stable until the 1.0 release.** Following true semver spec, this crate will
~~likely~~ definitely make breaking changes between development versions. The 1.0 release will come when I think this
crate is fully-featured.

## Simple example with winit

This example makes a window, a renderer, and only the boilerplate necessary to correctly interface with with polystrip.

```rs
let event_loop = winit::event_loop::EventLoop::new();
let window = winit::window::Window::new(&event_loop).unwrap();

let window_size = window.inner_size().to_logical(window.scale_factor());
let mut renderer = WindowTarget::new(Renderer::new().wrap(), &window, (window_size.width, window_size.height));

event_loop.run(move |event, _, control_flow| {
    match event {
        Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
            let window_size = new_size.to_logical(window.scale_factor());
            renderer.resize((window_size.width, window_size.height));
        },
        Event::MainEventsCleared => {
            let mut frame = renderer.next_frame();
            // Render in here
        },
        _ => {}
    }
});
```

More examples can be found in the [examples](examples) directory, and can be run with `cargo run --example <name>`.