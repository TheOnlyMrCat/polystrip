struct VertexInput {
    @builtin(vertex_index) index: u32,
    @location(0) position: vec2<f32>,
    @location(1) depth: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec4<f32>,
}

struct Camera {
    projection: mat4x4<f32>,
}

struct Colours {
    colours: array<vec4<f32>>
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage> colours: Colours;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = camera.projection * vec4<f32>(in.position, 0.0, 1.0);
    out.colour = colours.colours[u32(in.depth)];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.colour;
}
