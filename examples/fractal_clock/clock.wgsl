// More aptly named "rounded_lines.wgsl"

struct VertexInput {
    @builtin(vertex_index) index: u32,
    @location(0) start: f32,
    @location(1) end: f32,
    @location(2) direction: f32,
    @location(3) half_thickness: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) coordinates: vec2<f32>,
    @location(1) length: f32,
    @location(2) half_thickness: f32,
}

struct Camera {
    projection: mat4x4<f32>
}

@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let length = in.end - in.start + in.half_thickness * 2.0;
    var out: VertexOutput;
    out.position = camera.projection * vec4<f32>(
        mat2x2<f32>(
            cos(-in.direction), sin(-in.direction),
            -sin(-in.direction), cos(-in.direction),
        ) *
        vec2<f32>(
            in.half_thickness * (f32(in.index & 2u) - 1.0),
            in.start + length * f32(in.index & 1u) - in.half_thickness,
        ),
        0.0,
        1.0
    );
    out.coordinates = vec2<f32>(f32(in.index & 2u) - 1.0, length * f32(in.index & 1u) - in.half_thickness);
    out.length = length - in.half_thickness * 2.0;
    out.half_thickness = in.half_thickness;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let half_length = in.length / 2.0;
    let circle_y = max(abs(in.coordinates.y - half_length) - half_length, 0.0) / in.half_thickness;
    if (sqrt(pow(in.coordinates.x, 2.0) + pow(circle_y, 2.0)) > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        // return vec4<f32>(in.coordinates.x, circle_y, 1.0, 1.0);
    } else {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
        // return vec4<f32>(in.coordinates.x, circle_y, 0.0, 1.0);
    }
}
