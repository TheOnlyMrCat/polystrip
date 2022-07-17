struct Vertex {
    position: vec2<f32>,
    depth: f32,
    direction: f32,
}

struct Vertices {
    vertices: array<Vertex>,
}

struct Uniform {
    depth: f32,
    time_sec: f32,
    time_min: f32,
    time_hr: f32,
}

@group(0) @binding(0) var<storage, read_write> vertices: Vertices;
@group(1) @binding(0) var<uniform> invocation: Uniform;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.x + global_invocation_id.y * 256u;
    let depth = invocation.depth;

    let buffer_offset = u32(pow(3.0, depth + 1.0)) / 2u + index * 3u; // OEIS A003462
    let base_vertex_offset = (buffer_offset - 1u) / 3u;

    let base_vertex = vertices.vertices[base_vertex_offset];
    let base_direction = base_vertex.direction;
    let base_position = base_vertex.position;

    let length_multiplier = pow(0.7, invocation.depth);
    let len_min_sec = 0.4 * length_multiplier;
    let len_hr = 0.175 * length_multiplier;

    let dir_sec = base_direction + invocation.time_sec;
    vertices.vertices[buffer_offset].direction = dir_sec;
    vertices.vertices[buffer_offset].position = base_position + vec2<f32>(sin(dir_sec), cos(dir_sec)) * len_min_sec;

    let dir_min = base_direction + invocation.time_min;
    vertices.vertices[buffer_offset + 1u].direction = dir_min;
    vertices.vertices[buffer_offset + 1u].position = base_position + vec2<f32>(sin(dir_min), cos(dir_min)) * len_min_sec;

    let dir_hr = base_direction + invocation.time_hr;
    vertices.vertices[buffer_offset + 2u].direction = dir_hr;
    vertices.vertices[buffer_offset + 2u].position = base_position + vec2<f32>(sin(dir_hr), cos(dir_hr)) * len_hr;

    vertices.vertices[buffer_offset].depth = depth;
    vertices.vertices[buffer_offset + 1u].depth = depth;
    vertices.vertices[buffer_offset + 2u].depth = depth;
}
