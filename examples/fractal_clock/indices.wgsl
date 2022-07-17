struct Indices {
    indices: array<u32>,
}

struct Offset {
    offset: u32,
}

@group(0) @binding(0) var<storage, read_write> indices: Indices;
@group(1) @binding(0) var<uniform> offset: Offset;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // Index of the vertex to draw lines from
    let common_index = global_invocation_id.x + global_invocation_id.y * 256u + offset.offset;
    let base_index = common_index * 3u;

    let buffer_offset = base_index * 2u;

    indices.indices[buffer_offset] = base_index + 1u;
    indices.indices[buffer_offset + 1u] = common_index;
    indices.indices[buffer_offset + 2u] = base_index + 2u;
    indices.indices[buffer_offset + 3u] = common_index;
    indices.indices[buffer_offset + 4u] = base_index + 3u;
    indices.indices[buffer_offset + 5u] = common_index;
}
