#version 450

// y-coordinate is flipped on Metal and DX backends
layout(constant_id=0) const bool FLIP_Y = false;
layout(constant_id=1) const uint TRANSFORM_ARRAY_SIZE = 1;

// Between 0.0 and 1.0, is the texture coordinates for this glyph.
layout(location=0) in vec2 in_position;

layout(location=0) out vec2 out_tex_position;

layout(push_constant)
uniform Transform {
    mat4 w_transform;
    mat4 o_transform[TRANSFORM_ARRAY_SIZE];
}

void main() {
    out_tex_position = in_position;
    gl_Position = w_transform * o_transform[gl_InstanceIndex] * vec4(in_position, 0.0, 1.0);
}
