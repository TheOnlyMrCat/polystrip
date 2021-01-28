#version 450

layout(constant_id=0) const bool FLIP_Y = false;

layout(location=0) in vec3 in_position;
layout(location=1) in vec2 in_texture_coords;

layout(location=0) out vec2 frag_texture_coords;
layout(location=1) out float frag_depth;

layout(push_constant)
uniform Transform {
	mat3 o_transform;
};

void main() {
	gl_Position = vec4(o_transform * vec3(in_position.x, FLIP_Y ? -in_position.y : in_position.y, 0.0), 1.0);
	frag_texture_coords = in_texture_coords;
	frag_depth = in_position.z;
}