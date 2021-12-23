#version 450

#define MAX_TRANSFORMS 127

layout(location=0) in vec3 in_position;
layout(location=1) in vec2 in_texture_coords;

layout(location=0) out vec2 frag_texture_coords;
layout(location=1) out float frag_depth;

layout(push_constant)
uniform Transform {
	mat4 w_transform;
	mat4 o_transform[MAX_TRANSFORMS];
};

void main() {
	vec4 transform_position = w_transform * o_transform[gl_InstanceIndex] * vec4(in_position.xy, 0.0, 1.0);
	gl_Position = vec4(transform_position.x, transform_position.y, transform_position.zw);
	frag_texture_coords = in_texture_coords;
	frag_depth = in_position.z;
}