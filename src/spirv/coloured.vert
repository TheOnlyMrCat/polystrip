#version 450

layout(constant_id=0) const bool FLIP_Y = false;
layout(constant_id=1) const bool REAL_Z = false;
layout(constant_id=2) const uint TRANSFORM_ARRAY_SIZE = 1;

layout(location=0) in vec3 in_position;
layout(location=1) in vec4 in_colour;

layout(location=0) out vec4 frag_colour;
layout(location=1) out float frag_depth;

layout(push_constant)
uniform Transform {
	mat4 w_transform;
	mat4 o_transform[TRANSFORM_ARRAY_SIZE];
};

void main() {
	vec4 transform_position = w_transform * o_transform[gl_InstanceIndex] * vec4(in_position.xy, REAL_Z ? in_position.z : 0.0, 1.0);
	gl_Position = vec4(transform_position.x, FLIP_Y ? -transform_position.y : transform_position.y, transform_position.zw);
	frag_colour = in_colour;
	frag_depth = REAL_Z ? transform_position.z : in_position.z;
}