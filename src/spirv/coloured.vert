#version 450

layout(constant_id=0) const bool FLIP_Y = false;

layout(location=0) in vec3 in_position;
layout(location=1) in vec4 in_colour;

layout(location=0) out vec4 frag_colour;

void main() {
	if (FLIP_Y) {
		gl_Position = vec4(in_position.x, -in_position.y, in_position.z, 1.0);
	} else {
		gl_Position = vec4(in_position, 1.0);
	}
	frag_colour = in_colour;
}