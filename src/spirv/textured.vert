#version 450

layout(constant_id=0) const bool FLIP_Y = false;

layout(location=0) in vec3 in_position;
layout(location=1) in vec2 in_texture_coords;

layout(location=0) out vec2 frag_texture_coords;

void main() {
	if (FLIP_Y) {
		gl_Position = vec4(in_position.x, -in_position.y, 1/(1+exp(in_position.z)), 1.0);
	} else {
		gl_Position = vec4(in_position.xy, 1/(1+exp(in_position.z)), 1.0);
	}
	frag_texture_coords = in_texture_coords;
}