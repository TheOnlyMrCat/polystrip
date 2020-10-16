#version 450

layout(location=0) in vec3 in_position;
layout(location=1) in vec2 in_texture_coords;

layout(location=0) out vec2 frag_texture_coords;

void main() {
	gl_Position = vec4(in_position, 1.0);
	frag_texture_coords = in_texture_coords;
}