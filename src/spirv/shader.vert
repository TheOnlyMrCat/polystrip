#version 450

layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_colour;
layout(location=2) in vec2 in_tex_coords;
layout(location=3) in uint in_tex_index;

layout(location=0) out vec3 frag_colour;
layout(location=1) out vec2 frag_tex_coords;
layout(location=2) out uint frag_tex_index;

void main() {
	gl_Position = vec4(in_position, 1.0);
	frag_colour = in_colour;
	frag_tex_coords = in_tex_coords;
	frag_tex_index = in_tex_index;
}