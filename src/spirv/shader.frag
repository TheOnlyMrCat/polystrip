#version 450

layout(location=0) in vec3 frag_colour;
layout(location=1) in vec2 frag_tex_coords;
layout(location=2) flat in uint frag_tex_index;

layout(location=0) out vec4 out_colour;

void main() {
	out_colour = vec4(frag_colour, 1.0);
}