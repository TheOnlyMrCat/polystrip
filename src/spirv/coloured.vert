#version 450

layout(location=0) in vec3 in_position;
layout(location=1) in vec4 in_colour;

layout(location=0) out vec4 frag_colour;

void main() {
	gl_Position = vec4(in_position, 1.0);
	frag_colour = in_colour;
}