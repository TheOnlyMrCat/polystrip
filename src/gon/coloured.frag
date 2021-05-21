#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location=0) in vec4 frag_colour;
layout(location=1) in float frag_depth;

layout(location=0) out vec4 out_colour;

void main() {
	out_colour = frag_colour;
	gl_FragDepth = frag_depth;
}