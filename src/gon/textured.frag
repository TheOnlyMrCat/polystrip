#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location=0) in vec2 frag_texture_coords;
layout(location=1) in float frag_depth;

layout(location=0) out vec4 out_colour;

layout(set=0, binding=0) uniform texture2D tex;
layout(set=0, binding=1) uniform sampler samp;

void main() {
	out_colour = texture(sampler2D(tex, samp), frag_texture_coords);
	gl_FragDepth = frag_depth;
}