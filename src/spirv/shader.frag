#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location=0) in vec3 frag_colour;
layout(location=1) in vec2 frag_tex_coords;
layout(location=2) flat in uint frag_tex_index;

layout(location=0) out vec4 out_colour;

layout(set=0, binding=0) uniform sampler samp;
layout(set=0, binding=1) uniform texture2D textures[8];

void main() {
	if (frag_tex_index == 4294967295) {
		out_colour = vec4(frag_colour, 1.0);
	} else {
		out_colour = texture(sampler2D(textures[frag_tex_index], samp), frag_tex_coords);
	}
}