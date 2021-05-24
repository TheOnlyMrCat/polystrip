#version 450

layout(location=0) in vec2 frag_texture_coords;

layout(location=0) out vec4 out_colour;

layout(set=0, binding=0) uniform texture2D tex;
layout(set=0, binding=1) uniform sampler samp;

layout(push_constant)
uniform Colours {
    vec4 fg_colour;
    vec4 bg_colour;
}

void main() {
    float alpha = texture(sampler2D(tex, samp), frag_texture_coords).x;
    out_colour = fg_colour * alpha + bg_colour * (1.0 - alpha);
}