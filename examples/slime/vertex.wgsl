struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

@vertex
fn vert(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32(vertex_index >> 1u) * 4.0 - 1.0;
    let y = f32(vertex_index & 1u) * 4.0 - 1.0;
	var output: VertexOutput;
	output.position = vec4<f32>(x, y, 0.0, 1.0);
	output.uv = abs(vec2<f32>(x, -y) + vec2<f32>(1.0, -1.0)) / 2.0;
    return output;
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn frag(input: VertexOutput) -> @location(0) vec4<f32> {
	return textureSample(tex, samp, input.uv);
}
