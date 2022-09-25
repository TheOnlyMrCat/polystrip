let decay_rate = vec4<f32>(0.05, 0.05, 0.05, 0.0);
let diffuse_weight = 0.9;

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(8, 8)
fn canvas(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let index = global_invocation_id.x + global_invocation_id.y * 256u;

	let dim = vec2<i32>(textureDimensions(input_texture));

	var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
	let original = textureLoad(input_texture, vec2<i32>(global_invocation_id.xy), 0);
	for (var offsetX: i32 = -1; offsetX <= 1; offsetX += 1) {
		for (var offsetY: i32 = -1; offsetY <= 1; offsetY += 1) {
			let sampleX = min(dim.x - 1, max(0, i32(global_invocation_id.x) + offsetX));
			let sampleY = min(dim.y - 1, max(0, i32(global_invocation_id.y) + offsetY));
			sum += textureLoad(input_texture, vec2<i32>(sampleX, sampleY), 0);
		}
	}

	textureStore(output_texture, vec2<i32>(global_invocation_id.xy), max(vec4<f32>(0.0, 0.0, 0.0, 0.0), sum / 9.0 * diffuse_weight + original * (1.0 - diffuse_weight) - decay_rate));
}
