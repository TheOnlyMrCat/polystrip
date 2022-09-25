let tau_8 = 0.78539816;
let pi = 3.1415927;
let tau = 6.2831853;

struct Agent {
	position: vec2<f32>,
	direction: f32,
}

struct Agents {
	agents: array<Agent>,
}

struct Uniform {
	time: u32,
}

@group(0) @binding(0) var<storage, read_write> agent_buffer: Agents;
@group(0) @binding(1) var<uniform> invocation: Uniform;
@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

// Hash function www.cs.ubc.ca/~rbridson/docs/schechter-sca08-turbulence.pdf
fn hash(state: u32) -> u32 {
	var hash = state;
    hash ^= 2747636419u;
    hash *= 2654435769u;
    hash ^= state >> 16u;
    hash *= 2654435769u;
    hash ^= state >> 16u;
    hash *= 2654435769u;
    return hash;
}

fn sense(agent: Agent, offset: f32) -> f32 {
	let sensor_angle = agent.direction + offset;
	let sensor_dir = vec2<f32>(cos(sensor_angle), sin(sensor_angle));
	let sensor_pos = agent.position + sensor_dir * 7.0;

	let dim = vec2<i32>(textureDimensions(input_texture));

	var sum: f32 = 0.0;
	for (var offsetX: i32 = -4; offsetX <= 4; offsetX += 1) {
		for (var offsetY: i32 = -4; offsetY <= 4; offsetY += 1) {
			let sampleX = min(f32(dim.x - 1), max(0.0, sensor_pos.x + f32(offsetX)));
			let sampleY = min(f32(dim.y - 1), max(0.0, sensor_pos.y + f32(offsetY)));
			sum += length(textureLoad(input_texture, vec2<i32>(i32(sampleX), i32(sampleY)), 0));
		}
	}

	return sum;
}

@compute
@workgroup_size(128)
fn agents(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let index = global_invocation_id.x + global_invocation_id.y * 256u;
	let dim = vec2<f32>(textureDimensions(input_texture));

	var agent = agent_buffer.agents[index];

	let random = hash(u32(agent.position.x) ^ hash(u32(agent.position.y) ^ hash(invocation.time)));
	let randomTurn = f32(random) / 4294967295.0;

	let left = sense(agent, tau_8);
	let forward = sense(agent, 0.0);
	let right = sense(agent, -tau_8);

	if (forward > left && forward > right) {
		// Do nothing; go straight ahead
	} else if (forward < left && forward < right) {
		agent.direction += 0.2 * (randomTurn - 0.5);
	} else if (left > right) {
		agent.direction += 0.2 * randomTurn;
	} else if (right > left) {
		agent.direction -= 0.2 * randomTurn;
	}

	agent.position += vec2<f32>(cos(agent.direction), sin(agent.direction));
	if (agent.position.x < 0.0) {
		agent.position.x = -agent.position.x;
		agent.direction = pi - agent.direction;
	}
	if (agent.position.x > dim.x) {
		agent.position.x = dim.x * 2.0 - agent.position.x;
		agent.direction = pi - agent.direction;
	}
	if (agent.position.y < 0.0) {
		agent.position.y = -agent.position.y;
		agent.direction = -agent.direction;
	}
	if (agent.position.y > dim.y) {
		agent.position.y = dim.y * 2.0 - agent.position.y;
		agent.direction = -agent.direction;
	}

	agent_buffer.agents[index] = agent;
	let old_trail = textureLoad(input_texture, vec2<i32>(agent.position.xy), 0);
	textureStore(output_texture, vec2<i32>(agent.position.xy), min(vec4<f32>(1.0, 1.0, 1.0, 1.0), old_trail + vec4<f32>(0.5, 0.5, 0.5, 1.0)));
}
