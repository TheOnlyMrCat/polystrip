GLSLC := glslc

shaders: src/spirv/shader.frag.spv src/spirv/shader.vert.spv

src/spirv/shader.frag.spv: src/spirv/shader.frag
	$(GLSLC) -o src/spirv/shader.frag.spv src/spirv/shader.frag

src/spirv/shader.vert.spv: src/spirv/shader.vert
	$(GLSLC) -o src/spirv/shader.vert.spv src/spirv/shader.vert