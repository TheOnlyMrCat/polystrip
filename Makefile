GLSLC := glslc

shaders: src/spirv/coloured.frag.spv src/spirv/coloured.vert.spv src/spirv/textured.frag.spv src/spirv/textured.vert.spv

src/spirv/coloured.frag.spv: src/spirv/coloured.frag
	$(GLSLC) -o src/spirv/coloured.frag.spv src/spirv/coloured.frag

src/spirv/coloured.vert.spv: src/spirv/coloured.vert
	$(GLSLC) -o src/spirv/coloured.vert.spv src/spirv/coloured.vert

src/spirv/textured.frag.spv: src/spirv/textured.frag
	$(GLSLC) -o src/spirv/textured.frag.spv src/spirv/textured.frag

src/spirv/textured.vert.spv: src/spirv/textured.vert
	$(GLSLC) -o src/spirv/textured.vert.spv src/spirv/textured.vert