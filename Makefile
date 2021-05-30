GLSLC := glslc

shaders: gen/gon/coloured.frag.spv gen/gon/coloured.vert.spv gen/gon/textured.frag.spv gen/gon/textured.vert.spv

gen/%.frag.spv: src/%.frag
	$(GLSLC) -o $@ $<

gen/%.vert.spv: src/%.vert
	$(GLSLC) -o $@ $<