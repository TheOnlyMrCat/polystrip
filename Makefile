GLSLC := glslc

shaders: gen/coloured.frag.spv gen/coloured.vert.spv gen/textured.frag.spv gen/textured.vert.spv

gen/coloured.frag.spv: src/gon/coloured.frag
	$(GLSLC) -o gen/coloured.frag.spv src/gon/coloured.frag

gen/coloured.vert.spv: src/gon/coloured.vert
	$(GLSLC) -o gen/coloured.vert.spv src/gon/coloured.vert

gen/textured.frag.spv: src/gon/textured.frag
	$(GLSLC) -o gen/textured.frag.spv src/gon/textured.frag

gen/textured.vert.spv: src/gon/textured.vert
	$(GLSLC) -o gen/textured.vert.spv src/gon/textured.vert

gen/character.frag.spv: src/gon/textured.vert
	$(GLSLC) -o gen/character.frag.spv src/tui/character.frag

gen/chaacter.vert.spv: src/gon/textured.vert
	$(GLSLC) -o gen/character.vert.spv src/tui/character.vert