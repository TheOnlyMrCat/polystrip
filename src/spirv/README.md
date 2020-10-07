# Shaders

The shaders in this directory are precompiled. If any of them are updated, the corresponding `.spv` should be updated
as well. The Makefile in the top level of this repository will update any modified shaders, and this is part of the
build process if using [runscript](https://github.com/TheOnlyMrCat/runscript).

Building these shaders requires a copy of a compiler such as the Google-owned [shaderc](https://github.com/google/shaderc).
The Makefile will try running the compiler program by name `glslc`. If this is not the correct name, run the Makefile with
`make GLSLC=name`.