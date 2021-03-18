# 0.6
## 0
- Update `gfx-hal` to 0.7
- Restructure `RendererContext` and `Renderer` into `Renderer` and `WindowTarget`, respectively
- Allow rendering to and getting data from `Texture`s
- Add global transform matrices
- Add `RendererBuilder`, with three modifiable settings
- Allow multiple frames to be buffered for render simultaneously
- Collapse `data` module into `vertex` module

# 0.5
## 1
- Fix transformations on textured shapes

## 0
- Add stroked shapes: outlined shapes
- Add shape depth: vertices can have a depth dimension which is interpolated and takes precedence over draw order.
- Use 4x4 matrices to transform shapes. 3D operations do not affect shape depth.

# 0.4
## 0
Start of changelog