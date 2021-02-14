//! Vertices and shapes, the core of the rendering process.
//! 
//! # Linear algebra libraries
//! A number of linear algebra libraries exist for rust. `polystrip` provides basic implementations of
//! `Vector2`, `Vector3`, and `Matrix4` in the [`algebra` module](algebra/index.html), but can instead use definitions from
//! [`cgmath`](https://docs.rs/cgmath) or [`mint`](https://docs.rs/mint) by enabling the respective features.
//! 
//! # Coordinates
//! ## Screen space
//! `(0.0, 0.0)` is the screen center. `(1.0, 1.0)` is the top-right corner.
//! `(-1.0, -1.0)` is the bottom-left corner.
//! 
//! ## Texture space
//! `(0.0, 0.0)` is the top-left corner
//! `(1.0, 1.0)` is the bottom-right corner

/// A vertex describing a position and a position on a texture.
/// 
/// Texture coordinates are interpolated linearly between vertices.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TextureVertex {
	pub position: Vector3,
	pub tex_coords: Vector2,
}

unsafe impl bytemuck::Pod for TextureVertex {}
unsafe impl bytemuck::Zeroable for TextureVertex {}

impl TextureVertex {
	pub(crate) fn desc<'a>() -> &'a [gfx_hal::pso::AttributeDesc] {
		use std::mem::size_of;
		
		&[
			gfx_hal::pso::AttributeDesc {
				location: 0,
				binding: 0,
				element: gfx_hal::pso::Element {
					format: gfx_hal::format::Format::Rgb32Sfloat,
					offset: 0,
				},
			},
			gfx_hal::pso::AttributeDesc {
				location: 1,
				binding: 0,
				element: gfx_hal::pso::Element {
					format: gfx_hal::format::Format::Rg32Sfloat,
					offset: size_of::<[f32; 3]>() as u32,
				},
			},
		]
	}
}

/// A color in the sRGB color space, with red, green, blue, and alpha components all represented with `u8`s
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct Color {
	pub r: u8,
	pub g: u8,
	pub b: u8,
	pub a: u8,
}

impl Color {
	pub fn new(r: u8, g: u8, b: u8, a: u8) -> Color {
		Color { r, g, b, a }
	}
}

/// A vertex describing a position and a color.
/// 
/// Colors are interpolated linearly between vertices.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct ColorVertex {
	pub position: Vector3,
	pub color: Color,
}

unsafe impl bytemuck::Pod for ColorVertex {}
unsafe impl bytemuck::Zeroable for ColorVertex {}

impl ColorVertex {
	pub(crate) fn desc<'a>() -> &'a [gfx_hal::pso::AttributeDesc] {
		use std::mem::size_of;
		
		&[
			gfx_hal::pso::AttributeDesc {
				location: 0,
				binding: 0,
				element: gfx_hal::pso::Element {
					format: gfx_hal::format::Format::Rgb32Sfloat,
					offset: 0,
				},
			},
			gfx_hal::pso::AttributeDesc {
				location: 1,
				binding: 0,
				element: gfx_hal::pso::Element {
					format: gfx_hal::format::Format::Rgba8Unorm,
					offset: size_of::<[f32; 3]>() as u32,
				},
			},
		]
	}
}

/// A set of vertices and indices describing an outlined geometric shape as a set of lines.
///
/// The colors of the lines are determined by interpolating the colors at each
/// [`ColorVertex`](struct.ColorVertex).
#[derive(Clone, Copy, Debug)]
pub struct StrokedShape<'a> {
	pub vertices: &'a [ColorVertex],
	/// A list of pairs of vertices which specify which vertices should have lines drawn between them
	pub indices: &'a [[u16; 2]],
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
///
/// The color of the shape is determined by interpolating the colors at each
/// [`ColorVertex`](struct.ColorVertex).
#[derive(Clone, Copy, Debug)]
pub struct ColoredShape<'a> {
	pub vertices: &'a [ColorVertex],
	/// A list of sets of three vertices which specify how the vertices should be rendered as triangles.
	pub indices: &'a [[u16; 3]], //TODO: Work out if it should need to be CCW or not
}

/// A set of vertices and indices describing a geometric shape as a set of triangles.
/// 
/// The color of the shape is determined by interpolating the texture coordinates at each
/// [`TextureVertex`](struct.TextureVertex).
/// 
/// A `TexturedShape` does not store the texture it is to draw. This must be specified in the
/// arguments to [`Frame::draw_textured`](../renderer/struct.Frame#method.draw_textured)
#[derive(Clone, Copy, Debug)]
pub struct TexturedShape<'a> {
	pub vertices: &'a [TextureVertex],
	/// A list of sets of three vertices which specify how the vertices should be rendered as triangles.
	pub indices: &'a [[u16; 3]], //TODO: As above
}

/// A rectangle in pixel coordinates. (x, y) is the top-left corner; (w, h) expanding rightward and downward.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct Rect {
	pub x: i32,
	pub y: i32,
	pub w: i32,
	pub h: i32,
}

impl Rect {
	pub fn new(x: i32, y: i32, w: i32, h: i32) -> Rect {
		Rect { x, y, w, h }
	}
}

/// Basic implementations of linear algebra types
pub mod algebra {
	/// A basic implementation of a 4 x 4, column major matrix
	#[repr(C)]
	#[derive(Clone, Copy, Debug)]
	pub struct Matrix4 {
		/// The first column of the matrix
		pub x: [f32; 4],
		/// The second column of the matrix
		pub y: [f32; 4],
		/// The third column of the matrix
		pub z: [f32; 4],
		/// The fourth column of the matrix
		pub w: [f32; 4],
	}

	impl Matrix4 {
		/// Returns the [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix).
		pub fn identity() -> Matrix4 {
			Matrix4 {
				x: [1.0, 0.0, 0.0, 0.0],
				y: [0.0, 1.0, 0.0, 0.0],
				z: [0.0, 0.0, 1.0, 0.0],
				w: [0.0, 0.0, 0.0, 1.0],
			}
		}

		pub fn translate(x: f32, y: f32) -> Matrix4 {
			Matrix4 {
				x: [1.0, 0.0, 0.0, 0.0],
				y: [0.0, 1.0, 0.0, 0.0],
				z: [0.0, 0.0, 1.0, 0.0],
				w: [ x ,  y , 0.0, 1.0],
			}
		}

		pub fn row(&self, i: usize) -> [f32; 4] {
			[self.x[i], self.y[i], self.z[i], self.w[i]]
		}
	}

	fn dot(lhs: [f32; 4], rhs: [f32; 4]) -> f32 {
		lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2] + lhs[3] * rhs[3]
	}

	impl std::ops::Mul<Matrix4> for Matrix4 {
		type Output = Matrix4;
		fn mul(self, rhs: Matrix4) -> Matrix4 {
			Matrix4 {
				x: [dot(self.row(0), rhs.x), dot(self.row(1), rhs.x), dot(self.row(2), rhs.x), dot(self.row(3), rhs.x)],
				y: [dot(self.row(0), rhs.y), dot(self.row(1), rhs.y), dot(self.row(2), rhs.y), dot(self.row(3), rhs.y)],
				z: [dot(self.row(0), rhs.z), dot(self.row(1), rhs.z), dot(self.row(2), rhs.z), dot(self.row(3), rhs.z)],
				w: [dot(self.row(0), rhs.w), dot(self.row(1), rhs.w), dot(self.row(2), rhs.w), dot(self.row(3), rhs.w)],
			}
		}
	}

	impl Into<[[f32; 4]; 4]> for Matrix4 {
		fn into(self) -> [[f32; 4]; 4] {
			[self.x, self.y, self.z, self.w]
		}
	}

	/// A basic implementation of a two-dimensional vector.
	#[repr(C)]
	#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
	pub struct Vector2 {
		pub x: f32,
		pub y: f32,
	}

	impl Vector2 {
		pub fn new(x: f32, y: f32) -> Vector2 {
			Vector2 { x, y }
		}

		pub fn with_height(self, z: f32) -> Vector3 {
			Vector3 { x: self.x, y: self.y, z }
		}
	}

	/// A basic implementation of a three-dimensional vector.
	#[repr(C)]
	#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
	pub struct Vector3 {
		pub x: f32,
		pub y: f32,
		pub z: f32,
	}
	
	impl Vector3 {
		pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
			Vector3 { x, y, z }
		}
	}
}

macro_rules! reexport_algebra {
	($name:ident, $cgm:ty, $mint:ty, $docs:tt) => {
		#[cfg(not(any(feature="cgmath", feature="mint")))]
		#[doc=$docs]
		pub type $name = algebra::$name;
		
		#[cfg(all(feature="cgmath", not(feature="mint")))]
		#[doc=$docs]
		pub type $name = $cgm;
		
		#[cfg(feature="mint")]
		#[doc=$docs]
		pub type $name = $mint;
	}
}

reexport_algebra! {
	Vector2,
	cgmath::Vector2<f32>,
	mint::Vector2<f32>,
	"\
A 2D vector in screen space\
	"
}

reexport_algebra! {
	Vector3,
	cgmath::Vector3<f32>,
	mint::Vector3<f32>,
	"\
A 3D vector in screen space
	
# Height
The `z` coordinate of this vector, is uncapped linear height, used to affect the render output.
Out of a set of shapes drawn at the same height, the one drawn last appears on top.
Out of a set of shapes drawn at different heights, the one with the greatest height appears on top.

Additionally, height interpolates linearly between vertices.\
	"
}

reexport_algebra! {
	Matrix4,
	cgmath::Matrix4,
	mint::ColumnMatrix4,
	"\
A 4 x 4 column major matrix in screen space\
	"
}

pub fn with_height(vec2: Vector2, height: f32) -> Vector3 {
	Vector3 {
		x: vec2.x,
		y: vec2.y,
		z: height,
	}
}