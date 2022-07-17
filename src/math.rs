//! Vertices and shapes, the core of the rendering process.
//!
//! # Linear algebra libraries
//! A number of linear algebra libraries exist for rust. `polystrip` provides `Vector2`, `Vector3`, and `Matrix4` as wrappers
//! around definitions provided by the [`mint`](https://docs.rs/mint) library, which is compatible with most of these linear
//! algebra libraries
//!
//! # Coordinates
//! ## Screen space
//! `(0.0, 0.0)` is the screen center. `(1.0, 1.0)` is the top-right corner.
//! `(-1.0, -1.0)` is the bottom-left corner.
//!
//! ## Texture space
//! `(0.0, 0.0)` is the top-left corner
//! `(1.0, 1.0)` is the bottom-right corner

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

/// A color in the sRGB color space, with red, green, blue, and alpha components all represented with `u8`s
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

unsafe impl bytemuck::Zeroable for Color {}
unsafe impl bytemuck::Pod for Color {}

impl Color {
    pub const ZERO: Color = Color::new(0, 0, 0, 0);
    pub const RED: Color = Color::new(255, 0, 0, 255);
    pub const YELLOW: Color = Color::new(255, 255, 0, 255);
    pub const GREEN: Color = Color::new(0, 255, 0, 255);
    pub const CYAN: Color = Color::new(0, 255, 255, 255);
    pub const BLUE: Color = Color::new(0, 0, 255, 255);
    pub const MAGENTA: Color = Color::new(255, 0, 255, 255);
    pub const WHITE: Color = Color::new(255, 255, 255, 255);
    pub const BLACK: Color = Color::new(0, 0, 0, 255);

    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Color {
        Color { r, g, b, a }
    }
}

/// A rectangle in pixel coordinates. (x, y) is the top-left corner; (w, h) expanding rightward and downward.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

/// A 2D vector in screen space
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_math", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct Vector2(pub mint::Vector2<f32>);

unsafe impl bytemuck::Zeroable for Vector2 {}
unsafe impl bytemuck::Pod for Vector2 {}

impl Vector2 {
    pub const fn new(x: f32, y: f32) -> Vector2 {
        Vector2(mint::Vector2 { x, y })
    }

    pub const fn with_height(self, height: f32) -> Vector3 {
        Vector3(mint::Vector3 {
            x: self.0.x,
            y: self.0.y,
            z: height,
        })
    }
}

impl From<mint::Vector2<f32>> for Vector2 {
    fn from(v: mint::Vector2<f32>) -> Self {
        Self(v)
    }
}

impl From<Vector2> for mint::Vector2<f32> {
    fn from(v: Vector2) -> Self {
        v.0
    }
}

impl From<[f32; 2]> for Vector2 {
    fn from(v: [f32; 2]) -> Self {
        Self(mint::Vector2::from(v))
    }
}

impl From<Vector2> for [f32; 2] {
    fn from(v: Vector2) -> Self {
        v.0.into()
    }
}

impl Deref for Vector2 {
    type Target = mint::Vector2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Vector2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A 3D vector in screen space
///
/// # Height
/// The `z` coordinate of this vector, is uncapped linear height, used to affect the render output.
/// Out of a set of shapes drawn at the same height, the one drawn last appears on top.
/// Out of a set of shapes drawn at different heights, the one with the greatest height appears on top.
///
/// Additionally, height interpolates linearly between vertices.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_math", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct Vector3(pub mint::Vector3<f32>);

unsafe impl bytemuck::Zeroable for Vector3 {}
unsafe impl bytemuck::Pod for Vector3 {}

impl Vector3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3(mint::Vector3 { x, y, z })
    }
}

impl From<mint::Vector3<f32>> for Vector3 {
    fn from(v: mint::Vector3<f32>) -> Self {
        Self(v)
    }
}

impl From<Vector3> for mint::Vector3<f32> {
    fn from(v: Vector3) -> Self {
        v.0
    }
}

impl From<[f32; 3]> for Vector3 {
    fn from(v: [f32; 3]) -> Self {
        Self(mint::Vector3::from(v))
    }
}

impl From<Vector3> for [f32; 3] {
    fn from(v: Vector3) -> Self {
        v.0.into()
    }
}

impl Deref for Vector3 {
    type Target = mint::Vector3<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Vector3 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A 4 x 4 column major matrix in screen space
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_math", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct Matrix4(pub mint::ColumnMatrix4<f32>);

unsafe impl bytemuck::Zeroable for Matrix4 {}
unsafe impl bytemuck::Pod for Matrix4 {}

impl Matrix4 {
    /// Returns the [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix).
    pub fn identity() -> Matrix4 {
        Matrix4(mint::ColumnMatrix4::from([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))
    }

    /// Returns a matrix that translates by the given X and Y values, in screen space
    pub fn translate(v: Vector2) -> Matrix4 {
        Matrix4(mint::ColumnMatrix4::from([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [v.x, v.y, 0.0, 1.0],
        ]))
    }

    /// Returns a matrix that rotates the XY plane counterclockwise about the origin by the given angle.
    ///
    /// Interprets the angle to be in radians.
    pub fn rotate(r: f32) -> Matrix4 {
        Matrix4(mint::ColumnMatrix4::from([
            [r.cos(), -r.sin(), 0.0, 0.0],
            [r.sin(), r.cos(), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))
    }

    /// Returns a matrix that scales the XY plane by the given factor
    pub fn scale(f: f32) -> Matrix4 {
        Matrix4(mint::ColumnMatrix4::from([
            [f, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))
    }

    /// Returns a matrix that scales the X coordinate by the given factor and the Y coordinate by the given factor.
    pub fn scale_nonuniform(x: f32, y: f32) -> Matrix4 {
        Matrix4(mint::ColumnMatrix4::from([
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))
    }

    /// Returns a simple perspective matrix from the four passed parameters. Equivalent to the
    /// [`gluPerspective`](https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml) function in OpenGL.
    ///
    /// # Parameters
    /// * `fovy`: The angle of the field of view up and down, in radians. Must be nonzero and cannot be a multiple of 2π (τ)
    /// * `aspect`: The ratio of screen width to screen height. Should be updated when the window is resized. Must be nonzero.
    /// * `near`: Distance from the camera to the near clipping plane, in screen-space coordinates.
    /// * `far`: Distance from the camera to the far clipping plane, in screen-space coordinates. Must not be equal to `near`.
    pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Matrix4 {
        let f = (fovy / 2.).cos() / (fovy / 2.).sin();
        Matrix4(mint::ColumnMatrix4::from([
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), -1.0],
            [0.0, 0.0, (2. * far * near) / (near - far), 0.0],
        ]))
    }

    pub fn row(&self, i: usize) -> [f32; 4] {
        [
            self.x.as_ref()[i],
            self.y.as_ref()[i],
            self.z.as_ref()[i],
            self.w.as_ref()[i],
        ]
    }
}

fn dot(lhs: [f32; 4], rhs: [f32; 4]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2] + lhs[3] * rhs[3]
}

impl std::ops::Mul<Matrix4> for Matrix4 {
    type Output = Matrix4;
    fn mul(self, rhs: Matrix4) -> Matrix4 {
        Matrix4(mint::ColumnMatrix4::from([
            [
                dot(self.row(0), rhs.x.into()),
                dot(self.row(1), rhs.x.into()),
                dot(self.row(2), rhs.x.into()),
                dot(self.row(3), rhs.x.into()),
            ],
            [
                dot(self.row(0), rhs.y.into()),
                dot(self.row(1), rhs.y.into()),
                dot(self.row(2), rhs.y.into()),
                dot(self.row(3), rhs.y.into()),
            ],
            [
                dot(self.row(0), rhs.z.into()),
                dot(self.row(1), rhs.z.into()),
                dot(self.row(2), rhs.z.into()),
                dot(self.row(3), rhs.z.into()),
            ],
            [
                dot(self.row(0), rhs.w.into()),
                dot(self.row(1), rhs.w.into()),
                dot(self.row(2), rhs.w.into()),
                dot(self.row(3), rhs.w.into()),
            ],
        ]))
    }
}

impl From<mint::ColumnMatrix4<f32>> for Matrix4 {
    fn from(v: mint::ColumnMatrix4<f32>) -> Self {
        Self(v)
    }
}

impl From<Matrix4> for mint::ColumnMatrix4<f32> {
    fn from(v: Matrix4) -> Self {
        v.0
    }
}

impl From<[[f32; 4]; 4]> for Matrix4 {
    fn from(v: [[f32; 4]; 4]) -> Self {
        Self(mint::ColumnMatrix4::from(v))
    }
}

impl From<Matrix4> for [[f32; 4]; 4] {
    fn from(v: Matrix4) -> Self {
        v.0.into()
    }
}

impl Deref for Matrix4 {
    type Target = mint::ColumnMatrix4<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Matrix4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
