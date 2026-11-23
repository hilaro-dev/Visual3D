// MathUtils.hpp -- minimal, dependency-free vector / matrix math (column-major).
#pragma once

#include <algorithm>
#include <cmath>

namespace v3d {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTau = 2.0f * kPi;

struct Vec2 {
    float x = 0.f, y = 0.f;
    constexpr Vec2() = default;
    constexpr Vec2(float x_, float y_) : x(x_), y(y_) {}
    constexpr Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
    constexpr Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
    constexpr Vec2 operator*(float s) const { return {x * s, y * s}; }
    constexpr Vec2 operator/(float s) const { return {x / s, y / s}; }
    Vec2& operator+=(Vec2 o) { x += o.x; y += o.y; return *this; }
};

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;
    constexpr Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    constexpr Vec3 operator+(Vec3 o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(Vec3 o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    constexpr Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(Vec3 o) { x += o.x; y += o.y; z += o.z; return *this; }
};

struct Vec4 {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
    constexpr Vec4() = default;
    constexpr Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    constexpr Vec4(Vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
    constexpr Vec4 operator*(float s) const { return {x * s, y * s, z * s, w * s}; }
};

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float length(Vec3 v) { return std::sqrt(dot(v, v)); }
inline float length(Vec2 v) { return std::sqrt(dot(v, v)); }
inline float distance(Vec3 a, Vec3 b) { return length(a - b); }
inline float distance(Vec2 a, Vec2 b) { return length(a - b); }
inline Vec3 normalize(Vec3 v) {
    const float l = length(v);
    return l > 1e-8f ? v / l : Vec3{0.f, 0.f, 0.f};
}
inline Vec2 normalize(Vec2 v) {
    const float l = length(v);
    return l > 1e-8f ? v / l : Vec2{0.f, 0.f};
}
inline float clampf(float v, float lo, float hi) { return std::min(std::max(v, lo), hi); }
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }
inline Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }
inline Vec2 lerp(Vec2 a, Vec2 b, float t) { return a + (b - a) * t; }
inline float smoothstep(float e0, float e1, float x) {
    const float t = clampf((x - e0) / (e1 - e0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}
/// Frame-rate independent exponential smoothing factor.
inline float damp(float rate, float dt) { return 1.f - std::exp(-rate * dt); }

inline Vec3 hsv2rgb(float h, float s, float v) {
    h = h - std::floor(h);
    const float i = std::floor(h * 6.f);
    const float f = h * 6.f - i;
    const float p = v * (1.f - s), q = v * (1.f - f * s), t = v * (1.f - (1.f - f) * s);
    switch (static_cast<int>(i) % 6) {
        case 0: return {v, t, p};
        case 1: return {q, v, p};
        case 2: return {p, v, t};
        case 3: return {p, q, v};
        case 4: return {t, p, v};
        default: return {v, p, q};
    }
}
inline float rgb2hue(Vec3 c) {
    const float mx = std::max({c.x, c.y, c.z}), mn = std::min({c.x, c.y, c.z});
    const float d = mx - mn;
    if (d < 1e-5f) return 0.f;
    float h;
    if (mx == c.x) h = std::fmod((c.y - c.z) / d, 6.f);
    else if (mx == c.y) h = (c.z - c.x) / d + 2.f;
    else h = (c.x - c.y) / d + 4.f;
    h /= 6.f;
    return h < 0.f ? h + 1.f : h;
}

/// 4x4 matrix, column-major (OpenGL convention): m[col * 4 + row].
struct Mat4 {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    static Mat4 identity() { return Mat4{}; }

    static Mat4 translate(Vec3 t) {
        Mat4 r;
        r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
        return r;
    }
    static Mat4 scale(float s) { return scale(Vec3{s, s, s}); }
    static Mat4 scale(Vec3 s) {
        Mat4 r;
        r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z;
        return r;
    }
    /// Rodrigues rotation around an arbitrary axis.
    static Mat4 rotate(Vec3 axis, float angle) {
        const Vec3 a = normalize(axis);
        const float c = std::cos(angle), s = std::sin(angle), t = 1.f - c;
        Mat4 r;
        r.m[0] = t * a.x * a.x + c;       r.m[4] = t * a.x * a.y - s * a.z; r.m[8] = t * a.x * a.z + s * a.y;
        r.m[1] = t * a.x * a.y + s * a.z; r.m[5] = t * a.y * a.y + c;       r.m[9] = t * a.y * a.z - s * a.x;
        r.m[2] = t * a.x * a.z - s * a.y; r.m[6] = t * a.y * a.z + s * a.x; r.m[10] = t * a.z * a.z + c;
        return r;
    }
    /// Construct a pure 3x3 rotation matrix from orthonormal columns (X, Y, Z).
    static Mat4 fromBasis(Vec3 xCol, Vec3 yCol, Vec3 zCol) {
        Mat4 r;
        r.m[0] = xCol.x; r.m[4] = yCol.x; r.m[8]  = zCol.x;
        r.m[1] = xCol.y; r.m[5] = yCol.y; r.m[9]  = zCol.y;
        r.m[2] = xCol.z; r.m[6] = yCol.z; r.m[10] = zCol.z;
        return r;
    }
    /// Transpose of the upper-left 3x3 rotation block (inverse for orthonormal matrices).
    Mat4 transposed3x3() const {
        Mat4 r;
        r.m[0] = m[0]; r.m[4] = m[1]; r.m[8]  = m[2];
        r.m[1] = m[4]; r.m[5] = m[5]; r.m[9]  = m[6];
        r.m[2] = m[8]; r.m[6] = m[9]; r.m[10] = m[10];
        return r;
    }
    /// Gram-Schmidt orthonormalization of the upper-left 3x3 block to prevent drift.
    Mat4 orthonormalized3x3() const {
        Vec3 x{m[0], m[1], m[2]};
        Vec3 y{m[4], m[5], m[6]};
        x = normalize(x);
        if (length(x) < 1e-5f) x = {1.f, 0.f, 0.f};
        y = normalize(y - x * dot(x, y));
        if (length(y) < 1e-5f) y = {0.f, 1.f, 0.f};
        const Vec3 z = cross(x, y);
        return fromBasis(x, y, z);
    }
    static Mat4 perspective(float fovY, float aspect, float zNear, float zFar) {
        const float f = 1.f / std::tan(fovY * 0.5f);
        Mat4 r;
        for (float& v : r.m) v = 0.f;
        r.m[0] = f / aspect;
        r.m[5] = f;
        r.m[10] = (zFar + zNear) / (zNear - zFar);
        r.m[11] = -1.f;
        r.m[14] = 2.f * zFar * zNear / (zNear - zFar);
        return r;
    }
    static Mat4 ortho(float l, float r_, float b, float t, float zNear, float zFar) {
        Mat4 r;
        r.m[0] = 2.f / (r_ - l);
        r.m[5] = 2.f / (t - b);
        r.m[10] = -2.f / (zFar - zNear);
        r.m[12] = -(r_ + l) / (r_ - l);
        r.m[13] = -(t + b) / (t - b);
        r.m[14] = -(zFar + zNear) / (zFar - zNear);
        return r;
    }
    static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
        const Vec3 f = normalize(center - eye);
        const Vec3 s = normalize(cross(f, up));
        const Vec3 u = cross(s, f);
        Mat4 r;
        r.m[0] = s.x;  r.m[4] = s.y;  r.m[8] = s.z;
        r.m[1] = u.x;  r.m[5] = u.y;  r.m[9] = u.z;
        r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
        r.m[12] = -dot(s, eye);
        r.m[13] = -dot(u, eye);
        r.m[14] = dot(f, eye);
        return r;
    }

    Mat4 operator*(const Mat4& b) const {
        Mat4 r;
        for (int c = 0; c < 4; ++c)
            for (int row = 0; row < 4; ++row) {
                float sum = 0.f;
                for (int k = 0; k < 4; ++k) sum += m[k * 4 + row] * b.m[c * 4 + k];
                r.m[c * 4 + row] = sum;
            }
        return r;
    }
    Vec4 operator*(Vec4 v) const {
        return {m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w,
                m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w,
                m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
                m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w};
    }
    const float* data() const { return m; }
};

}  // namespace v3d
