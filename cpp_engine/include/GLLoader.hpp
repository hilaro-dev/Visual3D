// GLLoader.hpp -- OpenGL 3.3 core function loading.
//
// Two interchangeable back-ends:
//   * V3D_USE_GLEW defined  -> classic GLEW (glewInit) path.
//   * otherwise             -> a tiny self-contained loader that resolves every
//                              entry point used by the engine through
//                              glfwGetProcAddress (zero extra dependencies).
//
// Both paths expose the standard gl* names to the rest of the code base.
#pragma once

#if defined(V3D_USE_GLEW)

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <string>

namespace v3d::gl {
inline bool load(std::string& error) {
    glewExperimental = GL_TRUE;
    const GLenum status = glewInit();
    (void)glGetError();  // GLEW may raise a benign GL_INVALID_ENUM on core profiles.
    if (status != GLEW_OK) {
        error = reinterpret_cast<const char*>(glewGetErrorString(status));
        return false;
    }
    return true;
}
inline const char* loaderName() { return "GLEW"; }
}  // namespace v3d::gl

#else  // ---------------------------------------------------------- built-in

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cstddef>
#include <string>

#if defined(_WIN32)
#define V3D_GLAPIENTRY __stdcall
#else
#define V3D_GLAPIENTRY
#endif

typedef unsigned int GLenum;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef void GLvoid;
typedef int GLint;
typedef unsigned int GLuint;
typedef int GLsizei;
typedef float GLfloat;
typedef char GLchar;
typedef unsigned char GLubyte;
typedef std::ptrdiff_t GLsizeiptr;
typedef std::ptrdiff_t GLintptr;

// --- enums used by the engine ------------------------------------------------
#define GL_FALSE 0
#define GL_TRUE 1
#define GL_NO_ERROR 0
#define GL_ZERO 0
#define GL_ONE 1
#define GL_LINES 0x0001
#define GL_TRIANGLES 0x0004
#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_SRC_ALPHA 0x0302
#define GL_ONE_MINUS_SRC_ALPHA 0x0303
#define GL_CULL_FACE 0x0B44
#define GL_DEPTH_TEST 0x0B71
#define GL_BLEND 0x0BE2
#define GL_UNPACK_ALIGNMENT 0x0CF5
#define GL_TEXTURE_2D 0x0DE1
#define GL_UNSIGNED_BYTE 0x1401
#define GL_UNSIGNED_INT 0x1405
#define GL_FLOAT 0x1406
#define GL_HALF_FLOAT 0x140B
#define GL_RED 0x1903
#define GL_RGB 0x1907
#define GL_RGBA 0x1908
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_NEAREST 0x2600
#define GL_LINEAR 0x2601
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_WRAP_S 0x2802
#define GL_TEXTURE_WRAP_T 0x2803
#define GL_RGB8 0x8051
#define GL_RGBA8 0x8058
#define GL_FUNC_ADD 0x8006
#define GL_MULTISAMPLE 0x809D
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_R8 0x8229
#define GL_ALIASED_LINE_WIDTH_RANGE 0x846E
#define GL_TEXTURE0 0x84C0
#define GL_RGBA16F 0x881A
#define GL_ARRAY_BUFFER 0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_STREAM_DRAW 0x88E0
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_SHADING_LANGUAGE_VERSION 0x8B8C
#define GL_READ_FRAMEBUFFER 0x8CA8
#define GL_DRAW_FRAMEBUFFER 0x8CA9
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_FRAMEBUFFER 0x8D40
#define GL_RENDERBUFFER 0x8D41
#define GL_MAX_SAMPLES 0x8D57

// --- X-macro list of every entry point -------------------------------------
#define V3D_GL_FUNCTIONS(X)                                                                         \
    X(void, glClear, (GLbitfield))                                                                  \
    X(void, glClearColor, (GLfloat, GLfloat, GLfloat, GLfloat))                                     \
    X(void, glViewport, (GLint, GLint, GLsizei, GLsizei))                                           \
    X(void, glEnable, (GLenum))                                                                     \
    X(void, glDisable, (GLenum))                                                                    \
    X(void, glBlendFunc, (GLenum, GLenum))                                                          \
    X(void, glBlendFuncSeparate, (GLenum, GLenum, GLenum, GLenum))                                  \
    X(void, glBlendEquation, (GLenum))                                                              \
    X(void, glDepthMask, (GLboolean))                                                               \
    X(void, glDrawArrays, (GLenum, GLint, GLsizei))                                                 \
    X(void, glDrawElements, (GLenum, GLsizei, GLenum, const void*))                                 \
    X(void, glLineWidth, (GLfloat))                                                                 \
    X(const GLubyte*, glGetString, (GLenum))                                                        \
    X(void, glGetIntegerv, (GLenum, GLint*))                                                        \
    X(void, glGetFloatv, (GLenum, GLfloat*))                                                        \
    X(GLenum, glGetError, (void))                                                                   \
    X(void, glPixelStorei, (GLenum, GLint))                                                         \
    X(void, glGenTextures, (GLsizei, GLuint*))                                                      \
    X(void, glDeleteTextures, (GLsizei, const GLuint*))                                             \
    X(void, glBindTexture, (GLenum, GLuint))                                                        \
    X(void, glActiveTexture, (GLenum))                                                              \
    X(void, glTexParameteri, (GLenum, GLenum, GLint))                                               \
    X(void, glTexImage2D, (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*)) \
    X(void, glTexSubImage2D, (GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*)) \
    X(void, glGenBuffers, (GLsizei, GLuint*))                                                       \
    X(void, glDeleteBuffers, (GLsizei, const GLuint*))                                              \
    X(void, glBindBuffer, (GLenum, GLuint))                                                         \
    X(void, glBufferData, (GLenum, GLsizeiptr, const void*, GLenum))                                \
    X(void, glBufferSubData, (GLenum, GLintptr, GLsizeiptr, const void*))                           \
    X(GLuint, glCreateShader, (GLenum))                                                             \
    X(void, glShaderSource, (GLuint, GLsizei, const GLchar* const*, const GLint*))                  \
    X(void, glCompileShader, (GLuint))                                                              \
    X(void, glGetShaderiv, (GLuint, GLenum, GLint*))                                                \
    X(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*))                               \
    X(void, glDeleteShader, (GLuint))                                                               \
    X(GLuint, glCreateProgram, (void))                                                              \
    X(void, glAttachShader, (GLuint, GLuint))                                                       \
    X(void, glDetachShader, (GLuint, GLuint))                                                       \
    X(void, glLinkProgram, (GLuint))                                                                \
    X(void, glGetProgramiv, (GLuint, GLenum, GLint*))                                               \
    X(void, glGetProgramInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*))                              \
    X(void, glDeleteProgram, (GLuint))                                                              \
    X(void, glUseProgram, (GLuint))                                                                 \
    X(GLint, glGetUniformLocation, (GLuint, const GLchar*))                                         \
    X(void, glUniform1i, (GLint, GLint))                                                            \
    X(void, glUniform1f, (GLint, GLfloat))                                                          \
    X(void, glUniform2f, (GLint, GLfloat, GLfloat))                                                 \
    X(void, glUniform3f, (GLint, GLfloat, GLfloat, GLfloat))                                        \
    X(void, glUniform4f, (GLint, GLfloat, GLfloat, GLfloat, GLfloat))                               \
    X(void, glUniformMatrix4fv, (GLint, GLsizei, GLboolean, const GLfloat*))                        \
    X(void, glEnableVertexAttribArray, (GLuint))                                                    \
    X(void, glVertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const void*))        \
    X(void, glGenVertexArrays, (GLsizei, GLuint*))                                                  \
    X(void, glDeleteVertexArrays, (GLsizei, const GLuint*))                                         \
    X(void, glBindVertexArray, (GLuint))                                                            \
    X(void, glGenFramebuffers, (GLsizei, GLuint*))                                                  \
    X(void, glDeleteFramebuffers, (GLsizei, const GLuint*))                                         \
    X(void, glBindFramebuffer, (GLenum, GLuint))                                                    \
    X(void, glFramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint))                        \
    X(GLenum, glCheckFramebufferStatus, (GLenum))                                                   \
    X(void, glGenRenderbuffers, (GLsizei, GLuint*))                                                 \
    X(void, glDeleteRenderbuffers, (GLsizei, const GLuint*))                                        \
    X(void, glBindRenderbuffer, (GLenum, GLuint))                                                   \
    X(void, glRenderbufferStorage, (GLenum, GLenum, GLsizei, GLsizei))                              \
    X(void, glRenderbufferStorageMultisample, (GLenum, GLsizei, GLenum, GLsizei, GLsizei))          \
    X(void, glFramebufferRenderbuffer, (GLenum, GLenum, GLenum, GLuint))                            \
    X(void, glBlitFramebuffer, (GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum))

// Function pointers live in their own namespace so they never collide with the
// symbols exported by opengl32.dll / libGL. A using-directive makes the plain
// gl* names available everywhere this header is included.
namespace v3dgl {
#define V3D_GL_DECLARE(ret, name, params)                    \
    typedef ret(V3D_GLAPIENTRY* PFN_##name) params;          \
    inline PFN_##name name = nullptr;
V3D_GL_FUNCTIONS(V3D_GL_DECLARE)
#undef V3D_GL_DECLARE
}  // namespace v3dgl
using namespace v3dgl;

namespace v3d::gl {
inline bool load(std::string& error) {
    bool ok = true;
#define V3D_GL_LOAD(ret, name, params)                                                    \
    v3dgl::name = reinterpret_cast<v3dgl::PFN_##name>(glfwGetProcAddress(#name));         \
    if (!v3dgl::name) {                                                                   \
        ok = false;                                                                       \
        error += std::string(error.empty() ? "missing: " : ", ") + #name;                 \
    }
    V3D_GL_FUNCTIONS(V3D_GL_LOAD)
#undef V3D_GL_LOAD
    return ok;
}
inline const char* loaderName() { return "built-in"; }
}  // namespace v3d::gl

#endif
