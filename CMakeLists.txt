cmake_minimum_required(VERSION 3.16)
project(Visual3D_Engine VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp_engine/src/main.cpp")
    set(V3D_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/cpp_engine")
else()
    set(V3D_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

# Raylib discovery (supports Config and direct library lookup)
find_package(raylib QUIET CONFIG)
if(TARGET raylib)
    set(V3D_RAYLIB_TARGET raylib)
else()
    find_path(RAYLIB_INCLUDE_DIR NAMES raylib.h HINTS "C:/msys64/ucrt64/include" "C:/raylib/raylib/src")
    find_library(RAYLIB_LIBRARY NAMES raylib libraylib HINTS "C:/msys64/ucrt64/lib" "C:/raylib/raylib/src")
    if(NOT RAYLIB_INCLUDE_DIR OR NOT RAYLIB_LIBRARY)
        message(FATAL_ERROR "Raylib not found. Install raylib (e.g. pacman -S mingw-w64-ucrt-x86_64-raylib).")
    endif()
    add_library(v3d_raylib UNKNOWN IMPORTED)
    set_target_properties(v3d_raylib PROPERTIES
        IMPORTED_LOCATION "${RAYLIB_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${RAYLIB_INCLUDE_DIR}")
    set(V3D_RAYLIB_TARGET v3d_raylib)
endif()

set(V3D_SOURCES
    "${V3D_ROOT}/src/main.cpp"
    "${V3D_ROOT}/src/FileDialog.cpp"
)
if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/resource.rc")
    list(APPEND V3D_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/resource.rc")
endif()

add_executable(visual3d_engine ${V3D_SOURCES})

target_include_directories(visual3d_engine PRIVATE "${V3D_ROOT}/include")
target_link_libraries(visual3d_engine PRIVATE
    ${V3D_RAYLIB_TARGET}
    OpenGL::GL
    Threads::Threads
)

if(WIN32)
    target_link_libraries(visual3d_engine PRIVATE ws2_32 gdi32 winmm user32 shell32 comdlg32)
endif()

if(MSVC)
    target_compile_options(visual3d_engine PRIVATE /W4 /O2)
    target_link_options(visual3d_engine PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
else()
    target_compile_options(visual3d_engine PRIVATE -Wall -Wextra -O2)
    if(WIN32)
        target_link_options(visual3d_engine PRIVATE -mwindows)
    endif()
endif()

# Copy shaders, icon, and package double-clickable executables into project root
add_custom_command(TARGET visual3d_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${V3D_ROOT}/shaders"
            "$<TARGET_FILE_DIR:visual3d_engine>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/icon.png"
            "$<TARGET_FILE_DIR:visual3d_engine>/icon.png"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:visual3d_engine>"
            "$<TARGET_FILE_DIR:visual3d_engine>/3DModeller.exe"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:visual3d_engine>"
            "${CMAKE_SOURCE_DIR}/3DModeller.exe"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:visual3d_engine>"
            "${CMAKE_SOURCE_DIR}/visual3d_engine.exe"
)

# Copy required MSYS2 runtime DLLs alongside both the root and build executables
if(WIN32)
    foreach(_dll IN ITEMS libraylib.dll glfw3.dll libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll)
        if(EXISTS "C:/msys64/ucrt64/bin/${_dll}")
            add_custom_command(TARGET visual3d_engine POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "C:/msys64/ucrt64/bin/${_dll}"
                        "${CMAKE_SOURCE_DIR}/${_dll}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "C:/msys64/ucrt64/bin/${_dll}"
                        "$<TARGET_FILE_DIR:visual3d_engine>/${_dll}"
            )
        endif()
    endforeach()
endif()
