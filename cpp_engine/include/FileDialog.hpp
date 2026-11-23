// FileDialog.hpp -- Native OS File Explorer dialogs isolated from Raylib headers.
#pragma once

#include <string>

namespace v3d {

std::string OpenOBJFileDialog(void* ownerHwnd = nullptr);
std::string SaveOBJFileDialog(const char* defaultName = "scene_export.obj",
                              void* ownerHwnd = nullptr);

std::string OpenProjectFileDialog(void* ownerHwnd = nullptr);
std::string SaveProjectFileDialog(const char* defaultName = "untitled.3dmproj",
                                  void* ownerHwnd = nullptr);

}  // namespace v3d
