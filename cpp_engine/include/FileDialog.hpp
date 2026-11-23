// FileDialog.hpp -- Native OS File Explorer dialogs isolated from Raylib headers.
#pragma once

#include <string>

namespace v3d {

std::string OpenOBJFileDialog(void* ownerHwnd = nullptr);
std::string SaveOBJFileDialog(const char* defaultName = "scene_export.obj",
                              void* ownerHwnd = nullptr);

}  // namespace v3d
