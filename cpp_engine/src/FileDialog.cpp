// FileDialog.cpp -- Native Windows File Explorer Open/Save dialogs (`<commdlg.h>`).
// Kept in its own translation unit so Win32 headers never collide with `<raylib.h>`.
#include "FileDialog.hpp"

#include <cctype>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

namespace v3d {

namespace {

bool hasObjSuffixCI(const std::string& path) {
    if (path.size() < 4) return false;
    const size_t n = path.size();
    return std::tolower(static_cast<unsigned char>(path[n - 4])) == '.' &&
           std::tolower(static_cast<unsigned char>(path[n - 3])) == 'o' &&
           std::tolower(static_cast<unsigned char>(path[n - 2])) == 'b' &&
           std::tolower(static_cast<unsigned char>(path[n - 1])) == 'j';
}

}  // namespace

std::string OpenOBJFileDialog(void* ownerHwnd) {
#if defined(_WIN32)
    char fileBuf[4096]{};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(OPENFILENAMEA);
    ofn.hwndOwner = static_cast<HWND>(ownerHwnd);
    ofn.lpstrFilter = "Wavefront OBJ (*.obj)\0*.obj\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = fileBuf;
    ofn.nMaxFile = static_cast<DWORD>(sizeof(fileBuf));
    ofn.nFilterIndex = 1;
    ofn.lpstrDefExt = "obj";
    ofn.lpstrTitle = "Import Wavefront .OBJ Model";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (::GetOpenFileNameA(&ofn)) {
        return std::string(fileBuf);
    }
    return {};
#else
    (void)ownerHwnd;
    return {};
#endif
}

std::string SaveOBJFileDialog(const char* defaultName, void* ownerHwnd) {
#if defined(_WIN32)
    char fileBuf[4096]{};
    if (defaultName && defaultName[0] != '\0') {
        std::snprintf(fileBuf, sizeof(fileBuf), "%s", defaultName);
    }
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(OPENFILENAMEA);
    ofn.hwndOwner = static_cast<HWND>(ownerHwnd);
    ofn.lpstrFilter = "Wavefront OBJ (*.obj)\0*.obj\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = fileBuf;
    ofn.nMaxFile = static_cast<DWORD>(sizeof(fileBuf));
    ofn.nFilterIndex = 1;
    ofn.lpstrDefExt = "obj";
    ofn.lpstrTitle = "Export Wavefront .OBJ";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (::GetSaveFileNameA(&ofn)) {
        std::string chosen(fileBuf);
        if (!chosen.empty() && !hasObjSuffixCI(chosen)) {
            chosen += ".obj";
        }
        return chosen;
    }
    return {};
#else
    (void)defaultName;
    (void)ownerHwnd;
    return {};
#endif
}

}  // namespace v3d
