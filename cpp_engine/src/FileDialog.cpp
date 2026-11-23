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

bool hasSuffixCI(const std::string& path, const char* suffix) {
    const size_t slen = std::strlen(suffix);
    if (path.size() < slen) return false;
    const size_t start = path.size() - slen;
    for (size_t i = 0; i < slen; ++i) {
        if (std::tolower(static_cast<unsigned char>(path[start + i])) !=
            std::tolower(static_cast<unsigned char>(suffix[i]))) {
            return false;
        }
    }
    return true;
}

std::string runWin32OpenDialog(void* ownerHwnd,
                               const char* filter,
                               const char* defExt,
                               const char* title) {
#if defined(_WIN32)
    char fileBuf[4096]{};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(OPENFILENAMEA);
    ofn.hwndOwner = static_cast<HWND>(ownerHwnd);
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = fileBuf;
    ofn.nMaxFile = static_cast<DWORD>(sizeof(fileBuf));
    ofn.nFilterIndex = 1;
    ofn.lpstrDefExt = defExt;
    ofn.lpstrTitle = title;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (::GetOpenFileNameA(&ofn)) {
        return std::string(fileBuf);
    }
    return {};
#else
    (void)ownerHwnd;
    (void)filter;
    (void)defExt;
    (void)title;
    return {};
#endif
}

std::string runWin32SaveDialog(const char* defaultName,
                               void* ownerHwnd,
                               const char* filter,
                               const char* defExt,
                               const char* requiredSuffix,
                               const char* title) {
#if defined(_WIN32)
    char fileBuf[4096]{};
    if (defaultName && defaultName[0] != '\0') {
        std::snprintf(fileBuf, sizeof(fileBuf), "%s", defaultName);
    }
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(OPENFILENAMEA);
    ofn.hwndOwner = static_cast<HWND>(ownerHwnd);
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = fileBuf;
    ofn.nMaxFile = static_cast<DWORD>(sizeof(fileBuf));
    ofn.nFilterIndex = 1;
    ofn.lpstrDefExt = defExt;
    ofn.lpstrTitle = title;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (::GetSaveFileNameA(&ofn)) {
        std::string chosen(fileBuf);
        if (!chosen.empty() && requiredSuffix && !hasSuffixCI(chosen, requiredSuffix)) {
            chosen += requiredSuffix;
        }
        return chosen;
    }
    return {};
#else
    (void)defaultName;
    (void)ownerHwnd;
    (void)filter;
    (void)defExt;
    (void)requiredSuffix;
    (void)title;
    return {};
#endif
}

constexpr const char kObjFilter[] = "Wavefront OBJ (*.obj)\0*.obj\0All Files (*.*)\0*.*\0";
constexpr const char kProjFilter[] = "3DModeller Project (*.3dmproj)\0*.3dmproj\0All Files (*.*)\0*.*\0";

}  // namespace

std::string OpenOBJFileDialog(void* ownerHwnd) {
    return runWin32OpenDialog(ownerHwnd, kObjFilter, "obj", "Import Wavefront .OBJ Model");
}

std::string SaveOBJFileDialog(const char* defaultName, void* ownerHwnd) {
    return runWin32SaveDialog(defaultName, ownerHwnd, kObjFilter, "obj", ".obj",
                              "Export Wavefront .OBJ");
}

std::string OpenProjectFileDialog(void* ownerHwnd) {
    return runWin32OpenDialog(ownerHwnd, kProjFilter, "3dmproj",
                              "Open 3DModeller Project (.3dmproj)");
}

std::string SaveProjectFileDialog(const char* defaultName, void* ownerHwnd) {
    return runWin32SaveDialog(defaultName, ownerHwnd, kProjFilter, "3dmproj", ".3dmproj",
                              "Save 3DModeller Project (.3dmproj)");
}

}  // namespace v3d
