// ProcessManager.hpp -- Auto-launches and lifecycle-manages the Python hand-tracking
// backend (`python_tracker\hand_tracker.py`) from the C++ engine on Windows.
//
// Features:
//   * Resolves the executable's directory via `GetModuleFileNameA` to locate
//     `python_tracker\hand_tracker.py` whether launched from the project root
//     (`3DModeller.exe`) or from `\build\` (`visual3d_engine.exe`).
//   * Uses `C:\raylib\emsdk\python\3.13.3_64bit\python.exe` with fallback to
//     `python.exe` on PATH.
//   * Spawns silently with `CREATE_NO_WINDOW` (no console popup).
//   * Attaches the child process to a Windows Job Object configured with
//     `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, guaranteeing zero zombie webcam
//     processes even if the C++ application crashes.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#ifndef NOUSER
#define NOUSER
#endif
#include <windows.h>

#ifdef DrawText
#undef DrawText
#endif
#ifdef LoadImage
#undef LoadImage
#endif
#ifdef CloseWindow
#undef CloseWindow
#endif
#ifdef ShowCursor
#undef ShowCursor
#endif
#ifdef PlaySound
#undef PlaySound
#endif
#endif

namespace v3d {

class ProcessManager {
public:
    ProcessManager() = default;
    ~ProcessManager() { stop(); }

    ProcessManager(const ProcessManager&) = delete;
    ProcessManager& operator=(const ProcessManager&) = delete;

    bool start(uint16_t udpPort = 5005) {
        stop();
#if defined(_WIN32)
        const std::string exeDir = executableDirectory();
        std::string workDir;
        std::string scriptPath;
        if (!locateTrackerScript(exeDir, workDir, scriptPath)) {
            status_ = "TRACKER NOT FOUND";
            return false;
        }

        const std::string pythonExe = resolvePythonExecutable();

        // Create a Job Object that automatically terminates the child process tree
        // when the job handle is closed (or when the parent process exits/crashes).
        jobHandle_ = ::CreateJobObjectA(nullptr, nullptr);
        if (jobHandle_) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!::SetInformationJobObject(jobHandle_, JobObjectExtendedLimitInformation,
                                           &info, sizeof(info))) {
                ::CloseHandle(jobHandle_);
                jobHandle_ = nullptr;
            }
        }

        std::string cmd = "\"" + pythonExe + "\" \"" + scriptPath + "\" --port " +
                          std::to_string(udpPort) + " --quiet";
        std::vector<char> cmdBuf(cmd.begin(), cmd.end());
        cmdBuf.push_back('\0');

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESHOWWINDOW;
        si.wShowWindow = 0;  // SW_HIDE (0), avoids including Win32 USER headers

        PROCESS_INFORMATION pi{};
        BOOL created = ::CreateProcessA(
            nullptr,
            cmdBuf.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB,
            nullptr,
            workDir.c_str(),
            &si,
            &pi);

        if (!created) {
            // If parent job forbids breakaway, retry without CREATE_BREAKAWAY_FROM_JOB.
            std::vector<char> retryBuf(cmd.begin(), cmd.end());
            retryBuf.push_back('\0');
            created = ::CreateProcessA(
                nullptr,
                retryBuf.data(),
                nullptr,
                nullptr,
                FALSE,
                CREATE_NO_WINDOW,
                nullptr,
                workDir.c_str(),
                &si,
                &pi);
        }

        if (!created) {
            status_ = "TRACKER SPAWN FAILED";
            if (jobHandle_) {
                ::CloseHandle(jobHandle_);
                jobHandle_ = nullptr;
            }
            return false;
        }

        hProcess_ = pi.hProcess;
        hThread_ = pi.hThread;
        pid_ = static_cast<uint32_t>(pi.dwProcessId);

        if (jobHandle_) {
            ::AssignProcessToJobObject(jobHandle_, hProcess_);
        }

        status_ = "TRACKER PID " + std::to_string(pid_);
        return true;
#else
        (void)udpPort;
        status_ = "MANUAL TRACKER (NON-WIN32)";
        return false;
#endif
    }

    void stop() {
#if defined(_WIN32)
        if (hProcess_) {
            ::TerminateProcess(hProcess_, 0);
            ::WaitForSingleObject(hProcess_, 1000);
            if (hThread_) {
                ::CloseHandle(hThread_);
                hThread_ = nullptr;
            }
            ::CloseHandle(hProcess_);
            hProcess_ = nullptr;
        }
        if (hThread_) {
            ::CloseHandle(hThread_);
            hThread_ = nullptr;
        }
        if (jobHandle_) {
            ::CloseHandle(jobHandle_);
            jobHandle_ = nullptr;
        }
        pid_ = 0;
#endif
    }

    bool isRunning() const {
#if defined(_WIN32)
        if (!hProcess_) return false;
        DWORD code = 0;
        if (!::GetExitCodeProcess(hProcess_, &code)) return false;
        return code == STILL_ACTIVE;
#else
        return false;
#endif
    }

    uint32_t pid() const { return pid_; }
    const std::string& status() const { return status_; }

private:
#if defined(_WIN32)
    HANDLE jobHandle_ = nullptr;
    HANDLE hProcess_ = nullptr;
    HANDLE hThread_ = nullptr;
#endif
    uint32_t pid_ = 0;
    std::string status_ = "NOT STARTED";

#if defined(_WIN32)
    static bool fileExists(const std::string& path) {
        const DWORD attr = ::GetFileAttributesA(path.c_str());
        return (attr != INVALID_FILE_ATTRIBUTES) && !(attr & FILE_ATTRIBUTE_DIRECTORY);
    }

    static std::string parentDir(const std::string& dir) {
        if (dir.empty()) return dir;
        const size_t pos = dir.find_last_of("\\/");
        if (pos == std::string::npos) return "";
        if (pos == 2 && dir.size() >= 3 && dir[1] == ':') return dir.substr(0, 3);
        return dir.substr(0, pos);
    }

    static std::string executableDirectory() {
        char buf[MAX_PATH]{};
        const DWORD len = ::GetModuleFileNameA(nullptr, buf, MAX_PATH);
        if (len == 0 || len >= MAX_PATH) return ".";
        std::string full(buf, len);
        const size_t slash = full.find_last_of("\\/");
        return (slash == std::string::npos) ? "." : full.substr(0, slash);
    }

    static bool locateTrackerScript(const std::string& exeDir,
                                    std::string& outWorkDir,
                                    std::string& outScriptPath) {
        std::vector<std::string> roots;
        roots.push_back(exeDir);
        const std::string p1 = parentDir(exeDir);
        if (!p1.empty()) roots.push_back(p1);
        const std::string p2 = parentDir(p1);
        if (!p2.empty()) roots.push_back(p2);

        for (const std::string& root : roots) {
            const std::string candidate = root + "\\python_tracker\\hand_tracker.py";
            if (fileExists(candidate)) {
                outWorkDir = root;
                outScriptPath = candidate;
                return true;
            }
        }
        return false;
    }

    static std::string resolvePythonExecutable() {
        const std::string preferred = "C:\\raylib\\emsdk\\python\\3.13.3_64bit\\python.exe";
        if (fileExists(preferred)) return preferred;
        return "python.exe";
    }
#endif
};

}  // namespace v3d
