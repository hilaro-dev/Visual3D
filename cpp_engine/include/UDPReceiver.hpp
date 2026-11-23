// UDPReceiver.hpp -- Lock-Free, True Non-Blocking UDP Socket Receiver.
//
// * Drains all queued UDP datagrams in a `while (true)` loop every frame using
//   a true non-blocking socket (`ioctlsocket(FIONBIO)` on Windows, `O_NONBLOCK`
//   on Linux/macOS) and retains ONLY the most recent `#pragma pack(push, 1)`
//   `TelemetryPacket` before the 3D scene updates.
// * Also reassembles incoming low-res RGB webcam chunks (`V3DV`) so Raylib can
//   upload the live camera `Texture2D` for Mode A (Fullscreen AR) and Mode B
//   (Studio 3D Picture-in-Picture).
#pragma once

#include "Protocol.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define NOGDI
#define NOUSER
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace v3d {

using TelemetryPacket = proto::TelemetryPacket;
using HandPacket = proto::HandPacket;

struct VideoFrame {
    int width = 0, height = 0, channels = 3;
    uint64_t version = 0;
    std::vector<uint8_t> pixels;
};

struct LinkStats {
    bool listening = false;
    bool connected = false;
    double udpLatencyMs = 0.0;
    double pipelineLatencyMs = 0.0;
    double packetsPerSec = 0.0;
    double lossPct = 0.0;
    double secondsSinceLast = 1e9;
    double secondsSinceVideo = 1e9;
    uint64_t packets = 0;
    uint64_t drainedStale = 0;
    uint64_t malformed = 0;
    std::string error;
};

class UDPReceiver {
public:
    explicit UDPReceiver(uint16_t port = 5005, std::string bindAddress = "127.0.0.1")
        : port_(port), bindAddress_(std::move(bindAddress)) {}
    ~UDPReceiver() { stop(); }
    UDPReceiver(const UDPReceiver&) = delete;
    UDPReceiver& operator=(const UDPReceiver&) = delete;

    bool start() {
        if (stats_.listening) return true;
#if defined(_WIN32)
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return fail("WSAStartup failed");
        wsaStarted_ = true;
#endif
        sock_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (!validSocket()) return fail("socket() failed");

        int yes = 1;
        ::setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
        int rcvbuf = 512 * 1024;
        ::setsockopt(sock_, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&rcvbuf), sizeof(rcvbuf));

#if defined(_WIN32)
        u_long mode = 1;
        if (::ioctlsocket(sock_, FIONBIO, &mode) != 0) {
            return fail("ioctlsocket(FIONBIO) failed");
        }
#else
        const int flags = ::fcntl(sock_, F_GETFL, 0);
        if (flags < 0 || ::fcntl(sock_, F_SETFL, flags | O_NONBLOCK) < 0) {
            return fail("fcntl(O_NONBLOCK) failed");
        }
#endif

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        if (::inet_pton(AF_INET, bindAddress_.c_str(), &addr.sin_addr) != 1)
            return fail("invalid bind address");
        if (::bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
            return fail("bind() failed on " + bindAddress_ + ":" + std::to_string(port_));

        windowStart_ = steadySeconds();
        stats_.listening = true;
        stats_.error.clear();
        return true;
    }

    void stop() {
        closeSocket();
        stats_.listening = false;
#if defined(_WIN32)
        if (wsaStarted_) {
            WSACleanup();
            wsaStarted_ = false;
        }
#endif
    }

    bool latest(TelemetryPacket& out, uint64_t* version = nullptr) {
        drainSocketBuffer();
        if (packetVersion_ == 0) return false;
        out = latestPacket_;
        if (version) *version = packetVersion_;
        return true;
    }

    bool latestVideo(VideoFrame& out, uint64_t haveVersion) {
        if (video_.version == 0 || video_.version == haveVersion) return false;
        out = video_;
        return true;
    }

    LinkStats stats() {
        const double now = steadySeconds();
        stats_.secondsSinceLast = lastPacketSteady_ > 0.0 ? (now - lastPacketSteady_) : 1e9;
        stats_.secondsSinceVideo = lastVideoSteady_ > 0.0 ? (now - lastVideoSteady_) : 1e9;
        stats_.connected = stats_.secondsSinceLast < 0.5;
        if (stats_.secondsSinceLast > 1.5) stats_.packetsPerSec = 0.0;
        return stats_;
    }

    size_t drainSocketBuffer() {
        if (!validSocket()) return 0;

        alignas(8) uint8_t rawBuf[16384];
        TelemetryPacket newestPkt{};
        bool gotTelemetry = false;
        size_t drainedCount = 0;

        while (true) {
#if defined(_WIN32)
            const int bytes = ::recvfrom(sock_, reinterpret_cast<char*>(rawBuf),
                                         static_cast<int>(sizeof(rawBuf)), 0, nullptr, nullptr);
            if (bytes <= 0) break;  // WSAEWOULDBLOCK: OS socket buffer is empty
#else
            const ssize_t bytes = ::recvfrom(sock_, rawBuf, sizeof(rawBuf), 0, nullptr, nullptr);
            if (bytes <= 0) break;  // EWOULDBLOCK / EAGAIN
#endif
            const size_t n = static_cast<size_t>(bytes);
            if (n >= 4 && std::memcmp(rawBuf, "V3DH", 4) == 0) {
                TelemetryPacket candidate{};
                if (proto::parseTelemetry(rawBuf, n, candidate)) {
                    if (!gotTelemetry || isNewerSeq(candidate.seq, newestPkt.seq)) {
                        newestPkt = candidate;
                        gotTelemetry = true;
                    }
                    ++drainedCount;
                } else {
                    ++stats_.malformed;
                }
            } else if (n >= proto::kVideoHeaderSize && std::memcmp(rawBuf, "V3DV", 4) == 0) {
                handleVideoChunk(rawBuf, n);
            } else {
                ++stats_.malformed;
            }
        }

        if (gotTelemetry) {
            commitLatestPacket(newestPkt, drainedCount);
        }
        updateRateWindow();
        return drainedCount;
    }

private:
#if defined(_WIN32)
    using socket_t = SOCKET;
    static constexpr socket_t kInvalid = INVALID_SOCKET;
#else
    using socket_t = int;
    static constexpr socket_t kInvalid = -1;
#endif

    static double epochSeconds() {
        using namespace std::chrono;
        return duration<double>(system_clock::now().time_since_epoch()).count();
    }

    static double steadySeconds() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }

    static bool isNewerSeq(uint32_t a, uint32_t b) {
        return static_cast<int32_t>(a - b) >= 0;
    }

    bool validSocket() const { return sock_ != kInvalid; }

    void closeSocket() {
        if (!validSocket()) return;
#if defined(_WIN32)
        ::closesocket(sock_);
#else
        ::close(sock_);
#endif
        sock_ = kInvalid;
    }

    bool fail(const std::string& msg) {
        closeSocket();
        stats_.error = msg;
        stats_.listening = false;
        return false;
    }

    void commitLatestPacket(const TelemetryPacket& pkt, size_t drainedCount) {
        const double nowEpoch = epochSeconds();
        const double nowSteady = steadySeconds();

        if (haveSeq_) {
            const uint32_t expected = lastSeq_ + 1;
            const uint32_t gap = pkt.seq - expected;
            if (gap > 0 && gap < 1000 && gap >= drainedCount) {
                windowLost_ += (gap - (drainedCount - 1));
            }
        }
        haveSeq_ = true;
        lastSeq_ = pkt.seq;

        latestPacket_ = pkt;
        ++packetVersion_;
        lastPacketSteady_ = nowSteady;

        const double udpMs = std::max(0.0, (nowEpoch - pkt.sendTs) * 1000.0);
        const double pipeMs = std::max(0.0, (nowEpoch - pkt.captureTs) * 1000.0);
        const bool first = (stats_.packets == 0);
        stats_.udpLatencyMs = first ? udpMs : (stats_.udpLatencyMs + 0.15 * (udpMs - stats_.udpLatencyMs));
        stats_.pipelineLatencyMs =
            first ? pipeMs : (stats_.pipelineLatencyMs + 0.15 * (pipeMs - stats_.pipelineLatencyMs));
        stats_.packets += drainedCount;
        if (drainedCount > 1) stats_.drainedStale += (drainedCount - 1);
        windowPackets_ += drainedCount;
    }

    void handleVideoChunk(const uint8_t* data, size_t n) {
        proto::VideoChunkHeader h;
        if (!proto::parseVideoHeader(data, n, h)) return;
        const size_t rowBytes = static_cast<size_t>(h.width) * h.channels;
        const size_t payload = n - proto::kVideoHeaderSize;
        if (h.rowStart + h.rowCount > h.height || payload < rowBytes * h.rowCount) return;

        if (h.width != asmW_ || h.height != asmH_ || h.channels != asmC_) {
            asmW_ = h.width;
            asmH_ = h.height;
            asmC_ = h.channels;
            assembly_.assign(rowBytes * h.height, 0);
            asmRows_ = 0;
            asmFrame_ = h.frameId;
            asmPublished_ = false;
        } else if (h.frameId != asmFrame_) {
            if (asmRows_ * 4 >= static_cast<size_t>(asmH_) * 3 && !asmPublished_) publishVideo();
            asmFrame_ = h.frameId;
            asmRows_ = 0;
            asmPublished_ = false;
        }
        std::memcpy(assembly_.data() + rowBytes * h.rowStart, data + proto::kVideoHeaderSize,
                    rowBytes * h.rowCount);
        asmRows_ += h.rowCount;
        if (asmRows_ >= asmH_ && !asmPublished_) publishVideo();
    }

    void publishVideo() {
        asmPublished_ = true;
        video_.width = asmW_;
        video_.height = asmH_;
        video_.channels = asmC_;
        video_.pixels = assembly_;
        ++video_.version;
        lastVideoSteady_ = steadySeconds();
    }

    void updateRateWindow() {
        const double now = steadySeconds();
        const double span = now - windowStart_;
        if (span >= 1.0) {
            stats_.packetsPerSec = windowPackets_ / span;
            const uint64_t total = windowPackets_ + windowLost_;
            stats_.lossPct = total ? (100.0 * static_cast<double>(windowLost_) / total) : 0.0;
            windowStart_ = now;
            windowPackets_ = 0;
            windowLost_ = 0;
        }
    }

    uint16_t port_;
    std::string bindAddress_;
    socket_t sock_ = kInvalid;
#if defined(_WIN32)
    bool wsaStarted_ = false;
#endif

    TelemetryPacket latestPacket_{};
    uint64_t packetVersion_ = 0;
    LinkStats stats_{};
    double lastPacketSteady_ = 0.0;
    double lastVideoSteady_ = 0.0;
    bool haveSeq_ = false;
    uint32_t lastSeq_ = 0;
    double windowStart_ = 0.0;
    uint64_t windowPackets_ = 0;
    uint64_t windowLost_ = 0;

    std::vector<uint8_t> assembly_;
    uint16_t asmW_ = 0, asmH_ = 0;
    uint8_t asmC_ = 3;
    uint32_t asmFrame_ = 0;
    size_t asmRows_ = 0;
    bool asmPublished_ = false;
    VideoFrame video_{};
};

}  // namespace v3d
