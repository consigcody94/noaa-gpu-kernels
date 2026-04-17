/**
 * Fast parallel CSV pivot: read 142K nex-*_output.csv files in parallel,
 * write a single forcings.bin (timestep x reach) tensor.
 *
 * This replaces the 32.7s single-thread Python parse_qlat.py.
 * Expected speedup: near-linear with thread count on modern hardware,
 * closing the I/O gap with JoshCu's 32-core Rust pipeline (~2.5s for the
 * equivalent pivot in rs_route).
 *
 * Strategy:
 *   - List all nex-*_output.csv paths (one pass over directory).
 *   - Launch N worker threads, each reading a chunk of files.
 *   - Each worker parses 25 rows per CSV (timestep, time, qlat).
 *   - Writes to its own row of the [timestep, reach] output tensor.
 *
 * Single binary output matches our existing forcings.bin format:
 *   int32 n_timesteps
 *   float32 qlat[n_timesteps, n_reaches]
 *   float32 qup0[n_reaches]         (zeros)
 *   float32 qdp0[n_reaches]         (zeros)
 *   float32 dp0[n_reaches]          (half-meter default)
 *
 * Build: g++ -O3 -std=c++17 -fopenmp parse_qlat_fast_cpp.cpp -o parse_qlat_fast
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <unordered_map>

namespace fs = std::filesystem;

static int strip_prefix(const std::string& name) {
    // "nex-1328591_output.csv" -> 1328591
    size_t dash = name.find('-');
    if (dash == std::string::npos) return -1;
    size_t underscore = name.find('_', dash + 1);
    if (underscore == std::string::npos) {
        size_t dot = name.find('.', dash + 1);
        if (dot == std::string::npos) return -1;
        underscore = dot;
    }
    try {
        return std::stoi(name.substr(dash + 1, underscore - dash - 1));
    } catch (...) {
        return -1;
    }
}

// Parse one CSV line looking like "0, 2017-01-01 00:00:00, 0.00582754"
// Returns true if parsed timestep + value successfully
static bool parse_line(const char* line, size_t len, int& ts, float& val) {
    // Find first comma (after ts)
    size_t i = 0;
    while (i < len && line[i] != ',') i++;
    if (i == len) return false;
    ts = std::atoi(line);  // stops at comma
    // Find last comma (before val)
    size_t j = len;
    while (j > 0 && line[j - 1] != ',') j--;
    if (j == 0 || j >= len) return false;
    val = (float)std::atof(line + j);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <csv_dir> <topo_to_wbid.npy> <out_forcings.bin> [n_timesteps=24] [threads=0 (auto)]\n", argv[0]);
        return 1;
    }
    std::string csv_dir = argv[1];
    std::string wbid_path = argv[2];
    std::string out_path = argv[3];
    int n_timesteps = (argc >= 5) ? std::atoi(argv[4]) : 24;
    int n_threads = (argc >= 6) ? std::atoi(argv[5]) : 0;
    if (n_threads <= 0) n_threads = (int)std::thread::hardware_concurrency();

    auto t0 = std::chrono::steady_clock::now();

    // Load topo_to_wbid: numpy .npy file, int64
    // Skip numpy header (128-byte boundary)
    std::ifstream f_npy(wbid_path, std::ios::binary);
    if (!f_npy) { fprintf(stderr, "cant open %s\n", wbid_path.c_str()); return 1; }
    char magic[6];
    f_npy.read(magic, 6);
    // Skip version
    char ver[2]; f_npy.read(ver, 2);
    uint16_t header_len;
    f_npy.read((char*)&header_len, 2);
    std::vector<char> header(header_len);
    f_npy.read(header.data(), header_len);
    // Figure out n_reaches from the header (look for 'shape': (N,))
    // For simplicity, read rest of file and divide by 8 bytes (int64)
    auto header_end = f_npy.tellg();
    f_npy.seekg(0, std::ios::end);
    auto total = f_npy.tellg();
    size_t n_reaches = (size_t)(total - header_end) / 8;
    f_npy.seekg(header_end);
    std::vector<int64_t> topo_to_wbid(n_reaches);
    f_npy.read((char*)topo_to_wbid.data(), n_reaches * 8);
    f_npy.close();

    printf("[cpp] loaded topo_to_wbid: %zu reaches\n", n_reaches);

    // Build wbid -> topo index map
    std::unordered_map<int64_t, int> wbid_to_topo;
    wbid_to_topo.reserve(n_reaches * 2);
    for (size_t i = 0; i < n_reaches; ++i) {
        wbid_to_topo[topo_to_wbid[i]] = (int)i;
    }

    // List all nex-*_output.csv files
    std::vector<std::string> paths;
    paths.reserve(150000);
    for (auto& entry : fs::directory_iterator(csv_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.rfind("nex-", 0) != 0) continue;
        if (name.find("_output.csv") == std::string::npos) continue;
        paths.push_back(entry.path().string());
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("[cpp] listed %zu nex csv files in %.1f ms\n", paths.size(),
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Allocate qlat tensor
    std::vector<float> qlat((size_t)n_timesteps * n_reaches, 0.0f);
    std::atomic<size_t> matched{0}, missing{0};

    auto t2 = std::chrono::steady_clock::now();

    // Parallel parse
    size_t per_thread = (paths.size() + n_threads - 1) / n_threads;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    for (int tid = 0; tid < n_threads; ++tid) {
        size_t start = tid * per_thread;
        size_t end = std::min(start + per_thread, paths.size());
        if (start >= end) break;
        workers.emplace_back([&, start, end]() {
            // Use a per-thread buffer to avoid lock contention on reading
            std::vector<char> buffer(4096);
            for (size_t pi = start; pi < end; ++pi) {
                const std::string& path = paths[pi];
                std::string name = fs::path(path).filename().string();
                int wbid = strip_prefix(name);
                if (wbid < 0) { missing.fetch_add(1); continue; }
                auto it = wbid_to_topo.find((int64_t)wbid);
                if (it == wbid_to_topo.end()) { missing.fetch_add(1); continue; }
                int tidx = it->second;

                // Read whole file (small, ~500 bytes)
                FILE* fp = std::fopen(path.c_str(), "rb");
                if (!fp) { missing.fetch_add(1); continue; }
                std::fseek(fp, 0, SEEK_END);
                long size = std::ftell(fp);
                std::fseek(fp, 0, SEEK_SET);
                if ((long)buffer.size() < size + 1) buffer.resize(size + 1);
                size_t got = std::fread(buffer.data(), 1, size, fp);
                buffer[got] = 0;
                std::fclose(fp);

                // Split by lines, parse each
                size_t i = 0;
                while (i < got) {
                    size_t lstart = i;
                    while (i < got && buffer[i] != '\n') i++;
                    size_t llen = i - lstart;
                    if (i < got) i++;  // skip '\n'
                    if (llen < 5) continue;  // skip empty
                    int ts;
                    float val;
                    if (!parse_line(buffer.data() + lstart, llen, ts, val)) continue;
                    if (ts >= 0 && ts < n_timesteps) {
                        qlat[(size_t)ts * n_reaches + tidx] = val;
                    }
                }
                matched.fetch_add(1);
            }
        });
    }
    for (auto& w : workers) w.join();

    auto t3 = std::chrono::steady_clock::now();
    double parse_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("[cpp] parsed %zu csv (missing %zu) in %.1f ms using %d threads (%.0f files/s)\n",
           matched.load(), missing.load(), parse_ms, n_threads,
           paths.size() / (parse_ms / 1000.0));

    // Write forcings.bin in our existing format
    std::vector<float> qup0(n_reaches, 0.0f);
    std::vector<float> qdp0(n_reaches, 0.0f);
    std::vector<float> dp0(n_reaches, 0.5f);

    FILE* of = std::fopen(out_path.c_str(), "wb");
    if (!of) { fprintf(stderr, "cant open %s for write\n", out_path.c_str()); return 1; }
    int32_t nt = n_timesteps;
    std::fwrite(&nt, 4, 1, of);
    std::fwrite(qlat.data(), 4, qlat.size(), of);
    std::fwrite(qup0.data(), 4, n_reaches, of);
    std::fwrite(qdp0.data(), 4, n_reaches, of);
    std::fwrite(dp0.data(), 4, n_reaches, of);
    std::fclose(of);

    auto t4 = std::chrono::steady_clock::now();
    printf("[cpp] wrote %s (%.1f MB) in %.1f ms\n", out_path.c_str(),
           fs::file_size(out_path) / 1e6,
           std::chrono::duration<double, std::milli>(t4 - t3).count());
    printf("[cpp] TOTAL end-to-end: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t4 - t0).count());

    return 0;
}
