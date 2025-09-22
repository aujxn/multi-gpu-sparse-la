#include "utils.h"

#include <fstream>
#include <string>

// Helper to read a numeric token from a file (cgroups)
static long long read_num_token(const std::string& p)
{
    std::ifstream f(p);
    if (!f.good()) return -1;
    std::string tok; f >> tok; if (!f.good()) return -1;
    if (tok == "max") return -1; // unlimited in cgroup v2
    try { return std::stoll(tok); } catch (...) { return -1; }
}

void StageLogger::log(const char* stage)
{
    using namespace std::chrono;
    auto now = clock::now();
    double dt_prev = duration<double>(now - last_).count();
    double dt_total = duration<double>(now - start_).count();
    last_ = now;

    if (!debug_mem_) {
        logging::debug_logf(rank_, "[stage] %s | dt=%.6f s total=%.6f s", stage, dt_prev, dt_total);
        return;
    }

    // Host memory (RSS/VmSize)
    long rss_kb = -1, vms_kb = -1;
    {
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("VmRSS:", 0) == 0) {
                std::string num; for (char c : line) if ((c>='0'&&c<='9')||c==' ') num.push_back(c);
                try { rss_kb = std::stol(num); } catch (...) { rss_kb = -1; }
            } else if (line.rfind("VmSize:", 0) == 0) {
                std::string num; for (char c : line) if ((c>='0'&&c<='9')||c==' ') num.push_back(c);
                try { vms_kb = std::stol(num); } catch (...) { vms_kb = -1; }
            }
            if (rss_kb >= 0 && vms_kb >= 0) break;
        }
    }
    double rss_gib = rss_kb > 0 ? (double)rss_kb / (1024.0*1024.0) : -1.0;
    double vms_gib = vms_kb > 0 ? (double)vms_kb / (1024.0*1024.0) : -1.0;

    // CGroup memory (current/max)
    struct { long long cur=-1, max=-1; bool ok=false; } cgm;
    {
        std::ifstream cg("/proc/self/cgroup");
        std::string line, v2_path, v1_mem_path;
        while (std::getline(cg, line)) {
            size_t c1 = line.find(':'), c2 = (c1==std::string::npos)?std::string::npos:line.find(':', c1+1);
            if (c1 == std::string::npos || c2 == std::string::npos) continue;
            std::string ctrls = line.substr(c1+1, c2-c1-1);
            std::string path  = line.substr(c2+1);
            if (ctrls.empty()) { v2_path = path; }
            if (ctrls.find("memory") != std::string::npos) { v1_mem_path = path; }
        }
        if (!v2_path.empty()) {
            std::string base = std::string("/sys/fs/cgroup") + v2_path;
            long long cur = read_num_token(base + "/memory.current");
            long long max = read_num_token(base + "/memory.max");
            if (cur >= 0) { cgm = {cur, max, true}; }
        }
        if (!cgm.ok && !v1_mem_path.empty()) {
            std::string base = std::string("/sys/fs/cgroup/memory") + v1_mem_path;
            long long cur = read_num_token(base + "/memory.usage_in_bytes");
            long long max = read_num_token(base + "/memory.limit_in_bytes");
            if (cur >= 0) { cgm = {cur, max, true}; }
        }
    }
    double cg_cur_gib = cgm.ok ? (double)cgm.cur/(1024.0*1024.0*1024.0) : -1.0;
    double cg_max_gib = (cgm.ok && cgm.max>0 && cgm.max < (1LL<<60)) ? (double)cgm.max/(1024.0*1024.0*1024.0) : -1.0;

    // GPU memory (optional)
    size_t free_b=0,total_b=0; double free_gib=-1.0,total_gib=-1.0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        free_gib = (double)free_b / (1024.0*1024.0*1024.0);
        total_gib = (double)total_b / (1024.0*1024.0*1024.0);
    }

    if (cgm.ok) {
        if (cg_max_gib > 0.0)
            logging::debug_logf(rank_, "[stage] %s | dt=%.6f s total=%.6f s | Host: RSS=%.2f GiB VMS=%.2f GiB | CGroup mem=%.2f/%.2f GiB | GPU free=%.2f/%.2f GiB",
                                stage, dt_prev, dt_total, rss_gib, vms_gib, cg_cur_gib, cg_max_gib, free_gib, total_gib);
        else
            logging::debug_logf(rank_, "[stage] %s | dt=%.6f s total=%.6f s | Host: RSS=%.2f GiB VMS=%.2f GiB | CGroup mem=%.2f GiB | GPU free=%.2f/%.2f GiB",
                                stage, dt_prev, dt_total, rss_gib, vms_gib, cg_cur_gib, free_gib, total_gib);
    } else {
        logging::debug_logf(rank_, "[stage] %s | dt=%.6f s total=%.6f s | Host: RSS=%.2f GiB VMS=%.2f GiB | GPU free=%.2f/%.2f GiB",
                            stage, dt_prev, dt_total, rss_gib, vms_gib, free_gib, total_gib);
    }
}

