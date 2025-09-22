#include "utils.h"

#include <memory>
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <filesystem>

static std::string sanitize_job_name(const char* name) {
    if (!name || !*name) return std::string("debug");
    std::string out; out.reserve(64);
    for (const char* p = name; *p; ++p) {
        char c = *p;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) out = "debug";
    return out;
}

static std::shared_ptr<spdlog::logger> get_or_create_logger(int pe) {
    char lname[32];
    snprintf(lname, sizeof(lname), "PE%d", pe);
    auto logger = spdlog::get(lname);
    if (logger) return logger;
    try {
        const char* job = std::getenv("SLURM_JOB_ID");
        const char* jname = std::getenv("SLURM_JOB_NAME");
        std::string base = sanitize_job_name(jname);
        char path[512];
        // Ensure logs directory exists (ignore errors)
        std::error_code ec;
        std::filesystem::create_directories("logs", ec);
        if (job && *job) {
            snprintf(path, sizeof(path), "logs/%s.%s.rank%04d.out", base.c_str(), job, pe);
        } else {
            snprintf(path, sizeof(path), "logs/%s.rank%04d.out", base.c_str(), pe);
        }
        auto sink = std::make_shared<spdlog::sinks::basic_file_sink_st>(path, true);
        auto new_logger = std::make_shared<spdlog::logger>(lname, sink);
        new_logger->set_pattern("%v");
        new_logger->flush_on(spdlog::level::trace);
        spdlog::register_logger(new_logger);
        return new_logger;
    } catch (...) {
        return nullptr;
    }
}

void debug_logf(int pe, const char* fmt, ...) {
    auto logger = get_or_create_logger(pe);
    if (!logger) return;
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    logger->info("{}", buf);
    logger->flush();
}

namespace logging {
void debug_logf(int pe, const char* fmt, ...) {
    auto logger = get_or_create_logger(pe);
    if (!logger) return;
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    logger->info("{}", buf);
    logger->flush();
}
} // namespace logging
