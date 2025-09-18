#include "utils.h"

#include <memory>
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

static std::shared_ptr<spdlog::logger> get_or_create_logger(int pe) {
    char lname[32];
    snprintf(lname, sizeof(lname), "PE%d", pe);
    auto logger = spdlog::get(lname);
    if (logger) return logger;
    try {
        const char* job = std::getenv("SLURM_JOB_ID");
        char path[512];
        if (job && *job) {
            snprintf(path, sizeof(path), "logs/debug.%s.rank%04d.out", job, pe);
        } else {
            snprintf(path, sizeof(path), "logs/debug.rank%04d.out", pe);
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
