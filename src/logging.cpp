#include "utils.h"

#include <memory>
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

void debug_logf(int pe, const char* fmt, ...) {
    char lname[32];
    snprintf(lname, sizeof(lname), "PE%d", pe);
    auto logger = spdlog::get(lname);
    if (!logger) {
        try {
            char path[256];
            snprintf(path, sizeof(path), "logs/debug.rank%04d.out", pe);
            auto sink = std::make_shared<spdlog::sinks::basic_file_sink_st>(path, true);
            logger = std::make_shared<spdlog::logger>(lname, sink);
            logger->set_pattern("%v");
            logger->flush_on(spdlog::level::trace);
            spdlog::register_logger(logger);
        } catch (...) { return; }
    }
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    logger->info("{}", buf);
    logger->flush();
}
