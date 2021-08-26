#define CATCH_CONFIG_RUNNER
#include "PLOG/PlogSetup.h"
#include "PROG/RunTimeInfo.hpp"
#include "catch.hpp"

using namespace KCT;

int main(int argc, char* argv[])
{
    // global setup...
    bool logging = true;
    if(logging)
    {
        plog::Severity verbosityLevel
            = plog::debug; // Set to debug to see the debug messages, info messages
        std::string csvLogFile = "/tmp/imageRegistrationLog.csv"; // Set NULL to disable
        bool logToConsole = true;
        plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
        plogSetup.initLogging();
    }
    int result = Catch::Session().run(argc, argv);

    // global clean-up...

    return result;
}

std::string basedir()
{
    util::RunTimeInfo rti;
    std::string path = rti.getExecutableDirectoryPath(); // build dir
    return io::getParent(path);
}

TEST_CASE("LOGGING SETUP", "catch_main.cpp") {}
