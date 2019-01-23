// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ctpl_stl.h" //Threadpool

// Internal libraries
#include "ARGPARSE/parseArgs.h"
#include "SMA/BufferedSparseMatrixDoubleReader.hpp"
#include "SMA/BufferedSparseMatrixFloatReader.hpp"
#include "rawop.h"

using namespace CTL;

struct Args
{
    std::vector<std::string> typeStrings;
    Args()
    {
        typeStrings.push_back("float");
        typeStrings.push_back("double");
    }
    std::string inputVolume;
    std::string inputSystemMatrix;
    std::string systemMatrixType;
    std::string outputProjection;
    uint32_t threads = 1;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    uint16_t projectionSizeX = 616;
    uint16_t projectionSizeY = 480;
    uint16_t projectionSizeZ = 1;
    // Here (0,0,0) is in the center of the volume
    uint16_t volumeSizeX = 256;
    uint16_t volumeSizeY = 256;
    uint16_t volumeSizeZ = 199;

    bool force = false;
    int parseArguments(int argc, char* argv[]);
    std::string checkTypeConsistency(const std::string& s)
    {
        if(std::find(typeStrings.begin(), typeStrings.end(), s) == typeStrings.end())
        {
            return io::xprintf("The string that represents type is float or double not %s!",
                               s.c_str());
        }
        return "";
    }
};

int Args::parseArguments(int argc, char* argv[])
{
    // Checking type of system matrix
    std::function<std::string(const std::string&)> f
        = std::bind(&Args::checkTypeConsistency, this, std::placeholders::_1);

    CLI::App app{ "Project volume based on sparse system matrix." };
    app.add_option("input_volume", inputVolume,
                   "Volume in a DEN format to process. It is expected that the type is float.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("system_matrix", inputSystemMatrix,
                   "System matrix to process to project volume data.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("system_matrix_type", systemMatrixType,
                   "Type of the system matrix, that is float or double.")
        ->required()
        ->check(f);
    app.add_option("output_projection", outputProjection,
                   "DEN file with the projections to output as floats.")
        ->required();

    app.add_option("-j,--threads", threads, "Number of extra threads that application can use.")
        ->check(CLI::Range(1, 65535));
    app.add_flag("-f,--force", force, "Overwrite output_projection if it exists.");
    CLI::Option* px
        = app.add_option("--projx", projectionSizeX, "Dimension of detector, defaults to 616.");
    CLI::Option* py
        = app.add_option("--projy", projectionSizeY, "Dimension of detector, defaults to 480.");
    CLI::Option* vx
        = app.add_option("--volumex", volumeSizeX, "Dimension of volume, defaults to 256.");
    CLI::Option* vy
        = app.add_option("--volumey", volumeSizeY, "Dimension of volume, defaults to 256.");
    CLI::Option* vz
        = app.add_option("--volumez", volumeSizeZ, "Dimension of volume, defaults to 199.");
    px->needs(py);
    py->needs(px);
    vx->needs(vy)->needs(vz);
    vy->needs(vx)->needs(vz);
    vz->needs(vx)->needs(vy);
    app.parse(argc, argv);
    try
    {
        app.parse(argc, argv);
        if(!force)
        {
            if(io::fileExists(outputProjection))
            {
                std::string msg
                    = "Error: output file already exists, use --force to force overwrite.";
                LOGE << msg;
                return 1;
            }
        }
    } catch(const CLI::CallForHelp e)
    {
        app.exit(e); // Prints help message
        return 1;
    } catch(const CLI::ParseError& e)
    {
        int exitcode = app.exit(e);
        LOGE << io::xprintf("There was perse error with exit code %d catched.\n %s", exitcode,
                            app.help().c_str());
        return -1;
    } catch(...)
    {
        LOGE << "Unknown exception catched";
        return -1;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel = plog::debug; // debug, info, ...
    std::string csvLogFile = io::xprintf(
        "/tmp/%s.csv", io::getBasename(std::string(argv[0])).c_str()); // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    // Argument parsing
    Args a;
    int parseResult = a.parseArguments(argc, argv);
    if(parseResult != 0)
    {
        if(parseResult > 0)
        {
            return 0; // Exited sucesfully, help message printed
        } else
        {
            return -1; // Exited somehow wrong
        }
    }
    LOGI << io::xprintf("START %s", argv[0]);
    // Argument parsing

    // Frames to process
    uint64_t totalVolumeSize = uint64_t(a.volumeSizeX) * a.volumeSizeY * a.volumeSizeZ;
    float* volume = new float[totalVolumeSize];
    io::readBytesFrom(a.inputVolume, 6, (uint8_t*)volume, totalVolumeSize * 4);
    uint64_t totalProjectionSize = uint64_t(a.projectionSizeX) * a.projectionSizeY * a.projectionSizeZ;
    float* projection = new float[totalProjectionSize](); // Initialized by zeros
    if(a.systemMatrixType == "float")
    {
        uint32_t i, j;
        float v;
        matrix::BufferedSparseMatrixFloatReader A(a.inputSystemMatrix, 16384);
        uint64_t elements = A.getNumberOfElements();

        while(elements != 0)
        {
            A.readNextValue(&i, &j, &v);
            projection[j] += volume[i] * v;
            elements--;
        }
    } else if(a.systemMatrixType == "double")
    {
        uint32_t i, j;
        double v;
        matrix::BufferedSparseMatrixDoubleReader A(a.inputSystemMatrix, 16384);
        uint64_t elements = A.getNumberOfElements();
        while(elements != 0)
        {
            A.readNextValue(&i, &j, &v);
            projection[j] += float(volume[i] * v);
            elements--;
        }
    }
    uint16_t buf[3];
    buf[0] = a.projectionSizeY;
    buf[1] = a.projectionSizeX;
    buf[2] = a.projectionSizeZ;
    io::createEmptyFile(a.outputProjection, 0, true); // Try if this is faster
    io::appendBytes(a.outputProjection, (uint8_t*)buf, 6);
    io::appendBytes(a.outputProjection, (uint8_t*)projection, totalProjectionSize * 4);
    delete[] volume;
    delete[] projection;
    LOGI << io::xprintf("END %s", argv[0]);
}
