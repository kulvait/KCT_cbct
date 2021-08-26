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
#include "SMA/BufferedSparseMatrixFloatReader.hpp"
#include "rawop.h"

using namespace KCT;

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel = plog::debug; // Set to debug to see the
                                                 // debug messages, info
                                                 // messages
    std::string csvLogFile = "/tmp/dacprojector.csv"; // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "START";
    // Argument parsing
    int a_threads = 1;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    int projectionSizeX = 616;
    int projectionSizeY = 480;
    int projectionSizeZ = 1;
    // Here (0,0,0) is in the center of the volume
    int volumeSizeX = 256;
    int volumeSizeY = 256;
    int volumeSizeZ = 199;

    std::string a_inputVolume;
    std::string a_inputSystemMatrix;
    std::string a_projectionFile;

    CLI::App app{ "Using divide and conquer techniques to construct CT system matrix.." };
    app.add_option("-j,--threads", a_threads, "Number of extra threads that application can use.")
        ->check(CLI::Range(1, 65535));
    app.add_option("-n,--number_of_projections", projectionSizeZ,
                   "Number of projections, defaults to 1.")
        ->check(CLI::Range(1, 65535));
    app.add_option("input_volume", a_inputVolume,
                   "Files in a DEN format to process. These files represents projection matrices.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("system_matrix", a_inputSystemMatrix,
                   "Files in a DEN format to process. These files represents projection matrices.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("output_file", a_projectionFile, "File in a sparse matrix format to output.")
        ->required();
    app.parse(argc, argv);
    // Frames to process
    int totalVolumeSize = volumeSizeX * volumeSizeY * volumeSizeZ;
    float* volume = new float[volumeSizeX * volumeSizeY * volumeSizeZ];
    io::readBytesFrom(a_inputVolume, 6, (uint8_t*)volume, totalVolumeSize * 4);
    uint32_t i, j;
    float v;

    matrix::BufferedSparseMatrixFloatReader A(a_inputSystemMatrix, 16384);
    uint64_t elements = A.getNumberOfElements();
    uint32_t totalProjectionSize = projectionSizeX * projectionSizeY * projectionSizeZ;
    float* projection = new float[totalProjectionSize](); // Initialized by zeros

    uint16_t buf[3];
    buf[0] = projectionSizeY;
    buf[1] = projectionSizeX;
    buf[2] = projectionSizeZ;
    while(elements != 0)
    {
        A.readNextValue(&i, &j, &v);
        projection[j] += volume[i] * v;
        elements--;
    } /*
         float curvol;
         uint32_t previ = totalProjectionSize;
     while(elements != 0)
     {
         A.readNextValue(&i, &j, &v);
         if(i!=previ)
         {
                 curvol = volume[i];
                 previ = i;
         }
         projection[j] += curvol * v;
         elements--;
     }*/
    io::createEmptyFile(a_projectionFile, 0, true); // Try if this is faster
    io::appendBytes(a_projectionFile, (uint8_t*)buf, 6);
    io::appendBytes(a_projectionFile, (uint8_t*)projection, totalProjectionSize * 4);
    delete[] volume;
    delete[] projection;
    LOGI << "END";
}
