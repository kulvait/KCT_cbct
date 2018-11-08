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
#include "DEN/DenProjectionMatrixReader.hpp"
#include "SMA/BufferedSparseMatrixWritter.hpp"

#include "DivideAndConquerFootprintExecutor.hpp"

using namespace CTL;

int main(int argc, char* argv[])
{
    plog::Severity verbosityLevel = plog::debug; // Set to debug to see the
                                                 // debug messages, info
                                                 // messages
    std::string csvLogFile = "/tmp/dacprojector.csv"; // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "projector";
    // Argument parsing
    std::string a_frameSpecs = "";
    int a_eachkth = 1;
    int a_threads = 1;
    // It is evaluated from -0.5, pixels are centerred at integer coordinates
    int projectionSizeX = 616;
    int projectionSizeY = 480;
    // Here (0,0,0) is in the center of the volume
    int volumeSizeX = 256;
    int volumeSizeY = 256;
    int volumeSizeZ = 199;

    std::string a_outputSystemMatrix;
    std::string a_projectionMatrices;
    CLI::App app{ "Using divide and conquer techniques to construct CT system matrix.." };
    app.add_option(
        "-f,--frames", a_frameSpecs,
        "Specify only particular projection matrices to process. You can input range i.e. 0-20 or "
        "also individual comma separated frames i.e. 1,8,9. Order does matter. Accepts "
        "end literal that means total number of slices of the input.");
    app.add_option("-k,--each-kth", a_eachkth,
                   "Process only each k-th frame intended to output. The frames to output "
                   "are then 1st specified, 1+kN, N=1...\\infty if such frame exists. Parameter k "
                   "must be positive integer.")
        ->check(CLI::Range(1, 65535));
    app.add_option("-j,--threads", a_threads, "Number of extra threads that application can use.")
        ->check(CLI::Range(1, 65535));
    app.add_option("output_system_matrix", a_outputSystemMatrix,
                   "File in a sparse matrix format to output.")
        ->required()
        ->check(CLI::NonexistentPath);
    app.add_option("input_matrices", a_projectionMatrices,
                   "Files in a DEN format to process. These files represents projection matrices.")
        ->required()
        ->check(CLI::ExistingFile);
    CLI::Option* px = app.add_option("--projx", projectionSizeX, "Dimension of detector");
    CLI::Option* py = app.add_option("--projy", projectionSizeY, "Dimension of detector");
    CLI::Option* vx = app.add_option("--volumex", projectionSizeY, "Dimension of volume");
    CLI::Option* vy = app.add_option("--volumey", projectionSizeY, "Dimension of volume");
    CLI::Option* vz = app.add_option("--volumez", projectionSizeY, "Dimension of volume");
    px->needs(py);
    py->needs(px);
    vx->needs(vy)->needs(vz);
    vy->needs(vx)->needs(vz);
    vz->needs(vx)->needs(vy);
    CLI11_PARSE(app, argc, argv);
    LOGD << io::xprintf("Optional parameters: frames=%s, eachkth=%d, threads=%d.",
                        a_frameSpecs.c_str(), a_eachkth, a_threads);
    // Frames to process
    std::shared_ptr<io::DenProjectionMatrixReader> dr
        = std::make_shared<io::DenProjectionMatrixReader>(a_projectionMatrices);
    int count = dr->count();
    if(uint64_t(volumeSizeX) * uint64_t(volumeSizeY) * uint64_t(volumeSizeZ) > INT_MAX)
    {
        io::throwerr("Implement indexing by uint64_t matrix dimension overflow of voxels count.");
    }
    if(uint64_t(projectionSizeX) * uint64_t(projectionSizeY) * uint64_t(count) > INT_MAX)
    {
        io::throwerr(
            "Implement indexing by uint64_t matrix dimension overflow of projection pixels count.");
    }
    // LOGD << io::xprintf("The file %s has dimensions (x,y,z)=(%d, %d, %d)",
    //                    a_inputDenFiles[0].c_str(), dimx, dimy, dimz);
    std::vector<int> framesToProcess = util::processFramesSpecification(a_frameSpecs, count);
    std::vector<int> framesToOutput;
    for(int i = 0; i != framesToProcess.size(); i++)
    {
        if(i % a_eachkth == 0)
        {
            framesToOutput.push_back(framesToProcess[i]);
        }
    }
    LOGD << io::xprintf("Number of projections to process is %d.", framesToOutput.size());
    // End parsing arguments
    std::shared_ptr<matrix::BufferedSparseMatrixWritter> matrixWritter
        = std::make_shared<matrix::BufferedSparseMatrixWritter>(a_outputSystemMatrix);
    util::DivideAndConquerFootprintExecutor dfe(matrixWritter, projectionSizeX, projectionSizeY,
                                          volumeSizeX, volumeSizeY, volumeSizeZ, a_threads);
    uint32_t projnum;
    for(int i = 0; i != framesToOutput.size(); i++)
   {	
        projnum = framesToOutput[i];
LOGD << io::xprintf("Output of %d matrix.", projnum);
        uint32_t pixelIndexOffset = projnum * projectionSizeX * projectionSizeY;
        util::ProjectionMatrix pm = dr->readMatrix(projnum);
	dfe.insertMatrixProjections(pm, pixelIndexOffset);
    }
}
