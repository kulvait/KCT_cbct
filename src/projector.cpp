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
#include "SMA/BufferedSparseMatrixDoubleWritter.hpp"

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
    util::ProjectionMatrix pm = dr->readMatrix(0);
    double pixelSpacingX = 0.616;
    double pixelSpacingY = 0.616;

    std::array<double, 3> sourcePosition = pm.sourcePosition();
    std::array<double, 3> normalToDetector = pm.normalToDetector();

    double x1, x2, y1, y2;
    pm.project(sourcePosition[0] + normalToDetector[0], sourcePosition[1] + normalToDetector[1],
               sourcePosition[2] + normalToDetector[2], &x1, &y1);
    pm.project(100.0, 100.0, 100.0, &x2, &y2);
    double xspacing2 = pixelSpacingX * pixelSpacingX;
    double yspacing2 = pixelSpacingY * pixelSpacingY;
    double distance
        = std::sqrt((x1 - x2) * (x1 - x2) * xspacing2 + (y1 - y2) * (y1 - y2) * yspacing2);
    double x = 100.0 - sourcePosition[0];
    double y = 100.0 - sourcePosition[1];
    double z = 100.0 - sourcePosition[2];
    double norma = std::sqrt(x * x + y * y + z * z);
    x /= norma;
    y /= norma;
    z /= norma;
    double cos = normalToDetector[0] * x + normalToDetector[1] * y + normalToDetector[2] * z;
    double theta = std::acos(cos);
    double distToDetector = std::abs(distance / std::tan(theta));
    double scalingFactor = distToDetector * distToDetector / pixelSpacingX / pixelSpacingY;
    LOGI << io::xprintf("Distance to the detector is %fmm therefore scaling factor is %f.",
                        distToDetector, scalingFactor);

    bool submatrices = true;
    if(!submatrices)
    {
        LOGD << io::xprintf("Number of projections to process is %d.", framesToOutput.size());
        // End parsing arguments
        std::shared_ptr<matrix::BufferedSparseMatrixWritter> matrixWritter
            = std::make_shared<matrix::BufferedSparseMatrixWritter>(a_outputSystemMatrix);
        util::DivideAndConquerFootprintExecutor dfe(matrixWritter, projectionSizeX, projectionSizeY,
                                                    volumeSizeX, volumeSizeY, volumeSizeZ,
                                                    scalingFactor, a_threads);
        uint32_t projnum;
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            dfe.startThreadpool();
            projnum = framesToOutput[i];
            LOGD << io::xprintf("Processing projections from %dth position.", projnum);
            uint32_t pixelIndexOffset = projnum * projectionSizeX * projectionSizeY;
            util::ProjectionMatrix pm = dr->readMatrix(projnum);
            dfe.insertMatrixProjections(pm, pixelIndexOffset);
            dfe.stopThreadpool();
            dfe.reportNumberOfWrites();
        }
    } else
    {
        // Write individual submatrices

        std::shared_ptr<matrix::BufferedSparseMatrixWritter> matrixWritter;
        uint32_t projnum;
        for(int i = 0; i != framesToOutput.size(); i++)
        {
            matrixWritter = std::make_shared<matrix::BufferedSparseMatrixWritter>(
                io::xprintf("file%s_%03d.sm", a_outputSystemMatrix.c_str(), i), 8192, true);
            util::DivideAndConquerFootprintExecutor dfe(matrixWritter, projectionSizeX,
                                                        projectionSizeY, volumeSizeX, volumeSizeY,
                                                        volumeSizeZ, scalingFactor, a_threads);
            dfe.startThreadpool();
            projnum = framesToOutput[i];
            LOGD << io::xprintf("Processing projections from %dth position.", projnum);
            uint32_t pixelIndexOffset = projnum * projectionSizeX * projectionSizeY;
            util::ProjectionMatrix pm = dr->readMatrix(projnum);
            dfe.insertMatrixProjections(pm, pixelIndexOffset);
            dfe.stopThreadpool();
            matrixWritter->flush();
            dfe.reportNumberOfWrites();
        }
    }
}
