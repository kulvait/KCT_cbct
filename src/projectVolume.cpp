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
#include "SMA/BufferedSparseMatrixReader.hpp"
#include "rawop.h"

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
    CLI11_PARSE(app, argc, argv);
    // Frames to process
    int totalVolumeSize = volumeSizeX * volumeSizeY * volumeSizeZ;
    float* volume = new float[volumeSizeX * volumeSizeY * volumeSizeZ];
    io::readBytesFrom(a_inputVolume, 6, (uint8_t*)volume, totalVolumeSize * 4);
    int i, j;
    double v;

    matrix::BufferedSparseMatrixReader A(a_inputSystemMatrix);
    uint64_t elements = A.getNumberOfElements();
    uint64_t totalProjectionSize = projectionSizeX * projectionSizeY * projectionSizeZ;
    float* projection = new float[totalProjectionSize](); // Initialized by zeros
    bool* acessed = new bool[totalProjectionSize](); // Initialized by zeros

    uint8_t buf[6];
    
    float * testVolumeSlice = new float[volumeSizeX * volumeSizeY]();

    std::memcpy(testVolumeSlice, &volume[198*volumeSizeX*volumeSizeY], volumeSizeX*volumeSizeY*4);
        util::putUint16((uint16_t)volumeSizeY, &buf[0]);
        util::putUint16((uint16_t)volumeSizeX, &buf[2]);
        util::putUint16((uint16_t)1, &buf[4]);
    io::createEmptyFile("/b/git/DivideConquerProjector/build/xxx", 6+ volumeSizeX*volumeSizeY*4,
    true); io::writeFirstBytes("/b/git/DivideConquerProjector/build/xxx", buf, 6);
    io::writeBytesFrom("/b/git/DivideConquerProjector/build/xxx", 6, (uint8_t*)testVolumeSlice,
    volumeSizeX*volumeSizeY*4);
    
	float addme;
    while(elements != 0)
    {
        A.readNextValue(&i, &j, &v);
	if(i>=totalVolumeSize)
	{
		LOGE << io::xprintf("Coordinates vol=(%d, %d, %d) are invalid!", i%volumeSizeX, (i/volumeSizeX)%volumeSizeY, (i/volumeSizeX)/volumeSizeY);  
	}
/*
	if(j%projectionSizeX == 2 && j/projectionSizeX ==2)
	{
		LOGE << io::xprintf("v=%e,i=%d,vol=(%d, %d, %d), volume[i] =%e, v*volume[i]=%e ", v,i, i%volumeSizeX, (i/volumeSizeX)%volumeSizeY, (i/volumeSizeX)/volumeSizeY, volume[i], v*volume[i]);
		LOGE << io::xprintf("vol(0,0,199)=%e", volume[199*volumeSizeX*volumeSizeY]);
	}
*/
	if(j>=totalProjectionSize)
		{
			LOGD << "BIG";
		}
        //	LOGD << io::xprintf("elements = %lu, i=%d, j=%d, v=%e.", elements, i, j, v);
	addme = volume[i] * v;
        projection[j] += addme;
	acessed[j] = true;
        elements--;
    }
/*
	for(j = 0; j!= totalProjectionSize; j++)
	{
		
		if(acessed[j] == true)
			LOGD << io::xprintf("The (i,j) = (%d, %d) was acessed", j%projectionSizeX, j/projectionSizeX);
	}
  */  uint64_t totalFileSize = uint64_t(6) + totalProjectionSize * 4;
    io::createEmptyFile(a_projectionFile, totalFileSize, true);
    /// io::createEmptyFile(a_projectionFile, 0, true); //Try if this is faster
    util::putUint16((uint16_t)projectionSizeY, &buf[0]);
    util::putUint16((uint16_t)projectionSizeX, &buf[2]);
    util::putUint16((uint16_t)projectionSizeZ, &buf[4]);
    io::writeFirstBytes(a_projectionFile, buf, 6);
    io::writeBytesFrom(a_projectionFile, 6, (uint8_t*)projection, totalProjectionSize * 4);
    // io::appendBytes(a_projectionFile, buf, 6);
    // io::appendBytes(a_projectionFile, projection, totalProjectionSize*4);
    delete[] volume;
    delete[] projection;
    LOGI << "END";
}
