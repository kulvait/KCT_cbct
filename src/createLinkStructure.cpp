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
#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

using namespace KCT;

void insertvectopix(std::string file, uint16_t num) {}

int main(int argc, char* argv[])
{

    uint16_t* vectopixbuf = new uint16_t[1024];
    uint32_t* pixtonumbuf = new uint8_t[5 * 1024];
    uint8_t* consecutivebuf = new uint8_t[1024];
    float* floatbuf = new float[1024];
    int vectopixpos = 0, pixtonumpos = 0, consetutivepos = 0, floatpos = 0;
    plog::Severity verbosityLevel = plog::debug; // Set to debug to see the
                                                 // debug messages, info
                                                 // messages
    std::string csvLogFile = "/tmp/dacprojector.csv"; // Set NULL to disable
    bool logToConsole = true;
    plog::PlogSetup plogSetup(verbosityLevel, csvLogFile, logToConsole);
    plogSetup.initLogging();
    LOGI << "sortsortsort";
    // Argument parsing
    std::string a_sortedFloat;
    std::string a_vectopix;
    std::string a_pixtoorder;
    std::string a_floats;
    CLI::App app{ "Using divide and conquer techniques to construct CT system matrix.." };
    app.add_option("sorted_float", a_sortedFloat,
                   "Files in a DEN format to process. These files represents projection matrices.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("vectopix", a_vectopix, "Contains number of pixel records in pixtoorder.")
        ->required()
        ->check(CLI::NonexistentPath);
    app.add_option("pixtoorder", a_pixtoorder,
                   "Contains pixel index and number of records for this pixel.")
        ->required()
        ->check(CLI::NonexistentPath);
    app.add_option("pixtoorder", a_floats, "Contains floats.")
        ->required()
        ->check(CLI::NonexistentPath);
    app.parse(argc, argv);
    // Frames to process
    matrix::BufferedSparseMatrixFloatReader readmatrix(a_sortedFloat);
    io::createEmptyFile(a_vectopix, 0, true); // Try if this is faster
    io::createEmptyFile(a_pixtoorder, 0, true); // Try if this is faster
    io::createEmptyFile(a_floats, 0, true); // Try if this is faster
    io::appendBytes(a_projectionFile, (uint8_t*)projection, totalProjectionSize * 4);
    uint32_t previ = 256 * 256 * 199;
    uint32_t prevj = 616 * 480 * 248;
    uint32_t initseqj = 616 * 480 * 248;
    uint16_t pixrecords = 0;
    uint8_t consecutive = 0;
    uint32_t i, j;
    float v;
    uint64_t totalfloatpos = 0;
    while(!readmatrix.atEnd())
    {
        readmatrix.readNextValue(&i, &j, &v);
        if(floatpos != 1024)
        {
            floatbuf[floatpos] = v;
            floatpos++;
        } else
        {
            io::appendBytes(a_floats, (uint8_t*)floatbuf, 1024 * 4);
            floatbuf[0] = v;
            floatpos = 1;
        }
        if(i == previ)
        {
            if(j == prevj + 1)
            {
                consecutive++;
            } else
            {
                if(consecutive != 0)
                {
                    if(pixtonumpos != 1024)
                    {
                        std::memcpy(pixtonumbuf + 5 * pixtonumpos, &initseqj, 4);
                        std::memcpy(pixtonumbuf + 5 * pixtonumpos + 4, &consecutive, 1);
                        pixtonumpos++;

                    } else
                    {
                        io::appendBytes(a_pixtoorder, (uint8_t*)pixtonumbuf, 5 * 1024);
                        std::memcpy(pixtonumbuf, &initseqj, 4);
                        std::memcpy(pixtonumbuf + 4, &consecutive, 1);
                        pixtonumpos = 1;
                    }
                }
                initseqj = j;
                consecutive = 1;
                pixrecords++;
            }
        } else
        {

            if(consecutive != 0)
            {
                if(pixtonumpos != 1024)
                {
                    std::memcpy(pixtonumbuf + 5 * pixtonumpos, &initseqj, 4);
                    std::memcpy(pixtonumbuf + 5 * pixtonumpos + 4, &consecutive, 1);
                    pixtonumpos++;

                } else
                {
                    io::appendBytes(a_pixtoorder, (uint8_t*)pixtonumbuf, 5 * 1024);
                    std::memcpy(pixtonumbuf, &initseqj, 4);
                    std::memcpy(pixtonumbuf + 4, &consecutive, 1);
                    pixtonumpos = 1;
                }
            }
            initseqj = j;
            consecutive = 1;
            if(pixrecords != 0)
            {
                if(vectopixpos != 1024)
                {
                    std::memcpy(vectopixbuf + vectopixpos, &pixrecords, 2);
                    vectopixpos++;
                } else
                {
                    io::appendBytes(a_vectopixbuf, (uint8_t*)vectopixbuf, 2 * 1024);
                }
            }
            pixrecords = 0;
        }
    }

    if(vectopixpos != 0)
    {
        io::appendBytes(a_vectopixbuf, (uint8_t*)vectopixbuf, 2 * vectopixpos);
    }
    if(pixtonumpos != 0)
    {
        io::appendBytes(a_pixtoorder, (uint8_t*)pixtonumbuf, 5 * pixtonumpos);
    }
    if(floatpos != 0)
    {
        io::appendBytes(a_floats, (uint8_t*)floatbuf, 4 * floatpos);
    }
    LOGI << "END";
}
