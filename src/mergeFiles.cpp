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
    LOGI << "sortsortsort";
    // Argument parsing
    std::vector<std::string> a_unsortedDoubles;
    std::string a_sortedFloat;
    CLI::App app{ "Using divide and conquer techniques to construct CT system matrix.." };
    app.add_option("sorted_float", a_sortedFloat,
                   "Files in a DEN format to process. These files represents projection matrices.")
        ->required()
        ->check(CLI::NonexistentPath);
    app.add_option("unsorted_double", a_unsortedDoubles,
                   "File in a sparse matrix format to output.")
        ->required()
        ->check(CLI::ExistingFile);
    app.parse(argc, argv);
    // Frames to process
    std::vector<std::shared_ptr<matrix::BufferedSparseMatrixFloatReader>> readers;
    for(std::string const& v : a_unsortedDoubles)
    {
        std::shared_ptr<matrix::BufferedSparseMatrixFloatReader> r
            = std::make_shared<matrix::BufferedSparseMatrixFloatReader>(v, 16384);
        readers.push_back(r);
    }
    matrix::BufferedSparseMatrixFloatWritter output(a_sortedFloat, 16384);

    // Find the minimum value of i in a readers
    int mink = 0;
    uint32_t i, j;
    float v;
    while(true)
    {
        uint32_t mini = 256 * 256 * 199;
        for(std::size_t k = 0; k != a_unsortedDoubles.size(); k++)
        {
            readers[k]->getNextValue(&i, &j, &v);
            if(i < mini)
            {
                mini = i;
                mink = k;
            }
        }
        if(mini == 256 * 256 * 199)
        {
            for(std::size_t k = 0; k != a_unsortedDoubles.size(); k++)
            {
                if(!readers[k]->atEnd())
                {
                    LOGD << io::xprintf("Not at end in %f.", a_unsortedDoubles[k]);
                }
            }
            break;
        }
        while(true)
        {
            readers[mink]->getNextValue(&i, &j, &v);
            if(i == mini)
            {
                output.insertValue(i, j, v);
                readers[mink]->increaseCounter();
            } else
            {
                break;
            }
        }
    }
    output.flush();
    LOGI << "END";
}
