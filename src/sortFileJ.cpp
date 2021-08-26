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
#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

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
    LOGI << "sortsortsort";
    // Argument parsing
    std::string a_unsortedDouble;
    std::string a_sortedFloat;
    CLI::App app{ "Using divide and conquer techniques to construct CT system matrix.." };
    app.add_option("unsorted_double", a_unsortedDouble, "File in a sparse matrix format to output.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("sorted_float", a_sortedFloat,
                   "Files in a DEN format to process. These files represents projection matrices.")
        ->required()
        ->check(CLI::NonexistentPath);
    app.parse(argc, argv);
    // Frames to process
    matrix::BufferedSparseMatrixDoubleReader input(a_unsortedDouble);
    uint64_t numberOfElements = input.getNumberOfElements();
    std::vector<matrix::ElementDouble> elements;
    for(uint64_t k = 0; k != numberOfElements; k++)
    {
        elements.push_back(input.readNextElement());
    }
    std::sort(elements.begin(), elements.end(), std::greater<matrix::ElementDouble>());
    matrix::BufferedSparseMatrixFloatWritter output(a_sortedFloat);
    for(uint64_t k = 0; k != numberOfElements; k++)
    {
        output.insertValue(elements[k].i, elements[k].j, elements[k].v);
    }
    output.flush();
    LOGI << "END";
}
