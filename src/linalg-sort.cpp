// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

// External libraries
#include "CLI/CLI.hpp" //Command line parser
#include "ctpl_stl.h" //Threadpool

// Internal libraries
#include "SMA/BufferedSparseMatrixDoubleReader.hpp"
#include "SMA/BufferedSparseMatrixDoubleWritter.hpp"
#include "SMA/BufferedSparseMatrixFloatReader.hpp"
#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

using namespace KCT;

struct Args
{
    std::vector<std::string> typeStrings;
    Args()
    {
        typeStrings.push_back("float");
        typeStrings.push_back("double");
    }
    std::string unsortedMatrix;
    std::string unsortedType;
    std::string sortedMatrix;
    std::string sortedType;
    bool force = false;
    bool bySecondIndex = false;
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
    CLI::App app{ "Sort sparse matrices." };
    app.add_flag("-f,--force", force, "Overwrite outputFile if it exists.");
    app.add_flag("-j,--bySecondIndex", bySecondIndex,
                 "Sort by the second index, if this flag is not specified sorting will by by the "
                 "first index.");

    std::function<std::string(const std::string&)> f
        = std::bind(&Args::checkTypeConsistency, this, std::placeholders::_1);
    app.add_option("unsorted_matrix", unsortedMatrix, "File with the unsorted matrix.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("unsorted_type", unsortedType,
                   "Type of the unsorted matrix, that is float or double.")
        ->required()
        ->check(f);
    app.add_option("sorted_matrix", sortedMatrix, "Output file with the sorted matrix.")
        ->required();
    app.add_option("sorted_type", sortedType, "Type of the sorted matrix, that is float or double.")
        ->required()
        ->check(f);
    try
    {
        app.parse(argc, argv);
        if(!force)
        {
            if(io::pathExists(sortedMatrix))
            {
                std::string msg
                    = "Error: output file already exists, use --force to force overwrite.";
                LOGE << msg;
                return 1;
            }
        }
    } catch(const CLI::CallForHelp& e)
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
    // Frames to process
    if(a.unsortedType == "double")
    {
        matrix::BufferedSparseMatrixDoubleReader input(a.unsortedMatrix);
        uint64_t numberOfElements = input.getNumberOfElements();
        std::vector<matrix::ElementDouble> elements;
        for(uint64_t k = 0; k != numberOfElements; k++)
        {
            elements.push_back(input.readNextElement());
        }
        if(a.bySecondIndex)
        {
            std::sort(elements.begin(), elements.end(), std::greater<matrix::ElementDouble>());
        } else
        {
            std::sort(elements.begin(), elements.end());
        }
        if(a.sortedType == "double")
        {
            matrix::BufferedSparseMatrixDoubleWritter output(a.sortedMatrix);
            for(uint64_t k = 0; k != numberOfElements; k++)
            {
                output.insertValue(elements[k].i, elements[k].j, elements[k].v);
            }
            output.flush();
        } else if(a.sortedType == "float")
        {
            matrix::BufferedSparseMatrixFloatWritter output(a.sortedMatrix);
            for(uint64_t k = 0; k != numberOfElements; k++)
            {
                output.insertValue(elements[k].i, elements[k].j, elements[k].v);
            }
            output.flush();
        } else
        {
            io::throwerr("Unsupported output type");
        }
    } else if(a.unsortedType == "float")
    {
        matrix::BufferedSparseMatrixFloatReader input(a.unsortedMatrix);
        uint64_t numberOfElements = input.getNumberOfElements();
        std::vector<matrix::ElementFloat> elements;
        for(uint64_t k = 0; k != numberOfElements; k++)
        {
            elements.push_back(input.readNextElement());
        }
        if(a.bySecondIndex)
        {
            std::sort(elements.begin(), elements.end(), std::greater<matrix::ElementFloat>());
        } else
        {
            std::sort(elements.begin(), elements.end());
        }
        if(a.sortedType == "double")
        {
            matrix::BufferedSparseMatrixDoubleWritter output(a.sortedMatrix);
            for(uint64_t k = 0; k != numberOfElements; k++)
            {
                output.insertValue(elements[k].i, elements[k].j, elements[k].v);
            }
            output.flush();
        } else if(a.sortedType == "float")
        {
            matrix::BufferedSparseMatrixFloatWritter output(a.sortedMatrix);
            for(uint64_t k = 0; k != numberOfElements; k++)
            {
                output.insertValue(elements[k].i, elements[k].j, elements[k].v);
            }
            output.flush();
        } else
        {
            io::throwerr("Unsupported output type");
        }
    } else
    {
        io::throwerr("Unsupported input type");
    }
    LOGI << io::xprintf("END %s", argv[0]);
}
