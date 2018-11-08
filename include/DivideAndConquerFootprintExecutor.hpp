#pragma once
#include "SMA/BufferedSparseMatrixWritter.hpp"

namespace CTL {
namespace util {

    struct Cube
    {
        // Index of the pixel to which the corner of the Cube is projected
        std::array<int, 8> pixelIndex;
        std::array<double, 3> leftLowerBottomCorner;
        double edgeLength;
    };

    class DivideAndConquerFootprintExecutor
    {
    public:
        DivideAndConquerFootprintExecutor(std::shared_ptr<matrix::BufferedSparseMatrixWritter> w,
                                          uint32_t pdimx,
                                          uint32_t pdimy,
                                          uint32_t vdimx,
                                          uint32_t vdimy,
                                          uint32_t vdimz,
                                          int threads)
        {
            this->w = w;
            this->pdimx = pdimx;
            this->pdimy = pdimy;
            this->vdimx = vdimx;
            this->vdimy = vdimy;
            this->vdimz = vdimz;
            this->threads = threads;
        }

        void insertMatrixProjections(ProjectionMatrix pm, uint32_t pixelIndexOffset)
        {
            std::array<double, 3> sourcePosition = pm.sourcePosition();
            double xcoord, ycoord, zcoord;
            // First I try to precompute indices of each corner that is
            // (vdimx+1)x(vdimy+1)x(vdimz+1)
            uint32_t voxelCornerNum = (vdimx + 1) * (vdimy + 1) * (vdimz + 1);
            uint32_t* resultingIndices = new uint32_t[voxelCornerNum];
            xcoord = -(double(vdimx) / 2.0) - 0.5;
            ycoord = -(double(vdimy) / 2.0) - 0.5;
            zcoord = -(double(vdimz) / 2.0) - 0.5;
            uint32_t voxelindex = 0;
            double px, py;
            uint32_t pi, pj;
            uint32_t numberOfWrites = 0;
            uint32_t nonwrites = 0;
            for(uint32_t k = 0; k != vdimz + 1; k++)
            {
                for(uint32_t j = 0; j != vdimy + 1; j++)
                {
                    for(uint32_t i = 0; i != vdimx + 1; i++)
                    {
                        pm.project(xcoord, ycoord, zcoord, &px, &py);
                        pi = (int)(px + 0.5);
                        pj = (int)(py + 0.5); // Rounding to integer
                        if(pi >= 0 && pj >= 0 && pi < pdimx && pj < pdimy)
                        {
                            resultingIndices[voxelindex] = pj * pdimx + pi;
                            w->insertValue(voxelindex, pj * vdimx + pi, 1.0);
                            numberOfWrites++;
                        } else
                        {
                            resultingIndices[voxelindex] = pdimx * pdimy;
                            nonwrites++;
                        }
                        xcoord += 1.0;
                        voxelindex++;
                    }
                    xcoord = -(double(vdimx) / 2.0) - 0.5;
                    ycoord += 1.0;
                }
                xcoord = -(double(vdimx) / 2.0) - 0.5;
                ycoord = -(double(vdimy) / 2.0) - 0.5;
                zcoord += 1.0;
            }
            delete[] resultingIndices;
            w->flush();
            LOGI << io::xprintf("There were %d writes and %d non writes to the matrix that should "
                                "result in the increase of its size by %d bytes.",
                                numberOfWrites, nonwrites, numberOfWrites * 16);

            /*ctpl::thread_pool* threadpool = new ctpl::thread_pool(threads);
                        double sourcePosition[3];

                        for(int i = 0; i != vdimx; i++)
                            for(int j = 0; j != vidimy; j++)
                                for(int k = 0; k != vdimz; k++)
                                {

                                    threadpool->push([&, this, cube](int id) { asyncFitting(k,
               writters); }); threadPool.insertNewRow
                                }
                        threadpool->stop(true);
                        delete[] threadpool;*/
        }

    private:
        std::shared_ptr<matrix::BufferedSparseMatrixWritter> w;
        // It is evaluated from -0.5, pixels are centerred at integer coordinates
        uint32_t pdimx = 616;
        uint32_t pdimy = 480;
        // Here (0,0,0) is in the center of the volume
        uint32_t vdimx = 256;
        uint32_t vdimy = 256;
        uint32_t vdimz = 199;
        int threads = 1;
    };
} // namespace util
} // namespace CTL
