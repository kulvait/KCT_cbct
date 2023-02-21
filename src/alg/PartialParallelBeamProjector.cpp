#include "PartialParallelBeamProjector.hpp"

namespace KCT {

int PartialParallelBeamProjector::arrayIntoBuffer(float* c_array,
                                                  cl::Buffer cl_buffer,
                                                  uint64_t size)
{
    uint64_t bufferSize = size * sizeof(float);
    std::vector<std::shared_ptr<cl::CommandQueue>> Q = CT->getCommandQueues();
    cl_int err = CL_SUCCESS;
    if(c_array != nullptr)
    {
        err = Q[0]->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0, bufferSize, (void*)c_array);
    } else
    {
        KCTERR("Null pointer exception!");
    }
    if(err != CL_SUCCESS)
    {
        std::string msg = io::xprintf("Failed arrayIntoBuffer with code %d!", err);
        KCTERR(msg);
    }
    return 0;
}

int PartialParallelBeamProjector::bufferIntoArray(cl::Buffer cl_buffer,
                                                  float* c_array,
                                                  uint64_t size)
{
    std::vector<std::shared_ptr<cl::CommandQueue>> Q = CT->getCommandQueues();
    uint64_t bufferSize = size * sizeof(float);
    cl_int err = CL_SUCCESS;
    if(c_array != nullptr)
    {
        err = Q[0]->enqueueReadBuffer(cl_buffer, CL_TRUE, 0, bufferSize, (void*)c_array);
    } else
    {
        KCTERR("Null pointer exception!");
    }
    if(err != CL_SUCCESS)
    {
        std::string msg = io::xprintf("Failed bufferIntoArray with code %d!", err);
        KCTERR(msg);
    }
    return 0;
}

int PartialParallelBeamProjector::fillBufferByConstant(cl::Buffer cl_buffer,
                                                       float constant,
                                                       uint64_t bytecount)
{
    std::vector<std::shared_ptr<cl::CommandQueue>> Q = CT->getCommandQueues();
    return Q[0]->enqueueFillBuffer<cl_float>(cl_buffer, constant, 0, bytecount);
}

int PartialParallelBeamProjector::project_partial(float* volume, float* projection)
{
    std::vector<std::shared_ptr<cl::CommandQueue>> Q = CT->getCommandQueues();
    CT->allocateXBuffers(1);
    CT->allocateBBuffers(1);
    std::shared_ptr<cl::Buffer> x_buf, b_buf;
    x_buf = CT->getXBuffer(0);
    b_buf = CT->getBBuffer(0);
    // uint32_t pdimx, pdimy, pFrameSize, pdimz, pdimz_partial, vdimz_partial_last pzblocks;
    // uint32_t vdimx, vdimy, vFrameSize, vdimz, vdimz_partial, pdimz_partial_last, vzblocks;
    // uint64_t XDIM, BDIM, XDIM_partial, BDIM_partial;
    // uint64_t XBYTES, BBYTES, XBYTES_partial, BBYTES_partial;
    float *volumePtr, *projectionPtr;
    uint64_t geometriesFrom;
    uint64_t geometriesTo;
    float voxelzCenterOffset;
    uint32_t pdimz_partial_now, vdimz_partial_now;
    for(uint64_t PIN = 0; PIN != pzblocks; PIN++)
    {
        projectionPtr = projection + PIN * pFrameSize * (uint64_t)pdimz_partial;
        if(PIN != pzblocks - 1)
        {
            pdimz_partial_now = pdimz_partial;
        } else
        {
            pdimz_partial_now = pdimz_partial_last;
        }
        geometriesFrom = PIN * pdimz_partial;
        geometriesTo = geometriesFrom + pdimz_partial_now;
        // Projection buffer is filled by zeros
        fillBufferByConstant(*b_buf, 0.0f, BDIM_partial * sizeof(float));
        for(uint64_t VIN = 0; VIN != vzblocks; VIN++)
        {
            volumePtr = volume + VIN * vFrameSize * (uint64_t)vdimz_partial;
            std::cout << io::xprintf("START VIN=%d\n", VIN);
            if(VIN != vzblocks - 1)
            {
                vdimz_partial_now = vdimz_partial;
                arrayIntoBuffer(volumePtr, *x_buf, XDIM_partial);
            } else
            {
                vdimz_partial_now = vdimz_partial_last;
                arrayIntoBuffer(volumePtr, *x_buf, XDIM_partial_last);
            }
            std::cout << io::xprintf("BUFREAD VIN=%d\n", VIN);
            voxelzCenterOffset = float(vdimz) * 0.5f
                - (float(VIN * vdimz_partial) + float(vdimz_partial_now) * 0.5f);
            CT->updateReductionParameters(pdimx, pdimy, pdimz_partial_now, vdimx, vdimy,
                                          vdimz_partial_now, workGroupSize);

            std::cout << io::xprintf("START Partial projection PIN=%d [0, %d) VIN = %d [0, %d).\n", PIN,
                                pzblocks, VIN, vzblocks);
            Q[0]->finish();
            CT->project_partial(*x_buf, *b_buf, voxelzCenterOffset, geometriesFrom, geometriesTo);
            Q[0]->finish();
            std::cout << io::xprintf("END Partial projection PIN=%d [0, %d) VIN = %d [0, %d).\n", PIN,
                                pzblocks, VIN, vzblocks);
        }
        if(PIN != pzblocks - 1)
        {
            bufferIntoArray(*b_buf, projectionPtr, BDIM_partial);
        } else
        {
            bufferIntoArray(*b_buf, projectionPtr, BDIM_partial_last);
        }
    }
    return 0;
}
int PartialParallelBeamProjector::project_print_discrepancy(float* volume,
                                                            float* projection,
                                                            float* rhs)
{
    /*
        CT->allocateXBuffers(1);
        CT->allocateBBuffers(2);
        std::shared_ptr<cl::Buffer> x_buf, b_buf, bb_buf;
        x_buf = getXBuffer(0);
        b_buf = getBBuffer(0);
        bb_buf = getBBuffer(1);
        arrayIntoBuffer(volume, *x_buf, XDIM);
        arrayIntoBuffer(rhs, *bb_buf, BDIM);
        BasePBCTOperator::project(*x_buf, *b_buf);
        double normbb = std::sqrt(normBBuffer_barrier_double(*bb_buf));
        algFLOATvector_A_equals_A_plus_cB(*bb_buf, *b_buf, -1.0f, BDIM);
        double normDiscrepancy = std::sqrt(normBBuffer_barrier_double(*bb_buf));
        LOGI << io::xprintf("|Ax-rhs|=%f, |rhs|=%f, discrepancy is %f%% of the initial norm.",
                            normDiscrepancy, normbb, 100.0 * normDiscrepancy / normbb);
        bufferIntoArray(*b_buf, projection, BDIM);
    */
    LOGE << "Unimplemented";
    return 0;
}
int PartialParallelBeamProjector::backproject(float* projection, float* volume)
{
    /*
        allocateXBuffers(1);
        allocateBBuffers(1);
        std::shared_ptr<cl::Buffer> x_buf, b_buf;
        x_buf = getXBuffer(0);
        b_buf = getBBuffer(0);
        arrayIntoBuffer(projection, *b_buf, BDIM);
        BasePBCTOperator::backproject(*b_buf, *x_buf);
        bufferIntoArray(*x_buf, volume, XDIM);*/
    LOGE << "Unimplemented";
    return 0;
}

double PartialParallelBeamProjector::normSquare(float* v, uint32_t pdimx, uint32_t pdimy)
{
    /*
        size_t vecsize = sizeof(float) * pdimx * pdimy;
        if(tmpBuffer == nullptr || vecsize != tmpBuffer_size)
        {
            tmpBuffer_size = vecsize;
            tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE |
       CL_MEM_COPY_HOST_PTR, tmpBuffer_size, (void*)v); } else
        {
            Q[0]->enqueueWriteBuffer(*tmpBuffer, CL_TRUE, 0, tmpBuffer_size, (void*)v);
        }
        cl::Buffer onedouble(*context, CL_MEM_READ_WRITE, sizeof(double), nullptr);
        double sum;
        uint32_t framesize = pdimx * pdimy;
        cl::EnqueueArgs eargs(*Q[0], cl::NDRange(1));
        (*vector_NormSquarePartial)(eargs, *tmpBuffer, onedouble, framesize).wait();
        Q[0]->enqueueReadBuffer(onedouble, CL_TRUE, 0, sizeof(double), &sum);
        return sum;
    */
    LOGE << "Unimplemented";
    return 0;
}

double PartialParallelBeamProjector::normSquareDifference(float* v, uint32_t pdimx, uint32_t pdimy)
{
    /*
        size_t vecsize = sizeof(float) * pdimx * pdimy;
        if(projectionBuffer == nullptr)
        {
            std::string msg = "Comparing to empty buffer is not possible.";
            KCTERR(msg);
        }
        if(vecsize != totalProjectionBufferSize)
        {
            std::string msg = "Can not compare buffers of incompatible dimensions.";
            KCTERR(msg);
        }

        if(tmpBuffer == nullptr || vecsize != tmpBuffer_size)
        {
            tmpBuffer_size = vecsize;
            tmpBuffer = std::make_shared<cl::Buffer>(*context, CL_MEM_READ_WRITE |
       CL_MEM_COPY_HOST_PTR, tmpBuffer_size, (void*)v); } else
        {
            Q[0]->enqueueWriteBuffer(*tmpBuffer, CL_TRUE, 0, tmpBuffer_size, (void*)v);
        }
        uint32_t framesize = pdimx * pdimy;
        cl::EnqueueArgs eargs(*Q[0], cl::NDRange(framesize));
        float factor = -1.0;
        (*FLOATvector_A_equals_Ac_plus_B)(eargs, *tmpBuffer, *projectionBuffer, factor).wait();
        cl::Buffer onedouble(*context, CL_MEM_READ_WRITE, sizeof(double), nullptr);
        double sum;
        cl::EnqueueArgs ear(*Q[0], cl::NDRange(1));
        (*vector_NormSquarePartial)(ear, *tmpBuffer, onedouble, framesize).wait();
        Q[0]->enqueueReadBuffer(onedouble, CL_TRUE, 0, sizeof(double), &sum);
        return sum;*/
    LOGE << "Unimplemented";
    return 0;
}

} // namespace KCT
