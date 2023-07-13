#include "PartialParallelBeam2DProjector.hpp"

namespace KCT {

int PartialParallelBeam2DProjector::arrayIntoBuffer(uint32_t QID,
                                                  float* c_array,
                                                  cl::Buffer cl_buffer,
                                                  uint64_t size)
{
    uint64_t bufferSize = size * sizeof(float);
    std::shared_ptr<cl::CommandQueue> Q = CT->getCommandQueues()[QID];
    cl_int err = CL_SUCCESS;
    if(c_array != nullptr)
    {
        err = Q->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0, bufferSize, (void*)c_array);
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

int PartialParallelBeam2DProjector::bufferIntoArray(uint32_t QID,
                                                  cl::Buffer cl_buffer,
                                                  float* c_array,
                                                  uint64_t size)
{
    std::shared_ptr<cl::CommandQueue> Q = CT->getCommandQueues()[QID];
    uint64_t bufferSize = size * sizeof(float);
    cl_int err = CL_SUCCESS;
    if(c_array != nullptr)
    {
        err = Q->enqueueReadBuffer(cl_buffer, CL_TRUE, 0, bufferSize, (void*)c_array);
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

int PartialParallelBeam2DProjector::fillBufferByConstant(uint32_t QID,
                                                       cl::Buffer cl_buffer,
                                                       float constant,
                                                       uint64_t bytecount)
{
    std::shared_ptr<cl::CommandQueue> Q = CT->getCommandQueues()[QID];
    return Q->enqueueFillBuffer<cl_float>(cl_buffer, constant, 0, bytecount);
}

std::queue<uint32_t> QID_queue;
std::mutex ql;
std::condition_variable gpuavail;

int PartialParallelBeam2DProjector::project_pzblock(float* volume, float* projection, uint64_t PIN)
{
    std::mt19937_64 eng{std::random_device{}()};  // Randomizing I/O
    std::uniform_int_distribution<> dist{0, 3000};
    std::this_thread::sleep_for(std::chrono::milliseconds{dist(eng)});


    float *volumePtr, *projectionPtr;
    uint64_t geometriesFrom;
    uint64_t geometriesTo;
    uint64_t projectionArrayOffset;
    float voxelzCenterOffset;
    uint32_t pdimz_partial_now, vdimz_partial_now;
    uint64_t BDIM_now;
    projectionArrayOffset = PIN * pFrameSize * (uint64_t)pdimz_partial;
    projectionPtr = projection + projectionArrayOffset;
    if(PIN != pzblocks - 1)
    {
        pdimz_partial_now = pdimz_partial;
        BDIM_now = BDIM_partial;
    } else
    {
        pdimz_partial_now = pdimz_partial_last;
        BDIM_now = BDIM_partial_last;
    }
    geometriesFrom = PIN * pdimz_partial;
    geometriesTo = geometriesFrom + pdimz_partial_now;
    std::shared_ptr<cl::CommandQueue> Q;
    uint32_t QID;
    // Obtain QID of the GPU on which to execute
    {
        std::unique_lock qll(ql);
        // It first tests then waits, see
        // https://stackoverflow.com/questions/75591190/strange-behavior-of-condition-variable-in-c
        gpuavail.wait(qll, [] { return !QID_queue.empty(); });
        QID = QID_queue.front();
        QID_queue.pop();
        Q = CT->getCommandQueues()[QID];
        LOGI << io::xprintf("START QID %d to project PIN %d [0,%d).", QID, PIN, pzblocks);
    }
    std::shared_ptr<cl::Buffer> x_buf, b_buf;
    x_buf = CT->getXBuffer(QID);
    b_buf = CT->getBBuffer(QID);
    // Projection buffer is filled by zeros
    fillBufferByConstant(QID, *b_buf, 0.0f, BDIM_now * sizeof(float));
    for(uint64_t VIN = 0; VIN != vzblocks; VIN++)
    {
        volumePtr = volume + VIN * vFrameSize * (uint64_t)vdimz_partial;
        if(VIN != vzblocks - 1)
        {
            vdimz_partial_now = vdimz_partial;
            arrayIntoBuffer(QID, volumePtr, *x_buf, XDIM_partial);
        } else
        {
            vdimz_partial_now = vdimz_partial_last;
            arrayIntoBuffer(QID, volumePtr, *x_buf, XDIM_partial_last);
        }
        voxelzCenterOffset
            = float(vdimz) * 0.5f - (float(VIN * vdimz_partial) + float(vdimz_partial_now) * 0.5f);
        // CT->updateReductionParameters(pdimx, pdimy, pdimz_partial_now, vdimx, vdimy,
        //                              vdimz_partial_now, workGroupSize);

        CT->project_partial(QID, *x_buf, *b_buf, vdimz_partial_now, voxelzCenterOffset,
                            geometriesFrom, geometriesTo);
    }

    bufferIntoArray(QID, *b_buf, projectionPtr, BDIM_now);

    LOGD << io::xprintf("END PIN %d [0,%d) written %lu values to [%lu, %lu).", PIN, pzblocks,
                        BDIM_now, projectionArrayOffset, projectionArrayOffset + BDIM_now);
    // Push QID into the queue and notify waiting threads
    {
        std::unique_lock qll(ql);
        QID_queue.push(QID);
    }
    gpuavail.notify_one();
    return PIN;
}

int PartialParallelBeam2DProjector::project_partial(float* volume, float* projection)
{
    std::vector<std::shared_ptr<cl::CommandQueue>> Q = CT->getCommandQueues();
    // Fill queue with QID of available GPUs in the queue
    uint32_t qsize = Q.size();
    QID_queue = std::queue<uint32_t>();

    CT->allocateXBuffers(1 * qsize);
    CT->allocateBBuffers(1 * qsize);

    std::vector<cl::Memory> mem;
    std::shared_ptr<cl::Buffer> x_buf, b_buf;
    for(uint32_t i = 0; i != qsize; i++)
    {
        QID_queue.push(i);
        x_buf = CT->getXBuffer(i);
        b_buf = CT->getBBuffer(i);
        mem.clear();
        mem.emplace_back(*x_buf);
        mem.emplace_back(*b_buf);
        Q[i]->enqueueMigrateMemObjects(mem, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED);
    }

    // uint32_t pdimx, pdimy, pFrameSize, pdimz, pdimz_partial, vdimz_partial_last pzblocks;
    // uint32_t vdimx, vdimy, vFrameSize, vdimz, vdimz_partial, pdimz_partial_last, vzblocks;
    // uint64_t XDIM, BDIM, XDIM_partial, BDIM_partial;
    // uint64_t XBYTES, BBYTES, XBYTES_partial, BBYTES_partial;
    std::vector<std::future<int>> computedTasks;
    for(uint64_t PIN = 0; PIN != pzblocks; PIN++)
    {
        std::future<int> future
            = std::async(std::launch::async, &PartialParallelBeam2DProjector::project_pzblock, this,
                         volume, projection, PIN);
        computedTasks.emplace_back(std::move(future));
    }
    for(std::future<int>& f : computedTasks)
    {
        f.get(); // To finish all the computations
    }
    return 0;
}
int PartialParallelBeam2DProjector::project_print_discrepancy(float* volume,
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
int PartialParallelBeam2DProjector::backproject(float* projection, float* volume)
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

double PartialParallelBeam2DProjector::normSquare(float* v, uint32_t pdimx, uint32_t pdimy)
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

double PartialParallelBeam2DProjector::normSquareDifference(float* v, uint32_t pdimx, uint32_t pdimy)
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
