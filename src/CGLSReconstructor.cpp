#include "CGLSReconstructor.hpp"

namespace CTL {

int CGLSReconstructor::reconstruct(uint32_t maxIterations, float errCondition)
{
    LOGD << printTime("WELCOME TO CGLS, init", false, true);
    uint32_t iteration = 1;

    // Initialization
    double norm, residualNorm2_old, residualNorm2_now, AdirectionNorm2, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barier_double(*b_buf));
    double NR0, NX;
    LOGI << io::xprintf("||b||=%f", NB0);
    std::shared_ptr<cl::Buffer> directionVector_xbuf, residualVector_xbuf; // X buffers
    allocateXBuffers(2);
    directionVector_xbuf = getXBuffer(0);
    residualVector_xbuf = getXBuffer(1);
    allocateBBuffers(2);
    std::shared_ptr<cl::Buffer> discrepancy_bbuf, AdirectionVector_bbuf; // B buffers
    discrepancy_bbuf = getBBuffer(0);
    AdirectionVector_bbuf = getBBuffer(1);
    // discrepancy_bbuf stores initial discrepancy
    algFLOATvector_copy(*b_buf, *discrepancy_bbuf, BDIM);
    if(useVolumeAsInitialX0)
    {
        project(*x_buf, *AdirectionVector_bbuf);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -1.0f, BDIM);
        reportTime("Projection x0", false, true);
    } else
    {
        Q[0]->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, XDIM * sizeof(float));
    }
    backproject(*discrepancy_bbuf, *residualVector_xbuf);
    algFLOATvector_copy(*residualVector_xbuf, *directionVector_xbuf, XDIM);
    residualNorm2_old = normXBuffer_barier_double(*residualVector_xbuf);
    reportTime("Backprojection 0", false, true);
    NR0 = std::sqrt(residualNorm2_old);
    project(*directionVector_xbuf, *AdirectionVector_bbuf);
    AdirectionNorm2 = normBBuffer_barier_double(*AdirectionVector_bbuf);
    reportTime("Projection 1", false, true);
    alpha = residualNorm2_old / AdirectionNorm2;
    algFLOATvector_A_equals_A_plus_cB(*x_buf, *directionVector_xbuf, alpha, XDIM);
    algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
    norm = std::sqrt(normBBuffer_barier_double(*discrepancy_bbuf));
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", progressPrefixPath.c_str(),
                                iteration);
            writeVolume(*x_buf,
                        io::xprintf("%sx_it%02d.den", progressPrefixPath.c_str(), iteration));
        }
        // DEBUG
        if(iteration % 1000 == 0)
        {
            project(*x_buf, *discrepancy_bbuf);
            algFLOATvector_A_equals_Ac_plus_B(*discrepancy_bbuf, *b_buf, -1.0, BDIM);
            double norm2 = std::sqrt(normBBuffer_barier_double(*discrepancy_bbuf));
            reportTime(io::xprintf("Reothrogonalization projection %d", iteration), false, true);
            LOGE << io::xprintf(
                "Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|, norms "
                "loss of orthogonality %f%%.",
                iteration, norm2, 100.0 * norm2 / NB0, 100 * (norm2 - norm) / norm);
        }
        // DEBUG
        backproject(*discrepancy_bbuf, *residualVector_xbuf);
        residualNorm2_now = normXBuffer_barier_double(*residualVector_xbuf);
        reportTime(io::xprintf("Backprojection %d", iteration), false, true);
        // Delayed update of residual vector
        beta = residualNorm2_now / residualNorm2_old;
        NX = std::sqrt(residualNorm2_now);
        LOGE << io::xprintf("Iteration %d: |Ax-b|=%0.1f that is %0.2f%% of |b|, |AT(Ax-b)|=%0.2f "
                            "that is %0.3f%% of |AT(Ax0-b)|.",
                            iteration, norm, 100.0 * norm / NB0, NX, 100 * NX / NR0);
        algFLOATvector_A_equals_Ac_plus_B(*directionVector_xbuf, *residualVector_xbuf, beta, XDIM);
        // Delayed update of direction vector
        iteration = iteration + 1;
        residualNorm2_old = residualNorm2_now;
        project(*directionVector_xbuf, *AdirectionVector_bbuf);
        AdirectionNorm2 = normBBuffer_barier_double(*AdirectionVector_bbuf);
        reportTime(io::xprintf("Projection %d", iteration), false, true);
        alpha = residualNorm2_old / AdirectionNorm2;
        algFLOATvector_A_equals_A_plus_cB(*x_buf, *directionVector_xbuf, alpha, XDIM);
        algFLOATvector_A_equals_A_plus_cB(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
        norm = std::sqrt(normBBuffer_barier_double(*discrepancy_bbuf));
    }
    LOGE << io::xprintf("Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|.", iteration,
                        norm, 100.0 * norm / NB0);
    Q[0]->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

int CGLSReconstructor::reconstructDiagonalPreconditioner(
    std::shared_ptr<cl::Buffer> invertedpreconditioner_xbuf,
    uint32_t maxIterations,
    float errCondition)
{
    bool blockingReport = true;
    reportTime("WELCOME TO CGLS WITH PRECONDITIONING, init", blockingReport, true);
    LOGI << io::xprintf("Maximum number of iterations is set to %d and errCondition %f",
                        maxIterations, errCondition);
    uint32_t iteration = 1;

    // Initialization
    double norm, residualNorm2_old, residualNorm2_now, AdirectionNorm2, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barier_double(*b_buf));
    double NR0, NX;
    LOGI << io::xprintf("||b||=%f", NB0);
    std::shared_ptr<cl::Buffer> directionVector_xbuf, residualVector_xbuf,
        preconditionedResidualVector_xbuf; // X buffers
    allocateXBuffers(3);
    directionVector_xbuf = getXBuffer(0);
    residualVector_xbuf = getXBuffer(1);
    preconditionedResidualVector_xbuf = getXBuffer(2);
    allocateBBuffers(2);
    std::shared_ptr<cl::Buffer> discrepancy_bbuf, AdirectionVector_bbuf; // B buffers
    discrepancy_bbuf = getBBuffer(0);
    AdirectionVector_bbuf = getBBuffer(1);

    // INITIALIZATION x_0 is initialized typically by zeros but in general by supplied array
    // c_0 is filled by b
    // v_0=w_0=BACKPROJECT(c_0)
    // writeProjections(*discrepancy_bbuf, io::xprintf("/tmp/cgls/c_0.den"));
    copyFloatVector(*b_buf, *discrepancy_bbuf, BDIM); // discrepancy_bbuf stores initial discrepancy
    if(useVolumeAsInitialX0)
    {
        setTimestamp(blockingReport);
        project(*x_buf, *AdirectionVector_bbuf);
        reportTime("Projection x0", blockingReport, true);
        addIntoFirstVectorSecondVectorScaled(*discrepancy_bbuf, *AdirectionVector_bbuf, -1.0, BDIM);
    } else
    {
        Q[0]->enqueueFillBuffer<cl_float>(*x_buf, FLOATZERO, 0, XDIM * sizeof(float));
    }
    setTimestamp(blockingReport);
    backproject(*discrepancy_bbuf, *residualVector_xbuf);
    reportTime("Backprojection 0", blockingReport, true);
    copyFloatVector(*residualVector_xbuf, *directionVector_xbuf, XDIM);
    multiplyVectorsIntoFirstVector(*directionVector_xbuf, *invertedpreconditioner_xbuf, XDIM);
    copyFloatVector(*directionVector_xbuf, *preconditionedResidualVector_xbuf, XDIM);
    residualNorm2_old = scalarProductXBuffer_barier_double(*residualVector_xbuf,
                                                           *preconditionedResidualVector_xbuf);
    NR0 = std::sqrt(residualNorm2_old);
    // DEBUG
    NR0 = std::sqrt(normXBuffer_barier_double(*residualVector_xbuf));
    setTimestamp(blockingReport);
    project(*directionVector_xbuf, *AdirectionVector_bbuf);
    reportTime("Projection 1", blockingReport, true);
    AdirectionNorm2 = normBBuffer_barier_double(*AdirectionVector_bbuf);
    alpha = residualNorm2_old / AdirectionNorm2;
    addIntoFirstVectorSecondVectorScaled(*x_buf, *directionVector_xbuf, alpha, XDIM);
    addIntoFirstVectorSecondVectorScaled(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha, BDIM);
    norm = std::sqrt(normBBuffer_barier_double(*discrepancy_bbuf));
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        if(reportKthIteration > 0 && iteration % reportKthIteration == 0)
        {
            LOGD << io::xprintf("Writing file %sx_it%02d.den", progressPrefixPath.c_str(),
                                iteration);
            writeVolume(*x_buf,
                        io::xprintf("%sx_it%02d.den", progressPrefixPath.c_str(), iteration));
        }
        // DEBUG
        if(iteration % 10 == 0)
        {
            setTimestamp(blockingReport);
            project(*x_buf, *discrepancy_bbuf);
            reportTime(io::xprintf("Reothrogonalization projection %d", iteration), blockingReport,
                       true);
            addIntoFirstVectorScaledSecondVector(*discrepancy_bbuf, *b_buf, -1.0, BDIM);
            double norm2 = std::sqrt(normBBuffer_barier_double(*discrepancy_bbuf));

            LOGE << io::xprintf(
                "Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|, norms "
                "loss of orthogonality %f%%.",
                iteration, norm2, 100.0 * norm2 / NB0, 100 * (norm2 - norm) / norm);
        }
        // DEBUG
        setTimestamp(blockingReport);
        backproject(*discrepancy_bbuf, *residualVector_xbuf);
        vectorA_multiple_B_equals_C(*residualVector_xbuf, *invertedpreconditioner_xbuf,
                                    *preconditionedResidualVector_xbuf, XDIM);
        writeVolume(
            *residualVector_xbuf,
            io::xprintf("%sresidualVector_xbuf_it%02d.den", progressPrefixPath.c_str(), iteration));
        writeVolume(*preconditionedResidualVector_xbuf,
                    io::xprintf("%spreconditionedResidualVector_xbuf_it%02d.den",
                                progressPrefixPath.c_str(), iteration));
        reportTime(io::xprintf("Backprojection %d", iteration), blockingReport, true);
        residualNorm2_now = scalarProductXBuffer_barier_double(*residualVector_xbuf,
                                                               *preconditionedResidualVector_xbuf);
        // Delayed update of residual vector
        beta = residualNorm2_now / residualNorm2_old;
        LOGI << io::xprintf("Beta=%f/%f is %f", residualNorm2_now, residualNorm2_old, beta);
        NX = std::sqrt(residualNorm2_now);
        // DEBUG
        NX = std::sqrt(normXBuffer_barier_double(*residualVector_xbuf));
        // DEBUG
        LOGE << io::xprintf("Iteration %d: |Ax-b|=%0.1f that is %0.2f%% of |b|, |AT(Ax-b)|=%0.2f "
                            "that is %0.3f%% of |AT(Ax0-b)|.",
                            iteration, norm, 100.0 * norm / NB0, NX, 100 * NX / NR0);
        addIntoFirstVectorScaledSecondVector(*directionVector_xbuf,
                                             *preconditionedResidualVector_xbuf, beta, XDIM);
        // Delayed update of direction vector
        iteration = iteration + 1;
        residualNorm2_old = residualNorm2_now;
        setTimestamp(blockingReport);
        project(*directionVector_xbuf, *AdirectionVector_bbuf);
        reportTime(io::xprintf("Projection %d", iteration), blockingReport, true);
        AdirectionNorm2 = normBBuffer_barier_double(*AdirectionVector_bbuf);
        alpha = residualNorm2_old / AdirectionNorm2;
        addIntoFirstVectorSecondVectorScaled(*x_buf, *directionVector_xbuf, alpha, XDIM);
        addIntoFirstVectorSecondVectorScaled(*discrepancy_bbuf, *AdirectionVector_bbuf, -alpha,
                                             BDIM);
        norm = std::sqrt(normBBuffer_barier_double(*discrepancy_bbuf));
    }
    LOGE << io::xprintf("Iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of |b|.", iteration,
                        norm, 100.0 * norm / NB0);
    Q[0]->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

int CGLSReconstructor::reconstruct_experimental(uint32_t maxIterations, float errCondition)
{
    LOGI << io::xprintf(
        "WELCOME TO Experimental implementation CGLS using nonstandard cone beam operator");
    std::shared_ptr<cl::Buffer> v_buf, w_buf; // X buffers
    allocateXBuffers(2);
    v_buf = getXBuffer(0);
    w_buf = getXBuffer(1);
    allocateBBuffers(2);
    std::shared_ptr<cl::Buffer> c_buf, d_buf; // B buffers
    c_buf = getBBuffer(0);
    d_buf = getBBuffer(1);
    reportTime("CGLS INIT", false, true);
    if(verbose)
    {
        // writeProjections(*b_buf, io::xprintf("%sb.den", progressPrefixPath.c_str()));
        // writeVolume(*x_buf, io::xprintf("%sx_0.den", progressPrefixPath.c_str()));
    }
    uint32_t iteration = 0;
    double norm, vnorm2_old, vnorm2_now, dnorm2_old, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barier_double(*b_buf));
    norm = NB0;
    project(*x_buf, *tmp_b_buf);
    reportTime("X_0 projection", false, true);
    addIntoFirstVectorSecondVectorScaled(*tmp_b_buf, *b_buf, -1.0, BDIM);
    norm = std::sqrt(normBBuffer_barier_double(*tmp_b_buf));
    // Experimental
    cl_int err;
    std::shared_ptr<cl::Buffer> v_proj = std::make_shared<cl::Buffer>(
        *context, CL_MEM_READ_WRITE, sizeof(float) * pdimx * pdimy, nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    std::shared_ptr<cl::Buffer> w_proj = std::make_shared<cl::Buffer>(
        *context, CL_MEM_READ_WRITE, sizeof(float) * pdimx * pdimy, nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    std::shared_ptr<cl::Buffer> x_proj = std::make_shared<cl::Buffer>(
        *context, CL_MEM_READ_WRITE, sizeof(float) * pdimx * pdimy, nullptr, &err);
    if(err != CL_SUCCESS)
    {
        LOGE << io::xprintf("Unsucessful initialization of buffer with error code %d!", err);
        return -1;
    }
    Q[0]->enqueueFillBuffer<cl_float>(*v_proj, FLOATZERO, 0,
                                      uint32_t(pdimx) * pdimy * sizeof(float));
    Q[0]->enqueueFillBuffer<cl_float>(*w_proj, FLOATZERO, 0,
                                      uint32_t(pdimx) * pdimy * sizeof(float));
    Q[0]->enqueueFillBuffer<cl_float>(*x_proj, FLOATZERO, 0,
                                      uint32_t(pdimx) * pdimy * sizeof(float));
    // Experimental
    LOGI << io::xprintf("Initial norm of b is %f and initial |Ax-b| is %f.", NB0, norm);
    // INITIALIZATION x_0 is initialized typically by zeros but in general by supplied array
    // c_0 is filled by b
    // v_0=w_0=BACKPROJECT(c_0)
    // writeProjections(*c_buf, io::xprintf("/tmp/cgls/c_0.den"));
    backproject(*c_buf, *v_buf);
    // Experimental
    LOGI << "Backprojection correction vector";
    Q[0]->enqueueFillBuffer<cl_float>(*v_proj, FLOATZERO, 0, pdimx * pdimy * sizeof(float));
    for(unsigned int i = 0; i != pdimz; i++)
    {
        B_equals_A_plus_B_offsets(*c_buf, i * pdimx * pdimy, *v_proj, 0, uint32_t(pdimx) * pdimy);
    }
    // Experimental
    reportTime("v_0 backprojection", false, true);
    if(verbose)
    {
        // writeVolume(*v_buf, io::xprintf("%sv_0.den", progressPrefixPath.c_str()));
    }
    vnorm2_old = normXBuffer_barier_double(*v_buf);
    // EXPERIMENTAL
    double sum;
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs1(*Q[0], cl::NDRange(1));
    (*vector_NormSquarePartial)(eargs1, *v_proj, *tmp_x_red1, framesize).wait();
    Q[0]->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(double), &sum);
    vnorm2_old += sum;
    // EXPERIMENTAL
    copyFloatVector(*v_buf, *w_buf, XDIM);
    // EXPERIMENTAL
    copyFloatVector(*v_proj, *w_proj, pdimx * pdimy);
    // EXPERIMENTAL
    if(verbose)
    {
        // writeVolume(*w_buf, io::xprintf("%sw_0.den", progressPrefixPath.c_str()));
    }
    project(*w_buf, *d_buf);
    // Experimental
    for(unsigned int i = 0; i != pdimz; i++)
    {
        B_equals_A_plus_B_offsets(*w_proj, 0, *d_buf, i * pdimx * pdimy, pdimx * pdimy);
    }
    // Experimental
    reportTime("d_0 projection", false, true);
    if(verbose)
    {
        // writeProjections(*d_buf, io::xprintf("%sd_0.den", progressPrefixPath.c_str()));
    }
    dnorm2_old = normBBuffer_barier_double(*d_buf);
    while(norm / NB0 > errCondition && iteration < maxIterations)
    {
        // Iteration
        iteration = iteration + 1;
        alpha = vnorm2_old / dnorm2_old;
        LOGI << io::xprintf("After iteration %d, |v|^2=%E, |d|^2=%E, alpha=%E", iteration - 1,
                            vnorm2_old, dnorm2_old, float(alpha));
        addIntoFirstVectorSecondVectorScaled(*x_buf, *w_buf, alpha, XDIM);
        // EXPERIMENTAL
        addIntoFirstVectorSecondVectorScaled(*x_proj, *w_proj, alpha, pdimx * pdimy);
        // EXPERIMENTAL
        if(verbose)
        {
            writeVolume(*x_buf, io::xprintf("%sx_%d.den", progressPrefixPath.c_str(), iteration));
        }
        addIntoFirstVectorSecondVectorScaled(*c_buf, *d_buf, -alpha, BDIM);
        if(verbose)
        {
            //    writeProjections(*c_buf,
            //                     io::xprintf("%sc_%d.den", progressPrefixPath.c_str(),
            //                     iteration));
        }
        backproject(*c_buf, *v_buf);
        // Experimental
        Q[0]->enqueueFillBuffer<cl_float>(*v_proj, FLOATZERO, 0, pdimx * pdimy * sizeof(float));
        for(unsigned int i = 0; i != pdimz; i++)
        {
            B_equals_A_plus_B_offsets(*c_buf, i * pdimx * pdimy, *v_proj, 0, pdimx * pdimy);
        }
        // Experimental
        reportTime(io::xprintf("v_%d backprojection", iteration), false, true);

        if(verbose)
        {
            //    writeVolume(*v_buf, io::xprintf("%sv_%d.den", progressPrefixPath.c_str(),
            //    iteration));
        }
        vnorm2_now = normXBuffer_barier_double(*v_buf);
        // EXPERIMENTAL
        cl::EnqueueArgs eargs1(*Q[0], cl::NDRange(1));
        (*vector_NormSquarePartial)(eargs1, *v_proj, *tmp_x_red1, framesize).wait();
        Q[0]->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(double), &sum);
        vnorm2_now += sum;
        // EXPERIMENTAL
        beta = vnorm2_now / vnorm2_old;
        LOGI << io::xprintf("In iteration %d, |v_now|^2=%E, |v_old|^2=%E, beta=%0.2f", iteration,
                            vnorm2_now, vnorm2_old, beta);
        vnorm2_old = vnorm2_now;
        addIntoFirstVectorScaledSecondVector(*w_buf, *v_buf, beta, XDIM);
        // EXPERIMENTAL
        addIntoFirstVectorScaledSecondVector(*w_proj, *v_proj, beta, pdimx * pdimy);
        // EXPERIMENTAL
        if(verbose)
        {
            //    writeVolume(*w_buf, io::xprintf("%sw_%d.den", progressPrefixPath.c_str(),
            //    iteration));
        }
        addIntoFirstVectorSecondVectorScaled(*tmp_b_buf, *d_buf, alpha, BDIM);
        project(*w_buf, *d_buf);
        // EXPERIMENTAL
        for(unsigned int i = 0; i != pdimz; i++)
        {
            B_equals_A_plus_B_offsets(*w_proj, 0, *d_buf, i * pdimx * pdimy, pdimx * pdimy);
        }
        // EXPERIMENTAL
        reportTime(io::xprintf("d_%d projection", iteration), false, true);
        if(verbose)
        {
            //    writeProjections(*d_buf,
            //                     io::xprintf("%sd_%d.den", progressPrefixPath.c_str(),
            //                     iteration));
        }
        dnorm2_old = normBBuffer_barier_double(*d_buf);

        norm = std::sqrt(normBBuffer_barier_double(*tmp_b_buf));
        LOGE << io::xprintf("After iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of NB0.",
                            iteration, norm, 100.0 * norm / NB0);
    }
    // Optionally write even more converged solution
    alpha = vnorm2_old / dnorm2_old;
    LOGI << io::xprintf("Finally |v|^2=%E, |d|^2=%E, alpha=%E", iteration - 1, vnorm2_old,
                        dnorm2_old, float(alpha));
    addIntoFirstVectorSecondVectorScaled(*x_buf, *w_buf, alpha, XDIM);
    // EXPERIMENTAL
    addIntoFirstVectorSecondVectorScaled(*x_proj, *w_proj, alpha, pdimx * pdimy);
    // EXPERIMENTAL
    Q[0]->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

int CGLSReconstructor::reconstructJacobi(uint32_t maxIterations, float errCondition)
{
    std::shared_ptr<cl::Buffer> preconditioner_xbuf; // X buffers
    allocateXBuffers(4);
    preconditioner_xbuf = getXBuffer(3);
    precomputeJacobiPreconditioner(preconditioner_xbuf);
    invertFloatVector(*preconditioner_xbuf, XDIM);
    LOGD << io::xprintf("Writing file %s_preconditioner.inv", progressPrefixPath.c_str());
    writeVolume(*preconditioner_xbuf,
                io::xprintf("%sx_preconditioner.inv", progressPrefixPath.c_str()));
    reconstructDiagonalPreconditioner(preconditioner_xbuf, maxIterations, errCondition);
    return 0;
}

void CGLSReconstructor::precomputeJacobiPreconditioner(std::shared_ptr<cl::Buffer> X)
{
    Q[0]->enqueueFillBuffer<cl_float>(*X, FLOATZERO, 0, XDIM * sizeof(float));
    cl::EnqueueArgs eargs(*Q[0], cl::NDRange(vdimz, vdimy, vdimx));
    for(uint32_t k = 0; k != cameraVector.size(); k++)
    {
        std::shared_ptr<CameraI> c = cameraVector[k];
        std::array<double, 2> focalLength = c->focalLength();
        // Kernel parameters
        cl_double16 CM;
        cl_double3 sourcePosition, normalToDetector;
        float scalingFactor = focalLength[0] * focalLength[1];
        c->projectionMatrixAsVector12((double*)&CM);
        c->normalToDetector((double*)&normalToDetector);
        c->sourcePosition((double*)&sourcePosition);
        (*FLOATcutting_voxel_jacobiPreconditionerVector)(eargs, *X, CM, sourcePosition,
                                                         normalToDetector, vdims, voxelSizes,
                                                         volumeCenter, pdims, scalingFactor);
    }
    LOGD << io::xprintf("Writing file %s_preconditioner.den", progressPrefixPath.c_str());
    writeVolume(*X, io::xprintf("%sx_preconditioner.den", progressPrefixPath.c_str()));
}

int CGLSReconstructor::reconstructDiagonalPreconditioner(float* invertedpreconditioner,
                                                         uint32_t maxIterations,
                                                         float errCondition)
{
    std::shared_ptr<cl::Buffer> preconditioner_xbuf; // X buffers
    allocateXBuffers(4);
    preconditioner_xbuf = getXBuffer(3);
    vectorIntoBuffer(*preconditioner_xbuf, invertedpreconditioner, XDIM);
    double norm = std::sqrt(normXBuffer_barier_double(*preconditioner_xbuf));
    LOGE << "Norm " << norm;
    reconstructDiagonalPreconditioner(preconditioner_xbuf, maxIterations, errCondition);
    return 0;
}

} // namespace CTL
