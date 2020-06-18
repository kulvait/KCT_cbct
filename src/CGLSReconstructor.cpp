#include "CGLSReconstructor.hpp"

namespace CTL {

int CGLSReconstructor::reconstruct(std::shared_ptr<io::DenProjectionMatrixReader> matrices,
                                   uint32_t maxIterations,
                                   float errCondition)
{
    LOGI << io::xprintf("WELCOME TO CGLS");
    reportTime("CGLS INIT");
    std::shared_ptr<cl::Buffer> v_buf, w_buf; // X buffers
    allocateXBuffers(2);
    v_buf = getXBuffer(0);
    w_buf = getXBuffer(1);
    allocateBBuffers(2);
    std::shared_ptr<cl::Buffer> c_buf, d_buf; // B buffers
    c_buf = getBBuffer(0);
    d_buf = getBBuffer(1);

    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
    std::vector<cl_double16> ICM = inverseProjectionMatrices(PM);
    std::vector<float> scalingFactors = computeScalingFactors(PM);
    uint32_t iteration = 1;
    double norm, vnorm2_old, vnorm2_now, dnorm2_old, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barier_double(*b_buf));
    LOGI << io::xprintf("Initial norm of b is %f.", NB0);
    // INITIALIZATION x_0 is initialized typically by zeros but in general by supplied array
    // c_0 is filled by b
    // v_0=w_0=BACKPROJECT(c_0)
    // writeProjections(*c_buf, io::xprintf("/tmp/cgls/c_0.den"));
    backproject(*c_buf, *v_buf, PM, ICM, scalingFactors);
    reportTime("backprojection 0");
    vnorm2_old = normXBuffer_barier_double(*v_buf);
    copyFloatVector(*v_buf, *w_buf, XDIM);
    project(*w_buf, *d_buf, PM, ICM, scalingFactors);
    reportTime("projection 0");
    dnorm2_old = normBBuffer_barier_double(*d_buf);
    alpha = vnorm2_old / dnorm2_old;
    addIntoFirstVectorSecondVectorScaled(*x_buf, *w_buf, alpha, XDIM);
    addIntoFirstVectorSecondVectorScaled(*c_buf, *d_buf, -alpha, BDIM);
    norm = std::sqrt(normBBuffer_barier_double(*c_buf));
    LOGE << io::xprintf("After iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of NB0.",
                        iteration, norm, 100.0 * norm / NB0);
    while(std::sqrt(dnorm2_old) / NB0 > errCondition && iteration < maxIterations)
    {
        // Iteration
        iteration = iteration + 1;
        backproject(*c_buf, *v_buf, PM, ICM, scalingFactors);
        reportTime(io::xprintf("v_%d backprojection", iteration));
        vnorm2_now = normXBuffer_barier_double(*v_buf);
        beta = vnorm2_now / vnorm2_old;
        LOGI << io::xprintf("In iteration %d, |v_now|^2=%E, |v_old|^2=%E, beta=%0.2f", iteration,
                            vnorm2_now, vnorm2_old, beta);
        vnorm2_old = vnorm2_now;
        addIntoFirstVectorScaledSecondVector(*w_buf, *v_buf, beta, XDIM);
        project(*w_buf, *d_buf, PM, ICM, scalingFactors);
        reportTime(io::xprintf("d_%d projection", iteration));
        dnorm2_old = normBBuffer_barier_double(*d_buf);
        alpha = vnorm2_old / dnorm2_old;
        addIntoFirstVectorSecondVectorScaled(*x_buf, *w_buf, alpha, XDIM);
        addIntoFirstVectorSecondVectorScaled(*c_buf, *d_buf, -alpha, BDIM);
        norm = std::sqrt(normBBuffer_barier_double(*c_buf));
        LOGE << io::xprintf(
            "After the iteration %d, the norm of |Ax-b| is %f that is %0.2f%% of NB0.", iteration,
            norm, 100.0 * norm / NB0);
    }
    Q->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

int CGLSReconstructor::reconstruct_experimental(
    std::shared_ptr<io::DenProjectionMatrixReader> matrices,
    uint32_t maxIterations,
    float errCondition)
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
    reportTime("CGLS INIT");
    if(reportProgress)
    {
        // writeProjections(*b_buf, io::xprintf("%sb.den", progressPrefixPath.c_str()));
        // writeVolume(*x_buf, io::xprintf("%sx_0.den", progressPrefixPath.c_str()));
    }
    std::vector<matrix::ProjectionMatrix> PM = encodeProjectionMatrices(matrices);
    std::vector<cl_double16> ICM = inverseProjectionMatrices(PM);
    std::vector<float> scalingFactors = computeScalingFactors(PM);
    uint32_t iteration = 0;
    double norm, vnorm2_old, vnorm2_now, dnorm2_old, alpha, beta;
    double NB0 = std::sqrt(normBBuffer_barier_double(*b_buf));
    norm = NB0;
    project(*x_buf, *tmp_b_buf, PM, ICM, scalingFactors);
    reportTime("X_0 projection");
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
    Q->enqueueFillBuffer<cl_float>(*v_proj, FLOATZERO, 0, uint32_t(pdimx) * pdimy * sizeof(float));
    Q->enqueueFillBuffer<cl_float>(*w_proj, FLOATZERO, 0, uint32_t(pdimx) * pdimy * sizeof(float));
    Q->enqueueFillBuffer<cl_float>(*x_proj, FLOATZERO, 0, uint32_t(pdimx) * pdimy * sizeof(float));
    // Experimental
    LOGI << io::xprintf("Initial norm of b is %f and initial |Ax-b| is %f.", NB0, norm);
    // INITIALIZATION x_0 is initialized typically by zeros but in general by supplied array
    // c_0 is filled by b
    // v_0=w_0=BACKPROJECT(c_0)
    // writeProjections(*c_buf, io::xprintf("/tmp/cgls/c_0.den"));
    backproject(*c_buf, *v_buf, PM, ICM, scalingFactors);
    // Experimental
    LOGI << "Backprojection correction vector";
    Q->enqueueFillBuffer<cl_float>(*v_proj, FLOATZERO, 0, pdimx * pdimy * sizeof(float));
    for(unsigned int i = 0; i != pdimz; i++)
    {
        copyFloatVectorOffset(*c_buf, i * pdimx * pdimy, *v_proj, 0, uint32_t(pdimx) * pdimy);
    }
    // Experimental
    reportTime("v_0 backprojection");
    if(reportProgress)
    {
        // writeVolume(*v_buf, io::xprintf("%sv_0.den", progressPrefixPath.c_str()));
    }
    vnorm2_old = normXBuffer_barier_double(*v_buf);
    // EXPERIMENTAL
    double sum;
    uint32_t framesize = pdimx * pdimy;
    cl::EnqueueArgs eargs1(*Q, cl::NDRange(1));
    (*NormSquare)(eargs1, *v_proj, *tmp_x_red1, framesize).wait();
    Q->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(double), &sum);
    vnorm2_old += sum;
    // EXPERIMENTAL
    copyFloatVector(*v_buf, *w_buf, XDIM);
    // EXPERIMENTAL
    copyFloatVector(*v_proj, *w_proj, pdimx * pdimy);
    // EXPERIMENTAL
    if(reportProgress)
    {
        // writeVolume(*w_buf, io::xprintf("%sw_0.den", progressPrefixPath.c_str()));
    }
    project(*w_buf, *d_buf, PM, ICM, scalingFactors);
    // Experimental
    for(unsigned int i = 0; i != pdimz; i++)
    {
        copyFloatVectorOffset(*w_proj, 0, *d_buf, i * pdimx * pdimy, pdimx * pdimy);
    }
    // Experimental
    reportTime("d_0 projection");
    if(reportProgress)
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
        if(reportProgress)
        {
            writeVolume(*x_buf, io::xprintf("%sx_%d.den", progressPrefixPath.c_str(), iteration));
        }
        addIntoFirstVectorSecondVectorScaled(*c_buf, *d_buf, -alpha, BDIM);
        if(reportProgress)
        {
            //    writeProjections(*c_buf,
            //                     io::xprintf("%sc_%d.den", progressPrefixPath.c_str(), iteration));
        }
        backproject(*c_buf, *v_buf, PM, ICM, scalingFactors);
        // Experimental
        Q->enqueueFillBuffer<cl_float>(*v_proj, FLOATZERO, 0, pdimx * pdimy * sizeof(float));
        for(unsigned int i = 0; i != pdimz; i++)
        {
            copyFloatVectorOffset(*c_buf, i * pdimx * pdimy, *v_proj, 0, pdimx * pdimy);
        }
        // Experimental
        reportTime(io::xprintf("v_%d backprojection", iteration));

        if(reportProgress)
        {
            //    writeVolume(*v_buf, io::xprintf("%sv_%d.den", progressPrefixPath.c_str(),
            //    iteration));
        }
        vnorm2_now = normXBuffer_barier_double(*v_buf);
        // EXPERIMENTAL
        cl::EnqueueArgs eargs1(*Q, cl::NDRange(1));
        (*NormSquare)(eargs1, *v_proj, *tmp_x_red1, framesize).wait();
        Q->enqueueReadBuffer(*tmp_x_red1, CL_TRUE, 0, sizeof(double), &sum);
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
        if(reportProgress)
        {
            //    writeVolume(*w_buf, io::xprintf("%sw_%d.den", progressPrefixPath.c_str(),
            //    iteration));
        }
        addIntoFirstVectorSecondVectorScaled(*tmp_b_buf, *d_buf, alpha, BDIM);
        project(*w_buf, *d_buf, PM, ICM, scalingFactors);
        // EXPERIMENTAL
        for(unsigned int i = 0; i != pdimz; i++)
        {
            copyFloatVectorOffset(*w_proj, 0, *d_buf, i * pdimx * pdimy, pdimx * pdimy);
        }
        // EXPERIMENTAL
        reportTime(io::xprintf("d_%d projection", iteration));
        if(reportProgress)
        {
            //    writeProjections(*d_buf,
            //                     io::xprintf("%sd_%d.den", progressPrefixPath.c_str(), iteration));
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
    Q->enqueueReadBuffer(*x_buf, CL_TRUE, 0, sizeof(float) * XDIM, x);
    return 0;
}

} // namespace CTL
