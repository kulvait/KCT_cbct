#pragma once
#include "MATRIX/ProjectionMatrix.hpp"
#include "SMA/BufferedSparseMatrixDoubleWritter.hpp"

namespace CTL {
namespace util {

    struct Elm
    {
        uint32_t pindex;
        double val;
        Elm(uint32_t pindex, double val)
            : pindex(pindex)
            , val(val)
        {
        }
        bool operator<(const Elm& e) { return pindex < e.pindex; }
    };

    struct Cube
    {
        // Index of the pixel to which the corner of the Cube is projected
        std::array<uint32_t, 8> pixelIndex;

        // There will also be
        std::array<uint32_t, 19> subPixelIndex;
        // Left lower bottom corner of the cube, needs to be increased by edgelength to get other
        // coords
        std::array<double, 3> corner;
        double edgeLength;
        double halfLength;
        uint32_t pdimx, pdimy;
        uint32_t detectorPixels;
        Cube(double lx, double ly, double lz, double edgeLength, uint32_t pdimx, uint32_t pdimy)
        {
            corner[0] = lx;
            corner[1] = ly;
            corner[2] = lz;
            this->edgeLength = edgeLength;
            this->halfLength = edgeLength / 2.0;
            this->pdimx = pdimx;
            this->pdimy = pdimy;
            detectorPixels = pdimx * pdimy;
        }
        // I index here such that 000 means the pixel index that belongs to the corner
        // 001 corner + dx
        // 010 corner + dy
        // 100 corner + dz
        // 00h corner + dx/2
        uint32_t get000() { return pixelIndex[0]; }
        void set000(uint32_t v) { pixelIndex[0] = v; }
        uint32_t get001() { return pixelIndex[1]; }
        void set001(uint32_t v) { pixelIndex[1] = v; }
        uint32_t get010() { return pixelIndex[2]; }
        void set010(uint32_t v) { pixelIndex[2] = v; }
        uint32_t get011() { return pixelIndex[3]; }
        void set011(uint32_t v) { pixelIndex[3] = v; }
        uint32_t get100() { return pixelIndex[4]; }
        void set100(uint32_t v) { pixelIndex[4] = v; }
        uint32_t get101() { return pixelIndex[5]; }
        void set101(uint32_t v) { pixelIndex[5] = v; }
        uint32_t get110() { return pixelIndex[6]; }
        void set110(uint32_t v) { pixelIndex[6] = v; }
        uint32_t get111() { return pixelIndex[7]; }
        void set111(uint32_t v) { pixelIndex[7] = v; }

        uint32_t get00H() { return subPixelIndex[0]; }
        void set00H(uint32_t v) { subPixelIndex[0] = v; }
        uint32_t get0H0() { return subPixelIndex[1]; }
        void set0H0(uint32_t v) { subPixelIndex[1] = v; }
        uint32_t get0HH() { return subPixelIndex[2]; }
        void set0HH(uint32_t v) { subPixelIndex[2] = v; }
        uint32_t get0H1() { return subPixelIndex[3]; }
        void set0H1(uint32_t v) { subPixelIndex[3] = v; }
        uint32_t get01H() { return subPixelIndex[4]; }
        void set01H(uint32_t v) { subPixelIndex[4] = v; }
        uint32_t getH00() { return subPixelIndex[5]; }
        void setH00(uint32_t v) { subPixelIndex[5] = v; }
        uint32_t getH0H() { return subPixelIndex[6]; }
        void setH0H(uint32_t v) { subPixelIndex[6] = v; }
        uint32_t getH01() { return subPixelIndex[7]; }
        void setH01(uint32_t v) { subPixelIndex[7] = v; }
        uint32_t getHH0() { return subPixelIndex[8]; }
        void setHH0(uint32_t v) { subPixelIndex[8] = v; }
        uint32_t getHHH() { return subPixelIndex[9]; }
        void setHHH(uint32_t v) { subPixelIndex[9] = v; }
        uint32_t getHH1() { return subPixelIndex[10]; }
        void setHH1(uint32_t v) { subPixelIndex[10] = v; }
        uint32_t getH10() { return subPixelIndex[11]; }
        void setH10(uint32_t v) { subPixelIndex[11] = v; }
        uint32_t getH1H() { return subPixelIndex[12]; }
        void setH1H(uint32_t v) { subPixelIndex[12] = v; }
        uint32_t getH11() { return subPixelIndex[13]; }
        void setH11(uint32_t v) { subPixelIndex[13] = v; }
        uint32_t get10H() { return subPixelIndex[14]; }
        void set10H(uint32_t v) { subPixelIndex[14] = v; }
        uint32_t get1H0() { return subPixelIndex[15]; }
        void set1H0(uint32_t v) { subPixelIndex[15] = v; }
        uint32_t get1HH() { return subPixelIndex[16]; }
        void set1HH(uint32_t v) { subPixelIndex[16] = v; }
        uint32_t get1H1() { return subPixelIndex[17]; }
        void set1H1(uint32_t v) { subPixelIndex[17] = v; }
        uint32_t get11H() { return subPixelIndex[18]; }
        void set11H(uint32_t v) { subPixelIndex[18] = v; }
        // Usually the biggest differences could be on the diagonal
        bool indicesAreEqual()
        {
            for(int i = 7; i > 0; i--)
            {
                if(pixelIndex[0] != pixelIndex[i])
                    return false;
            }
            return true;
        }

        void fillSubindices(ProjectionMatrix pm)
        {
            set00H(getIndex(pm, corner[0] + halfLength, corner[1], corner[2]));
            set0H0(getIndex(pm, corner[0], corner[1] + halfLength, corner[2]));
            set0HH(getIndex(pm, corner[0] + halfLength, corner[1] + halfLength, corner[2]));
            set0H1(getIndex(pm, corner[0] + edgeLength, corner[1] + halfLength, corner[2]));
            set01H(getIndex(pm, corner[0] + halfLength, corner[1] + edgeLength, corner[2]));
            setH00(getIndex(pm, corner[0], corner[1], corner[2] + halfLength));
            setH0H(getIndex(pm, corner[0] + halfLength, corner[1], corner[2] + halfLength));
            setH01(getIndex(pm, corner[0] + edgeLength, corner[1], corner[2] + halfLength));
            setHH0(getIndex(pm, corner[0], corner[1] + halfLength, corner[2] + halfLength));
            setHHH(getIndex(pm, corner[0] + halfLength, corner[1] + halfLength,
                            corner[2] + halfLength));
            setHH1(getIndex(pm, corner[0] + edgeLength, corner[1] + halfLength,
                            corner[2] + halfLength));
            setH10(getIndex(pm, corner[0], corner[1] + edgeLength, corner[2] + halfLength));
            setH1H(getIndex(pm, corner[0] + halfLength, corner[1] + edgeLength,
                            corner[2] + halfLength));
            setH11(getIndex(pm, corner[0] + edgeLength, corner[1] + edgeLength,
                            corner[2] + halfLength));
            set10H(getIndex(pm, corner[0] + halfLength, corner[1], corner[2] + edgeLength));
            set1H0(getIndex(pm, corner[0], corner[1] + halfLength, corner[2] + edgeLength));
            set1HH(getIndex(pm, corner[0] + halfLength, corner[1] + halfLength,
                            corner[2] + edgeLength));
            set1H1(getIndex(pm, corner[0] + edgeLength, corner[1] + halfLength,
                            corner[2] + edgeLength));
            set11H(getIndex(pm, corner[0] + halfLength, corner[1] + edgeLength,
                            corner[2] + edgeLength));
        }

        void fillSubcubes(ProjectionMatrix pm,
                          Cube* c000,
                          Cube* c001,
                          Cube* c010,
                          Cube* c011,
                          Cube* c100,
                          Cube* c101,
                          Cube* c110,
                          Cube* c111)
        {
            fillSubindices(pm);

            c000->set000(get000());
            c000->set001(get00H());
            c000->set010(get0H0());
            c000->set011(get0HH());
            c000->set100(getH00());
            c000->set101(getH0H());
            c000->set110(getHH0());
            c000->set111(getHHH());

            c001->set000(get00H());
            c001->set001(get001());
            c001->set010(get0HH());
            c001->set011(get0H1());
            c001->set100(getH0H());
            c001->set101(getH01());
            c001->set110(getHHH());
            c001->set111(getHH1());

            c010->set000(get0H0());
            c010->set001(get0HH());
            c010->set010(get010());
            c010->set011(get01H());
            c010->set100(getHH0());
            c010->set101(getHHH());
            c010->set110(getH10());
            c010->set111(getH1H());

            c011->set000(get0HH());
            c011->set001(get0H1());
            c011->set010(get01H());
            c011->set011(get011());
            c011->set100(getHHH());
            c011->set101(getHH1());
            c011->set110(getH1H());
            c011->set111(getH11());

            c100->set000(getH00());
            c100->set001(getH0H());
            c100->set010(getHH0());
            c100->set011(getHHH());
            c100->set100(get100());
            c100->set101(get10H());
            c100->set110(get1H0());
            c100->set111(get1HH());

            c101->set000(getH0H());
            c101->set001(getH01());
            c101->set010(getHHH());
            c101->set011(getHH1());
            c101->set100(get10H());
            c101->set101(get101());
            c101->set110(get1HH());
            c101->set111(get1H1());

            c110->set000(getHH0());
            c110->set001(getHHH());
            c110->set010(getH10());
            c110->set011(getH1H());
            c110->set100(get1H0());
            c110->set101(get1HH());
            c110->set110(get110());
            c110->set111(get11H());

            c111->set000(getHHH());
            c111->set001(getHH1());
            c111->set010(getH1H());
            c111->set011(getH11());
            c111->set100(get1HH());
            c111->set101(get1H1());
            c111->set110(get11H());
            c111->set111(get111());
        }

        uint32_t centerIndex(ProjectionMatrix pm)
        {
            return getIndex(pm, corner[0] + halfLength, corner[1] + halfLength,
                            corner[2] + halfLength);
        }

        uint32_t getIndex(ProjectionMatrix pm, double x, double y, double z)
        {
            double px, py;
            int pi, pj;
            pm.project(x, y, z, &px, &py);
            pi = (int)(std::lround(
                px)); // 0.5 is correct since the grid of the projector starts on -0.5,
                      // -0.5
            pj = (int)(std::lround(py));
            if(pi >= 0 && pj >= 0 && pi < (int)pdimx && pj < (int)pdimy)
            {
                return pj * pdimx + pi;
            } else
            {
                return pdimx * pdimy;
            }
        }
    };

    class DivideAndConquerFootprintExecutor
    {
    public:
        DivideAndConquerFootprintExecutor(
            std::shared_ptr<matrix::BufferedSparseMatrixDoubleWritter> w,
            uint32_t pdimx,
            uint32_t pdimy,
            uint32_t vdimx,
            uint32_t vdimy,
            uint32_t vdimz,
            double scalingFactor,
            int threads,
            double terminatingEdgeLength = 0.25)
        {
            this->w = w;
            this->pdimx = pdimx;
            this->pdimy = pdimy;
            this->vdimx = vdimx;
            this->vdimy = vdimy;
            this->vdimz = vdimz;
            this->threads = threads;
            this->totalWritesExact = 0;
            this->totalWritesInexact = 0;
            this->voxelCornerNum = (vdimx + 1) * (vdimy + 1) * (vdimz + 1);
            this->resultingIndices = new uint32_t[voxelCornerNum];
            this->scalingFactor = scalingFactor;
            this->threadpool = nullptr;
            this->terminatingEdgeLength = terminatingEdgeLength;
        }

        ~DivideAndConquerFootprintExecutor()
        {
            delete[] resultingIndices;
            if(threadpool != nullptr)
            {
                threadpool->stop(true);
                delete threadpool;
            }
        }

        uint32_t getPixelIndex(ProjectionMatrix pm, float x, float y, float z)
        {
            float px, py;
            int pi, pj;
            pm.project(x, y, z, &px, &py);
            pi = (int)(std::lround(
                px)); // 0.5 is correct since the grid of the projector starts on -0.5,
                      // -0.5
            pj = (int)(std::lround(py));
            if(pi >= 0 && pj >= 0 && pi < (int)pdimx && pj < (int)pdimy)
            {
                return pj * pdimx + pi;
            } else
            {
                return pdimx * pdimy;
            }
        }

        void insertWeightFactors(std::vector<Elm>& vec,
                                 Cube& c,
                                 ProjectionMatrix pm,
                                 std::array<double, 3>& sourcePosition,
                                 std::array<double, 3>& normalToDetector)
        {
            bool equalIndices = c.indicesAreEqual();
            if(c.edgeLength > terminatingEdgeLength && !equalIndices)
            {
                Cube c000(c.corner[0], c.corner[1], c.corner[2], c.halfLength, pdimx, pdimy);
                Cube c001(c.corner[0] + c.halfLength, c.corner[1], c.corner[2], c.halfLength, pdimx,
                          pdimy);
                Cube c010(c.corner[0], c.corner[1] + c.halfLength, c.corner[2], c.halfLength, pdimx,
                          pdimy);
                Cube c011(c.corner[0] + c.halfLength, c.corner[1] + c.halfLength, c.corner[2],
                          c.halfLength, pdimx, pdimy);
                Cube c100(c.corner[0], c.corner[1], c.corner[2] + c.halfLength, c.halfLength, pdimx,
                          pdimy);
                Cube c101(c.corner[0] + c.halfLength, c.corner[1], c.corner[2] + c.halfLength,
                          c.halfLength, pdimx, pdimy);
                Cube c110(c.corner[0], c.corner[1] + c.halfLength, c.corner[2] + c.halfLength,
                          c.halfLength, pdimx, pdimy);
                Cube c111(c.corner[0] + c.halfLength, c.corner[1] + c.halfLength,
                          c.corner[2] + c.halfLength, c.halfLength, pdimx, pdimy);
                c.fillSubcubes(pm, &c000, &c001, &c010, &c011, &c100, &c101, &c110, &c111);

                insertWeightFactors(vec, c000, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c001, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c010, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c011, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c100, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c101, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c110, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c111, pm, sourcePosition, normalToDetector);
            } else
            {
                uint32_t pixelIndex;
                if(equalIndices)
                {
                    pixelIndex = c.pixelIndex[0];
                    totalWritesExact++;
                } else
                {
                    pixelIndex = c.centerIndex(pm);
                    totalWritesInexact++;
                }
                if(pixelIndex != c.detectorPixels)
                {
                    double volume = c.edgeLength * c.edgeLength * c.edgeLength;
                    double v_x = c.corner[0] + c.halfLength - sourcePosition[0];
                    double v_y = c.corner[1] + c.halfLength - sourcePosition[1];
                    double v_z = c.corner[2] + c.halfLength - sourcePosition[2];
                    double distsquare = v_x * v_x + v_y * v_y + v_z * v_z;
                    double norm = std::sqrt(distsquare);
                    double cos = (normalToDetector[0] * v_x + normalToDetector[1] * v_y
                                  + normalToDetector[2] * v_z)
                        / norm;
                    double cos3 = cos * cos * cos;
                    // LOGD << io::xprintf("Edge length is %f, volume is %f, scaling factor is %f,
                    // distsquare %f and cos3 is %f.", c.edgeLength, volume, scalingFactor,
                    // distsquare, cos3);
                    vec.push_back(Elm(pixelIndex, volume * scalingFactor / (cos3 * distsquare)));
                }
            }
        }

        void computeWeightFactors(Cube c,
                                  ProjectionMatrix pm,
                                  uint32_t voxelIndex,
                                  uint32_t pixelIndexOffset,
                                  std::array<double, 3> sourcePosition,
                                  std::array<double, 3> normalToDetector)
        {

            bool equalIndices = c.indicesAreEqual();
            if(c.edgeLength > terminatingEdgeLength && !equalIndices)
            {
                std::vector<Elm> vec;
                Cube c000(c.corner[0], c.corner[1], c.corner[2], c.halfLength, pdimx, pdimy);
                Cube c001(c.corner[0] + c.halfLength, c.corner[1], c.corner[2], c.halfLength, pdimx,
                          pdimy);
                Cube c010(c.corner[0], c.corner[1] + c.halfLength, c.corner[2], c.halfLength, pdimx,
                          pdimy);
                Cube c011(c.corner[0] + c.halfLength, c.corner[1] + c.halfLength, c.corner[2],
                          c.halfLength, pdimx, pdimy);
                Cube c100(c.corner[0], c.corner[1], c.corner[2] + c.halfLength, c.halfLength, pdimx,
                          pdimy);
                Cube c101(c.corner[0] + c.halfLength, c.corner[1], c.corner[2] + c.halfLength,
                          c.halfLength, pdimx, pdimy);
                Cube c110(c.corner[0], c.corner[1] + c.halfLength, c.corner[2] + c.halfLength,
                          c.halfLength, pdimx, pdimy);
                Cube c111(c.corner[0] + c.halfLength, c.corner[1] + c.halfLength,
                          c.corner[2] + c.halfLength, c.halfLength, pdimx, pdimy);
                c.fillSubcubes(pm, &c000, &c001, &c010, &c011, &c100, &c101, &c110, &c111);
                insertWeightFactors(vec, c000, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c001, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c010, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c011, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c100, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c101, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c110, pm, sourcePosition, normalToDetector);
                insertWeightFactors(vec, c111, pm, sourcePosition, normalToDetector);

                std::sort(vec.begin(), vec.end());
                uint32_t prevind = c.detectorPixels;
                double sum = 0.0;
                // LOGI << "Calling vector iteration";
                for(auto const& e : vec)
                {
                    //	LOGD << io::xprintf("Vector index = %d, val = %f.", e.pindex, e.val);
                    if(e.pindex == prevind)
                    {
                        sum += e.val;
                    } else
                    {
                        if(prevind != c.detectorPixels && sum != 0.0)
                        {
                            w->insertValue(voxelIndex, prevind + pixelIndexOffset, sum);
                        }
                        prevind = e.pindex;
                        sum = e.val;
                    }
                }
                if(prevind != c.detectorPixels && sum != 0.0)
                {
                    w->insertValue(voxelIndex, prevind + pixelIndexOffset, sum);
                    sum = 0.0;
                }
            } else
            {
                uint32_t pixelIndex;
                if(equalIndices)
                {
                    pixelIndex = c.pixelIndex[0];
                    totalWritesExact++;
                } else
                {
                    pixelIndex = c.centerIndex(pm);
                    totalWritesInexact++;
                }
                if(pixelIndex != c.detectorPixels)
                {
                    double volume = c.edgeLength * c.edgeLength * c.edgeLength;
                    double v_x = c.corner[0] + c.halfLength - sourcePosition[0];
                    double v_y = c.corner[1] + c.halfLength - sourcePosition[1];
                    double v_z = c.corner[2] + c.halfLength - sourcePosition[2];
                    double distsquare = v_x * v_x + v_y * v_y + v_z * v_z;
                    double norm = std::sqrt(distsquare);
                    double cos = (normalToDetector[0] * v_x + normalToDetector[1] * v_y
                                  + normalToDetector[2] * v_z)
                        / norm;
                    double cos3 = cos * cos * cos;
                    // LOGD << io::xprintf("Edge length is %f, volume is %f, scaling factor is
                    // %f, distsquare %f and cos3 is %f.", c.edgeLength, volume, scalingFactor,
                    // distsquare, cos3);
                    //   double d_x, d_y;
                    //   pm.project(v_x, v_y, v_z, &d_x, &d_y);
                    // LOGD << io::xprintf("(v_x, v_y, v_z) = (%f, %f, %f), (d_x, d_y) = (%f,
                    // %f), normal=(%f, %f, %f).", v_x, v_y, v_z, d_x, d_y, normalToDetector[0],
                    // normalToDetector[1], normalToDetector[2]);
                    w->insertValue(voxelIndex, pixelIndex + pixelIndexOffset,
                                   volume * scalingFactor / (distsquare * cos3));
                }
            }
        }

        void insertMatrixProjections(ProjectionMatrix pm, uint32_t pixelIndexOffset)
        {
            if(!threadpoolstarted)
            {
                startThreadpool();
            }
            std::array<double, 3> sourcePosition = pm.sourcePosition();
            std::array<double, 3> normalToDetector = pm.normalToDetector();

            LOGD << io::xprintf(
                "Source position is [%f, %f, %f] and normal to detector [%f, %f, %f]",
                sourcePosition[0], sourcePosition[1], sourcePosition[2], normalToDetector[0],
                normalToDetector[1], normalToDetector[2]);
            double xcoord, ycoord, zcoord;
            // First I try to precompute indices of each corner that is
            // (vdimx+1)x(vdimy+1)x(vdimz+1)
            uint32_t voxelindex = 0;
            zcoord = -(double(vdimz) / 2.0);
            for(uint32_t k = 0; k != vdimz + 1; k++)
            {
                ycoord = -(double(vdimy) / 2.0);
                for(uint32_t j = 0; j != vdimy + 1; j++)
                {
                    xcoord = -(double(vdimx) / 2.0);
                    for(uint32_t i = 0; i != vdimx + 1; i++)
                    {
                        resultingIndices[voxelindex]
                            = this->getPixelIndex(pm, xcoord, ycoord, zcoord);
                        xcoord += 1.0;
                        voxelindex++;
                    }
                    ycoord += 1.0;
                }
                zcoord += 1.0;
            }
            // w->flush();
            // LOGI << io::xprintf("There were %d writes and %d non writes to the matrix that
            // should
            // "
            //                    "result in the increase of its size by %d bytes.",
            //                    numberOfWrites, nonwrites, numberOfWrites * 16);
            Cube c(-(double(vdimx) / 2.0), -(double(vdimy) / 2.0), -(double(vdimz) / 2.0), 1.0,
                   pdimx, pdimy);
            // c.edgeLength = 1.0;
            // c.halfLength = 0.5;
            // c.corner[0] = -(double(vdimx) / 2.0) - 0.5;
            // c.corner[1] = -(double(vdimy) / 2.0) - 0.5;
            // c.corner[2] = -(double(vdimz) / 2.0) - 0.5;
            voxelindex = 0;
            // totalWrites = 0;
            c.corner[2] = -(double(vdimz) / 2.0);
            for(uint32_t k = 0; k != vdimz; k++)
            {
                c.corner[1] = -(double(vdimy) / 2.0);
                for(uint32_t j = 0; j != vdimy; j++)
                {
                    c.corner[0] = -(double(vdimx) / 2.0);
                    for(uint32_t i = 0; i != vdimx; i++)
                    {

                        c.set000(
                            resultingIndices[i + (vdimx + 1) * j + (vdimx + 1) * (vdimy + 1) * k]);
                        c.set001(resultingIndices[i + 1 + (vdimx + 1) * j
                                                  + (vdimx + 1) * (vdimy + 1) * k]);
                        c.set010(resultingIndices[i + (vdimx + 1) * (j + 1)
                                                  + (vdimx + 1) * (vdimy + 1) * k]);
                        c.set011(resultingIndices[i + 1 + (vdimx + 1) * (j + 1)
                                                  + (vdimx + 1) * (vdimy + 1) * k]);
                        c.set100(resultingIndices[i + (vdimx + 1) * j
                                                  + (vdimx + 1) * (vdimy + 1) * (k + 1)]);
                        c.set101(resultingIndices[i + 1 + (vdimx + 1) * j
                                                  + (vdimx + 1) * (vdimy + 1) * (k + 1)]);
                        c.set110(resultingIndices[i + (vdimx + 1) * (j + 1)
                                                  + (vdimx + 1) * (vdimy + 1) * (k + 1)]);
                        c.set111(resultingIndices[i + 1 + (vdimx + 1) * (j + 1)
                                                  + (vdimx + 1) * (vdimy + 1) * (k + 1)]);

                        threadpool->push([&, this, c, pm, voxelindex, pixelIndexOffset,
                                          sourcePosition, normalToDetector](int id) {
                            this->computeWeightFactors(c, pm, voxelindex, pixelIndexOffset,
                                                       sourcePosition, normalToDetector);
                        });
                        // computeWeightFactors(c, pm, voxelindex, pixelIndexOffset);
                        //            if(voxelindex != i + j * vdimx + k * vdimx * vdimy)
                        //            {
                        //                LOGD << "WRONG INDEX";
                        //            }
                        voxelindex++;
                        c.corner[0] += 1.0;
                    }
                    c.corner[1] += 1.0;
                }
                c.corner[2] += 1.0;
            }
        }

        void reportNumberOfWrites()
        {
            LOGD << io::xprintf("Performed %lu exact writes and %lu inexact writes.",
                                totalWritesExact, totalWritesInexact);
        }

        // To manage threadpooling from outside
        void startThreadpool()
        {
            if(threadpool != nullptr)
            {
                stopThreadpool();
            }
            threadpool = new ctpl::thread_pool(threads);
            threadpoolstarted = true;
        }

        // To manage threadpooling from outside
        void stopThreadpool()
        {
            if(threadpool != nullptr)
            {
                threadpool->stop(true);
                delete threadpool;
                threadpool = nullptr;
            }
            threadpoolstarted = false;
        }

    private:
        bool threadpoolstarted = false;
        ctpl::thread_pool* threadpool;
        std::shared_ptr<matrix::BufferedSparseMatrixDoubleWritter> w;
        uint32_t voxelCornerNum;
        uint32_t* resultingIndices;
        // It is evaluated from -0.5, pixels are centerred at integer coordinates
        uint32_t pdimx = 616;
        uint32_t pdimy = 480;
        // Here (0,0,0) is in the center of the volume
        uint32_t vdimx = 256;
        uint32_t vdimy = 256;
        uint32_t vdimz = 199;
        int threads = 1;
        double terminatingEdgeLength;
        uint64_t totalWritesExact, totalWritesInexact;
        // Square distance from source to detector divided by the area of pixel.
        double scalingFactor;
    }; // namespace util
} // namespace util
} // namespace CTL
