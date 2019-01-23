#pragma once
#include "MATH/round.h"
#include "MATRIX/ProjectionMatrix.hpp"
#include "SMA/BufferedSparseMatrixFloatWritter.hpp"

namespace CTL {
namespace util {

    struct ElmInt
    {
        int pindex;
        float val;
        ElmInt(int pindex, float val)
            : pindex(pindex)
            , val(val)
        {
        }
        bool operator<(const ElmInt& e) { return pindex < e.pindex; }
    };

    struct Point2D
    {
        float x;
        float y;

        Point2D()
            : x(0)
            , y(0)
        {
        }

        Point2D(float x, float y)
            : x(x)
            , y(y)
        {
        }
    };

    struct Prism
    {
        std::vector<Point2D> points;
        float surface;
        int pxindex;
        Prism(float surface, int pxindex)
            : surface(surface)
            , pxindex(pxindex)
        {
        }

        void addPoint(float x, float y) { points.push_back(Point2D(x, y)); }
        void addPoint(Point2D p) { points.push_back(p); }
    };

    struct Cube
    {
        // Index of the pixel to which the corner of the Cube is projected
        std::array<uint32_t, 8> pixelIndex;

        // There will also be
        std::array<uint32_t, 19> subPixelIndex;
        // Left lower bottom corner of the cube, needs to be increased by edgelength to get other
        // coords
        std::array<float, 3> corner;
        float edgeLength;
        float halfLength;
        uint32_t pdimx, pdimy;
        uint32_t detectorPixels;
        Cube(float lx, float ly, float lz, double edgeLength, uint32_t pdimx, uint32_t pdimy)
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

        void fillSubindices(matrix::ProjectionMatrix pm)
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

        void fillSubcubes(matrix::ProjectionMatrix pm,
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

        uint32_t centerIndex(matrix::ProjectionMatrix pm)
        {
            return getIndex(pm, corner[0] + halfLength, corner[1] + halfLength,
                            corner[2] + halfLength);
        }

        uint32_t getIndex(matrix::ProjectionMatrix pm, float x, float y, float z)
        {
            float px, py;
            int pi, pj;
            pm.project(x, y, z, &px, &py);
            pi = (int)(math::lroundLow(px)); // Pixel (0,0) has boundaries (-0.5,+0.5].
            pj = (int)(math::lroundLow(py));
            if(pi >= 0 && pj >= 0 && pi < (int)pdimx && pj < (int)pdimy)
            {
                return pj * pdimx + pi;
            } else
            {
                return pdimx * pdimy;
            }
        }
    };

    class VolumeFootprintExecutor
    {
    public:
        VolumeFootprintExecutor(std::shared_ptr<matrix::BufferedSparseMatrixFloatWritter> w,
                                uint32_t pdimx,
                                uint32_t pdimy,
                                uint32_t vdimx,
                                uint32_t vdimy,
                                uint32_t vdimz,
                                float scalingFactor,
                                int threads)
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
        }

        ~VolumeFootprintExecutor()
        {
            delete[] resultingIndices;
            if(threadpool != nullptr)
            {
                threadpool->stop(true);
                delete threadpool;
            }
        }

        uint32_t getProjectionIndex(matrix::ProjectionMatrix pm, float x, float y, float z)
        {
            float px, py;
            int pi, pj;
            pm.project(x, y, z, &px, &py);
            pi = (int)(math::lroundLow(px)); // 0.5 is correct
            pj = (int)(math::lroundLow(py));
            if(pi >= 0 && pj >= 0 && pi < (int)pdimx && pj < (int)pdimy)
            {
                return pj * pdimx + pi;
            } else
            {
                return pdimx * pdimy;
            }
        }

        // Now cut it in z slices
        void insertPrism(std::shared_ptr<Prism> prism,
                         Cube c,
                         uint32_t voxelIndex,
                         matrix::ProjectionMatrix pm,
                         uint32_t pixelIndexOffset,
                         std::array<float, 3> sourcePosition,
                         std::array<float, 3> normalToDetector)
        {
            // If prism projection x index is not on the detector, write nothing
            if(prism->pxindex < 0 || prism->pxindex > (int)pdimx)
            {
                return;
            } else
            {
                // We have x coordinate fixed and since we are on the boundary we are not working
                // with its projections because it could be close.
                std::vector<ElmInt> zindices;
                for(std::size_t i = 0; i != prism->points.size(); i++)
                {
                    float px, py, py1;
                    int py_i, py1_i; // Integer representation of the points
                    pm.project(prism->points[i].x, prism->points[i].y, (float)c.corner[2], &px,
                               &py); // Coordinate px should not differ when projecting points where
                                     // only z coordinate varies.
                    pm.project(prism->points[i].x, prism->points[i].y,
                               (float)(c.corner[2] + c.edgeLength), &px, &py1);
                    if(py == py1)
                    {
                        // There is no point to add along this line
                        return;
                    }
                    py_i = math::lroundLow(py);
                    py1_i = math::lroundLow(py1);
                    if(py_i == py1_i)
                    {
                        zindices.push_back(ElmInt(py_i, c.edgeLength));
                        continue;
                    }
                    int py_cur = py_i;
                    float stepLength
                        = c.edgeLength / (py1 - py); // Length of step for which py increases by one
                    float nextGridPy; // Which will be the next intersection with the projector grid
                                      // on the lixe [vx, vy, vz] -- [vx, vy, vz+edgeLength]
                    if(stepLength > 0)
                    {
                        nextGridPy = (float(py_i) + 0.5);
                    } else
                    {
                        nextGridPy = (float(py_i) - 0.5);
                    }
                    float intersection = stepLength * (nextGridPy - py);
                    zindices.push_back(ElmInt(py_cur, intersection));
                    while(intersection < c.edgeLength)
                    {
                        if(stepLength > 0)
                        {
                            py_cur++;
                        } else
                        {
                            py_cur--;
                        }
                        intersection += std::abs(stepLength);
                        if(intersection < c.edgeLength)
                        {
                            zindices.push_back(ElmInt(py_cur, std::abs(stepLength)));
                        } else
                        {
                            zindices.push_back(
                                ElmInt(py_cur, c.edgeLength - intersection + std::abs(stepLength)));
                        }
                    }
                }
                std::sort(zindices.begin(), zindices.end());
                assert(zindices.size() > 0);
                // Now duplicates will be represented by one object with the value that is sum of
                // values of
                int lastpindex = zindices[0].pindex;
                int previndex = 0;
                for(std::size_t i = 1; i < zindices.size(); i++)
                {
                    if(lastpindex == zindices[i].pindex)
                    {
                        zindices[previndex].val += zindices[i].val;
                        zindices[i].val = 0.0;
                    } else
                    {
                        lastpindex = zindices[i].pindex;
                        previndex = i;
                    }
                }
                // Remove all zero elements
                zindices.erase(std::remove_if(zindices.begin(), zindices.end(),
                                              [](const ElmInt& e) { return e.val == 0; }),
                               zindices.end());

                float totalSum = 0;
                for(std::size_t i = 0; i != zindices.size(); i++)
                {
                    totalSum += zindices[i].val;
                }
                for(std::size_t i = 0; i != zindices.size(); i++)
                {
                    zindices[i].val /= totalSum;
                }
                for(std::size_t i = 0; i != zindices.size(); i++)
                {
                    if(zindices[i].pindex >= 0 && zindices[i].pindex < (int)pdimy)
                    {
                        double volume = prism->surface * zindices[i].val;
                        double v_x = prism->points[0].x - sourcePosition[0];
                        double v_y = prism->points[0].y - sourcePosition[1];
                        double v_z = c.corner[2] + c.halfLength - sourcePosition[2];
                        double distsquare = v_x * v_x + v_y * v_y + v_z * v_z;
                        double norm = std::sqrt(distsquare);
                        double cos = (normalToDetector[0] * v_x + normalToDetector[1] * v_y
                                      + normalToDetector[2] * v_z)
                            / norm;
                        double cos3 = cos * cos * cos;
                        w->insertValue(voxelIndex,
                                       zindices[i].pindex * pdimx + prism->pxindex
                                           + pixelIndexOffset,
                                       float(scalingFactor * volume / (distsquare * cos3)));
                    }
                }
            }
        }

        /** Find the time of intersection of voxel and pixel grid in a plane projected to pixel x
         *coordinate.
         *
         *Lets a=(x,y) and b =(x1,y1), z coordinate of both points iz a z. Let's have t \in [0,1)
         *that parametrizes the segment (x+t(x1-x), y+t(y1-y), z). Suppose that the value of vx
         *coordinate on the projection plane is non decreasing with the value of x. Function finds
         *all the values of parameter t \in [0,1) on the segment (x, y, z) -- (x1, y1, z), for which
         *projection of (x+t(x1-x), y+t(y1-y), z) to (px, py) is exactly on the boundary of the
         *pixel, that means px + 0.5 is integer. It adds the values of t+pointOffset to the
         *boundaryPoints vector and adds the values of (x+t(x1-x), y+t(y1-y)) inte the points
         *vector.
         */
        void findBoundaryPoints(std::shared_ptr<std::vector<float>> boundaryPoints,
                                std::shared_ptr<std::vector<Point2D>> points,
                                matrix::ProjectionMatrix pm,
                                Point2D* a,
                                Point2D* b,
                                float z,
                                float pointOffset)
        {

            float px, px1, py;
            float px_boundary; // The highest px coordinate that maps to the same pixel (or to the
                               // boundary)
            int ax_i, bx_i;
            int boundarySteps;
            pm.project(a->x, a->y, z, &px, &py); // z coordinate is the same for all points
            pm.project(b->x, b->y, z, &px1, &py);
            px_boundary = math::roundLow(px) + 0.5; // Here it is beneficial to have the point as a
                                                    // float since I am working with it as with the
                                                    // coordinate of the boundary
            assert(px - math::roundLow(px) != -0.5);
            ax_i = math::lroundLow(px);
            bx_i = math::lroundLow(px1);
            boundarySteps = bx_i - ax_i;
            float edgeLength
                = std::sqrt((a->x - b->x) * (a->x - b->x) + (a->y - b->y) * (a->y - b->y));
            float stepLength = edgeLength / (px1 - px);
            if(stepLength < 0.0)
            {
                io::throwerr("Step length is negative!");
            }
            float pxtoedge = px_boundary - px; // How far is the boundary on the detector in x from
                                               // the place to which x,y,z maps
            float intersection = stepLength * pxtoedge; // Due to rounding errors rather use for
            for(int i = 0; i != boundarySteps; i++)
            {
                boundaryPoints->push_back(intersection + pointOffset);
                points->push_back(Point2D(a->x + intersection * (b->x - a->x),
                                          a->y + intersection * (b->y - a->y)));
                intersection += stepLength;
            }
        }

        void computeWeightFactors(Cube c,
                                  matrix::ProjectionMatrix pm,
                                  uint32_t voxelIndex,
                                  uint32_t pixelIndexOffset,
                                  std::array<float, 3> sourcePosition,
                                  std::array<float, 3> normalToDetector)
        {
            bool equalIndices = c.indicesAreEqual();
            if(equalIndices)
            {
                if(c.corner[0] == c.detectorPixels)
                {
                    return;
                }
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
                w->insertValue(voxelIndex, c.get000() + pixelIndexOffset,
                               float(volume * scalingFactor / (distsquare * cos3)));

            } else
            {
                // First I determine volumes of the objects that will have the same px
                // coordinate

                float py;
                float px00, px01, px10, px11;
                pm.project(c.corner[0], c.corner[1], c.corner[2], &px00, &py);
                pm.project(c.corner[0] + c.edgeLength, c.corner[1], c.corner[2], &px01, &py);
                pm.project(c.corner[0], c.corner[1] + c.edgeLength, c.corner[2], &px10, &py);
                pm.project(c.corner[0] + c.edgeLength, c.corner[1] + c.edgeLength, c.corner[2],
                           &px11, &py);

                // There is a corner with minimal px; ... find this corner
                std::shared_ptr<std::vector<float>> intersectionsCCW
                    = std::make_shared<std::vector<float>>();
                std::shared_ptr<std::vector<float>> intersectionsCW
                    = std::make_shared<std::vector<float>>();
                std::shared_ptr<std::vector<Point2D>> pointsCCW
                    = std::make_shared<std::vector<Point2D>>();
                std::shared_ptr<std::vector<Point2D>> pointsCW
                    = std::make_shared<std::vector<Point2D>>();
                int pxindex;
                // Four points in the four corners of the down square of voxel
                Point2D V00(c.corner[0], c.corner[1]);
                Point2D V01(c.corner[0] + c.edgeLength, c.corner[1]);
                Point2D V10(c.corner[0], c.corner[1] + c.edgeLength);
                Point2D V11(c.corner[0] + c.edgeLength, c.corner[1] + c.edgeLength);

                Point2D* V_max;
                Point2D* V_ccw[4];
                if(px00 == std::min({ px00, px01, px10, px11 }))
                {
                    pxindex = int(math::lroundLow(px00));
                    V_ccw[0] = &V00;
                    V_ccw[1] = &V01;
                    V_ccw[2] = &V11;
                    V_ccw[3] = &V10;
                } else if(px01 == std::min({ px00, px01, px10, px11 }))
                {
                    pxindex = int(math::lroundLow(px01));
                    V_ccw[0] = &V01;
                    V_ccw[1] = &V11;
                    V_ccw[2] = &V10;
                    V_ccw[3] = &V00;
                } else if(px10 == std::min({ px00, px01, px10, px11 }))
                {
                    pxindex = int(math::lroundLow(px10));
                    V_ccw[0] = &V10;
                    V_ccw[1] = &V00;
                    V_ccw[2] = &V01;
                    V_ccw[3] = &V11;
                } else // its px11
                {
                    pxindex = int(math::lroundLow(px11));
                    V_ccw[0] = &V11;
                    V_ccw[1] = &V10;
                    V_ccw[2] = &V00;
                    V_ccw[3] = &V01;
                }
                if(px00 == std::max({ px00, px01, px10, px11 }))
                {
                    V_max = &V00;
                } else if(px01 == std::max({ px00, px01, px10, px11 }))
                {
                    V_max = &V01;
                } else if(px10 == std::max({ px00, px01, px10, px11 }))
                {
                    V_max = &V10;
                } else // its px11
                {
                    V_max = &V11;
                }
                for(int i = 0; i != 4; i++)
                {
                    findBoundaryPoints(intersectionsCCW, pointsCCW, pm, V_ccw[i], V_ccw[i + 1],
                                       c.corner[2], float(i));
                    if(V_ccw[i + 1] == V_max)
                    {
                        break;
                    }
                }
                // Go against the clock until reaching p_max point
                for(int i = 0; i != 4; i++)
                {
                    findBoundaryPoints(intersectionsCW, pointsCW, pm, V_ccw[(4 - i) % 4],
                                       V_ccw[(4 - i - 1) % 4], c.corner[2], float(i));
                    if(V_ccw[(4 - i - 1) % 4] == V_max)
                    {
                        break;
                    }
                }
                if(intersectionsCCW->size() != intersectionsCW->size())
                {
                    assert(intersectionsCCW->size() == intersectionsCW->size());
                    assert(pointsCCW->size() == pointsCW->size());
                    assert(pointsCCW->size() == intersectionsCCW->size());
                }
                std::vector<float> volumes;
                // From the values of intersection I can compute the area of the square that
                // contains point 0,0 and points of intersections
                // Volumes are relative to unit square area of one.
                for(std::size_t i = 0; i != intersectionsCW->size(); i++)
                {
                    float a, b;
                    a = (*intersectionsCCW)[i];
                    b = (*intersectionsCW)[i];
                    if(a > b)
                    {
                        std::swap(a, b);
                    }
                    // a<=b
                    if(a <= 1.0)
                    {
                        if(b <= 1.0)
                        {
                            volumes.push_back(
                                a * b / 2.0); // Triangle that is bounded by corners (0, a, b)
                        } else if(b <= 2.0)
                        {
                            b = b - 1.0; // Upper edge length
                            volumes.push_back(a + (b - a) / 2.0);
                        } else if(b <= 3.0)
                        {
                            a = 1.0 - a;
                            b = 3.0 - b;
                            volumes.push_back(1.0 - (a * b) / 2.0); // Whole square but the area of
                                                                    // the triangle bounded by
                                                                    // corners (a, 1, b)
                        }
                    } else if(a <= 2.0)
                    {
                        assert(b <= 3.0);
                        a = 2.0 - a;
                        b = 2.0 - b;
                        volumes.push_back(1 - (a * b) / 2.0); // Whole square but the area of the
                                                              // triangle bounded by (a, 2, b)
                    }
                }
                volumes.push_back(1.0);
                assert(pointsCCW->size() + 1 == volumes.size());
                for(int i = intersectionsCW->size(); i > 0; i--)
                {
                    volumes[i] -= volumes[i - 1];
                }
                // Suppose that in volumes there is one point more than in intersectionsCCW
                for(std::size_t i = 0; i != volumes.size(); i++)
                {
                    std::shared_ptr<Prism> p = std::make_shared<Prism>(volumes[i], pxindex);
                    if(pxindex >= 0 && pxindex < (int)pdimx)
                    {
                        if(i == 0)
                        {
                            p->addPoint(*V_ccw[0]); // minimum
                            p->addPoint(pointsCCW->operator[](i));
                            p->addPoint(pointsCW->operator[](i));
                        } else if(i == volumes.size() - 1)
                        {
                            p->addPoint(pointsCCW->operator[](i - 1));
                            p->addPoint(pointsCW->operator[](i - 1));
                            p->addPoint(*V_max);
                        } else
                        {
                            p->addPoint(pointsCCW->operator[](i - 1));
                            p->addPoint(pointsCW->operator[](i - 1));
                            p->addPoint(pointsCCW->operator[](i));
                            p->addPoint(pointsCW->operator[](i));
                        }
                        insertPrism(p, c, voxelIndex, pm, pixelIndexOffset, sourcePosition,
                                    normalToDetector);
                    }
                    pxindex++;
                }
                // Now find parts of the square that maps into the given index or indices below
                // it
            }
        }

        void insertMatrixProjections(matrix::ProjectionMatrix pm, uint32_t pixelIndexOffset)
        {
            if(!threadpoolstarted)
            {
                startThreadpool();
            }
            std::array<double, 3> sourcePositionD = pm.sourcePosition();
            std::array<double, 3> normalToDetectorD = pm.normalToDetector();
            std::array<float, 3> sourcePosition;
            std::array<float, 3> normalToDetector;
            for(int i = 0; i != 3; i++)
            {
                sourcePosition[i] = sourcePositionD[i];
                normalToDetector[i] = normalToDetectorD[i];
            }

            LOGD << io::xprintf(
                "Source position is [%f, %f, %f] and normal to detector [%f, %f, %f]",
                sourcePosition[0], sourcePosition[1], sourcePosition[2], normalToDetector[0],
                normalToDetector[1], normalToDetector[2]);
            float xcoord, ycoord, zcoord;
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
                            = getProjectionIndex(pm, xcoord, ycoord, zcoord);
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
        std::shared_ptr<matrix::BufferedSparseMatrixFloatWritter> w;
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
        uint64_t totalWritesExact, totalWritesInexact;
        // Square distance from source to detector divided by the area of pixel.
        float scalingFactor;
    }; // namespace util
} // namespace util
} // namespace CTL
