#include "NDRange/CBCTLocalNDRangeFactory.hpp"

namespace KCT {

cl::NDRange CBCTLocalNDRangeFactory::getProjectorLocalNDRange(cl::NDRange lr, bool verbose) const {}
cl::NDRange CBCTLocalNDRangeFactory::getProjectorBarrierLocalNDRange(cl::NDRange lr,
                                                                     bool verbose) const
{
}
cl::NDRange CBCTLocalNDRangeFactory::getBackprojectorLocalNDRange(cl::NDRange lr,
                                                                  bool verbose) const
{
}
bool CBCTLocalNDRangeFactory::isLocalRangeAdmissible(cl::NDRange& localRange) const {}
void CBCTLocalNDRangeFactory::checkLocalRange(cl::NDRange& localRange, std::string name) const {}
cl::NDRange CBCTLocalNDRangeFactory::guessProjectorLocalNDRange(bool barrierCalls) const {}
cl::NDRange CBCTLocalNDRangeFactory::guessBackprojectorLocalNDRange() const {}

} // namespace KCT
