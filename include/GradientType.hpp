#pragma once

#include <string>

namespace KCT {

/**
 * @brief Enum for gradient types, e.g. central vs. forward difference and derivative type.
 */
enum class GradientType {
    CentralDifference3Point,
    CentralDifference5Point,
    ForwardDifference2Point,
    ForwardDifference3Point,
    ForwardDifference4Point,
    ForwardDifference5Point,
    ForwardDifference6Point,
    ForwardDifference7Point
};

// Free function to convert GradientType to string
inline std::string GradientTypeToString(GradientType type)
{
    switch(type)
    {
        // Sigma=tau=0.7
    case GradientType::CentralDifference3Point:
        return "CentralDifference3Point convergent with sigma=tau=0.7";
        // Sigma=tau=0.35
    case GradientType::CentralDifference5Point:
        return "CentralDifference5Point convergetnt with sigma=tau=0.35";
        // Sigma=tau=0.5
    case GradientType::ForwardDifference2Point:
        return "ForwardDifference2Point convergent with sigma=tau=0.5";
        // Sigma=tau=0.125
    case GradientType::ForwardDifference3Point:
        return "ForwardDifference3Point convergent with sigma=tau=0.125";
    case GradientType::ForwardDifference4Point:
        return "ForwardDifference4Point convergent with sigma=tau=0.125";
    case GradientType::ForwardDifference5Point:
        return "ForwardDifference5Point convergent with sigma=tau=0.125";
    case GradientType::ForwardDifference6Point:
        return "ForwardDifference6Point convergent with sigma=tau=0.125";
    case GradientType::ForwardDifference7Point:
        return "ForwardDifference7Point convergent with sigma=tau=0.125";
    default:
        return "Unknown GradientType";
    }
}

} // namespace KCT
