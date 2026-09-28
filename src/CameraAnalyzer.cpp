#include "CameraAnalyzer.hpp"

AnalysisResult CameraAnalyzer::analyze(
    const std::vector<Detection>& detections
) const {
    AnalysisResult result;

    result.totalObjects = static_cast<int>(detections.size());

    for (const Detection& detection : detections) {
        result.objectCounts[detection.className]++;
        if (detection.className == "person") {
            result.personCount++;
        }
    }

    result.occupied = result.personCount > 0;
    return result;
}
