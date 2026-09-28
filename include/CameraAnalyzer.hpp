#pragma once

#include <vector>

#include "Detection.hpp"
#include "AnalysisResult.hpp"

class CameraAnalyzer {
    public:
        AnalysisResult analyze(
            const std::vector<Detection>& detections
            ) const;
};