#pragma once

#include <map>
#include <string>

struct AnalysisResult {
    int totalObjects = 0;
    int personCount = 0;
    bool occupied = false;

    std::map<std::string, int> objectCounts;
};

