#pragma once

namespace ANNS {
// Return the first passing index in [1, upper], or 0 if the upper bound fails.
// Requires a monotonic predicate. Adjacent failing/passing endpoints are tested.
template<class Passes>
int calibration_binary_threshold(int upper, Passes passes) {
    if (upper < 1 || !passes(upper)) return 0;
    int low = 0, high = upper;
    while (high - low > 1) {
        int mid = low + (high - low) / 2;
        if (passes(mid)) high = mid;
        else low = mid;
    }
    return high;
}
} // namespace ANNS
