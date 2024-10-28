#include "fix_gpu.cuh"
#include "image.hh"
#include "kernel.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

void fix_image_gpu(const int image_size,rmm::device_uvector<int>& to_fix)
{

    // #1 Compact

    // Build predicate vector

    rmm::device_uvector predicate(to_fix.size()+1, to_fix.stream()); 
    creation_mask(to_fix,predicate);

    // Compute the exclusive sum of the predicate

    DecoupledLookBack_Scan(predicate);

    // Scatter to the corresponding addresses


    scatter(to_fix,predicate);

    // #2 Apply map to fix pixels

    for (int i = 0; i < image_size; ++i)
    {
        if (i % 4 == 0)
            to_fix.buffer[i] += 1;
        else if (i % 4 == 1)
            to_fix.buffer[i] -= 5;
        else if (i % 4 == 2)
            to_fix.buffer[i] += 3;
        else if (i % 4 == 3)
            to_fix.buffer[i] -= 8;
    }

    // #3 Histogram equalization

    // Histogram

    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
        ++histo[to_fix.buffer[i]];

    // Compute the inclusive sum scan of the histogram

    std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    // Find the first non-zero value in the cumulative histogram

    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero;

    // Apply the map transformation of the histogram equalization

    std::transform(to_fix.buffer, to_fix.buffer + image_size, to_fix.buffer,
        [image_size, cdf_min, &histo](int pixel)
            {
                return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
            }
    );
}
