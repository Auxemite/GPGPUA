#pragma once

#include <rmm/device_uvector.hpp>

void fix_image_gpu(const int image_size,rmm::device_uvector<int>& to_fix);
