#pragma once

#include "image.hh"
#include <rmm/device_uvector.hpp>

void fix_image_gpu(rmm::device_uvector<int>& d_buffer, const int image_size);