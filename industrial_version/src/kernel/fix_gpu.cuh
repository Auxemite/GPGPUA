#pragma once

#include <rmm/device_uvector.hpp>
#include "../image.hh"

void fix_image_gpu(Image& to_fix,cudaStream_t& stream);
