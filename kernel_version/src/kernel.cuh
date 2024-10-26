#pragma once

#include <rmm/device_uvector.hpp>


void DecoupledLookBack_Scan(rmm::device_uvector<int>& buffer);

void map_modulo(rmm::device_uvector<int>& buffer);
