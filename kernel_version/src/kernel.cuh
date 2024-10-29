#pragma once

#include <rmm/device_uvector.hpp>


void DecoupledLookBack_Scan(rmm::device_uvector<int>& buffer);

void map_modulo(rmm::device_uvector<int>& buffer,const int image_size);
void map_classique(rmm::device_uvector<int>& buffer,const int image_size);

void creation_mask(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& mask);
void scatter(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& predicate);
