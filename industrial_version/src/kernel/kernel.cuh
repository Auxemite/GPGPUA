#pragma once

#include <rmm/device_uvector.hpp>


void DecoupledLookBack_Scan(rmm::device_uvector<int>& buffer);

void map_classique(rmm::device_uvector<int>& buffer,const int image_size);

void creation_mask(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& mask);

void scatter(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& predicate,rmm::device_uvector<int>& res);

void fill_histo(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& histo);

void Scan_histo(rmm::device_uvector<int>& buffer);

int find_first_value(rmm::device_uvector<int>& buffer);

void last_mapping(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& histo,const int cdf);
