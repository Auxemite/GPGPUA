#pragma once

#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

void my_reduce(raft::device_span<const T> buffer, raft::device_span<T> total)