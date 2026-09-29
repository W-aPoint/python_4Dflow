#include "dfwavelet.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>

#if defined(_WIN32)
#define FLOW4D_EXPORT extern "C" __declspec(dllexport)
#else
#define FLOW4D_EXPORT extern "C"
#endif

namespace {

enum StatusCode {
    kSuccess = 0,
    kInvalidPointer = 1,
    kInvalidDimensions = 2,
    kInvalidMinimumSize = 3,
    kInvalidResolution = 4,
    kInvalidSpins = 5,
    kInvalidRandomShift = 6,
    kInvalidErrorBufferSize = 7,
    kVolumeTooLarge = 8,
    kExecutionError = 100,
};

void write_error(char* buffer, int buffer_size, const char* message) noexcept {
    if (buffer == nullptr || buffer_size <= 0) {
        return;
    }
    std::snprintf(buffer, static_cast<std::size_t>(buffer_size), "%s", message);
    buffer[buffer_size - 1] = '\0';
}

}  // namespace

FLOW4D_EXPORT int flow4d_dfwavelet_sure_mad_spin_3d(
    const double* vx,
    const double* vy,
    const double* vz,
    int dim0,
    int dim1,
    int dim2,
    const int min_size[3],
    const double resolution[3],
    int spins,
    int is_random_shift,
    double* out_vx,
    double* out_vy,
    double* out_vz,
    char* error_buffer,
    int error_buffer_size) {
    if (error_buffer_size < 0) {
        return kInvalidErrorBufferSize;
    }
    if (vx == nullptr || vy == nullptr || vz == nullptr || min_size == nullptr ||
        resolution == nullptr || out_vx == nullptr || out_vy == nullptr ||
        out_vz == nullptr || error_buffer == nullptr) {
        write_error(error_buffer, error_buffer_size, "A required pointer is null.");
        return kInvalidPointer;
    }
    if (error_buffer_size > 0) {
        error_buffer[0] = '\0';
    }

    if (dim0 <= 0 || dim1 <= 0 || dim2 <= 0) {
        write_error(error_buffer, error_buffer_size, "All dimensions must be positive.");
        return kInvalidDimensions;
    }
    const long long plane_size =
        static_cast<long long>(dim0) * static_cast<long long>(dim1);
    if (
        plane_size > std::numeric_limits<int>::max() ||
        plane_size > std::numeric_limits<int>::max() / dim2) {
        write_error(error_buffer, error_buffer_size, "Voxel count exceeds the native core limit.");
        return kVolumeTooLarge;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (min_size[axis] <= 0) {
            write_error(error_buffer, error_buffer_size, "minimum_size values must be positive.");
            return kInvalidMinimumSize;
        }
        if (!std::isfinite(resolution[axis]) || resolution[axis] <= 0.0) {
            write_error(error_buffer, error_buffer_size, "resolution values must be positive and finite.");
            return kInvalidResolution;
        }
    }
    if (spins <= 0) {
        write_error(error_buffer, error_buffer_size, "spins must be positive.");
        return kInvalidSpins;
    }
    if (is_random_shift != 0 && is_random_shift != 1) {
        write_error(error_buffer, error_buffer_size, "is_random_shift must be 0 or 1.");
        return kInvalidRandomShift;
    }

    dfwavelet_plan_s* plan = nullptr;
    try {
        int dimensions[3] = {dim0, dim1, dim2};
        int minimum_size_copy[3] = {min_size[0], min_size[1], min_size[2]};
        double resolution_copy[3] = {
            resolution[0], resolution[1], resolution[2]};
        plan = prepare_dfwavelet_plan(
            3, dimensions, minimum_size_copy, resolution_copy, 0);
        if (plan == nullptr) {
            write_error(error_buffer, error_buffer_size, "Failed to create DFW plan.");
            return kExecutionError;
        }
        dfwavelet_thresh_SURE_MAD_spin(
            plan,
            spins,
            is_random_shift,
            out_vx,
            out_vy,
            out_vz,
            const_cast<double*>(vx),
            const_cast<double*>(vy),
            const_cast<double*>(vz));
        dfwavelet_free(plan);
        return kSuccess;
    } catch (const std::exception& exception) {
        if (plan != nullptr) {
            dfwavelet_free(plan);
        }
        write_error(error_buffer, error_buffer_size, exception.what());
        return kExecutionError;
    } catch (...) {
        if (plan != nullptr) {
            dfwavelet_free(plan);
        }
        write_error(error_buffer, error_buffer_size, "Unknown native DFW execution error.");
        return kExecutionError;
    }
}
