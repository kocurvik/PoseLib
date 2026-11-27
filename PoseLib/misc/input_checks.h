// Copyright (c) 2026, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_MISC_INPUT_CHECKS_H_
#define POSELIB_MISC_INPUT_CHECKS_H_

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

// Input validation for the public estimators and refinement functions.
//
// Every public function that takes several containers describing the same set of correspondences should
// check them on entry, before any of them is indexed. Typical use, as the first lines of the function:
//
//     POSELIB_CHECK_SAME_SIZE(points2D, points3D);
//     POSELIB_CHECK_SIZE_OR_EMPTY(weights, points2D);
//     POSELIB_CHECK_INDEX(match.cam_id1, cameras1);
//
// A failed check throws std::invalid_argument (a ValueError in Python) with a message naming the function,
// the arguments and their sizes, e.g.
//
//     estimate_homography: x1, x2 must have the same size (got 9, 4)
//
// The checks only compare .size() of the arguments, so they work with any container (std::vector, Eigen, ...).

#define POSELIB_CHECK_SAME_SIZE(...) ::poselib::detail::check_same_size(__func__, #__VA_ARGS__, __VA_ARGS__)

// Optional per-correspondence inputs (e.g. weights) may be empty, otherwise they must match the reference.
#define POSELIB_CHECK_SIZE_OR_EMPTY(optional, reference)                                                               \
    ::poselib::detail::check_size_or_empty(__func__, #optional, #reference, optional, reference)

// Indices stored in the inputs (e.g. PairwiseMatches::cam_id1) must be valid positions in the container they index.
#define POSELIB_CHECK_INDEX(idx, container) ::poselib::detail::check_index(__func__, #idx, #container, idx, container)

namespace poselib {
namespace detail {

template <typename... Ts> void check_same_size(const char *func, const char *names, const Ts &...args) {
    static_assert(sizeof...(Ts) >= 2, "POSELIB_CHECK_SAME_SIZE needs at least two arguments");
    const size_t sizes[] = {static_cast<size_t>(args.size())...};
    for (size_t s : sizes) {
        if (s != sizes[0]) {
            std::ostringstream msg;
            msg << func << ": " << names << " must have the same size (got ";
            for (size_t k = 0; k < sizeof...(Ts); ++k) {
                msg << (k > 0 ? ", " : "") << sizes[k];
            }
            msg << ")";
            throw std::invalid_argument(msg.str());
        }
    }
}

template <typename T, typename Ref>
void check_size_or_empty(const char *func, const char *name, const char *ref_name, const T &optional,
                         const Ref &reference) {
    const size_t n = static_cast<size_t>(optional.size());
    const size_t n_ref = static_cast<size_t>(reference.size());
    if (n != 0 && n != n_ref) {
        std::ostringstream msg;
        msg << func << ": " << name << " must be empty or have the same size as " << ref_name << " (got " << n << ", "
            << n_ref << ")";
        throw std::invalid_argument(msg.str());
    }
}

template <typename Container>
void check_index(const char *func, const char *idx_name, const char *container_name, size_t idx,
                 const Container &container) {
    const size_t n = static_cast<size_t>(container.size());
    if (idx >= n) {
        std::ostringstream msg;
        msg << func << ": " << idx_name << " (" << idx << ") is out of range for " << container_name << " (size " << n
            << ")";
        throw std::invalid_argument(msg.str());
    }
}

} // namespace detail
} // namespace poselib

#endif
