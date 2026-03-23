/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "tensor/power_normalization.hpp"
#include "utils/error_macros.hpp"

namespace framework::tensor {

namespace {

[[nodiscard]] float compute_denominator(
        std::span<const float> input, const PowerNormalizationConfig &config) {
    FRAMEWORK_NV_THROW_IF(input.empty(), std::invalid_argument, "Power normalization input is empty");
    FRAMEWORK_NV_THROW_IF(config.p <= 0.0F, std::invalid_argument, "Power p must be strictly positive");
    FRAMEWORK_NV_THROW_IF(config.epsilon < 0.0F, std::invalid_argument, "Epsilon must be non-negative");

    float lp_power_sum = 0.0F;
    for (const float value : input) {
        lp_power_sum += std::pow(std::abs(value), config.p);
    }

    return std::pow(lp_power_sum + config.epsilon, 1.0F / config.p);
}

} // namespace

std::vector<float> power_normalize(
        std::span<const float> input, const PowerNormalizationConfig &config) {
    const float denominator = compute_denominator(input, config);

    std::vector<float> output(input.begin(), input.end());
    std::ranges::transform(output, output.begin(), [denominator](float value) {
        return value / denominator;
    });

    return output;
}

void power_normalize_in_place(std::span<float> input, const PowerNormalizationConfig &config) {
    const float denominator = compute_denominator(input, config);
    std::ranges::transform(input, input.begin(), [denominator](float value) {
        return value / denominator;
    });
}

} // namespace framework::tensor

