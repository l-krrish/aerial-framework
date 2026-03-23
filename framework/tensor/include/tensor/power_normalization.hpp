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

#ifndef FRAMEWORK_TENSOR_POWER_NORMALIZATION_HPP
#define FRAMEWORK_TENSOR_POWER_NORMALIZATION_HPP

#include <span>
#include <vector>

namespace framework::tensor {

/**
 * @brief Configuration for Lp power normalization.
 */
struct PowerNormalizationConfig {
    float p{2.0F};          //!< Positive power for the Lp norm
    float epsilon{1.0e-12F}; //!< Stability term to avoid divide-by-zero
};

/**
 * @brief Compute Lp power normalization of the input values.
 *
 * For input vector x and power p:
 * y_i = x_i / (sum_j |x_j|^p + epsilon)^(1/p)
 *
 * @param[in] input Input values to normalize
 * @param[in] config Normalization parameters
 * @return Normalized values with the same shape as input
 *
 * @throws std::invalid_argument if input is empty, p <= 0, or epsilon < 0
 */
[[nodiscard]] std::vector<float>
power_normalize(std::span<const float> input, const PowerNormalizationConfig &config = {});

/**
 * @brief In-place overload for Lp power normalization.
 *
 * @param[in,out] input Input values to normalize in place
 * @param[in] config Normalization parameters
 *
 * @throws std::invalid_argument if input is empty, p <= 0, or epsilon < 0
 */
void power_normalize_in_place(std::span<float> input, const PowerNormalizationConfig &config = {});

} // namespace framework::tensor

#endif // FRAMEWORK_TENSOR_POWER_NORMALIZATION_HPP