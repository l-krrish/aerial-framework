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

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/power_normalization.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

TEST(PowerNormalizationTests, L2NormalizationMatchesExpectedValues) {
    const std::vector<float> input = {3.0F, 4.0F};

    const auto output = framework::tensor::power_normalize(input);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_NEAR(output[0], 0.6F, 1.0e-6F);
    EXPECT_NEAR(output[1], 0.8F, 1.0e-6F);
}

TEST(PowerNormalizationTests, L1NormalizationUsesAbsoluteSum) {
    const std::vector<float> input = {-2.0F, 1.0F, 1.0F};
    framework::tensor::PowerNormalizationConfig config{};
    config.p = 1.0F;
    config.epsilon = 0.0F;

    const auto output = framework::tensor::power_normalize(input, config);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_NEAR(output[0], -0.5F, 1.0e-6F);
    EXPECT_NEAR(output[1], 0.25F, 1.0e-6F);
    EXPECT_NEAR(output[2], 0.25F, 1.0e-6F);
}

TEST(PowerNormalizationTests, InPlaceOverloadNormalizesInput) {
    std::vector<float> input = {1.0F, 2.0F, 2.0F};

    framework::tensor::power_normalize_in_place(input);

    EXPECT_NEAR(input[0], 1.0F / 3.0F, 1.0e-6F);
    EXPECT_NEAR(input[1], 2.0F / 3.0F, 1.0e-6F);
    EXPECT_NEAR(input[2], 2.0F / 3.0F, 1.0e-6F);
}

TEST(PowerNormalizationTests, ZeroVectorUsesEpsilonForStability) {
    const std::vector<float> input = {0.0F, 0.0F, 0.0F};
    framework::tensor::PowerNormalizationConfig config{};
    config.p = 2.0F;
    config.epsilon = 1.0e-6F;

    const auto output = framework::tensor::power_normalize(input, config);

    ASSERT_EQ(output.size(), input.size());
    EXPECT_FLOAT_EQ(output[0], 0.0F);
    EXPECT_FLOAT_EQ(output[1], 0.0F);
    EXPECT_FLOAT_EQ(output[2], 0.0F);
}

TEST(PowerNormalizationTests, ThrowsForInvalidConfigAndInput) {
    const std::vector<float> input = {1.0F, 2.0F};
    std::vector<float> empty;

    framework::tensor::PowerNormalizationConfig bad_p{};
    bad_p.p = 0.0F;
    bad_p.epsilon = 0.0F;

    framework::tensor::PowerNormalizationConfig bad_eps{};
    bad_eps.p = 2.0F;
    bad_eps.epsilon = -1.0F;

    framework::tensor::PowerNormalizationConfig ok{};
    ok.p = 2.0F;
    ok.epsilon = 0.0F;

    EXPECT_THROW(static_cast<void>(framework::tensor::power_normalize(input, bad_p)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(framework::tensor::power_normalize(input, bad_eps)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(framework::tensor::power_normalize(empty, ok)), std::invalid_argument);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace