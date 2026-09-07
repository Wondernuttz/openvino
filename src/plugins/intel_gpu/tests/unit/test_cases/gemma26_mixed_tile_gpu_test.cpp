// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "paged_attention_gpu_test.h"
#include <fstream>
#include <cstdlib>

// Lab-only 26B global attention: actual 16 query / 8 KV heads, head size 512.
// Exercise both sides of Q=32/64 boundaries and cached/current-key boundaries.
class gemma26_mixed_tile_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(gemma26_mixed_tile_test, matches_cpu_reference) {
    auto p = GetParam();
    ASSERT_TRUE(this->pam.has_value());
    std::cout << "GEMMA26_TILE precision=" << p.kv_cache_precision
              << " new=" << p.subsequences[0].num_tokens
              << " past=" << p.subsequences[0].past_len << std::endl;
    auto result = run_gpu_inference(*this->pam, p);
    auto inst = result.network->get_primitive("paged_attention");
    const auto dump = inst->get_impl()->get_kernels_dump_info(*inst->get_impl_params()).get_entries();
    ASSERT_NE(dump.find("sdpa_micro"), std::string::npos) << dump;
    const auto reference = PagedAttentionReference(*this->pam).get_reference(result.key_cache_mem);
    auto output = result.outputs.at("output_data").get_memory();
    double max_error = 0, squared_error = 0;
    {
        cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> actual(output, tests::get_test_stream());
        const auto& expected = std::get<0>(reference);
        ASSERT_EQ(output->count(), expected.size());
        if (const char* dump_path = std::getenv("GEMMA26_OUTPUT_DUMP")) {
            std::ofstream dump(dump_path, std::ios::binary);
            ASSERT_TRUE(dump.good());
            dump.write(reinterpret_cast<const char*>(actual.data()), expected.size() * sizeof(ov::float16));
            ASSERT_TRUE(dump.good());
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_TRUE(std::isfinite(static_cast<float>(actual[i]))) << "Nonfinite at " << i;
            double e = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
            max_error = std::max(max_error, e); squared_error += e * e;
        }
        std::cout << "GEMMA26_ERROR max_abs=" << max_error
                  << " rms=" << std::sqrt(squared_error / expected.size())
                  << " tolerance=" << this->tolerance << std::endl;
    }
    compare(output, nullptr, nullptr, reference);
}
static std::vector<paged_attention_test_params> gemma26_mixed_matrix() {
    std::vector<paged_attention_test_params> result;
    for (auto precision : {ov::element::f16, ov::element::i8, ov::element::u4}) {
        for (auto seq : {SubsequenceDescriptor{31, 34}, SubsequenceDescriptor{32, 34},
                         SubsequenceDescriptor{33, 34}, SubsequenceDescriptor{63, 128},
                         SubsequenceDescriptor{64, 128}, SubsequenceDescriptor{65, 128},
                         SubsequenceDescriptor{129, 255}, SubsequenceDescriptor{17, 16384}}) {
            paged_attention_test_params p{{seq}, 16, 8, 512, 512, 16, 0,
                precision != ov::element::f16, ov::internal::CacheQuantMode::BY_CHANNEL,
                DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2};
            p.kv_cache_precision = precision; result.push_back(p);
        }
    }
    return result;
}
INSTANTIATE_TEST_SUITE_P(local_gemma26_mixed, gemma26_mixed_tile_test,
                        ::testing::ValuesIn(gemma26_mixed_matrix()));
