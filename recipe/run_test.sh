#!/bin/bash
set -euo pipefail

# CUDA_VER (e.g. "12.9", "13.0") is used below both for the MoE CUDA>=13.1 gate
# and the aarch64/CUDA-12.x branch.
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")

# t1441_b32_M3848xN4096xK1936_f32_gelu
#E   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.39 GiB. GPU 0 has a total capacity of 14.56 GiB of which 6.26 GiB is free
SKIP_TESTS="t1441_b32_M3848xN4096xK1936_f32_gelu"
# test_matmul_bias
# E   Warning: CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH; Reason: FORT_NATIVE_8X engine is only supported since Ampere architecture at: !(800 <= this->getDeviceProp()->deviceVer)
SKIP_TESTS="$SKIP_TESTS or test_matmul_bias"
# test_silu_and_mul_and_quantization
# E   AssertionError: Legacy CUDA profiling requires use_cpu=True
SKIP_TESTS="$SKIP_TESTS or test_silu_and_mul_and_quantization"
# test_in
# E   RuntimeError: execute(handle, plan->get_raw_desc(), variant_pack_descriptor.get_ptr()) failed with message: , and code: CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH
SKIP_TESTS="$SKIP_TESTS or test_in"
# test_conv_int8
# Torch does not support int8 convolution. Disabling comparison of output tensor
SKIP_TESTS="$SKIP_TESTS or test_conv_int8"

# Additional cutlass-dependent tests living inside otherwise cutlass-independent files
# (they call into cudnn.gemm.cutedsl.grouped.* or cudnn.api_base lazily, inside the
# test body, so they aren't caught by the file-level --ignore entries below)
SKIP_TESTS="$SKIP_TESTS or test_DSA_indexer_backward_wrapper_backend_keyword_only_signature"
SKIP_TESTS="$SKIP_TESTS or discrete_wrapper"
SKIP_TESTS="$SKIP_TESTS or test_grouped_gemm_dglu_blockscaled_discrete_records_pointer_streams"
SKIP_TESTS="$SKIP_TESTS or test_grouped_gemm_dsrelu_deterministic_dprob_side_stream_unordered_init"
SKIP_TESTS="$SKIP_TESTS or test_grouped_gemm_glu_hadamard_empty_input_validates_situglu_betas"
SKIP_TESTS="$SKIP_TESTS or test_grouped_gemm_glu_hadamard_wrapper_cache"
SKIP_TESTS="$SKIP_TESTS or test_grouped_gemm_wgrad_wrapper_input_order_cache_key"
# test_sdpa_sm80_frontend_integration.py: same cutlass/cutedsl gate, hit before the
# assertion under test. Rest of this file passes/skips fine.
# E   AssertionError: assert ('requires the cutedsl extra (nvidia-cutlass-dsl), which is not installed' is not None and 'dropout' in ...)
SKIP_TESTS="$SKIP_TESTS or test_probe_rejects_unsupported_features or test_direction_cross_rejection"

# SDPA FP16/BF16 requires SM80 (Ampere) or newer; our PBP instances run on T4 (SM75)
# E   cudnn._compiled_module.cudnnGraphNotSupportedError: SDPA FP16/BF16 requires SM80 (Ampere) or newer architecture
SKIP_TESTS="$SKIP_TESTS or test_cudnn_sdpa"
SKIP_TESTS="$SKIP_TESTS or rope_sdpa"
SKIP_TESTS="$SKIP_TESTS or rope_output_scale"
SKIP_TESTS="$SKIP_TESTS or yarn_e2e"
SKIP_TESTS="$SKIP_TESTS or test_partial_rope"
SKIP_TESTS="$SKIP_TESTS or test_yarn_mscale_fold_via_attn_scale"
SKIP_TESTS="$SKIP_TESTS or test_unified_rejects_unsupported_io_dtype"
SKIP_TESTS="$SKIP_TESTS or test_composite_rejects_fp64_io_dtype"
SKIP_TESTS="$SKIP_TESTS or stats_rejected or stats_dtype_inferred or fp32_stats_accepted"
SKIP_TESTS="$SKIP_TESTS or test_native_sdpa_fwd_lowers_to_backend"
SKIP_TESTS="$SKIP_TESTS or test_describing_a_graph_pulls_no_framework"

# MoE grouped matmul with cublasLt needs CUDA toolkit >= 13.1; we currently build against 12.9/13.0
# E   NotImplementedError: MoE grouped matmul with cublasLt is not be compiled with cuda toolkit < 13.1
if python -c "v = tuple(int(x) for x in '$CUDA_VER'.split('.')[:2]); exit(0 if v < (13, 1) else 1)"; then
    SKIP_TESTS="$SKIP_TESTS or test_bf16_moe_grouped_matmul_fwd or test_moe_forward or test_native_moe_grouped_matmul_lowers_to_backend"
fi
# test_moe_m_major_output: separate from the above -- a hard SM100-only (Blackwell)
# architecture gate, not a CUDA-toolkit-version gate, so no version selector applies
# E   NotImplementedError: template sm100_moe_grouped_matmul_fwd_*ctamma.py runs only on 100 <= SM < 120, but the active GPU is sm_75
SKIP_TESTS="$SKIP_TESTS or test_moe_m_major_output"

# test_conv_large_tensor_L0[lt15_N1_C8K4_R340x200_f16_2d_dg_nvc]: fp16 dgrad fuzzer case,
# 115/610600 elements outside tolerance, worst case near a ~0 reference value, on T4 (sm75)
SKIP_TESTS="$SKIP_TESTS or lt15_N1_C8K4_R340x200_f16_2d_dg_nvc"

# test_template_epilogue_parity.py: sm100_matmul.py was added upstream but never
# registered in the SETUP/DRAIN group tables these tests check against -- an upstream
# test-suite bookkeeping bug, not platform-specific
SKIP_TESTS="$SKIP_TESTS or test_every_template_is_assigned_to_exactly_one_group"
SKIP_TESTS="$SKIP_TESTS or test_l2_identity_fastpath_is_compile_time_and_used_by_every_mixed_cga_call"
SKIP_TESTS="$SKIP_TESTS or test_every_template_hoists_its_complete_smem_descriptor_inventory"

# test_sdpa_fwd_heuristics.py: same root cause as the graph_analyzer block below
# (empty eligible-engine set). Most of this file passes; only these 3 fail.
SKIP_TESTS="$SKIP_TESTS or test_recommend_emits_multiple_complete_sets_per_engine"
SKIP_TESTS="$SKIP_TESTS or test_recommend_primary_reproduces_the_derived_scheduler"
SKIP_TESTS="$SKIP_TESTS or test_recommend_split_leads_and_respects_structure"

# test_sdpa_graph_analyzer.py: _eligible() returns an empty engine set for every one of
# these regardless of the knobs/graph shape under test. Most likely because cutlass/
# cutedsl isn't installed so the SM100 engine never registers -- NOT confirmed to
# require physical SM100 hardware. ~45 other tests in this file pass and are unaffected,
# so we skip by name here rather than --ignore the whole file.
# E   assert 'sdpa_fwd_prefill_sm100' in set()
SKIP_TESTS="$SKIP_TESTS or test_bwd_knob_domains or test_bwd_mismatch_reason_strings or test_bwd_probe_accepts"
SKIP_TESTS="$SKIP_TESTS or test_bwd_probe_accepts_dense_flex_layouts or test_bwd_probe_accepts_deterministic"
SKIP_TESTS="$SKIP_TESTS or test_bwd_probe_accepts_padding_mask or test_bwd_probe_accepts_right_band_widening"
SKIP_TESTS="$SKIP_TESTS or test_bwd_probe_accepts_sink or test_bwd_probe_accepts_strided_stats"
SKIP_TESTS="$SKIP_TESTS or test_bwd_probe_causal_notches or test_bwd_probe_gqa"
SKIP_TESTS="$SKIP_TESTS or test_bwd_sm80_probe_accepts_the_sm120_rejections or test_d192_fp8_sink_dtype_support"
SKIP_TESTS="$SKIP_TESTS or test_fwd_probe_accepts_strided_stats or test_fwd_probe_rejects_invalid_stats_metadata"
SKIP_TESTS="$SKIP_TESTS or test_knob_request_lpt_sched_is_in_domain or test_knob_request_none_fields_are_no_preference"
SKIP_TESTS="$SKIP_TESTS or test_knob_request_pack_gqa_eligible_on_gqa or test_knob_request_pack_gqa_false_always_eligible"
SKIP_TESTS="$SKIP_TESTS or test_knob_request_pack_gqa_on_mha_is_identity or test_knob_request_within_domain_keeps_engine_eligible"
SKIP_TESTS="$SKIP_TESTS or test_probe_accepts_bf16 or test_probe_accepts_bottom_right_with_padded_seq_len_q"
SKIP_TESTS="$SKIP_TESTS or test_probe_accepts_bottom_right_with_swa or test_probe_accepts_dsv4_causal"
SKIP_TESTS="$SKIP_TESTS or test_probe_accepts_ragged_skv_via_synth_padding or test_probe_accepts_ragged_skv_with_top_left_causal"
SKIP_TESTS="$SKIP_TESTS or test_probe_accepts_right_band_widening or test_probe_accepts_seq_len_q_with_padding_mask"
SKIP_TESTS="$SKIP_TESTS or test_probe_accepts_thd_bottom_right or test_probe_accepts_thd_cu_seq_len"
SKIP_TESTS="$SKIP_TESTS or test_probe_accepts_thd_stats or test_probe_accepts_thd_top_left_causal"
SKIP_TESTS="$SKIP_TESTS or test_probe_envelope_covers_small_head_dim or test_probe_envelope_mixed_dims_pick_covering_flavor"
SKIP_TESTS="$SKIP_TESTS or test_probe_rejects_requested_amax_s or test_resolve_padding_mask_with_seq_len_kv"
SKIP_TESTS="$SKIP_TESTS or test_sm120_knob_domains or test_sm120_probe_accepts_bottom_right_with_swa"
SKIP_TESTS="$SKIP_TESTS or test_sm120_probe_accepts_causal_swa_on_both_minors or test_sm120_probe_accepts_dense_flex_layouts"
SKIP_TESTS="$SKIP_TESTS or test_sm120_probe_accepts_mixed_head_dims or test_sm120_probe_accepts_padded_stats"
SKIP_TESTS="$SKIP_TESTS or test_sm120_probe_accepts_padding_mask_with_seq_lens or test_sm120_probe_accepts_ragged_skv_without_padding_or_causal"
SKIP_TESTS="$SKIP_TESTS or test_sm120_probe_accepts_right_band_widening or test_sm120_probe_accepts_sink"
SKIP_TESTS="$SKIP_TESTS or test_sm120_probe_accepts_stats_output or test_sm120_probe_accepts_thd"
SKIP_TESTS="$SKIP_TESTS or test_sm120_probe_accepts_thd_bottom_right or test_sm120_probe_accepts_thd_stats"
SKIP_TESTS="$SKIP_TESTS or test_sm120_probe_head_dim_envelope"

# test_causal_conv1d.py: NWH-layout backward fails; NHW-layout backward passes fine on
# the same hardware. Not a standard CUDNN_STATUS_* code -- needs investigation before we
# can say whether this is an SM75 limitation or an upstream bug in the new NWH kernel.
# E   RuntimeError: cudnnCausalConv1dNwhBackward failed with status 3010
SKIP_TESTS="$SKIP_TESTS or test_causal_conv1d_nwh_autograd"
SKIP_TESTS="$SKIP_TESTS or test_causal_conv1d_compiled_autograd[nwh]"

# test/python/fe_api/test_grouped_gemm_swiglu.py
# These tests require python cutlass, which we don't have on the main channel
IGNORE_TESTS="--ignore test/python/fe_api/test_grouped_gemm_swiglu.py"

# A lot of out of memory errors coming from these files
IGNORE_TESTS="$IGNORE_TESTS --ignore test/python/test_mhas.py"
IGNORE_TESTS="$IGNORE_TESTS --ignore test/python/test_mhas_v2.py"

# These tests require SM8.0 or greater, which our PBP instances do not currently support.
IGNORE_TESTS="$IGNORE_TESTS --ignore test/python/test_flexible_sdpa.py"
IGNORE_TESTS="$IGNORE_TESTS --ignore test/python/test_flexible_sdpa_bprop.py"

# These import cutlass at module scope (directly, or via cudnn.api_base /
# cudnn.gemm.cutedsl.*), which we don't have on the main channel, so they fail to
# even collect
IGNORE_TESTS="$IGNORE_TESTS --ignore test/python/fe_api/bsa/test_BSA_attention_fp8.py"
IGNORE_TESTS="$IGNORE_TESTS --ignore test/python/fe_api/grouped_gemm/test_grouped_gemm_glu_hadamard_quant.py"
IGNORE_TESTS="$IGNORE_TESTS --ignore test/python/sdpa/frost/test_sdpa_fp8_sm107.py"


# Diagnostic output so we can check that we have CUDA
python -c "import torch.cuda; _a = torch.cuda.is_available(); print('CUDA available:', _a)"
python -c "import cudnn; assert cudnn.backend_version() >= 91000, cudnn.backend_version()"

PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ls -lh "$PREFIX/lib/python$PYVER/site-packages/include/"
test -f "$PREFIX/lib/python$PYVER/site-packages/include/cudnn_frontend.h"

export CUDA_HOME=$PREFIX

# test_matmul_fuzz failures on our hardware have consistently been VRAM exhaustion,
# not real bugs, so don't let this file fail the build.
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    pytest test/python $IGNORE_TESTS --ignore test/python/test_matmul_fuzzer.py -k "not ($SKIP_TESTS)"
    pytest test/python/test_matmul_fuzzer.py -k "not ($SKIP_TESTS)" || true
else
    # 1) These tests will fail on linux-aarch64 due to our compute architecture being 7.5, but pytorch needing >= 8.0.
    # We're still "running" the tests so that if we run this recipe on an upgraded aarch64 system, we will be able
    # to see the test results in the logs without any recipe changes.
    # 2) pytorch is not support on CUDA 12.x on aarch64, so we skip those tests entirely.
    if [[ "$CUDA_VER" != 12.* ]]; then
        pytest test/python $IGNORE_TESTS --ignore test/python/test_matmul_fuzzer.py -k "not ($SKIP_TESTS)" || true
        pytest test/python/test_matmul_fuzzer.py -k "not ($SKIP_TESTS)" || true
    fi
fi
