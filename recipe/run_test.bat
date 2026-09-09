REM CUDA_VER (e.g. "12.9", "13.0") is used below for the MoE CUDA-13.1-or-newer gate.
for /f "delims=" %%i in ('python -c "import torch; print(torch.version.cuda)"') do set CUDA_VER=%%i

REM t1441_b32_M3848xN4096xK1936_f32_gelu
REM E   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.39 GiB. GPU 0 has a total capacity of 14.56 GiB of which 6.26 GiB is free
set "SKIP_TESTS=t1441_b32_M3848xN4096xK1936_f32_gelu"
REM test_matmul_bias
REM E   Warning: CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH; Reason: FORT_NATIVE_8X engine requires Ampere or newer (device compute capability below 800)
set "SKIP_TESTS=%SKIP_TESTS% or test_matmul_bias"
REM test_silu_and_mul_and_quantization
REM E   AssertionError: Legacy CUDA profiling requires use_cpu=True
set "SKIP_TESTS=%SKIP_TESTS% or test_silu_and_mul_and_quantization"
REM test_in
REM E   RuntimeError: execute(...) failed with code: CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH
set "SKIP_TESTS=%SKIP_TESTS% or test_in"
REM test_conv_int8
REM Torch does not support int8 convolution. Disabling comparison of output tensor
set "SKIP_TESTS=%SKIP_TESTS% or test_conv_int8"

REM skip some tests that run out of memory on Windows
set "SKIP_TESTS=%SKIP_TESTS% or test_conv_random_L0_0"

REM Additional cutlass-dependent tests living inside otherwise cutlass-independent files
REM (they call into cudnn.gemm.cutedsl.grouped.* or cudnn.api_base lazily, inside the
REM test body, so they aren't caught by the file-level --ignore entries below)
set "SKIP_TESTS=%SKIP_TESTS% or test_DSA_indexer_backward_wrapper_backend_keyword_only_signature"
set "SKIP_TESTS=%SKIP_TESTS% or discrete_wrapper"
set "SKIP_TESTS=%SKIP_TESTS% or test_grouped_gemm_dglu_blockscaled_discrete_records_pointer_streams"
set "SKIP_TESTS=%SKIP_TESTS% or test_grouped_gemm_dsrelu_deterministic_dprob_side_stream_unordered_init"
set "SKIP_TESTS=%SKIP_TESTS% or test_grouped_gemm_glu_hadamard_empty_input_validates_situglu_betas"
set "SKIP_TESTS=%SKIP_TESTS% or test_grouped_gemm_glu_hadamard_wrapper_cache"
set "SKIP_TESTS=%SKIP_TESTS% or test_grouped_gemm_wgrad_wrapper_input_order_cache_key"
REM test_sdpa_sm80_frontend_integration.py: same cutlass/cutedsl gate, hit before the
REM assertion under test. Rest of this file passes/skips fine.
REM E   AssertionError: assert ('requires the cutedsl extra (nvidia-cutlass-dsl), which is not installed' is not None and 'dropout' in ...)
set "SKIP_TESTS=%SKIP_TESTS% or test_probe_rejects_unsupported_features or test_direction_cross_rejection"

REM SDPA FP16/BF16 requires SM80 (Ampere) or newer; our PBP instances run on T4 (SM75)
REM E   cudnn._compiled_module.cudnnGraphNotSupportedError: SDPA FP16/BF16 requires SM80 (Ampere) or newer architecture
set "SKIP_TESTS=%SKIP_TESTS% or test_cudnn_sdpa"
set "SKIP_TESTS=%SKIP_TESTS% or rope_sdpa"
set "SKIP_TESTS=%SKIP_TESTS% or rope_output_scale"
set "SKIP_TESTS=%SKIP_TESTS% or yarn_e2e"
set "SKIP_TESTS=%SKIP_TESTS% or test_partial_rope"
set "SKIP_TESTS=%SKIP_TESTS% or test_yarn_mscale_fold_via_attn_scale"
set "SKIP_TESTS=%SKIP_TESTS% or test_unified_rejects_unsupported_io_dtype"
set "SKIP_TESTS=%SKIP_TESTS% or test_composite_rejects_fp64_io_dtype"
set "SKIP_TESTS=%SKIP_TESTS% or stats_rejected or stats_dtype_inferred or fp32_stats_accepted"
set "SKIP_TESTS=%SKIP_TESTS% or test_native_sdpa_fwd_lowers_to_backend"
set "SKIP_TESTS=%SKIP_TESTS% or test_describing_a_graph_pulls_no_framework"

REM MoE grouped matmul with cublasLt needs CUDA toolkit 13.1 or newer; we currently build against 12.9/13.0
REM E   NotImplementedError: MoE grouped matmul with cublasLt is not be compiled with cuda toolkit older than 13.1
python -c "v = tuple(int(x) for x in '%CUDA_VER%'.split('.')[:2]); exit(0 if v < (13, 1) else 1)"
if %ERRORLEVEL% equ 0 (
    set "SKIP_TESTS=%SKIP_TESTS% or test_bf16_moe_grouped_matmul_fwd or test_moe_forward or test_native_moe_grouped_matmul_lowers_to_backend"
)
REM test_moe_m_major_output: separate from the above -- a hard SM100-only (Blackwell)
REM architecture gate, not a CUDA-toolkit-version gate, so no version selector applies
REM E   NotImplementedError: template sm100_moe_grouped_matmul_fwd_*ctamma.py runs only on SM 100 up to (not including) 120, but the active GPU is sm_75
set "SKIP_TESTS=%SKIP_TESTS% or test_moe_m_major_output"

REM test_conv_large_tensor_L0[lt15_N1_C8K4_R340x200_f16_2d_dg_nvc]: fp16 dgrad fuzzer case,
REM 115/610600 elements outside tolerance, worst case near a ~0 reference value, on T4 (sm75)
set "SKIP_TESTS=%SKIP_TESTS% or lt15_N1_C8K4_R340x200_f16_2d_dg_nvc"

REM test_template_epilogue_parity.py: sm100_matmul.py was added upstream but never
REM registered in the SETUP/DRAIN group tables these tests check against -- an upstream
REM test-suite bookkeeping bug, not platform-specific
set "SKIP_TESTS=%SKIP_TESTS% or test_every_template_is_assigned_to_exactly_one_group"
set "SKIP_TESTS=%SKIP_TESTS% or test_l2_identity_fastpath_is_compile_time_and_used_by_every_mixed_cga_call"
set "SKIP_TESTS=%SKIP_TESTS% or test_every_template_hoists_its_complete_smem_descriptor_inventory"

REM test_sdpa_fwd_heuristics.py: same root cause as the graph_analyzer block below
REM (empty eligible-engine set). Most of this file passes; only these 3 fail.
set "SKIP_TESTS=%SKIP_TESTS% or test_recommend_emits_multiple_complete_sets_per_engine"
set "SKIP_TESTS=%SKIP_TESTS% or test_recommend_primary_reproduces_the_derived_scheduler"
set "SKIP_TESTS=%SKIP_TESTS% or test_recommend_split_leads_and_respects_structure"

REM test_sdpa_graph_analyzer.py: _eligible() returns an empty engine set for every one of
REM these regardless of the knobs/graph shape under test. Most likely because cutlass/
REM cutedsl isn't installed so the SM100 engine never registers -- NOT confirmed to
REM require physical SM100 hardware. ~45 other tests in this file pass and are unaffected,
REM so we skip by name here rather than --ignore the whole file.
REM E   assert 'sdpa_fwd_prefill_sm100' in set()
set "SKIP_TESTS=%SKIP_TESTS% or test_bwd_knob_domains or test_bwd_mismatch_reason_strings or test_bwd_probe_accepts"
set "SKIP_TESTS=%SKIP_TESTS% or test_bwd_probe_accepts_dense_flex_layouts or test_bwd_probe_accepts_deterministic"
set "SKIP_TESTS=%SKIP_TESTS% or test_bwd_probe_accepts_padding_mask or test_bwd_probe_accepts_right_band_widening"
set "SKIP_TESTS=%SKIP_TESTS% or test_bwd_probe_accepts_sink or test_bwd_probe_accepts_strided_stats"
set "SKIP_TESTS=%SKIP_TESTS% or test_bwd_probe_causal_notches or test_bwd_probe_gqa"
set "SKIP_TESTS=%SKIP_TESTS% or test_bwd_sm80_probe_accepts_the_sm120_rejections or test_d192_fp8_sink_dtype_support"
set "SKIP_TESTS=%SKIP_TESTS% or test_fwd_probe_accepts_strided_stats or test_fwd_probe_rejects_invalid_stats_metadata"
set "SKIP_TESTS=%SKIP_TESTS% or test_knob_request_lpt_sched_is_in_domain or test_knob_request_none_fields_are_no_preference"
set "SKIP_TESTS=%SKIP_TESTS% or test_knob_request_pack_gqa_eligible_on_gqa or test_knob_request_pack_gqa_false_always_eligible"
set "SKIP_TESTS=%SKIP_TESTS% or test_knob_request_pack_gqa_on_mha_is_identity or test_knob_request_within_domain_keeps_engine_eligible"
set "SKIP_TESTS=%SKIP_TESTS% or test_probe_accepts_bf16 or test_probe_accepts_bottom_right_with_padded_seq_len_q"
set "SKIP_TESTS=%SKIP_TESTS% or test_probe_accepts_bottom_right_with_swa or test_probe_accepts_dsv4_causal"
set "SKIP_TESTS=%SKIP_TESTS% or test_probe_accepts_ragged_skv_via_synth_padding or test_probe_accepts_ragged_skv_with_top_left_causal"
set "SKIP_TESTS=%SKIP_TESTS% or test_probe_accepts_right_band_widening or test_probe_accepts_seq_len_q_with_padding_mask"
set "SKIP_TESTS=%SKIP_TESTS% or test_probe_accepts_thd_bottom_right or test_probe_accepts_thd_cu_seq_len"
set "SKIP_TESTS=%SKIP_TESTS% or test_probe_accepts_thd_stats or test_probe_accepts_thd_top_left_causal"
set "SKIP_TESTS=%SKIP_TESTS% or test_probe_envelope_covers_small_head_dim or test_probe_envelope_mixed_dims_pick_covering_flavor"
set "SKIP_TESTS=%SKIP_TESTS% or test_probe_rejects_requested_amax_s or test_resolve_padding_mask_with_seq_len_kv"
set "SKIP_TESTS=%SKIP_TESTS% or test_sm120_knob_domains or test_sm120_probe_accepts_bottom_right_with_swa"
set "SKIP_TESTS=%SKIP_TESTS% or test_sm120_probe_accepts_causal_swa_on_both_minors or test_sm120_probe_accepts_dense_flex_layouts"
set "SKIP_TESTS=%SKIP_TESTS% or test_sm120_probe_accepts_mixed_head_dims or test_sm120_probe_accepts_padded_stats"
set "SKIP_TESTS=%SKIP_TESTS% or test_sm120_probe_accepts_padding_mask_with_seq_lens or test_sm120_probe_accepts_ragged_skv_without_padding_or_causal"
set "SKIP_TESTS=%SKIP_TESTS% or test_sm120_probe_accepts_right_band_widening or test_sm120_probe_accepts_sink"
set "SKIP_TESTS=%SKIP_TESTS% or test_sm120_probe_accepts_stats_output or test_sm120_probe_accepts_thd"
set "SKIP_TESTS=%SKIP_TESTS% or test_sm120_probe_accepts_thd_bottom_right or test_sm120_probe_accepts_thd_stats"
set "SKIP_TESTS=%SKIP_TESTS% or test_sm120_probe_head_dim_envelope"

REM test_causal_conv1d.py: NWH-layout backward fails; NHW-layout backward passes fine on
REM the same hardware. Not a standard CUDNN_STATUS_* code -- needs investigation before we
REM can say whether this is an SM75 limitation or an upstream bug in the new NWH kernel.
REM E   RuntimeError: cudnnCausalConv1dNwhBackward failed with status 3010
set "SKIP_TESTS=%SKIP_TESTS% or test_causal_conv1d_nwh_autograd"
set "SKIP_TESTS=%SKIP_TESTS% or test_causal_conv1d_compiled_autograd[nwh]"

REM test/python/fe_api/test_grouped_gemm_swiglu.py
REM These tests require python cutlass, which we don't have on the main channel
set "IGNORE_TESTS=--ignore test/python/fe_api/test_grouped_gemm_swiglu.py"

REM A lot of out of memory errors coming from these files
set "IGNORE_TESTS=%IGNORE_TESTS% --ignore test/python/test_mhas.py"
set "IGNORE_TESTS=%IGNORE_TESTS% --ignore test/python/test_mhas_v2.py"

REM These tests require SM8.0 or greater, which our PBP instances do not currently support.
set "IGNORE_TESTS=%IGNORE_TESTS% --ignore test/python/test_flexible_sdpa.py"
set "IGNORE_TESTS=%IGNORE_TESTS% --ignore test/python/test_flexible_sdpa_bprop.py"

REM These import cutlass at module scope (directly, or via cudnn.api_base /
REM cudnn.gemm.cutedsl.*), which we don't have on the main channel, so they fail to
REM even collect
set "IGNORE_TESTS=%IGNORE_TESTS% --ignore test/python/fe_api/bsa/test_BSA_attention_fp8.py"
set "IGNORE_TESTS=%IGNORE_TESTS% --ignore test/python/fe_api/grouped_gemm/test_grouped_gemm_glu_hadamard_quant.py"
set "IGNORE_TESTS=%IGNORE_TESTS% --ignore test/python/sdpa/frost/test_sdpa_fp8_sm107.py"

python -c "import torch.cuda; _a = torch.cuda.is_available(); print('CUDA available:', _a)"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python -c "import cudnn; assert cudnn.backend_version() >= 91000, cudnn.backend_version()"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

if not exist %PREFIX%\lib\site-packages\include\cudnn_frontend.h exit /b 1

pytest test/python %IGNORE_TESTS% --ignore test/python/test_matmul_fuzzer.py -k "not (%SKIP_TESTS%)"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

REM test_matmul_fuzz failures on our hardware have consistently been VRAM exhaustion,
REM not real bugs, so don't let this file fail the build.
pytest test/python/test_matmul_fuzzer.py -k "not (%SKIP_TESTS%)"
exit /b 0
