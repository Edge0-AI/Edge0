# cmake/strata_ggml_tq_overlay.cmake - MMQ prompt kernels for the Gyro trellis types (TQ2_T, TQK6, TQK7) on an UPSTREAM llama.cpp.
#
# The agentionai llama.cpp fork has these types in its ggml (ggml.h enum, ggml-cuda/tq.cuh, a tile loader, MMQ configs and
# routing, three template instances).  Upstream does not.  This script writes, at configure time, a small OVERLAY of the
# pinned ggml-cuda headers into the build tree: copies of ggml.h, mmq.cuh, mmq-load-tiles.cuh and the mmq-config-*.cuh with
# the trellis types added the way IQ2_XS is handled (same Q8_0_16 tile layout, same dot kernels, same configs - the fork's
# own numbers for them are copies of IQ2_XS's), plus tq.cuh / tq-mmq-tiles.cuh from third_party/ggml-tq and three template
# instances.  Only the strata_mmq target sees the overlay (it comes first on its include path); every other type's
# instances are compiled from the untouched pinned files, and the additions are additive (identity gate: Q2_0, IQ3_XXS).
#
# Every edit is anchored on pinned text; if an anchor is missing (another llama.cpp), the script says so and the trellis
# MMQ stays off (STRATA_TQ_OVERLAY_OK=OFF): the engine then reads trellis prompts through the FP16 path, as before.
#
#   strata_ggml_tq_overlay(<llama.cpp dir> <out dir>)  ->  STRATA_TQ_OVERLAY_OK (parent scope)

function(strata_ggml_tq_overlay LLAMA OUT)
  set(STRATA_TQ_OVERLAY_OK OFF PARENT_SCOPE)
  set(_cu "${LLAMA}/ggml/src/ggml-cuda")
  set(_inc "${LLAMA}/ggml/include")
  set(_tq "${CMAKE_SOURCE_DIR}/third_party/ggml-tq")
  foreach(_f "${_inc}/ggml.h" "${_cu}/mmq.cuh" "${_cu}/mmq-load-tiles.cuh" "${_cu}/common.cuh")
    if(NOT EXISTS "${_f}")
      message(STATUS "trellis MMQ overlay: ${_f} is missing: off")
      return()
    endif()
  endforeach()

  # ---- helpers: read a file with LF line ends; insert text before / after the single occurrence of an anchor ----------
  macro(_tq_read VAR PATH)
    file(READ "${PATH}" ${VAR})
    string(REPLACE "\r\n" "\n" ${VAR} "${${VAR}}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PATH}")
  endmacro()
  macro(_tq_edit VAR WHERE ANCHOR TEXT)   # WHERE = BEFORE | AFTER; the anchor must occur exactly once
    string(FIND "${${VAR}}" "${ANCHOR}" _p)
    if(_p LESS 0)
      set(_tq_fail "anchor not found: ${ANCHOR}")
    else()
      string(LENGTH "${ANCHOR}" _alen)
      math(EXPR _after "${_p} + ${_alen}")
      string(SUBSTRING "${${VAR}}" ${_after} -1 _tail)
      string(FIND "${_tail}" "${ANCHOR}" _p2)
      if(NOT _p2 LESS 0)
        set(_tq_fail "anchor is not unique: ${ANCHOR}")
      else()
        string(SUBSTRING "${${VAR}}" 0 ${_p} _head)
        if("${WHERE}" STREQUAL "AFTER")
          set(${VAR} "${_head}${ANCHOR}${TEXT}${_tail}")
        else()
          set(${VAR} "${_head}${TEXT}${ANCHOR}${_tail}")
        endif()
      endif()
    endif()
  endmacro()
  set(_tq_fail "")

  file(MAKE_DIRECTORY "${OUT}/include" "${OUT}/template-instances")

  # ---- the whole header set of ggml-cuda and ggml/include is copied: quote includes resolve next to the including file,
  # so a header left in the pinned directory would pull the pinned mmq.cuh / ggml.h back in beside the patched ones -------
  file(GLOB _all_cu "${_cu}/*.cuh" "${_cu}/*.h")
  foreach(_f IN LISTS _all_cu)
    get_filename_component(_n "${_f}" NAME)
    configure_file("${_f}" "${OUT}/${_n}" COPYONLY)
  endforeach()
  configure_file("${_cu}/quantize.cu" "${OUT}/quantize.cu" COPYONLY)   # compiled from here: it includes mmq.cuh
  if(IS_DIRECTORY "${_cu}/vendors")
    file(COPY "${_cu}/vendors" DESTINATION "${OUT}")
  endif()
  file(GLOB _all_inc "${_inc}/*.h")
  foreach(_f IN LISTS _all_inc)
    get_filename_component(_n "${_f}" NAME)
    configure_file("${_f}" "${OUT}/include/${_n}" COPYONLY)
  endforeach()

  # ---- ggml.h: the three enumerators (values above GGML_TYPE_COUNT: ggml's own tables never see them) ---------------------
  _tq_read(_h "${_inc}/ggml.h")
  _tq_edit(_h BEFORE "        GGML_TYPE_COUNT   = 43,"
    "        GGML_TYPE_TQ2_T   = 144, // Agention trellis types (Strata overlay; the fork's ids)
        GGML_TYPE_TQK6    = 145,
        GGML_TYPE_TQK7    = 146,
        GGML_TYPE_TQP5    = 147, // Gyro-L pair trellis (the same fork's 147-149)
        GGML_TYPE_TQP6    = 148,
        GGML_TYPE_TQP7    = 149,
")
  file(WRITE "${OUT}/include/ggml.h" "${_h}")

  # ---- mmq.cuh: routing of the trellis types as IQ2_XS ----------------------------------------------------------------
  _tq_read(_m "${_cu}/mmq.cuh")
  # #3217: llama.cpp from b11xxx on gives the MMQ helpers a ggml_prec argument (template <..., ggml_prec prec_src1> for the write-back and
  # MMA dot callbacks, a fourth parameter for ggml_cuda_mmq_get_nthreads / get_I / get_sram_stride).  The pinned commit has neither.  The
  # generated routing and the loader follow whichever signature mmq.cuh has; one that is neither turns the overlay off at configure time
  # (the engine then reads trellis prompts through the FP16 path) instead of failing later inside nvcc.
  set(_ps "")        # ", prec_src1" in the generated callbacks
  set(_pa "")        # ", GGML_PREC_Q8" in the loader's calls (the value upstream's own IQ2_XS loader passes)
  string(FIND "${_m}" "ggml_cuda_mmq_get_nthreads(ggml_type type, int J, bool fallback, ggml_prec prec_src1)" _sig_new)
  string(FIND "${_m}" "ggml_cuda_mmq_get_nthreads(ggml_type type, int J, bool fallback)" _sig_old)
  if(NOT _sig_new LESS 0)
    set(_ps ", prec_src1")
    set(_pa ", GGML_PREC_Q8")
  elseif(_sig_old LESS 0)
    set(_tq_fail "the MMQ helper signatures of this ggml-cuda are not the ones the overlay knows (ggml_cuda_mmq_get_nthreads)")
  endif()
  _tq_edit(_m AFTER "        case GGML_TYPE_IQ2_XXS:\n        case GGML_TYPE_IQ2_XS:\n"
    "        case GGML_TYPE_TQ2_T:\n        case GGML_TYPE_TQK6:\n        case GGML_TYPE_TQK7:\n        case GGML_TYPE_TQP5:\n        case GGML_TYPE_TQP6:\n        case GGML_TYPE_TQP7:\n")
  _tq_edit(_m AFTER "        case GGML_TYPE_IQ2_XS:  return MMQ_DP4A_TXS_Q8_0_16;\n"
    "        case GGML_TYPE_TQ2_T:   return MMQ_DP4A_TXS_Q8_0_16;\n        case GGML_TYPE_TQK6:    return MMQ_DP4A_TXS_Q8_0_16;\n        case GGML_TYPE_TQK7:    return MMQ_DP4A_TXS_Q8_0_16;\n        case GGML_TYPE_TQP5:    return MMQ_DP4A_TXS_Q8_0_16;\n        case GGML_TYPE_TQP6:    return MMQ_DP4A_TXS_Q8_0_16;\n        case GGML_TYPE_TQP7:    return MMQ_DP4A_TXS_Q8_0_16;\n")
  _tq_edit(_m BEFORE "            case GGML_TYPE_IQ3_XXS:\n                return ggml_cuda_mmq_util_funcs(\n                    VDR_IQ3_XXS_Q8_1_MMQ,"
    "            case GGML_TYPE_TQ2_T:\n            case GGML_TYPE_TQK6:\n            case GGML_TYPE_TQK7:\n            case GGML_TYPE_TQP5:\n            case GGML_TYPE_TQP6:\n            case GGML_TYPE_TQP7:\n                return ggml_cuda_mmq_util_funcs(\n                    VDR_IQ2_XS_Q8_1_MMQ,\n                    ggml_cuda_mmq_load_tiles_tq<type, J, fallback>,\n                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,\n                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback${_ps}>);\n")
  _tq_edit(_m BEFORE "        case GGML_TYPE_IQ3_XXS:\n            return ggml_cuda_mmq_util_funcs(\n                -1,\n                ggml_cuda_mmq_load_tiles_iq3_xxs<type, J, fallback>,"
    "        case GGML_TYPE_TQ2_T:\n        case GGML_TYPE_TQK6:\n        case GGML_TYPE_TQK7:\n        case GGML_TYPE_TQP5:\n        case GGML_TYPE_TQP6:\n        case GGML_TYPE_TQP7:\n            return ggml_cuda_mmq_util_funcs(\n                -1,\n                ggml_cuda_mmq_load_tiles_tq<type, J, fallback>,\n                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback${_ps}>,\n                ggml_cuda_mmq_write_back_mma<type, J, fallback${_ps}>);\n")
  _tq_edit(_m AFTER "extern DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);\n"
    "extern DECL_MMQ_CASE(GGML_TYPE_TQ2_T);\nextern DECL_MMQ_CASE(GGML_TYPE_TQK6);\nextern DECL_MMQ_CASE(GGML_TYPE_TQK7);\nextern DECL_MMQ_CASE(GGML_TYPE_TQP5);\nextern DECL_MMQ_CASE(GGML_TYPE_TQP6);\nextern DECL_MMQ_CASE(GGML_TYPE_TQP7);\n")
  file(WRITE "${OUT}/mmq.cuh" "${_m}")

  # ---- mmq-load-tiles.cuh: the loader, at the end -------------------------------------------------------------------
  _tq_read(_l "${_cu}/mmq-load-tiles.cuh")
  file(WRITE "${OUT}/mmq-load-tiles.cuh" "${_l}\n#include \"tq-mmq-tiles.cuh\"\n")
  configure_file("${_tq}/tq.cuh" "${OUT}/tq.cuh" COPYONLY)
  file(READ "${_tq}/tq-mmq-tiles.cuh" _tiles)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_tq}/tq-mmq-tiles.cuh")
  foreach(_fn get_nthreads get_I get_sram_stride)
    string(REPLACE "ggml_cuda_mmq_${_fn}(type, J, fallback)" "ggml_cuda_mmq_${_fn}(type, J, fallback${_pa})" _tiles "${_tiles}")
  endforeach()
  file(WRITE "${OUT}/tq-mmq-tiles.cuh" "${_tiles}")

  # ---- mmq-config-*.cuh: every IQ2_XS CASE line gets one copy per trellis type ---------------------------------------
  file(GLOB _cfgs "${_cu}/mmq-config-*.cuh")
  foreach(_c ${_cfgs})
    get_filename_component(_cn "${_c}" NAME)
    _tq_read(_t "${_c}")
    string(REPLACE ";" "<SEMI>" _t "${_t}")   # the CASE lines hold semicolons; a CMake list would split them
    string(REGEX MATCHALL "[^\n]*CASE\\(GGML_TYPE_IQ2_XS,[^\n]*\n" _lines "${_t}")
    list(REMOVE_DUPLICATES _lines)
    foreach(_ln IN LISTS _lines)
      set(_add "")
      foreach(_ty TQ2_T TQK6 TQK7 TQP5 TQP6 TQP7)
        string(REPLACE "GGML_TYPE_IQ2_XS," "GGML_TYPE_${_ty}," _x "${_ln}")
        string(APPEND _add "${_x}")
      endforeach()
      _tq_edit(_t AFTER "${_ln}" "${_add}")
    endforeach()
    string(REPLACE "<SEMI>" ";" _t "${_t}")
    file(WRITE "${OUT}/${_cn}" "${_t}")
  endforeach()

  # ---- template instances -------------------------------------------------------------------------------------------
  foreach(_ty tq2_t tqk6 tqk7 tqp5 tqp6 tqp7)
    string(TOUPPER "${_ty}" _TY)
    file(WRITE "${OUT}/template-instances/mmq-instance-${_ty}.cu"
         "// Strata trellis overlay (cmake/strata_ggml_tq_overlay.cmake)\n\n#include \"../mmq.cuh\"\n\nDECL_MMQ_CASE(GGML_TYPE_${_TY});\n")
  endforeach()

  if(_tq_fail)
    message(STATUS "trellis MMQ overlay off: ${_tq_fail}")
    return()
  endif()
  set(STRATA_TQ_OVERLAY_OK ON PARENT_SCOPE)
endfunction()
