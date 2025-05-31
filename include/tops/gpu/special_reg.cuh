#pragma once

#include <cstdint>

namespace tops {
namespace gpu {

static constexpr uint8_t WarpSize = 32;

#define DEFINE_READ_PTX_SREG_32(ptx_reg, func_name, ret_ty)                    \
  __device__ ret_ty func_name() {                                              \
    uint32_t reg;                                                              \
    asm("mov.u32 %0, %%" #ptx_reg ";" : "=r"(reg));                            \
    return reg;                                                                \
  }

#define DEFINE_READ_PTX_SREG_64(ptx_reg, func_name, ret_ty)                    \
  __device__ ret_ty func_name() {                                              \
    uint64_t reg;                                                              \
    asm("mov.u64 %0, %%" #ptx_reg ";" : "=l"(reg));                            \
    return reg;                                                                \
  }

#define DEFINE_READ_PTX_SREG_BOOL(ptx_reg, func_name)                          \
  __device__ bool func_name() {                                                \
    uint16_t reg;                                                              \
    asm("selp.u16 %0, 1, 0, " #ptx_reg ";" : "=h"(reg));                       \
    return reg;                                                                \
  }

DEFINE_READ_PTX_SREG_32(tid.x, get_thread_x_id, uint8_t)
DEFINE_READ_PTX_SREG_32(tid.y, get_thread_y_id, uint8_t)
DEFINE_READ_PTX_SREG_32(tid.z, get_thread_z_id, uint8_t)

DEFINE_READ_PTX_SREG_32(ntid.x, get_n_thread_x_id, uint8_t)
DEFINE_READ_PTX_SREG_32(ntid.y, get_n_thread_y_id, uint8_t)
DEFINE_READ_PTX_SREG_32(ntid.z, get_n_thread_z_id, uint8_t)

DEFINE_READ_PTX_SREG_32(laneid, get_lane_id, uint8_t)

DEFINE_READ_PTX_SREG_32(warpid, get_curr_phys_warp_id, uint32_t)
DEFINE_READ_PTX_SREG_32(nwarpid, get_n_phys_warp_id, uint32_t)

DEFINE_READ_PTX_SREG_32(ctaid.x, get_cta_x_id, uint32_t)
DEFINE_READ_PTX_SREG_32(ctaid.y, get_cta_y_id, uint16_t)
DEFINE_READ_PTX_SREG_32(ctaid.z, get_cta_z_id, uint16_t)

DEFINE_READ_PTX_SREG_32(nctaid.x, get_n_cta_x_id, uint32_t)
DEFINE_READ_PTX_SREG_32(nctaid.y, get_n_cta_y_id, uint16_t)
DEFINE_READ_PTX_SREG_32(nctaid.z, get_n_cta_z_id, uint16_t)

DEFINE_READ_PTX_SREG_32(smid, get_curr_phys_sm_id, uint32_t)
DEFINE_READ_PTX_SREG_32(nsmid, get_n_phys_sm_id, uint32_t)

DEFINE_READ_PTX_SREG_32(gridid, get_grid_id, uint64_t)

DEFINE_READ_PTX_SREG_BOOL(is_explicit_cluster, is_explicit_cluster)

DEFINE_READ_PTX_SREG_32(clusterid.x, get_cluster_x_id, uint32_t)
DEFINE_READ_PTX_SREG_32(clusterid.y, get_cluster_y_id, uint32_t)
DEFINE_READ_PTX_SREG_32(clusterid.z, get_cluster_z_id, uint32_t)

DEFINE_READ_PTX_SREG_32(nclusterid.x, get_n_cluster_x_id, uint32_t)
DEFINE_READ_PTX_SREG_32(nclusterid.y, get_n_cluster_y_id, uint32_t)
DEFINE_READ_PTX_SREG_32(nclusterid.z, get_n_cluster_z_id, uint32_t)

DEFINE_READ_PTX_SREG_32(cluster_ctaid.x, get_cluster_cta_x_id, uint32_t)
DEFINE_READ_PTX_SREG_32(cluster_ctaid.y, get_cluster_cta_y_id, uint32_t)
DEFINE_READ_PTX_SREG_32(cluster_ctaid.z, get_cluster_cta_z_id, uint32_t)

DEFINE_READ_PTX_SREG_32(cluster_nctaid.x, get_cluster_n_cta_x_id, uint32_t)
DEFINE_READ_PTX_SREG_32(cluster_nctaid.y, get_cluster_n_cta_y_id, uint32_t)
DEFINE_READ_PTX_SREG_32(cluster_nctaid.z, get_cluster_n_cta_z_id, uint32_t)

DEFINE_READ_PTX_SREG_32(cluster_ctarank, get_cluster_cta_rank, uint32_t)
DEFINE_READ_PTX_SREG_32(cluster_nctarank, get_cluster_n_cta_rank, uint32_t)

DEFINE_READ_PTX_SREG_32(lanemask_eq, get_lane_mask_eq, uint32_t)
DEFINE_READ_PTX_SREG_32(lanemask_le, get_lane_mask_le, uint32_t)
DEFINE_READ_PTX_SREG_32(lanemask_lt, get_lane_mask_lt, uint32_t)
DEFINE_READ_PTX_SREG_32(lanemask_ge, get_lane_mask_ge, uint32_t)
DEFINE_READ_PTX_SREG_32(lanemask_gt, get_lane_mask_gt, uint32_t)

DEFINE_READ_PTX_SREG_32(clock, get_clock, uint32_t)
DEFINE_READ_PTX_SREG_32(clock_hi, get_clock_hi, uint32_t)
DEFINE_READ_PTX_SREG_64(clock64, get_clock64, uint64_t)

DEFINE_READ_PTX_SREG_64(globaltimer, get_global_timer, uint64_t)
DEFINE_READ_PTX_SREG_32(globaltimer_lo, get_global_timer_low, uint32_t)
DEFINE_READ_PTX_SREG_32(globaltimer_hi, get_global_timer_high, uint32_t)

DEFINE_READ_PTX_SREG_32(reserved_smem_offset_begin,
                        get_reserved_smem_offset_begin, uint32_t)
DEFINE_READ_PTX_SREG_32(reserved_smem_offset_end, get_reserved_smem_offset_end,
                        uint32_t)
DEFINE_READ_PTX_SREG_32(reserved_smem_offset_cap, get_reserved_smem_offset_cap,
                        uint32_t)
DEFINE_READ_PTX_SREG_32(reserved_smem_offset_0, get_reserved_smem_offset_0,
                        uint32_t)
DEFINE_READ_PTX_SREG_32(reserved_smem_offset_1, get_reserved_smem_offset_1,
                        uint32_t)
DEFINE_READ_PTX_SREG_32(total_smem_size, get_total_smem_size, uint32_t)
DEFINE_READ_PTX_SREG_32(aggr_smem_size, get_aggregated_smem_size, uint32_t)
DEFINE_READ_PTX_SREG_32(dynamic_smem_size, get_dynamic_smem_size, uint32_t)

DEFINE_READ_PTX_SREG_64(current_graph_exec, get_current_graph_exec, uint64_t)

#undef DEFINE_READ_PTX_SREG_32
#undef DEFINE_READ_PTX_SREG_64
#undef DEFINE_READ_PTX_SREG_BOOL
} // namespace gpu
} // namespace tops
