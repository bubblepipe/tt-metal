// SPDX-FileCopyrightText: 2025 The Spatter Authors
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Runtime arguments
    uint32_t pattern_addr = get_arg_val<uint32_t>(0);
    uint32_t start_element = get_arg_val<uint32_t>(1);
    uint32_t end_element = get_arg_val<uint32_t>(2);
    uint32_t pattern_length = get_arg_val<uint32_t>(3);
  
    // Early exit if no work to do
    if (start_element >= end_element) {
        return;
    }
    
    // Compile-time buffer indices
    constexpr uint32_t cb_pattern = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pattern_gather = get_compile_time_arg_val(1);
    
    constexpr uint32_t tile_size_bytes = 2048;  // 32x32 BFloat16 elements
    constexpr uint32_t elements_per_tile = 1024;  // 32x32
    
    // L1 buffer addresses for caching
    uint32_t pattern_l1_addr = get_write_ptr(cb_pattern);
    uint32_t pattern_gather_l1_addr = get_write_ptr(cb_pattern_gather);
    
    // Cache for tiles to minimize DRAM reads
    uint32_t cached_pattern_tile_id = UINT32_MAX;
    uint32_t cached_pattern_gather_tile_id = UINT32_MAX;
    uint32_t cached_sparse_tile_id = UINT32_MAX;
    uint32_t cached_dense_tile_id = UINT32_MAX;
    
    // Process elements assigned to this coreth
    for (uint32_t elem_idx = start_element; elem_idx < end_element; elem_idx++) {
        uint32_t j = elem_idx % pattern_length;
        uint32_t i = elem_idx / pattern_length;
        
        
        // Get pattern_gather[j] value
        uint32_t* pattern_gather_data = reinterpret_cast<uint32_t*>(pattern_gather_l1_addr);
        uint32_t pattern_gather_idx = pattern_gather_data[j % elements_per_tile];
        
        // Load pattern tile if needed
        // problematic code starts here: 
        uint32_t pattern_tile_id = pattern_gather_idx / elements_per_tile;
        if (pattern_tile_id != cached_pattern_tile_id) {
            uint32_t pattern_tile_addr = pattern_addr + pattern_tile_id * tile_size_bytes;
            noc_async_read(get_noc_addr(pattern_tile_addr), pattern_l1_addr, tile_size_bytes);
            noc_async_read_barrier();
            cached_pattern_tile_id = pattern_tile_id;
        }
    }
}