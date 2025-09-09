// Minimal driver to reproduce the multi_gather board reset bug
// Exact reproduction of the multi_gather kernel from Spatter
#include <fmt/ostream.h>
#include <cstdint>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

using namespace tt::tt_metal;

int main() {

    // Initialize device
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    
    // Constants matching the kernel
    constexpr uint32_t tile_size_bytes = 2048;  // 32x32 BFloat16
    constexpr uint32_t elements_per_tile = 1024;  // 32x32
    constexpr uint32_t num_tiles = 4;  // Allocate 4 tiles for each buffer
    constexpr uint32_t buffer_size = tile_size_bytes * num_tiles;
    
    // Multi-gather specific parameters
    constexpr uint32_t pattern_length = 10;  // Small pattern length  
    constexpr uint32_t sparse_size = elements_per_tile * num_tiles;
    constexpr uint32_t delta = 1;
    constexpr uint32_t count = 1;
    constexpr uint32_t wrap = 1;
    
    fmt::print("1. Creating DRAM buffers (4 tiles each)...\n");
    
    // Create DRAM buffers for all arrays
    InterleavedBufferConfig dram_config{
        .device = device,
        .size = buffer_size,
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM
    };
    
    auto sparse_buffer = CreateBuffer(dram_config);
    auto dense_buffer = CreateBuffer(dram_config);
    auto pattern_buffer = CreateBuffer(dram_config);
    auto pattern_gather_buffer = CreateBuffer(dram_config);
    
    fmt::print("2. Initializing data arrays...\n");
    
    // Initialize sparse array with test data (as uint16_t for BFloat16)
    std::vector<uint16_t> sparse_data(sparse_size, 0);
    for (uint32_t i = 0; i < sparse_size; i++) {
        sparse_data[i] = i % 256;  // Simple test pattern
    }
    
    // Initialize dense array (output)
    std::vector<uint16_t> dense_data(sparse_size, 0);
    
    // Initialize pattern array with valid indices
    std::vector<uint32_t> pattern_data(sparse_size, 0);
    for (uint32_t i = 0; i < pattern_length; i++) {
        pattern_data[i] = i * 2;  // Valid indices: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
    }
    
    // Initialize pattern_gather with PROBLEMATIC indices
    std::vector<uint32_t> pattern_gather_data(sparse_size, 0);
    
    // Set up indices that will trigger the bug
    pattern_gather_data[0] = 0;       // Valid - within pattern_length
    pattern_gather_data[1] = 5;       // Valid - within pattern_length
    pattern_gather_data[2] = 9;       // Valid - last valid index
    pattern_gather_data[3] = 100;     // INVALID - beyond pattern_length!
    pattern_gather_data[4] = 5000;    // INVALID - way out of bounds!
    pattern_gather_data[5] = 0xDEADBEEF; // INVALID - garbage value!
    
    // Write buffers to device
    EnqueueWriteBuffer(cq, sparse_buffer, sparse_data, false);
    EnqueueWriteBuffer(cq, dense_buffer, dense_data, false);
    EnqueueWriteBuffer(cq, pattern_buffer, pattern_data, false);
    EnqueueWriteBuffer(cq, pattern_gather_buffer, pattern_gather_data, false);
    Finish(cq);
    
    // Create program
    Program program = CreateProgram();
    CoreCoord core = {0, 0};
    
    fmt::print("5. Creating multi_gather kernel on core (0,0)...\n");
    
    // Create circular buffers for L1 - matching kernel's compile-time args
    constexpr uint32_t single_tile_size = tile_size_bytes;
    CircularBufferConfig cb_config = CircularBufferConfig(single_tile_size, {{0, tt::DataFormat::UInt32}})
        .set_page_size(0, single_tile_size);
    
    auto cb_pattern = CreateCircularBuffer(program, core, cb_config);
    auto cb_pattern_gather = CreateCircularBuffer(program, core, cb_config);
    auto cb_sparse = CreateCircularBuffer(program, core, cb_config);
    auto cb_dense = CreateCircularBuffer(program, core, cb_config);
    
    // Create kernel - no compile-time args needed since we hardcoded CB indices
    auto kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/board_reset_issue/kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );
    
    // Set runtime arguments matching the kernel's exact expectations
    const std::vector<uint32_t> runtime_args = {
        sparse_buffer->address(),          // 0: sparse_addr
        dense_buffer->address(),            // 1: dense_addr
        pattern_buffer->address(),          // 2: pattern_addr
        pattern_gather_buffer->address(),   // 3: pattern_gather_addr
        0,                                  // 4: start_element
        6,                                  // 5: end_element (process 6 elements)
        pattern_length,                     // 6: pattern_length
        delta,                              // 7: delta
        count,                              // 8: count
        wrap,                               // 9: wrap
        sparse_size                         // 10: sparse_size_elements
    };
    
    SetRuntimeArgs(program, kernel_id, core, runtime_args);
    
    
    // EXECUTE - This should trigger the board reset!
    EnqueueProgram(cq, program, false);
    Finish(cq);  // This will likely crash
            
    CloseDevice(device);
    
    return 0;
}