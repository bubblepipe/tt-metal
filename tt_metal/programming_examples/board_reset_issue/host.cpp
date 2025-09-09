// Exact reproduction of Spatter's multi_gather kernel execution
// Matches the command: --tt-cores 1 -k multigather -pUNIFORM:8:1 -gUNIFORM:4:1 -l10000 -w 2
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
    
    // Constants matching Spatter kernel
    constexpr uint32_t tile_size_bytes = 2048;  // 32x32 BFloat16
    constexpr uint32_t elements_per_tile = 1024;  // 32x32
    
    // Parameters from command line args
    constexpr uint32_t num_elements = 10000;  // -l10000
    constexpr uint32_t wrap = 2;              // -w 2
    constexpr uint32_t delta = 1;             // from UNIFORM:8:1 (second param)
    constexpr uint32_t count = 1;             // default
    
    // Generate UNIFORM:8:1 pattern (stride=8, delta=1)
    // Pattern will be [0, 8, 16, 24, 32, ...]
    std::vector<uint32_t> pattern_indices;
    for (uint32_t i = 0; i < num_elements; i++) {
        pattern_indices.push_back(i * 8);  // stride = 8
    }
    uint32_t pattern_length = pattern_indices.size();
    
    // Generate UNIFORM:4:1 pattern_gather (stride=4, delta=1)  
    // Pattern_gather will be [0, 4, 8, 12, 16, ...]
    std::vector<uint32_t> pattern_gather_indices;
    for (uint32_t i = 0; i < num_elements; i++) {
        pattern_gather_indices.push_back(i * 4);  // stride = 4
    }
    
    // Calculate required buffer sizes
    uint32_t max_pattern_idx = pattern_indices.back();
    uint32_t max_pattern_gather_idx = pattern_gather_indices.back();
    
    // Calculate buffer sizes in tiles
    uint32_t pattern_tiles = (pattern_length * sizeof(uint32_t) + tile_size_bytes - 1) / tile_size_bytes;
    uint32_t pattern_gather_tiles = (pattern_length * sizeof(uint32_t) + tile_size_bytes - 1) / tile_size_bytes;
    
    // Create DRAM buffers
    InterleavedBufferConfig pattern_dram_config{
        .device = device,
        .size = pattern_tiles * tile_size_bytes,
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM
    };
    
    InterleavedBufferConfig pattern_gather_dram_config{
        .device = device,
        .size = pattern_gather_tiles * tile_size_bytes,
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM
    };
    
    
    auto pattern_buffer = CreateBuffer(pattern_dram_config);
    auto pattern_gather_buffer = CreateBuffer(pattern_gather_dram_config);
   
            
    // Initialize data
    std::vector<uint32_t> pattern_data(pattern_tiles * elements_per_tile, 0);
    std::vector<uint32_t> pattern_gather_data(pattern_gather_tiles * elements_per_tile, 0);
    
    // Copy patterns to buffers
    for (size_t i = 0; i < pattern_indices.size(); i++) {
        pattern_data[i] = pattern_indices[i];
    }
    for (size_t i = 0; i < pattern_gather_indices.size(); i++) {
        pattern_gather_data[i] = pattern_gather_indices[i];
    }
    
    
    // Write buffers to device
    EnqueueWriteBuffer(cq, pattern_buffer, pattern_data, false);
    EnqueueWriteBuffer(cq, pattern_gather_buffer, pattern_gather_data, false);
    Finish(cq);
        
    // Create program
    Program program = CreateProgram();
    CoreCoord core = {0, 0};  // Single core as per --tt-cores 1
    
    // Create L1 buffers (matching Spatter's approach)
    InterleavedBufferConfig l1_config{
        .device = device,
        .size = tile_size_bytes,
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::L1
    };
    
    auto l1_pattern_buffer = CreateBuffer(l1_config);
    auto l1_pattern_gather_buffer = CreateBuffer(l1_config);
            
    // Compile-time arguments for kernel (CB indices)
    std::vector<uint32_t> compile_args = {
        0,  // cb_pattern
        1,  // cb_pattern_gather
    };
    
    // Create kernel with compile-time args
    auto kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/board_reset_issue/kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_args
        }
    );
            
    // Set runtime arguments matching kernel expectations
    const std::vector<uint32_t> runtime_args = {
        pattern_buffer->address(),          // 0: pattern_addr
        0,                                  // 1: start_element
        num_elements,                       // 2: end_element (10000 from -l10000)
        pattern_length,                     // 3: pattern_length
    };
    
    SetRuntimeArgs(program, kernel_id, core, runtime_args);

    EnqueueProgram(cq, program, false);
    Finish(cq);  
            
    CloseDevice(device);
    

    return 0;
}