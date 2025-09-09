// Exact reproduction of Spatter's multi_gather kernel execution
// Matches the command: --tt-cores 1 -k multigather -pUNIFORM:8:1 -gUNIFORM:4:1 -l10000 -w 2
#include <fmt/ostream.h>
#include <cstdint>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

using namespace tt::tt_metal;

int main() {
    fmt::print("\n==========================================================\n");
    fmt::print("Multi-Gather Board Reset Bug - Exact Spatter Reproduction\n");
    fmt::print("==========================================================\n\n");
    
    fmt::print("Reproducing: --tt-cores 1 -k multigather -pUNIFORM:8:1 -gUNIFORM:4:1 -l10000 -w 2\n\n");
    
    try {
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
        uint32_t sparse_size = std::max(max_pattern_idx + num_elements * delta, 
                                       static_cast<uint32_t>(100000));  // Ensure large enough
        
        // Calculate buffer sizes in tiles
        uint32_t pattern_tiles = (pattern_length * sizeof(uint32_t) + tile_size_bytes - 1) / tile_size_bytes;
        uint32_t pattern_gather_tiles = (pattern_length * sizeof(uint32_t) + tile_size_bytes - 1) / tile_size_bytes;
        uint32_t sparse_tiles = (sparse_size * sizeof(uint16_t) + tile_size_bytes - 1) / tile_size_bytes;
        uint32_t dense_tiles = (num_elements * wrap * sizeof(uint16_t) + tile_size_bytes - 1) / tile_size_bytes;
        
        fmt::print("1. Pattern generation:\n");
        fmt::print("   - Pattern (UNIFORM:8:1): {} elements, max index = {}\n", pattern_length, max_pattern_idx);
        fmt::print("   - Pattern_gather (UNIFORM:4:1): {} elements, max index = {}\n", 
                   pattern_gather_indices.size(), max_pattern_gather_idx);
        fmt::print("   - Sparse buffer size: {} elements\n", sparse_size);
        fmt::print("   - Dense buffer size: {} elements\n", num_elements * wrap);
        
        // Show problematic double indirection
        fmt::print("\n2. Double indirection will cause issues:\n");
        for (uint32_t i = 0; i < 5 && i < pattern_gather_indices.size(); i++) {
            uint32_t pg_idx = pattern_gather_indices[i];
            fmt::print("   pattern_gather[{}] = {} -> ", i, pg_idx);
            if (pg_idx < pattern_indices.size()) {
                fmt::print("pattern[{}] = {}\n", pg_idx, pattern_indices[pg_idx]);
            } else {
                fmt::print("OUT OF BOUNDS! (pattern has only {} elements)\n", pattern_indices.size());
            }
        }
        
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
        
        InterleavedBufferConfig sparse_dram_config{
            .device = device,
            .size = sparse_tiles * tile_size_bytes,
            .page_size = tile_size_bytes,
            .buffer_type = BufferType::DRAM
        };
        
        InterleavedBufferConfig dense_dram_config{
            .device = device,
            .size = dense_tiles * tile_size_bytes,
            .page_size = tile_size_bytes,
            .buffer_type = BufferType::DRAM
        };
        
        auto pattern_buffer = CreateBuffer(pattern_dram_config);
        auto pattern_gather_buffer = CreateBuffer(pattern_gather_dram_config);
        auto sparse_buffer = CreateBuffer(sparse_dram_config);
        auto dense_buffer = CreateBuffer(dense_dram_config);
        
        fmt::print("\n3. Created DRAM buffers\n");
        
        // Initialize data
        std::vector<uint32_t> pattern_data(pattern_tiles * elements_per_tile, 0);
        std::vector<uint32_t> pattern_gather_data(pattern_gather_tiles * elements_per_tile, 0);
        std::vector<uint16_t> sparse_data(sparse_tiles * elements_per_tile, 0);
        std::vector<uint16_t> dense_data(dense_tiles * elements_per_tile, 0);
        
        // Copy patterns to buffers
        for (size_t i = 0; i < pattern_indices.size(); i++) {
            pattern_data[i] = pattern_indices[i];
        }
        for (size_t i = 0; i < pattern_gather_indices.size(); i++) {
            pattern_gather_data[i] = pattern_gather_indices[i];
        }
        
        // Initialize sparse array with test data
        for (uint32_t i = 0; i < sparse_size && i < sparse_data.size(); i++) {
            sparse_data[i] = i % 256;  // Simple test pattern
        }
        
        // Write buffers to device
        EnqueueWriteBuffer(cq, pattern_buffer, pattern_data, false);
        EnqueueWriteBuffer(cq, pattern_gather_buffer, pattern_gather_data, false);
        EnqueueWriteBuffer(cq, sparse_buffer, sparse_data, false);
        EnqueueWriteBuffer(cq, dense_buffer, dense_data, false);
        Finish(cq);
        
        fmt::print("4. Buffers written to DRAM\n");
        
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
        auto l1_sparse_buffer = CreateBuffer(l1_config);
        auto l1_dense_buffer = CreateBuffer(l1_config);
        
        fmt::print("5. Created L1 buffers\n");
        
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
        
        fmt::print("6. Created multi_gather kernel on core (0,0)\n");
        
        // Set runtime arguments matching kernel expectations
        const std::vector<uint32_t> runtime_args = {
            pattern_buffer->address(),          // 0: pattern_addr
            0,                                  // 1: start_element
            num_elements,                       // 2: end_element (10000 from -l10000)
            pattern_length,                     // 3: pattern_length
         
        };
        
        SetRuntimeArgs(program, kernel_id, core, runtime_args);
        
        fmt::print("\n7. EXECUTING KERNEL (EXPECT DEAD LOOP OR CRASH)...\n");
        fmt::print("   Pattern_gather indices will access out-of-bounds pattern indices\n");
        fmt::print("   Example: pattern_gather[2500] = {} -> pattern[{}] = OUT OF BOUNDS!\n",
                   pattern_gather_indices[2500], pattern_gather_indices[2500]);
        fmt::print("   This causes invalid tile_id calculation and DRAM access\n");
        fmt::print("\n   Enable debug: TT_METAL_DPRINT_CORES=0,0\n");
        fmt::print("   Expected: Dead loop or 'Read 0xffffffff from PCIE'\n\n");
        
        fmt::print("   Launching kernel...\n");
        
        // EXECUTE - This should trigger the issue!
        EnqueueProgram(cq, program, false);
        Finish(cq);  // This will likely hang or crash
        
        fmt::print("   Kernel completed (UNEXPECTED - should have crashed)\n");
        
        CloseDevice(device);
        
    } catch (const std::exception& e) {
        fmt::print("\n!!! CRASH !!!\n");
        fmt::print("Error: {}\n", e.what());
        fmt::print("\nBoard may be in unrecoverable state.\n");
        return 1;
    }
    
    return 0;
}