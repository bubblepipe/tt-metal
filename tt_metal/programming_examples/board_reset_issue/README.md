# Multi-Gather Board Reset Issue Reproduction

This reproduces the exact bug from Spatter's `multi_gather_kernel.cpp` where double indirection with out-of-bounds indices causes a board reset or dead loop.

## The Bug

The issue occurs in the kernel at line 72:
```cpp
uint32_t pattern_tile_id = pattern_gather_idx / elements_per_tile;
```

When `pattern_gather_idx` contains values that are out of bounds for the pattern array, it calculates an invalid `pattern_tile_id`, leading to invalid DRAM access.

## Reproduction Command

This code reproduces the exact behavior of:
```bash
./spatter --tt-cores 1 -k multigather -pUNIFORM:8:1 -gUNIFORM:4:1 -l10000 -w 2
```

Where:
- `-pUNIFORM:8:1`: Pattern with stride=8, generates [0, 8, 16, 24, ...]
- `-gUNIFORM:4:1`: Pattern_gather with stride=4, generates [0, 4, 8, 12, ...]  
- `-l10000`: Process 10,000 elements
- `-w 2`: Wrap factor of 2

## The Problem

With these patterns:
- Pattern has 10,000 elements (indices 0-9999)
- Pattern_gather also has 10,000 elements but values go up to 39,996
- When pattern_gather[2500] = 10,000, it tries to access pattern[10000] which is OUT OF BOUNDS
- This causes the kernel to calculate invalid DRAM addresses and crash

## Build and Run

```bash
cd /home/bubblepipe/tt/tt-metal/build_Release
cmake --build . --target board_reset_issue

# Run with debug output
TT_METAL_DPRINT_CORES=0,0 ./programming_examples/board_reset_issue
```

## Expected Behavior

The program should either:
1. Enter a dead loop (hangs indefinitely)
2. Crash with "Read 0xffffffff from PCIE: you should reset the board"

Both behaviors indicate the board has entered an unrecoverable state due to invalid memory access.

## Files

- `kernel.cpp`: Exact copy of Spatter's multi_gather_kernel.cpp
- `host.cpp`: Driver that sets up the exact conditions from the command-line args
- `CMakeLists.txt`: Build configuration