# Board Reset Issue Reproduction
## Build and Run

```bash
./build_metal.sh --build-programming-examples  

./build/programming_examples/board_reset_issue
```

## Actual Behavior
### First run
The program should enter a dead loop (hangs indefinitely). We can quit the program 
with `Ctrl^C`.

### Follow up runs / execute other kernel 
Crash with "Read 0xffffffff from PCIE: you should reset the board"

oard reset or dead loop.

## Expected Behavior
The board should not be in a state where a system reboot is required to reset. 