# cuda-clocking
Making it easy to add and analyze clockings in CUDA

## example

First run `pip install tabulate`; then run `./run.sh` and enjoy the pretty readout!

## usage

The only include is `clocker.cuh`

cuda-clocking currently consists of 5 kernel macros and one host struct. First the macros:
1. Add `TIMINGDETAIL_ARGS()` to the parameters of the kernel you want to profile.
2. Run `INITTHREADTIMER();` at the start of the kernel you want to profile. If you need more than 64 breakpoints, then you can pass in the number you need like `INITTHREADTIMER(200);`
3. Run `CLOCKRESET();` whenever you want to restart the running timer.
4. Run `CLOCKPOINT(ID, LABEL);` whenever you want to record elapsed time and restart the timer. You can put it in a loop, and it will count both calls and elapsed time. `ID` should be unique, >=0, and <(max number of breakpoints; default 64). The `LABEL` field is completely ignored by the compiler but is used by the analysis script. A valid example use would be `CLOCKPOINT(3, "global memory write");`
5. Run `FINISHTHREADTIMER();` at the end of the kernel to write results back to global memory.

The host struct `TimingData` should be initialized with the grid and block dimensions of the kernel, and also the requested number of breakpoints if more than 64. (Otherwise that parameter can be omitted). Pass in `timingdata_struct_name.data` as the argument corresponding to `TIMINGDETAIL_ARGS()` when running the kernel. Finally, call `timingdata_struct_name.write("path_to_cuda_file_containing_kernel");` to write an output profile.

You should be able to compile and run your code normally. The timing does use a relatively small number of registers but it is unlikely to interfere with most kernels.

Finally, run `python analysis.py` to see the results!