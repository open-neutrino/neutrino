"""Neutrino Probing Engine, CUDA Implementation"""

import os
import sys
import shutil
import subprocess
import traceback # usef for print backtrace to log file instead of stdout
import toml      # to load probes from envariables
from neutrino.common import Register, Probe, Map, load
from neutrino.probe import Ref, KernelParam

workdir = sys.argv[1]     # directory contains original.bin
log = open(os.path.join(workdir, "process.log"), 'w')

# a macro like terms
SUPPORTED_DATAMODEL = { "thread": 0, "warp": 1 }

# TODO move it to global variable or configurable
def get_arch() -> str:
    """get compute_arch of the gpu, like 'sm_89'
    # Run nvidia-smi command
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        stdout=subprocess.PIPE, 
        text=True)
    # sm_version like `8.9`
    sm_version = result.stdout.split("\n")[0].strip()
    major, minor = sm_version.split(".")
    """
    # NOTE sometimes auto-detection like above will fail, so manually fix at the time
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        stdout=subprocess.PIPE, 
        text=True)
    # sm_version like `8.9`
    sm_version = result.stdout.split("\n")[0].strip()
    major, minor = sm_version.split(".")
    return f"sm_{major}{minor}" 

def dump(workdir: str, name: str = "original", suffix: str = ".bin") -> str:
    """Extract PTX from cuda binaries (cubin or fatbin) via cuobjdump

    NOTE accept three kind of binary:
    1. fatbin @see https://docs.nvidia.com/cuda/nvfatbin/index.html
    2. cubin  @see https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
    3. PTX text file - no need to further process, just rename it to .ptx
    """
    bin_path = os.path.join(workdir, name) + suffix
    # first check if it's already a NULL-Terminated PTX (i.e., ASCII Text)
    result = subprocess.run(['file', bin_path], stdout=subprocess.PIPE, text=True)
    out = result.stdout
    if "ASCII text" in result.stdout: # raw PTX file, just read it all
        shutil.copyfile(bin_path, os.path.join(workdir, name) + ".ptx")
        print("[objdump] bin is ptx", file=log)
        with open(os.path.join(workdir, name) + ".ptx", "r") as outf:
            return outf.read()
    # then try cuobjdump -ptx flag
    result = subprocess.run(
        ['cuobjdump', '-ptx', bin_path], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True)
    out = result.stdout
    if len(result.stderr) > 0:
        print(result.stderr, file=log)
    if out.find(".version") != -1:
        start = out.index(".version") # ptx valid part starts with .version
        with open(os.path.join(workdir, name) + ".ptx", "w") as outf:
            outf.write(out[start:])
        print("[objdump] via cuobjdump -ptx", file=log)
        return out[start:]
    else:
    # finally try cuobjdump -elf to dump elf content and check .nv_debug_ptx_txt
        result = subprocess.run(['cuobjdump', '-elf', bin_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if len(result.stderr) > 0:
            print(result.stderr, file=log)
        sections = result.stdout.split(".section ") # don't remove the space
        for section in sections:
            if section.startswith(".nv_debug_ptx_txt"): # PTX Info      
                # write to the original.ptx
                start = section.index(".version")
                with open(os.path.join(workdir, name) + ".ptx", "w") as outf:
                    outf.write(section[start:])
                print("[objdump] via cuobjdump -elf", file=log)
                return section[start:]
        # if still not found
        raise ValueError("PTX Not Found in CUBIN")

def prune(ptx: str, entry_name: str) -> tuple[str, str, str, str]:
    """ a minimum parser to truncate the ptx for specific entry

    Use this function to locate a specific entry with entry_name.
    as Single PTX objdumped usually have > 1 entry, (try cuobjdump -ptx libcublas.so)

    NOTE verified on PTX from NVCC GCC Backend and LLVM PTX Backend
    """
    # first try to find entry_name and raise error if entry_name not found
    # .visible .entry and .entry corresponds to CUDA __global__
    length = len(ptx)
    # NOTE fix PyTorch problem
    entry_loc = ptx.find(f".visible .entry {entry_name}")
    if entry_loc == -1:
        entry_loc = ptx.find(f".entry {entry_name}") # try raw .entry instead
        if entry_loc == -1:
            found = False
            for i in range(1, 30):
                # ignore last few character for fuzzy finding
                entry_loc = ptx.find(f".entry {entry_name[:-i]}") # try raw .entry instead
                # print(entry_loc, entry_name[:-i])
                if entry_loc != -1: # if find!
                    entry_name = entry_name[:-i]
                    found = True
                    break
            if not found:
                raise ValueError(f"{entry_name} not found")

    # parse the first global section - shall be included for global info
    # global info are complicated and not involved in later processing, just keep them
    start_visible_entry = ptx.find(".visible .entry") if ptx.find(".visible .entry") != -1 else length
    start_entry = ptx.find(".entry") if ptx.find(".entry") != -1 else length
    start = min(start_visible_entry, start_entry)
    # include global_section functions like __assertfail and definitions such as gmems
    global_section = ptx[:start]

    # parse the .func which corresponds to CUDA __device__, might be used by entries
    start_func = start
    func_sections = []
    # tbh only very little code arrives here, so not much overhead
    while start_func != -1:
        start_func = ptx.find(".func", start_func)
        if start_func == -1:
            break
        # A .func directive ending in ';' is a prototype such as
        # `.extern .func foo(...);`, whose definition may sit in a later PTX
        # image of the cuobjdump output. Skip it: taking the next '{' as its
        # body would also capture unrelated text such as the Fatbin banner.
        next_semicolon = ptx.find(";", start_func)
        next_open = ptx.find("{", start_func)
        if next_semicolon != -1 and (next_open == -1 or next_semicolon < next_open):
            start_func = next_semicolon + 1
            continue
        if next_open == -1:
            raise ValueError("PTX .func definition has no body")
        # function entry could be complicated, just keep them
        pos = next_open + 1
        ket_count = 1
        next_bra = ptx.find("{", pos)
        if next_bra == -1: # not found, set to max := ptx length
            next_bra = length
        next_ket = ptx.find("}", pos)
        if next_ket == -1:
            raise ValueError("unterminated PTX .func body")
        # now parse end
        while ket_count > 0:
            if next_bra < next_ket:
                pos = next_bra + 1
                next_bra = ptx.find("{", pos) # find next {
                if next_bra == -1: # not found, set to max := ptx length
                    next_bra = length
                ket_count += 1
            else: # next_ket < next_bra, := close a bra
                pos = next_ket + 1
                next_ket = ptx.find("}", pos) # find next }
                if next_ket == -1: # not found, set to max := ptx length
                    next_ket = length
                ket_count -= 1
        # now pos is end_body
        func_sections.append(ptx[start_func:pos])
        start_func = pos # 
    func_section = "\n".join(func_sections)

    # given entry_loc try to parse the whole body
    pos = ptx.find("{", entry_loc) + 1
    ket_count = 1 # one } miss
    next_bra = ptx.find("{", pos)
    if next_bra == -1:
        next_bra = length
    next_ket = ptx.find("}", pos)
    while ket_count > 0:
        if next_bra < next_ket:
            pos = next_bra + 1
            next_bra = ptx.find("{", pos)
            if next_bra == -1: # not found, set to max := ptx length
                next_bra = length
            ket_count += 1
        elif next_bra > next_ket: # next_ket < next_bra, := close a bra
            pos = next_ket + 1 # not found, set to max := ptx length
            next_ket = ptx.find("}", pos)
            if next_ket == -1:
                next_ket = length
            ket_count -= 1
    
    entry_section = ptx[entry_loc:pos]

    return global_section, func_section, entry_section, entry_name

"""
NOTE: templates for thread-constant datamodel buffer calculation
These part shall be placed ONCE at the beginning of every kernel function definition 
if there's any thread-constant probes

Most registers below is duplicate and will be optimized by PTXAS
TODO Optimize calculation for 1D/2D Indexing (many kernel don't use 3D Indexing
"""
COMMON_BUFFER_CALC = """// begin buffer calculation
.reg .b32   %buf<20>;                       // b32 reg to record access, will be optimized by ptxas
mov.u32 	%buf2, %tid.x;                  // threadIdx.x
mov.u32 	%buf3, %tid.y;                  // threadIdx.y
mov.u32 	%buf4, %tid.z;                  // threadIdx.z
mov.u32 	%buf5, %ntid.x;                 // blockDim.x
mov.u32 	%buf6, %ntid.y;                 // blockDim.y
mov.u32 	%buf7, %ntid.z;                 // blockDim.z
mov.u32 	%buf8, %ctaid.x;                // blockIdx.x
mov.u32 	%buf9, %ctaid.y;                // blockIdx.y
mov.u32 	%buf10, %ctaid.z;               // blockIdx.z
mov.u32 	%buf11, %nctaid.x;              // gridDim.x
mov.u32 	%buf12, %nctaid.y;              // gridDim.y
mad.lo.s32 	%buf13, %buf6,  %buf4,  %buf3;  // blockDim.y * threadIdx.z + threadIdx.y
mad.lo.s32 	%buf15, %buf13, %buf5,  %buf2;  // thread_idx = (blockDim.y * threadIdx.z + threadIdx.y) * blockDim.x + threadIdx.x
mad.lo.s32 	%buf16, %buf12, %buf10, %buf9;  // gridDim.y * blockIdx.z + blockIdx.y
mad.lo.s32 	%buf17, %buf16, %buf11, %buf8;  // block_idx = (gridDim.y * blockIdx.z + blockIdx.y) * gridDim.x + blockIdx.x
mul.lo.s32  %buf18, %buf5,  %buf6;          // blockDim.x * blockDim.y
mul.lo.s32  %buf19, %buf18, %buf7;          // blockSize = blockDim.x * blockDim.y * blockDim.z
mad.lo.s32 	%buf1, %buf17, %buf19, %buf15;  // buf_idx = block_idx * blockSize + thread_idx
// end buffer calculation"""

"""
NOTE templates for warp-constant datamodel buffer calculation
These part shall be placed ONCE at the beginning of every kernel function definition 
if there's any warp-constant probes

Most registers below is duplicate and will be optimized by PTXAS
"""
WARP_BUFFER_CALC = """// begin buffer calculation
.reg .b32   %warpbuf<21>;                   // b32 reg to record access, will be optimized by ptxas
.reg .pred  %leader;                        // predicate register
.reg .pred  %joint_pred;                    // used to store AND result of %leader and instruction operand
mov.u32     %warpbuf2, %laneid;             // read lane id
setp.eq.u32 %leader, %warpbuf2, 0;          // check if thread is warp leader
@%leader mov.u32 %warpbuf3, %nwarpid;       // warpDim := number of warp in current group
@%leader mov.u32 %warpbuf4, %tid.x;         // threadIdx.x
@%leader mov.u32 %warpbuf5, %tid.y;         // threadIdx.y
@%leader mov.u32 %warpbuf6, %tid.z;         // threadIdx.z
@%leader mov.u32 %warpbuf7, %ntid.x;        // blockDim.x
@%leader mov.u32 %warpbuf8, %ntid.y;        // blockDim.y
@%leader mov.u32 %warpbuf18, %ntid.z;       // blockDim.z
@%leader mov.u32 %warpbuf9, %ctaid.x;       // blockIdx.x
@%leader mov.u32 %warpbuf10, %ctaid.y;      // blockIdx.y
@%leader mov.u32 %warpbuf11, %ctaid.z;      // blockIdx.z
@%leader mov.u32 %warpbuf12, %nctaid.x;     // gridDim.x
@%leader mov.u32 %warpbuf13, %nctaid.y;     // gridDim.y
@%leader mad.lo.s32 %warpbuf14, %warpbuf8,  %warpbuf6,  %warpbuf5;  // blockDim.y * threadIdx.z + threadIdx.y
@%leader mad.lo.s32 %warpbuf15, %warpbuf14, %warpbuf7,  %warpbuf4;  // thread_idx = (blockDim.y * threadIdx.z + threadIdx.y) * blockDim.x + threadIdx.x
@%leader div.s32 %warpbuf15, %warpbuf15, 32;                        // get persistent warpid instead of dynamic %warpid
@%leader mad.lo.s32 %warpbuf16, %warpbuf13, %warpbuf11, %warpbuf10; // gridDim.y * blockIdx.z + blockIdx.y
@%leader mad.lo.s32 %warpbuf17, %warpbuf16, %warpbuf12, %warpbuf9;  // block_idx = (gridDim.y * blockIdx.z + blockIdx.y) * gridDim.x + blockIdx.x
@%leader mul.lo.s32 %warpbuf19, %warpbuf7, %warpbuf8;
@%leader mul.lo.s32 %warpbuf20, %warpbuf19, %warpbuf18;
@%leader div.s32 %warpbuf20, %warpbuf20, 32;
@%leader mad.lo.s32 %warpbuf1,  %warpbuf17, %warpbuf20,  %warpbuf15; // buf_idx = block_idx * warpSize + warpIdx
// end buffer calculation"""

# NOTE buffer location for thread-local buffers, every probe has independent this part
THREAD_PROBE_BUFFER = """// begin {name} map
.reg .b64 %map_{name}<5>;                         // register group defn
mul.wide.s32  %map_{name}4, %buf1, {no_bytes};    // get buffer location, no_bytes is per thread
ld.param.u64  %map_{name}3, [param_{name}];       // load address from .param state space
cvta.to.global.u64 	%map_{name}2, %map_{name}3;   // convert address to .global state space
add.s64 %map_{name}1, %map_{name}2, %map_{name}4; // offset to get final thread-specific address
// end {name} map"""

# NOTE buffer of the dynamic stuffs
THREAD_PROBE_DYNAMIC_BUFFER = """// begin {name} dynamic map
.reg .b64 %map_{name}<5>;                         // register group defn
.reg .b32 %cnt_{name};                            // The dynamic count of buffer size
ld.param.u32  %cnt_{name}, [bytes_{name}];        // load sizes from .param state spaces
mul.wide.s32  %map_{name}4, %buf1, %cnt_{name};   // get buffer location, no_bytes is per thread
ld.param.u64  %map_{name}3, [param_{name}];       // load address from .param state space
cvta.to.global.u64 	%map_{name}2, %map_{name}3;   // convert address to .global state space
add.s64 %map_{name}1, %map_{name}2, %map_{name}4; // offset to get final thread-specific address
// end {name} dynamic map"""

# NOTE buffer location for warp-local buffers, every probe has independent this part
WARP_PROBE_BUFFER = """// begin {name} map
.reg .b64 %map_{name}<5>;                                   // register group defn
@%leader mul.wide.s32  %map_{name}4, %warpbuf1, {no_bytes}; // get buffer location, no_bytes is per thread
@%leader ld.param.u64  %map_{name}3, [param_{name}];        // load address from .param state space
@%leader cvta.to.global.u64 	%map_{name}2, %map_{name}3; // convert address to .global state space
@%leader add.s64 %map_{name}1, %map_{name}2, %map_{name}4;  // offset to get final thread-specific address
// end {name} map"""

# NOTE for every probe with datamodel not none
# only support .u64 and recommend use 16 bytes alignment, minimum is 8 bytes
PROBE_PARAM = ".param .u64 param_{name}"
COUNT_PARAM = ".param .u32 bytes_{name}"

# NOTE This is a special probe applied if dynamic = True, to be filled with count_inst and count_size
COUNT_PROBE = """regs = 1
[map.Count]
level = "thread"
type = "array"
size = 8
cap = 1

[map.Count.regs]
count = [ "u64", "None", ]

[probe.Save]
pos = "kernel"
level = "thread"
before = "mov.u64 %NR0, 0;"
after = "SAVE [ Count ] {{ %NR0 }};"

[probe.Count]
pos = "{count_inst}"
level = "thread"
before = "add.u64 %NR0, %NR0, {count_size};"
"""

def probing(asm: str, probes: list[Probe], maps: list[Map], regs: int) -> tuple[str, list[int]]:
    """Process the probes, the core function of probing engine"""

    # NOTE parse interesting locations
    # A mapping from location to probes, a probe can hook at multiple location
    positions: dict[str, list[Probe]] = dict()
    kernel_start_probes: list[Probe]  = []
    # NOTE turn kernel:end into ret:start for better matching
    for probe in probes:
        # different position split by ;, and inside split by : for start/end
        for position in probe.pos:
            if position == "kernel": # turn into listening instructions
                if probe.after is not None:
                    if "ret;" in positions:
                        positions["ret;"].append(probe)
                    else:
                        positions["ret;"] = [probe, ]
                if probe.before is not None:
                    kernel_start_probes.append(probe)
            else:
                if position in positions:
                    positions[position].append(probe)
                else:
                    positions[position] = [probe, ]
    
    # NOTE parse PTX Assembly
    ptx_lines = asm.split("\n") # let's do it line by line
    # first extract basic kernel signature
    entry_found: bool = False # line of .entry or .visible .entry
    entry_last_line : int = 0  # last line of entry, marked by ()
    param_end_line  : int = 0  # last line of  param declaration, for probe params
    body_start_line : int = 0  # first line of body
    idx = 0
    while idx < len(ptx_lines):
        line = ptx_lines[idx]
        if not entry_found and ".entry" in line: # entry not yet found
            entry_found = True
        if entry_found: # now entry is found
            # first check if the entry has been closed
            if ")" in line and entry_last_line == 0:
                entry_last_line = idx 
            # if entry is closed, time for body!, another if as ) { can in one line
            if body_start_line == 0 and "{" in line and entry_last_line >= 0:
                body_start_line = idx
            # if not yet reach the entry, then line with .param is param declaration
            if ".param" in line and entry_last_line == 0: 
                param_end_line = idx
            # here pattern matching positions TODO optimize performance here
            else:
                for position, probes in positions.items():
                    if position in line: # BUG might mismatch parameter with confused naming
                        # NOTE we got a match, then every probe will insert snippet before or after the line
                        # this might cause idx fluctuatting if we use idx to process it
                        line_idx = idx # a copy to fix the insertion position
                        for probe in probes: 
                            # specially handle ret;, we need to place it before ret or it won't be executed
                            if position == "ret;" and probe.after is not None:
                                ptx_lines.insert(line_idx, Ref(line=line, probe=probe, before_after=False))
                                idx += 1
                                line_idx += 1
                            else:
                                if probe.before is not None: 
                                    ptx_lines.insert(line_idx, Ref(line=line, probe=probe, before_after=True))
                                    idx += 1
                                    line_idx += 1
                                if probe.after is not None:
                                    ptx_lines.insert(line_idx + 1, Ref(line=line, probe=probe, before_after=False))
                                    idx += 1
        idx += 1

    # Now add the probes to PTX Assembly
    offset: int = 0 # adding every line need to offset 1 to make it correct
    # First let's add parameters
    ptx_lines[param_end_line] = ptx_lines[param_end_line] + "," # add , to indicate more param
    # NOTE parameter layouts: Parameters are pointers to buffer, or buffer size
    # We arange buffer pointers linearly in advance (u64), and later size (u32)
    params_added: list[str] = []
    count_params: list[str] = [] # NOTE used for dynamic counts only

    # NOTE save the map_sizes so Hook Driver has a way to allocate the map memory
    # we must make sure this is aligned with the order of parameter or will be illegal access
    map_sizes: list[tuple[str, int]] = [] # 
    
    levels: set[str] = set()
    for map_ in maps:
        if map_.cap != "dynamic":
            map_sizes.append((map_.level, map_.cap * map_.size))
            params_added.append(PROBE_PARAM.format(name=map_.name))
            levels.add(map_.level)
        else:
            map_sizes.append((map_.level, -1))
            params_added.append(PROBE_PARAM.format(name=map_.name))
            count_params.append(COUNT_PARAM.format(name=map_.name))
            levels.add(map_.level)

        # else just ignore
    params_added = params_added + count_params # formulate the layout
    ptx_lines.insert(param_end_line + 1, ",\n".join(params_added))
    offset += 1 # in total one line is added 
    # Now add the probe with kernel:start -> this shall not dump anything I think
    ptx_lines.insert(body_start_line + offset + 1, f".reg .u64 %NR<{regs + 1}>;")
    offset += 1
    for probe in kernel_start_probes:
        # NOTE kernel:start probe has no helpers and have no predicate
        ptx_lines.insert(body_start_line + offset + 1, probe.before) # None is checked before
        offset += 1
    # Now add the common buffer calculation
    if "thread" in levels:
        ptx_lines.insert(body_start_line + offset + 1, COMMON_BUFFER_CALC)
        offset += 1
    if "warp" in levels:
        ptx_lines.insert(body_start_line + offset + 1, WARP_BUFFER_CALC)
        offset += 1
    # Now add the individual buffer calculation
    # NOTE add the handle of u32 registers
    for map_ in maps:
        num_u32 = 0
        for reg in map_.regs:
            if reg.dtype == "u32":
                num_u32 += 1
        if map_.level == "thread":
            if map_.cap != "dynamic":
                buffer_asm = THREAD_PROBE_BUFFER.format(name=map_.name, no_bytes=str(map_.cap * map_.size))
                if num_u32 > 0:
                    buffer_asm = buffer_asm + f"\n.reg .u32 %{map_.name}_u32_<{num_u32 + 1}>;"
                ptx_lines.insert(body_start_line + offset + 1, buffer_asm)
                offset += 1
            else:
                buffer_asm = THREAD_PROBE_DYNAMIC_BUFFER.format(name=map_.name)
                if num_u32 > 0:
                    buffer_asm = buffer_asm + f"\n.reg .u32 %{map_.name}_u32_<{num_u32 + 1}>;"
                ptx_lines.insert(body_start_line + offset + 1, buffer_asm)
                offset += 1
        elif map_.level == "warp":
            if map_.cap != "dynamic":
                buffer_asm = WARP_PROBE_BUFFER.format(name=map_.name, no_bytes=str(map_.cap * map_.size))
                if num_u32 > 0:
                    buffer_asm = buffer_asm + f"\n.reg .u32 %{map_.name}_u32_<{num_u32 + 1}>;"
                ptx_lines.insert(body_start_line + offset + 1, buffer_asm)
                offset += 1
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    name_to_map: dict[str, Map] = {m.name: m for m in maps}
    # Now add the instruction listenings 
    for idx in range(len(ptx_lines)):
        # ignore most of line that is a string!
        if type(ptx_lines[idx]) == Ref: # NOTE isinstance is slow?
            line: str         = ptx_lines[idx].line
            probe: Probe      = ptx_lines[idx].probe
            before_after: str = ptx_lines[idx].before_after
            # parse instruction operands, operands are separated by space fundamentally
            tmp = line[:line.index(";")].split(",")
            operands: list[str] = []
            # NOTE handling vectorized operands with { and }
            merges = []
            merging: bool = False
            for operand in tmp:
                if "{" in operand and not "}" in operand:
                    merging = True
                    merges.append(operand)
                elif "}" in operand and not "{" in operand:
                    merges.append(operand) # FIX, now operand is the last one and shall be included
                    operands.append(",".join(merges).strip("{} ")) # we don't want {} remains
                    merges = []     # flush merges
                    merging = False # reset status
                else:
                    operands.append(operand) if not merging else merges.append(operand)
            # first operand also have pred, inst and the real first operand
            remaining = operands[0].strip() if len(operands) > 0 else print(line, tmp, operands, merges)
            # handle predicate -> used in final insertion
            if "@" in remaining:
                pred = remaining[:remaining.index(" ") + 1] # include the space!
                remaining = remaining[remaining.index(" ") + 1:].strip()
            else:
                pred = ""
            # TODO assert matching instruction
            mem_bytes: str = None
            out: str = None
            if remaining.find(" ") != -1:          
                inst = remaining[:remaining.index(" ")]
                # NOTE a helper to calculate bytes, ld and st's bytes are inferred not from operand
                # but the instruction body (likewise ld.global.v2.u64)
                if "ld" in inst or "st" in inst:
                    vec = 1
                    if "v2" in inst or "x2" in inst:
                        vec = 2
                    elif "v4" in inst or "x4" in inst:
                        vec = 4
                    # most dtypes are u32, no worries
                    dtypes = ["u32", "u64", "b16", "u16", "u8", "f32", "f64", "b128", "s32", "s64", "s16", "s8", "b32", "b64", "b8"]
                    for dtype in dtypes:
                        if dtype in inst:
                            mem_bytes = str(vec * int(dtype[1:]) // 8) # 8 "= size"
                            break
                out = remaining[remaining.index(" "):].strip()
                if "[" in out:
                    # NOTE handle [ addr ] used to indicate the memory address
                    out = out[out.index("[") + 1 : out.index("]")] if out is not None and "[" in out else out # fix
            in1:  str = operands[1] if len(operands) >= 2 else None
            # NOTE handle [ addr ] used to indicate the memory address
            in1 = in1[in1.index("[") + 1 : in1.index("]")] if in1 is not None and "[" in in1 else in1
            in2 = operands[2] if len(operands) >= 3 else None
            in3 = operands[3] if len(operands) >= 4 else None
            # TODO handle some weird syntax like + 0 used meaninglessly to locate correct places
            # Currently only a minimal solution
            if out is not None and "+" in out: out = out[:out.find("+")]
            if in1 is not None and "+" in in1: in1 = in1[:in1.find("+")]
            # print(line, out, in1, in2, sep=" / ")
            # now handles operand helpers by directly replacing the value
            snippet = probe.before if before_after else probe.after
            snippet = snippet.replace("OUT", out) if "OUT" in snippet else snippet
            snippet = snippet.replace("IN1", in1) if "IN1" in snippet else snippet
            snippet = snippet.replace("IN2", in2) if "IN2" in snippet else snippet
            snippet = snippet.replace("IN3", in3) if "IN3" in snippet else snippet
            # NOTE add a new helper named ADDR referencing gmem address
            if "ADDR" in snippet:
                if "ld" in operands[0] or "cp.async" in operands[0]:
                    snippet = snippet.replace("ADDR", in1)
                elif "st" in operands[0]: # st has 
                    snippet = snippet.replace("ADDR", out)
            if mem_bytes is not None:
                snippet = snippet.replace("BYTES", mem_bytes) if "BYTES" in snippet else snippet
            # now handles STORE helpers
            snippet_lines = snippet.split("\n")
            # NOTE special arrangements for warp datamodel
            org_pred = pred
            if probe.level == "warp":
                if pred == "":
                    pred = "@%leader " # apply filter that only leader works
                else:
                    pred = "@%joint_pred " # will be updated %leader AND pred

            for snippet_line_idx in range(len(snippet_lines)):
                snippet_line: str = snippet_lines[snippet_line_idx]
                if "SAVE" in snippet_line: # only one save, at the begin of line
                    # SAVE [ block_sched ] { R0, R1, R3 }
                    map_ = name_to_map[snippet_line[snippet_line.index("[") + 1: snippet_line.index("]")].strip()]
                    # dtype = snippet_line[snippet_line.find("SAVE") + 5: snippet_line.find("SAVE") + 8]
                    items = snippet_line[snippet_line.index("{") + 1:snippet_line.index("}")].split(",")
                    assert len(items) == len(map_.regs), f"{map_.name}.save not follow definition"
                    cvt_lines, u64s, u32s = [], [], []
                    u32_idx = 1 # NOTE must be 1 due to some syntax issue I think
                    for item, reg in zip(items, map_.regs):
                        if reg.dtype == "u64": 
                            u64s.append(item)
                        elif reg.dtype == "u32": # apply conversion
                            cvt_lines.append(f"{pred}cvt.u32.u64 %{map_.name}_u32_{u32_idx}, {item};\n")
                            u32s.append(f"%{map_.name}_u32_{u32_idx}")
                            u32_idx += 1
                        else:
                            raise NotImplementedError
                    assert len(u32s) % 2 == 0, "Must save 2n u32 registers for memory alignment, please promote to u64"
                    save_lines = []
                    for item_idx in range(len(u64s) // 2):
                        save_lines.append(f"{pred}st.global.v2.u64 [%map_{map_.name}1],  {{ {u64s[item_idx * 2]}, {u64s[item_idx * 2 + 1]} }};\n{pred}add.s64 %map_{map_.name}1, %map_{map_.name}1, 16;")
                    if len(u64s) % 2 != 0: # odd number -> one item left!
                        save_lines.append(f"{pred}st.global.u64 [%map_{map_.name}1], {u64s[-1]};\n{pred}add.s64 %map_{map_.name}1, %map_{map_.name}1, 8;")
                    for item_idx in range(len(u32s) // 4):
                        save_lines.append(f"{pred}st.global.v4.u32 [%map_{map_.name}1],  {{ {u32s[item_idx * 4]}, {u32s[item_idx * 4 + 1]}, {u32s[item_idx * 4 + 2]}, {u32s[item_idx * 4 + 3]} }};\n{pred}add.s64 %map_{map_.name}1, %map_{map_.name}1, 16;")
                    if len(u32s) % 4 != 0: # two items left...
                        save_lines.append(f"{pred}st.global.v2.u32 [%map_{map_.name}1],  {{ {u32s[-2]}, {u32s[-1]} }};\n{pred}add.s64 %map_{map_.name}1, %map_{map_.name}1, 8;")
                    snippet_lines[snippet_line_idx] = "\n".join(cvt_lines) + "\n".join(save_lines)
                else:
                    # or just add the pred!
                    # NOTE handling warp that having double buffer
                    if ".reg" not in snippet_line and snippet_line.strip() not in ("{", "}"):
                        snippet_lines[snippet_line_idx] = pred + snippet_line
                    else:
                        snippet_lines[snippet_line_idx] = snippet_line
            if probe.level == "warp" and org_pred != "":
                snippet_lines.insert(0, f"and.pred %tmp, %leader, {org_pred[1:]}; // joint prediction") # ignore the '@' signal at first
            snippet = "\n".join(snippet_lines)
            # finally replace the Ref with snippet to finish the probing!
            ptx_lines[idx] = snippet

    # Finally finished.1
    return "\n".join(ptx_lines)

def assemble(workdir: str, name: str) -> None:
    """compile the ptx into cubin via ptxas
    NOTE: ptxas command like `ptxas -arch=sm_80 --verbose -m64  "original.ptx"  -o "original.cubin"` 
    * This is not actually need for running because CUDA Driver cuModuleLoad can load PTX (JIT),
    * But is useful for checking as ptxas --verbose can give more info for debugging
    """
    ptx_path = os.path.join(workdir, name) + ".ptx"
    bin_path = os.path.join(workdir, name) + ".bin" # target binary
    command = ["ptxas", f'-arch={get_arch()}', '-m64', "--verbose", ptx_path, '-o', bin_path]
    print(" ".join(command), file=log)
    result = subprocess.run(
        command, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    # print debug and verbose information to the process.log
    if len(result.stderr) > 0:
        print(result.stderr.decode("utf-8"), file=log)
    if len(result.stdout) > 0:
        print(result.stdout.decode("utf-8"), file=log)

def parse_params(ptx: str) -> tuple[list[KernelParam], str]:
    """parse kernel function parameters
    @see https://docs.nvidia.com/cuda/parallel-thread-execution/#kernel-function-parameters

    NOTE this is because cuLaunchKernel receive void** as kernelParam and one can not infer
    the valid no.params from void** (NVIDIA driver also use similar parsing for that)
    """
    start = ptx.find("(")
    name_start = ptx.rfind(" ", 0, start)
    end = ptx.find(")", start)
    ptx_lines = ptx[start + 1 : end].split("\n")
    param_lines: list[str] = []
    params: list[KernelParam] = []

    for line in ptx_lines:
        if  ".param" in line:
            param_lines.append(line.strip(" ,"))
    for param_line in param_lines:
        tmp = param_line.split(" ")
        dtype = tmp[1][1:]   # .s32 .u64 ...
        name = tmp[-1]
        params.append(KernelParam(dtype, name))
    return params, ptx[name_start + 1:start] # + 1 := ignore space


def write_kernel_info(name: str, params: list[KernelParam], map_sizes: list[int], 
                      workdir: str, file_name: str = "kernel.info"):
    """write kernel info to workdir/file_name"""
    # TODO add support for vectorized items
    with open(os.path.join(workdir, file_name), "w") as f:
        # print kernel name
        print(name, file=f)
        # number of parameters, for parsing void** kernelParams
        print(len(params), file=f)
        # number of probes with memory
        print(len(map_sizes), file=f)
        # size of each memory section
        for probe_type, size in map_sizes:
            print(f"{SUPPORTED_DATAMODEL[probe_type]},{size}", file=f)
        # # NOTE: print the hook here, resolve relative path
        # if callback != "" and not callback.startswith("/"): 
        #     callback = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", callback)
        # print(callback, file=f)
        # # NOTE: following are referencing stuff not really used by hook driver
        # for param in params:
        #     print(f"{param.name},{param.dtype}", file=f)

# ENTRY for this tool
if __name__ == "__main__":
    # no argparse as the CLI is straightforward
    workdir = sys.argv[1]     # directory contains original.bin
    kernel_name = sys.argv[2].encode('utf-8', 'ignore').decode('utf-8', 'ignore') # for possible case with multiple entry in one binary

    probes: dict
    if len(sys.argv) > 3: # NOTE to facilitate debugging, not used in production
        probe_path = sys.argv[3]
        probes = toml.load(probe_path)
        print(probes)
    else: # the path in production
        # parse the environment variable to read the probes
        probe_envvar = os.environ.get("NEUTRINO_PROBES")
        if probe_envvar is None:
            raise ValueError("Can not read probes from envaraible 'NEUTRINO_PROBES'")
        # load it via toml
        probes = toml.loads(probe_envvar)
    
    # # filter out, probes are nested dict in TOML via [name]
    # probes: Dict[str, dict] = dict()
    # callback = probe_toml["CALLBACK"] if "callback" in probe_toml else ""
    # for key, value in probe_toml.items():
    #     if isinstance(value, dict):
    #         probes[key] = value

    # parse the environment variable for filtered out kernel, this is for
    # 1. Some buggy kernels caused system fails -> many GPU error is not recoverable
    # 2. Some uninterested kernels such as vectorized_elementwise for PyTorch
    filter_out = os.environ.get("NEUTRINO_FILTER", "")
    filter_out = filter_out.split(":") if len(filter_out) > 0 else None
    print(filter_out, file=log)
    
    filter_in = os.environ.get("NEUTRINO_KERNEL", "")
    filter_in = filter_in.split(":") if len(filter_in) > 0 else None
    print(filter_in, file=log)
    
    # NOTE check if some probe is defined as dynamic, if so, we need to add a counter
    #      for these probes in different arangements
    dynamic = bool(os.environ.get("NEUTRINO_DYNAMIC", 0)) 

    try:
        # first objdump binary to ptx
        ptx = dump(workdir)
        # then truncate ptx for entry_name
        global_section, func_section, entry_section, _ = prune(ptx, kernel_name)
        # split and process ptx lines and write kernel info
        params, kernel_name = parse_params(entry_section)

        # basic logging
        print(kernel_name, file=log)
        if filter_in:
            matched = False
            for tmp in filter_in:
                if tmp in kernel_name:
                    matched = True
            if not matched:
                print(f"{kernel_name} is not in {filter_in}", file=log)
                exit(1)
        if filter_out:
            for tmp in filter_out:
                if tmp != "" and tmp in kernel_name:
                    print(f"{kernel_name} filtered out from {filter_out}", file=log)
                    exit(1)

        # write pruned ptx to file
        pruned_ptx = global_section + "\n" + func_section + "\n" + entry_section
        with open(os.path.join(workdir, "pruned.ptx"), "w") as f:
            f.write(pruned_ptx)

        # convert probes from Python Dict to data structure
        probes, maps, regs = load(probes)

        # NOTE generate the trace reading code
        if dynamic:
            # First check the probe with size is dynamic, aka size = -1
            count_inst = []
            count_size = 0
            count_map  = ""
            for map_ in maps:
                if map_.cap == "dynamic":
                    count_size = map_.size
                    count_map  = map_.name
                    break
            for probe in probes:
                # NOTE there might be a bug that before and after can only save once
                if probe.before is not None:
                    if "SAVE" in probe.before and count_map in probe.before:
                        count_inst = count_inst + probe.pos
                if probe.after is not None:
                    if "SAVE" in probe.after and count_map in probe.after:
                        count_inst = count_inst + probe.pos
            assert len(count_inst) > 0
            count_inst = ":".join(count_inst)
            count_probe = COUNT_PROBE.format(count_inst = count_inst, count_size = count_size)
            count_probe, count_map, count_reg = load(toml.loads(count_probe))
            count_ptx = probing(entry_section, count_probe, count_map, count_reg)
            count_ptx = global_section + "\n" + func_section + "\n" + count_ptx
            with open(os.path.join(workdir, "countd.ptx"), "w") as f:
                f.write(count_ptx)

        map_sizes: list[tuple[str, int]] = []
        for map_ in maps:
            if isinstance(map_.cap, int):
                map_sizes.append((map_.level, map_.size * map_.cap))
            elif map_.cap == "dynamic":
                map_sizes.append((map_.level, -1))
            else:
                raise NotImplementedError

        # process ptx lines
        probed_ptx = probing(entry_section, probes, maps, regs)

        # merge global and func back
        probed_ptx  = global_section + "\n" + func_section + "\n" + probed_ptx

        # write ptx to file
        with open(os.path.join(workdir, "probed.ptx"), "w") as f:
            f.write(probed_ptx)
        
        # params = parse_params(ptx_lines)
        write_kernel_info(kernel_name, params, map_sizes, workdir)

        # compile ptx to binary, we want both probed and pruned
        assemble(workdir, "probed")
        assemble(workdir, "pruned")
        if dynamic:
            assemble(workdir, "countd")

    except Exception as e:
        traceback.print_exc(file=log)
        exit(1)