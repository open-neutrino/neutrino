"""Neutrino Probing Engine, CUDA Implementation"""

from typing import List, Tuple, Optional, Dict, Set
import os
import sys
import shutil
import subprocess
import traceback # usef for print backtrace to log file instead of stdout
import toml      # to load probes from envariables
from dataclasses import dataclass
from engine import Probe, Ref, safe_load_probes, TRACE_READING_CODE_PY

workdir = sys.argv[1]     # directory contains original.bin
log = open(os.path.join(workdir, "process.log"), 'w')

# a macro like terms
SUPPORTED_DATAMODEL = { "thread": 0, "warp": 1 }

@dataclass
class KernelParam:
    dtype: str
    name: str

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
        print("[decompile] bin is ptx", file=log)
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
        print("[decompile] via cuobjdump -ptx", file=log)
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
                print("[decompile] via cuobjdump -elf", file=log)
                return section[start:]
        # if still not found
        raise ValueError("PTX Not Found in CUBIN")

def prune(ptx: str, entry_name: str) -> Tuple[str, str, str, str]:
    """ a minimum parser to truncate the ptx for specific entry

    Use this function to locate a specific entry with entry_name.
    as Single PTX decompiled usually have > 1 entry, (try cuobjdump -ptx libcublas.so)

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
        # function entry could be complicated, just keep them
        pos = ptx.find("{", start_func) + 1
        ket_count = 1
        next_bra = ptx.find("{", pos)
        next_ket = ptx.find("}", pos)
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
PROBE_BUFFER = """// begin {name} buffer
.reg .b64 %buf_{name}<5>;                         // register group defn
mul.wide.s32  %buf_{name}4, %buf1, {no_bytes};    // get buffer location, no_bytes is per thread
ld.param.u64  %buf_{name}3, [param_{name}];       // load address from .param state space
cvta.to.global.u64 	%buf_{name}2, %buf_{name}3;   // convert address to .global state space
add.s64 %buf_{name}1, %buf_{name}2, %buf_{name}4; // offset to get final thread-specific address
// end {name} buffer"""

# NOTE buffer location for warp-local buffers, every probe has independent this part
WARP_PROBE_BUFFER = """// begin {name} buffer
.reg .b64 %buf_{name}<5>;                          // register group defn
@%leader mul.wide.s32  %buf_{name}4, %warpbuf1, {no_bytes}; // get buffer location, no_bytes is per thread
@%leader ld.param.u64  %buf_{name}3, [param_{name}];        // load address from .param state space
@%leader cvta.to.global.u64 	%buf_{name}2, %buf_{name}3;    // convert address to .global state space
@%leader add.s64 %buf_{name}1, %buf_{name}2, %buf_{name}4;  // offset to get final thread-specific address
// end {name} buffer"""

# NOTE for every probe with datamodel not none
# only support .u64 and recommend use 16 bytes alignment, minimum is 8 bytes
PROBE_PARAM = ".param .u64 param_{name}"

def probing(asm: str, probes: List[Probe]) -> Tuple[str, List[int], str]:
    """Process the probes, the core function of probing engine

    In general, Take several steps, from:
    0. Process the probe 
    1. Validate the probe
    2. Check the location for parsing
    3. Parsing the PTX Assembly
    4. Adding the PTX Assembly
    """

    # NOTE parse interesting locations
    # A mapping from location to probes, a probe can hook at multiple location
    positions: Dict[str, List[Probe]] = dict()
    kernel_start_probes: List[Probe]  = []
    kernel_end_probes:   List[Probe]  = []
    # NOTE turn kernel:end into ret:start for better matching
    for probe in probes:
        # different position split by ;, and inside split by : for start/end
        for position in probe.position:
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
    params_added: List[str] = []

    # NOTE save the probe_mem_sizes so Hook Driver has a way to load the stuff back
    # we must make sure this is aligned with the order of parameter or will be illegal access
    probe_mem_sizes: List[Tuple[str, int]] = [] # 
    
    processed: Set[str] = set() # a set to avoid repeated process same probe that leads to error
    datamodels: Set[str] = set()
    for probe in probes + kernel_start_probes:
        if probe.name not in processed and probe.datamodel is not None:
            probe_mem_sizes.append((probe.datamodel, int(probe.cap) * int(probe.no_bytes)))
            params_added.append(PROBE_PARAM.format(name=probe.name))
            processed.add(probe.name)
            datamodels.add(probe.datamodel)
        # else just ignore
    ptx_lines.insert(param_end_line + 1, ",\n".join(params_added))
    offset += 1 # in total one line is added 
    # Now add the probe with kernel:start -> this shall not dump anything I think
    for probe in kernel_start_probes:
        # NOTE kernel:start probe has no helpers and have no predicate
        ptx_lines.insert(body_start_line + offset + 1, probe.before) # None is checked before
        offset += 1
    # Now add the common buffer calculation
    if "thread" in datamodels:
        ptx_lines.insert(body_start_line + offset + 1, COMMON_BUFFER_CALC)
        offset += 1
    if "warp" in datamodels:
        ptx_lines.insert(body_start_line + offset + 1, WARP_BUFFER_CALC)
        offset += 1
    # Now add the individual buffer calculation
    processed = set()
    for probe in probes + kernel_start_probes:
        if probe.name not in processed:
            if probe.datamodel == "thread":
                no_bytes = str(int(probe.cap) * int(probe.no_bytes))
                ptx_lines.insert(body_start_line + offset + 1, 
                                PROBE_BUFFER.format(name=probe.name, no_bytes=no_bytes))
                offset += 1
            elif probe.datamodel == "warp":
                no_bytes = int(probe.cap) * int(probe.no_bytes)
                ptx_lines.insert(body_start_line + offset + 1,
                                 WARP_PROBE_BUFFER.format(name=probe.name, no_bytes=no_bytes))
                offset += 1
            processed.add(probe.name)
        # all rest is treated as no saving
    
    # NOTE for reading the trace into Python
    trace_reading_code = ""

    # Now add the instruction listenings 
    for idx in range(len(ptx_lines)):
        # ignore most of line that is a string!
        if type(ptx_lines[idx]) == Ref: # NOTE isinstance is slow?
            line: str         = ptx_lines[idx].line
            probe: Probe      = ptx_lines[idx].probe
            before_after: str = ptx_lines[idx].before_after
            # parse instruction operands, operands are separated by space fundamentally
            tmp = line[:line.index(";")].split(",")
            operands: List[str] = []
            # NOTE handling vectorized operands with { and }
            merges = []
            merging: bool = False
            for operand in tmp:
                if "{" in operand and not "}" in operand:
                    merging = True
                elif "}" in operand and not "{" in operand:
                    operands.append(",".join(merges).strip("{} ")) # we don't want {} remains
                    merges = []     # flush merges
                    merging = False # reset status
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
            in2:  str = operands[2] if len(operands) >= 3 else None
            in3:  str = operands[3] if len(operands) >= 4 else None
            # now handles operand helpers by directly replacing the value
            snippet = probe.before if before_after else probe.after
            snippet = snippet.replace("OUT", out) if "OUT" in snippet else snippet
            snippet = snippet.replace("IN1", in1) if "IN1" in snippet else snippet
            snippet = snippet.replace("IN2", in2) if "IN2" in snippet else snippet
            snippet = snippet.replace("IN3", in3) if "IN3" in snippet else snippet
            if mem_bytes is not None:
                snippet = snippet.replace("BYTES", mem_bytes) if "BYTES" in snippet else snippet
            # now handles STORE helpers
            snippet_lines = snippet.split("\n")
            # NOTE special arrangements for warp datamodel
            org_pred = pred
            if probe.datamodel == "warp":
                if pred == "":
                    pred = "@%leader " # apply filter that only leader works
                else:
                    pred = "@%joint_pred " # will be updated %leader AND pred
            
            # NOTE for reading the probe afterwards
            saved: List[Tuple[str, str]] = []

            for snippet_line_idx in range(len(snippet_lines)):
                snippet_line: str = snippet_lines[snippet_line_idx]
                if "SAVE" in snippet_line: # only one save, at the begin of line
                    # NOTE saving different types must be separated
                    # example SAVE.u64 {%gstart, %gend}; or SAVE.u32 {%smid, %nsmid}; 
                    dtype = snippet_line[snippet_line.find("SAVE") + 5: snippet_line.find("SAVE") + 8]
                    save_lines = [] # start a new string
                    items = snippet_line[snippet_line.index("{") + 1:snippet_line.index("}")].split(",")
                    if dtype == "u64":
                        for item_idx in range(len(items) // 2): # try to generate vectorized!
                            save_lines.append(f"{pred}st.global.v2.u64 [%buf_{probe.name}1],  {{ {items[item_idx * 2]}, {items[item_idx * 2 + 1]} }};\n{pred}add.s64 %buf_{probe.name}1, %buf_{probe.name}1, 16;")
                            # TODO Need to convert register starts with % to Python variable name, now only forget 1st one
                            saved.append((items[item_idx * 2], "Q")) # Q := u64 in Python struct
                            saved.append((items[item_idx * 2 + 1], "Q")) # Q := u64 in Python struct
                        if len(items) % 2 != 0: # odd number -> one item left!
                            save_lines.append(f"{pred}st.global.u64 [%buf_{probe.name}1], {items[-1]};\n{pred}add.s64 %buf_{probe.name}1, %buf_{probe.name}1, 8;")
                            saved.append((items[-1], "Q")) # Q := u64 in Python struct
                    elif dtype == "u32":
                        for item_idx in range(len(items) // 4): # try to generate vectorized!
                            save_lines.append(f"{pred}st.global.v4.u32 [%buf_{probe.name}1],  {{ {items[item_idx * 4]}, {items[item_idx * 4 + 1]}, {items[item_idx * 4 + 2]}, {items[item_idx * 4 + 3]} }};\n{pred}add.s64 %buf_{probe.name}1, %buf_{probe.name}1, 16;")
                            saved.append((items[item_idx * 4], "I")) # I := u32 in Python struct
                            saved.append((items[item_idx * 4 + 1], "I")) # I := u32 in Python struct
                            saved.append((items[item_idx * 4 + 2], "I")) # I := u32 in Python struct
                            saved.append((items[item_idx * 4 + 3], "I")) # I := u32 in Python struct
                        if len(items) % 4 != 0: # two items left...
                            save_lines.append(f"{pred}st.global.v2.u32 [%buf_{probe.name}1],  {{ {items[-2]}, {items[-1]} }};\n{pred}add.s64 %buf_{probe.name}1, %buf_{probe.name}1, 8;")
                            saved.append((items[-2], "I")) # I := u32 in Python struct
                            saved.append((items[-1], "I")) # I := u32 in Python struct
                    else:
                        raise ValueError(f"Unsupported dtype {dtype} in {probe.name}:{'before' if before_after else 'after'}")
                    snippet_lines[snippet_line_idx] = "\n".join(save_lines)
                else:
                    # or just add the pred!
                    # NOTE handling warp that having double buffer
                    snippet_lines[snippet_line_idx] = pred + snippet_line
            if probe.datamodel == "warp" and org_pred != "":
                snippet_lines.insert(0, f"and.pred %tmp, %leader, {org_pred[1:]}; // joint prediction") # ignore the '@' signal at first
            snippet = "\n".join(snippet_lines)
            # finally replace the Ref with snippet to finish the probing!
            ptx_lines[idx] = snippet
            # Finalizing Code Generation for Reading Probe Content
            if probe.datamodel is not None:
                saved_content = ""
                saved_reading = ""
                format_string = ""
                reading_bytes = 0
                for name, dtype in saved:
                    name = name.strip()
                    name = name[1:] if name.startswith("%") else name
                    saved_content += f"\t{name}: int\n"
                    saved_reading += f"{name}, "
                    format_string += dtype
                    reading_bytes += 8 if dtype == "Q" else 4
                trace_reading_code += TRACE_READING_CODE_PY.format(probe_name = probe.name, 
                    saved_content=saved_content, saved_reading=saved_reading[:-2], 
                    format_string=format_string, reading_bytes=reading_bytes,
                    warp_div="//32" if probe.datamodel.startswith("warp") else "")

    # Finally finished.1
    return "\n".join(ptx_lines), probe_mem_sizes, trace_reading_code

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

def parse_params(ptx: str) -> Tuple[List[KernelParam], str]:
    """parse kernel function parameters
    @see https://docs.nvidia.com/cuda/parallel-thread-execution/#kernel-function-parameters

    NOTE this is because cuLaunchKernel receive void** as kernelParam and one can not infer
    the valid no.params from void** (NVIDIA driver also use similar parsing for that)
    """
    start = ptx.find("(")
    name_start = ptx.rfind(" ", 0, start)
    end = ptx.find(")", start)
    ptx_lines = ptx[start + 1 : end].split("\n")
    param_lines: List[str] = []
    params: List[KernelParam] = []

    for line in ptx_lines:
        if  ".param" in line:
            param_lines.append(line.strip(" ,"))
    for param_line in param_lines:
        tmp = param_line.split(" ")
        dtype = tmp[1][1:]   # .s32 .u64 ...
        name = tmp[-1]
        params.append(KernelParam(dtype, name))
    return params, ptx[name_start + 1:start] # + 1 := ignore space


def write_kernel_info(name: str, params: List[KernelParam], probe_mem_sizes: List[int], 
                      workdir: str, analyze_hook: str = "", file_name: str = "kernel.info"):
    """write kernel info to workdir/file_name"""
    # TODO add support for vectorized items
    with open(os.path.join(workdir, file_name), "w") as f:
        # print kernel name
        print(name, file=f)
        # number of parameters, for parsing void** kernelParams
        print(len(params), file=f)
        # number of probes with memory
        print(len(probe_mem_sizes), file=f)
        # size of each memory section
        for probe_type, size in probe_mem_sizes:
            print(f"{SUPPORTED_DATAMODEL[probe_type]},{size}", file=f)
        # NOTE: print the hook here, resolve relative path
        if analyze_hook != "" and not analyze_hook.startswith("/"): 
            analyze_hook = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", analyze_hook)
        print(analyze_hook, file=f)
        # NOTE: following are referencing stuff not really used by hook driver
        for param in params:
            print(f"{param.name},{param.dtype}", file=f)

# ENTRY for this tool
if __name__ == "__main__":
    # no argparse as the CLI is straightforward
    workdir = sys.argv[1]     # directory contains original.bin
    kernel_name = sys.argv[2].encode('utf-8', 'ignore').decode('utf-8', 'ignore') # for possible case with multiple entry in one binary

    if len(sys.argv) > 3: # NOTE to facilitate debugging, not used in production
        probe_path = sys.argv[3]
        probe_toml = toml.load(probe_path)
    else: # the path in production
        # parse the environment variable to read the probes
        probe_envvar = os.environ.get("NEUTRINO_PROBES")
        if probe_envvar is None:
            raise ValueError("Can not read probes from envaraible 'NEUTRINO_PROBES'")
        # load it via toml
        probe_toml = toml.loads(probe_envvar)
    
    # filter out, probes are nested dict in TOML via [name]
    probes: Dict[str, dict] = dict()
    analyze_hook = probe_toml["analyze_hook"] if "analyze_hook" in probe_toml else ""
    for key, value in probe_toml.items():
        if isinstance(value, dict):
            probes[key] = value

    # parse the environment variable for filtered out kernel, this is for
    # 1. Some buggy kernels caused system fails -> many GPU error is not recoverable
    # 2. Some uninterested kernels such as vectorized_elementwise for PyTorch
    filter_out = os.environ.get("NEUTRINO_FILTER", "").split(":")
    print(filter_out, file=log)

    try:
        # first decompile binary to ptx
        ptx = dump(workdir)
        # then truncate ptx for entry_name
        global_section, func_section, entry_section, _ = prune(ptx, kernel_name)
        # split and process ptx lines and write kernel info
        params, kernel_name = parse_params(entry_section)

        # basic logging
        print(kernel_name, file=log)
        for tmp in filter_out:
            if tmp != "" and tmp in kernel_name:
                print(f"{kernel_name} filtered out from {filter_out}", file=log)
                exit(1)

        # write pruned ptx to file
        pruned_ptx = global_section + "\n" + func_section + "\n" + entry_section
        with open(os.path.join(workdir, "pruned.ptx"), "w") as f:
            f.write(pruned_ptx)

        # convert probes from Python Dict to data structure
        probes = safe_load_probes(raw_probes=probes)

        # process ptx lines
        probed_ptx, probe_mem_sizes, trace_reading_code = probing(entry_section, probes)

        # merge global and func back
        probed_ptx  = global_section + "\n" + func_section + "\n" + probed_ptx

        # write ptx to file
        with open(os.path.join(workdir, "probed.ptx"), "w") as f:
            f.write(probed_ptx)
        
        # params = parse_params(ptx_lines)
        write_kernel_info(kernel_name, params, probe_mem_sizes, workdir, analyze_hook)

        # compile ptx to binary, we want both probed and pruned
        assemble(workdir, "probed")
        assemble(workdir, "pruned")

        print(trace_reading_code, file=log)

    except Exception as e:
        traceback.print_exc(file=log)
        exit(1)