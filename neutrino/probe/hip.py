"""Neutrino Probing Engine, AMD ROCm HIP Implementation

HIP's AMDGCN (.amdgcn) has only one source: LLVM AMDGPU Backend
CITE https://llvm.org/docs/AMDGPUUsage.html

GCNAsm is similar to x86 assembly, kernel mainly has two parts (two .text):
1. GCNAsm Code in form of .section .text
2. AMD HSA Configuration or said `.amdgpu_metadata`

NOTE At the moment, we only support part of syntax because there's no official
     documentation on syntax / semantics of GCN Assembly (only instructions).

Important GCNAsm syntax for Neutrino developers:
1. s[0:1] holds pointer to kernargs, use s_load_dword to ld.param
2. v0 (32bit) holds 00 + threadIdx.z(30-20)+threadIDx.y(20-10)+threadIdx.x(10-0)
3. blockIdx.xyz is the next 3 registers after first three, gridDim.xyz is ttmp8/9/10
4. It seems blockDim can not be retrieved easily from special registers, only via 

It's worth noticed that v0 and s[0:1] can be changed by developers, i.e., their 
value (threadIdx.xyz) is only available at kernel begins.

NOTE Why not fully support?
Because we can not find AMD GPUs for testing or debugging (we are not AMD).
There's nearly no cloud providers for AMD GPUs (only MI300x on runpods.io). 

And part of AMD's ISA is ridiculous, for example, until CDNA3, add two u64
is finally supported on VGPRs, still not supported on SGPRs. Why? Tell me why?
"""

from typing import List, Tuple, Optional, Dict, Set
import os
import sys
import shutil
import subprocess
import traceback # usef for print backtrace to log file instead of stdout
import toml      # to load probes from envariables
import yaml      # AMD GCN ASM use YAML as METADATA Storage
from dataclasses import dataclass
from engine import Probe, Ref, safe_load_probes

workdir = sys.argv[1]     # directory contains original.bin
log = open(os.path.join(workdir, "process.log"), 'w')

# a macro like terms
SUPPORTED_DATAMODEL = { "thread": 0, "warp": 1 }

# NOTE applicable to CDNA GPUs but might not be applicable to GDNA GPUs
# TODO change to amdgpu_metadata['amdhsa.kernels'][0]['.wavefront_size']
WARP_SIZE = 64 

@dataclass
class KernelParam: # NOTE GCNASM has different defn
    value_kind: str
    size: int

# NOTE it's risky but safe as this is a CLI tool invoked for specific kernel
amdgpu_metadata: Dict = None

# TODO finalize rocm-smi toolchain
def get_arch() -> str: 
    """At the moment, we extract target arch from the assembly, but not sure
    if this may leads to misleading arch for codegen, will see"""
    ...

# TODO finalize llvm-objdump toolchain
def extract(workdir: str, name: str = "original", suffix: str = ".bin") -> str:
    bin_path = os.path.join(workdir, name) + suffix
    # first check if it's already a NULL-Terminated PTX (i.e., ASCII Text)
    result = subprocess.run(['file', bin_path], stdout=subprocess.PIPE, text=True)
    out = result.stdout
    if "ASCII text" in result.stdout: # raw PTX file, just read it all
        shutil.copyfile(bin_path, os.path.join(workdir, name) + ".asm")
        print("[decompile] bin is gcnasm", file=log)
        with open(os.path.join(workdir, name) + ".asm", "r") as outf:
            return outf.read()

# TODO add prune support
def prune(asm: str, entry_name: str) -> Tuple[str, str]:
    """A Minimum parser to truncate the gcn asm for specific entry
    
    Use this function to locate a specific entry with entry_name as
    single .asm / .s usually have > 1 entry!    
    """
    # First find the two single line of .text
    lines = asm.split("\n")
    sections = []
    target = None # assembler target like .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
    for idx in range(len(lines)):
        if ".text" in lines[idx]:
            sections.append(idx) # record the sections
        elif ".amdgcn_target" in lines[idx]:
            target = lines[idx]
    # reorganize sections
    kernels = []
    for idx in range(len(sections) - 2): # last section holds gcnasm
        if "@function" in "\n".join(lines[sections[idx] : sections[idx + 1]]):
            kernels.append(sections[idx])
    kernels.append(sections[-1])
    # Now locate the entry
    # TODO add rough matching!!!
    entry_section = None
    for idx in range(len(kernels) - 2):
        temp = "\n".join(lines[kernels[idx] : kernels[idx + 1]])
        if entry_name in temp:
            entry_section = temp
    assert entry_section is not None, "Fail to find"
    # adding target if not found
    if target not in entry_section:
        entry_section = entry_section.split("\n")
        entry_section = entry_section[0] + target + "\n".join(entry_section[1:])
    # fix the metadata section
    last_section = "\n".join(lines[sections[-1]:])
    assert ".amdgpu_metadata" in last_section
    metadata = last_section[last_section.index(".amdgpu_metadata") +  16: last_section.index(".end_amdgpu_metadata") - 1] # BUG -1 is a fix
    global amdgpu_metadata
    amdgpu_metadata = yaml.safe_load(metadata)
    for kernelmeta in amdgpu_metadata['amdhsa.kernels']:
        if kernelmeta['.name'] == entry_name:
            amdgpu_metadata['amdhsa.kernels'] = [kernelmeta, ] # only want this one
            break
    return entry_section, last_section

def parse_params() -> Tuple[List[KernelParam], str]:
    global amdgpu_metadata
    kernel_name = amdgpu_metadata['amdhsa.kernels'][0]['.name']
    params: List[KernelParam] = []
    for arg in amdgpu_metadata['amdhsa.kernels'][0]['.args']:
        params.append(KernelParam(arg['.value_kind'], arg['.size']))
    return params, kernel_name

"""
NOTE: templates for thread-constant datamodel buffer calculation
These part shall be placed ONCE at the beginning of every kernel function definition 
if there's any thread-constant probes

Most registers below is duplicate and will be optimized by AMD Assembler
"""

THREAD_BUFFER_COMMON = """;;# begin buffer calculation
V_MOV_B32 v{thread_buff}, v0 ;;# v0 holds threadIdx.x, don't know what's threadIdx.y, threadIdx.z
;;# end buffer calculation"""

WARP_BUFFER_COMMON = """;;# begin buffer calculation
V_LSHRREV_B32_E32 v{warp_buff}, 6, v0;; # shift 6 bits := // 64
;;# end buffer calculation"""

# NOTE buffer location for thread-local buffers, every probe has independent this part
THREAD_BUFFER = """;;# begin {name} buffer
S_LOAD_DWORDX2 s[{param_reg}], s[0:1], {param_offset};;# load buffer address into 64bit register (2x32)
V_MAD_I64_I32 v[{param_addr}], v{thread_buff}, {no_bytes}, s[{param_reg}];;# calculate the address
;;# end {name} buffer"""

# BUG it shall be possible to move everything into sgpr, but I don't know how to locate
# warpIdx in solely SGPR, please help me
WARP_BUFFER = """;;# begin {name} buffer
S_LOAD_DWORDX2 s[{param_reg}], s[0:1], {param_offset};;# load buffer address into 64bit register (2x32)
V_MAD_I64_I32 v[{param_addr}], v{warp_buff}, {no_bytes}, s[{param_reg}];;# calculate the address
;;# end {name} buffer"""

def probing(asm: str, probes: List[Probe]) -> Tuple[str, List[int], str]:
    """Probing the Assembly, the core of probing engine
    
    NOTE we assume probe is parsed and (security checked)"""
    
    # NOTE parse interesting locations
    # A mapping from location to probes, a probe can hook at multiple location
    positions: Dict[str, List[Probe]] = dict()
    kernel_start_probes: List[Probe]  = []
    # NOTE turn kernel:end into ret:start for better matching
    print(probes)
    for probe in probes:
        # different position split by ;, and inside split by : for start/end
        for position in probe.position:
            if position == "kernel": # turn into listening instructions
                if probe.after is not None:
                    if "s_endpgm" in positions: # AMD use s_endpgm to terminate
                        positions["s_endpgm"].append(probe)
                    else:
                        positions["s_endpgm"] = [probe, ]
                if probe.before is not None:
                    kernel_start_probes.append(probe)
            else: 
                if position in positions:
                    positions[position].append(probe)
                else:
                    positions[position] = [probe, ]
    
    # NOTE parse GCN Assembly
    gcn_lines = asm.split("\n") # let's do it line by line
    # first extract basic kernel signature
    body_start_line : int = 0  # first line of body
    idx = 0
    # NOTE specially handle kernel start probe

    while idx < len(gcn_lines):
        line = gcn_lines[idx]
        # First try to find ; %bb.0: NOTE just the behavior of hipcc not standard syntax
        # but we don't know what's the standard syntax, there's no documentation about this...
        # maybe this is the reason why AMD product is hard to use?
        if "%bb.0" in line:
            body_start_line = idx
            # BUG move it to the real beginning before loading and saving ? 
            for probe in kernel_start_probes:
                gcn_lines.insert(idx + 1, Ref(line=line, probe=probe, before_after=True)) # place after
                idx += 1
        # here pattern matching positions TODO optimize performance here
        else:
            for position, probes in positions.items():
                if position in line: # BUG might mismatch parameter with confused naming
                    # NOTE we got a match, then every probe will insert snippet before or after the line
                    # this might cause idx fluctuatting if we use idx to process it
                    line_idx = idx # a copy to fix the insertion position
                    for probe in probes:
                        # specially handle ret;, we need to place it before ret or it won't be executed
                        if position == "s_endpgm" and probe.after is not None:
                            gcn_lines.insert(line_idx, Ref(line=line, probe=probe, before_after=False))
                            idx += 1
                            line_idx += 1
                        else:
                            if probe.before is not None: 
                                gcn_lines.insert(line_idx, Ref(line=line, probe=probe, before_after=True))
                                idx += 1
                                line_idx += 1
                            if probe.after is not None:
                                gcn_lines.insert(line_idx + 1, Ref(line=line, probe=probe, before_after=False))
                                idx += 1
        idx += 1
    
    # work with register spaces, NOTE AMD GCN Asm don't have declartion syntax
    # for registers, just a flatten v[0:1], we need to manage them manually
    # GCN Asm has two register spaces:
    # 1. VGPR (v0), holding thread-spcific values
    # 2. SGPR (s0), holding warp-specifc values
    # TODO we can optimize warp probes to SGPR only, avoiding VGPR usage

    # Now add the probes to PTX Assembly
    offset: int = 0 # adding every line need to offset 1 to make it correct
    probe_mem_sizes: List[Tuple[str, int]] = [] # 
    # TODO parse these from meta
    global amdgpu_metadata
    # NOTE here the sgpr number is wrong, there'll be 6 more, I don't know why, I can only
    # record it at the moment.
    # BUG SGPR number from metadata doesn't match the actual usage. Always
    # 6 more is used. I don't know why but let's keep it.
    sgpr_all = amdgpu_metadata["amdhsa.kernels"][0]['.sgpr_count'] 
    sgpr = 0
    for idx in range(len(gcn_lines)):
        if type(gcn_lines[idx]) is str and ".amdhsa_next_free_sgpr" in gcn_lines[idx]:
            sgpr = int(gcn_lines[idx].strip().split()[1])
    sgpr_diff = sgpr_all - sgpr
    vgpr = amdgpu_metadata['amdhsa.kernels'][0]['.vgpr_count'] # used for new stuff
    param_off = amdgpu_metadata["amdhsa.kernels"][0]['.kernarg_segment_size']
    param_align = amdgpu_metadata["amdhsa.kernels"][0]['.kernarg_segment_align']
    param_off = ((param_off + param_align - 1) // param_align ) * param_align # round up
    params = []
    thread_buff_vgpr, warp_buff_vgpr = None, None # conform Python scope

    processed: Set[str] = set() # a set to avoid repeated process same probe that leads to error
    datamodels: Set[str] = set()
    for probe in probes:
        if probe.name not in processed and probe.datamodel is not None:
            probe_mem_sizes.append((probe.datamodel, int(probe.cap) * int(probe.no_bytes)))
            processed.add(probe.name)
            datamodels.add(probe.datamodel)

    if "thread" in datamodels:
        thread_buff_vgpr = f"{vgpr}"
        gcn_lines.insert(body_start_line + offset + 1, THREAD_BUFFER_COMMON.format(thread_buff=thread_buff_vgpr))
        offset += 1
        vgpr += 1
    if "warp" in datamodels:
        warp_buff_vgpr = f"{vgpr}"
        gcn_lines.insert(body_start_line + offset + 1, WARP_BUFFER_COMMON.format(warp_buff=warp_buff_vgpr))
        offset += 1
        vgpr += 1

    # Now add the individual buffer calculation
    processed = set()
    for probe in probes:
        if probe.name not in processed:
            if probe.datamodel == "thread":
                no_bytes = str(int(probe.cap) * int(probe.no_bytes))
                gcn_lines.insert(body_start_line + offset + 1, 
                    THREAD_BUFFER.format(name=probe.name, no_bytes=no_bytes, 
                        param_offset=param_off, param_reg=f"{sgpr}:{sgpr+1}", 
                        thread_buff=thread_buff_vgpr, param_addr=f"{vgpr}:{vgpr+1}"))
                probe.param_addr = f"{vgpr}:{vgpr+1}" # NOTE record the address 
                offset += 1
                sgpr += 2 # 2x32bit registers to hold 8bytes, specific to warp
                vgpr += 2 # 2x32bit registers to hold 8bytes, specific to thread
                params.append({'.address_space': 'global', '.size': 8, 
                    '.offset': param_off, '.value_kind': 'global_buffer'})
                param_off += 8 # only pass in pointers so 8bytes := 64bits
            elif probe.datamodel == "warp":
                no_bytes = str(int(probe.cap) * int(probe.no_bytes))
                gcn_lines.insert(body_start_line + offset + 1, 
                    WARP_BUFFER.format(name=probe.name, no_bytes=no_bytes, 
                        param_offset=param_off, param_reg=f"{sgpr}:{sgpr+1}", 
                        warp_buff=warp_buff_vgpr, param_addr=f"{vgpr}:{vgpr+1}"))
                probe.param_addr = f"{vgpr}:{vgpr+1}" # NOTE record the address 
                offset += 1
                sgpr += 2 # 2x32bit registers to hold 8bytes, specific to warp
                vgpr += 2 # 2x32bit registers to hold 8bytes, specific to thread
                params.append({'.address_space': 'global', '.size': 8, 
                    '.offset': param_off, '.value_kind': 'global_buffer'})
                param_off += 8 # only pass in pointers so 8bytes := 64bits
            for reg in probe.registers:
                if probe.registers[reg] == "b32":
                    if probe.datamodel == "warp":
                        probe.registers[reg] = f"s{sgpr}"
                        sgpr += 1
                    elif probe.datamodel == "thread":
                        probe.registers[reg] = f"v{vgpr}"
                        vgpr += 1
                elif probe.registers[reg] == "b64":
                    if probe.datamodel == "warp":
                        probe.registers[reg] = f"s[{sgpr}:{sgpr+1}]"
                        sgpr += 2
                    elif probe.datamodel == "thread":
                        probe.registers[reg] = f"v[{vgpr}:{vgpr+1}]"
                        vgpr += 2
            processed.add(probe.name)
        # all rest is treated as no saving
    
    # Now add the instruction listening
    for idx in range(len(gcn_lines)):
        # ignore most of line that is a string!
        if type(gcn_lines[idx]) == Ref: # NOTE isinstance is slow?
            line: str         = gcn_lines[idx].line.strip()
            probe: Probe      = gcn_lines[idx].probe
            before_after: str = gcn_lines[idx].before_after
            # parse instruction operands, operands are separated by comma
            if ";" in line: line = line[:line.find(";")]
            tmp = line.split(",")
            operands: List[str] = []
            inst, op1 = tmp[0].split(" ")[0], tmp[0].split(" ")[-1] # 
            operands.append(op1)
            for t in tmp[1:]:
                operands.append(t.strip().split(" ")[0])
            snippet = probe.before if before_after else probe.after
            if "OUT" in snippet: snippet = snippet.replace("OUT", operands[0])
            if "IN1" in snippet: snippet = snippet.replace("IN1", operands[1])
            if "IN2" in snippet: snippet = snippet.replace("IN2", operands[2])
            if "IN3" in snippet: snippet = snippet.replace("IN3", operands[3])
            
            # Adding support for SAVE.u64 statement
            # NOTE for reading the probe afterwards
            snippet_lines = snippet.split("\n")
            for snippet_line_idx in range(len(snippet_lines)):
                snippet_line: str = snippet_lines[snippet_line_idx]
                if "SAVE" in snippet_line: # only one save, at the begin of line
                    save_lines = [] # start a new string
                    items = snippet_line[snippet_line.index("{") + 1:snippet_line.index("}")].split(",")
                    dtype = snippet_line[snippet_line.find("SAVE") + 5: snippet_line.find("SAVE") + 8]
                    if dtype == "u64":
                        for item_idx in range(len(items)):
                            item_val = probe.registers[items[item_idx].strip()]
                            save_lines.append(f"\tGLOBAL_STORE_DWORDX2 v[{probe.param_addr}], {item_val} \n\tV_LSHL_ADD_U64  v[{probe.param_addr}], 0, 8")
                    elif dtype == "u32":
                        for item_idx in range(len(items)):
                            item_val = probe.registers[items[item_idx].strip()]
                            save_lines.append(f"\tGLOBAL_STORE_DWORD v[{probe.param_addr}], {item_val} \n\tV_LSHL_ADD_U64  v[{probe.param_addr}], 0, 4")
                    else:
                        raise ValueError("Only Support Saving u32 / u64")
                    snippet_lines[snippet_line_idx] = "\n".join(save_lines)
            snippet = "\n".join(snippet_lines)
            for reg in probe.registers:
                if reg in snippet: 
                    snippet = snippet.replace(reg, probe.registers[reg])
            # Finally replace the line
            gcn_lines[idx] = snippet
    
    # NOTE we need to modify the number of registers used in metasection
    # 1. Mofify the kernarg_size .amdhsa_kernarg_size 28
    # 2. Modify the SGPRs used   .amdhsa_next_free_sgpr 12
    # 3. Modify the VGPRs used   .amdhsa_next_free_vgpr 9
    # Something might need .amdhsa_user_sgpr_count 2
    for idx in range(len(gcn_lines)):
        if ".amdhsa_kernarg_size" in gcn_lines[idx]:
            gcn_lines[idx] = f"\t.amdhsa_kernarg_size {param_off}"
        elif ".amdhsa_next_free_sgpr" in gcn_lines[idx]:
            gcn_lines[idx] = f"\t.amdhsa_next_free_sgpr {sgpr}"
        elif ".amdhsa_next_free_vgpr" in gcn_lines[idx]:
            gcn_lines[idx] = f"\t.amdhsa_next_free_vgpr {vgpr}"
    
    # NOTE also modify the amdgpu_metadata, after all, becomes 
    amdgpu_metadata["amdhsa.kernels"][0]['.sgpr_count'] = sgpr + sgpr_diff
    amdgpu_metadata['amdhsa.kernels'][0]['.vgpr_count'] = vgpr
    amdgpu_metadata["amdhsa.kernels"][0]['.kernarg_segment_size'] = param_off 
    amdgpu_metadata["amdhsa.kernels"][0]['.args'] += params
    # Finally finished, we might need to finalize the metadata
    return "\n".join(gcn_lines), probe_mem_sizes

    # NOTE also add new parameters


def assemble(workdir: str, name: str) -> None:
    """Assemble the GCN Asm (probed.asm) into Machine Code (probed.bin)
    NOTE AMD assembler command is part of Clang LLVM like 
    clang -cc1as -triple amdgcn-amd-amdhsa -filetype obj -target-cpu gfx942 
    -mrelocation-model pic -v -mllvm -amdgpu-early-inline-all=true -mllvm 
    -amdgpu-function-calls=false -o probed.bin probed.asm
    """
    # TODO need to locate the clang of ROCm, unlike like ptxas of unique name
    asm_path = os.path.join(workdir, name) + ".asm"
    bin_path = os.path.join(workdir, name) + ".bin" # target binary
    command = ["clang", '-cc1as', '-triple', 'amdgcn-amd-amdhsa', '-filetype=obj', 
               f"-target-cpu={get_arch()}", '-mrelocation-model=pic', '--verbose', 
               '-mllvm', '-amdgpu-early-inline-all=true',
               '-mllvm', '-amdgpu-function-calls=falsep', 
               asm_path, '-o', bin_path]
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

# ENTRY for this tool
if __name__ == "__main__":
    # no argparse as the CLI is straightforward
    workdir = sys.argv[1]     # directory contains original.bin
    kernel_name = sys.argv[2].encode('utf-8', 'ignore').decode('utf-8', 'ignore') # for possible case with multiple entry in one binary

    if len(sys.argv) > 3: # NOTE to facilitate debugging, not used in production
        probe_path = sys.argv[3]
        probe_toml = toml.load(probe_path)
    else: # the pass in production
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
            probes[key] =value

    probes = safe_load_probes(probes)
    # apply a 

    try:
        # first decompile binary to ptx
        asm = extract(workdir)
        # then truncate ptx for entry_name
        entry_section, meta_section = prune(asm, kernel_name)
        
        # split and process ptx lines and write kernel info
        params, kernel_name = parse_params()

        # basic logging
        print(kernel_name, file=log)

        # write pruned gcnasm to file
        meta_section = meta_section[: meta_section.index(".amdgpu_metadata") +  16] + yaml.safe_dump(amdgpu_metadata) + meta_section[meta_section.index(".end_amdgpu_metadata") - 1:]
        pruned_ptx = entry_section + "\n" + meta_section
        with open(os.path.join(workdir, "pruned.asm"), "w") as f:
            f.write(pruned_ptx)

        probed_asm, probe_mem_sizes = probing(entry_section, probes)

        # NOTE we need to update the meta_section we updated
        # TODO split into multiple lines
        meta_section = meta_section[: meta_section.index(".amdgpu_metadata") +  16] + yaml.safe_dump(amdgpu_metadata) + meta_section[meta_section.index(".end_amdgpu_metadata") - 1:]
        
        # merge global and func back
        probed_asm  = probed_asm + "\n" + meta_section

        # write probed gcnasm to file
        with open(os.path.join(workdir, "probed.asm"), "w") as f:
            f.write(probed_asm)
        
    except Exception as e:
        traceback.print_exc(file=log)
        exit(1)