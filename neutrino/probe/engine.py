"""Neutrino Probing Engine Protocol

NOTE just a protocol for developers, don't import / export"""

from dataclasses import dataclass
from typing import List, Optional, Dict

__all__ = ["Probe", "Ref", "safe_load_probes", "TRACE_READING_CODE_PY"]

@dataclass
class Probe:
    """Neutrino Probes Data Structure"""
    name: str                       # name is the key in TOML
    position: List[str]             # := tracepoint in the paper
    registers: Optional[Dict] = None # register name -> dtype mapping
    datamodel: Optional[str]  = None # 
    no_bytes:  Optional[str]  = None # number of bytes per thread
    before:    Optional[str]  = None # snippet inserted before, one of before and after shall be given
    after:     Optional[str]  = None # snippet inserted after,  one of before and after shall be given
    cap:       Optional[int]  = -1   # only event-constant has the value
    no_pred:   Optional[bool] = False # NOTE ignore the original predictive of instruction -> Not recommended
    
@dataclass
class Ref:
    """Reference for replacement"""
    line: str          # Original line
    probe: str         # Probe name for matchine
    before_after: bool # True if before and False if after -> to distinguish which snippet is used

def safe_load_probes(raw_probes: Dict[str, dict]) -> List[Probe]:
    """Turn Raw Probe (dict from toml) into dataclass, also apply restrictions
    NOTE Position must be given, datamodel must be given if want to save, one of before and after must be given
    """
    probes: List[Probe] = []
    for probe_name, raw_probe in raw_probes.items():
        # first validate the 
        keys = raw_probe.keys()
        assert "position" in keys, f"[error] {probe_name} has no position (required)"
        # assert "datamodel" in keys, f"[error] "
        assert "before" in keys or "after" in keys, f"[error] {probe_name} is empty, one of before or start shall be given"
        # gradually process all things
        if "datamodel" in keys:
            datamodel: str = raw_probe["datamodel"]
            if datamodel.startswith("thread") or datamodel.startswith("warp"):
                tmp = datamodel.split(":")
                datamodel, no_bytes = tmp[0], tmp[1]
                cap = 1 if len(tmp) <= 2 else tmp[2] # by default save one item
            else:
                raise ValueError(f"[error] unsupported datamodel {datamodel}")
        else:
            datamodel, no_bytes, cap = None, None, -1 # all marked with None
        probe = Probe(name=probe_name, 
                      position=raw_probe["position"].split(":") if ":" in raw_probe["position"] else [raw_probe["position"], ],
                      registers=raw_probe['registers'] if "registers" in keys else None,
                      datamodel=datamodel,
                      no_bytes=no_bytes,
                      before=raw_probe["before"] if "before" in keys else None,
                      after=raw_probe["after"] if "after" in keys else None,
                      no_pred=raw_probe["no_pred"] if "no_pred" in keys else False,
                      cap = cap)
        probes.append(probe)
    return probes

# NOTE Template for Generating Trace Reading Code
TRACE_READING_CODE_PY = """# Neutrino Auto-Generated Code for Trace Reading
import struct
from typing import NamedTuple, List, Tuple
from neutrino import TraceHeader, TraceSection

class {probe_name}(NamedTuple):
{saved_content}

def parse(path: str) -> Tuple[TraceHeader, List[TraceSection], List[List[{probe_name}]]]:
    with open(path, "rb") as f:
        gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, numProbes = struct.unpack("iiiiiiii", f.read(32))
        header: TraceHeader = TraceHeader(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, numProbes)
        assert header.numProbes == 1 # currently only one saving probe is supported
        sections: List[TraceSection] = []
        for _ in range(header.numProbes):
            size, offset = struct.unpack("QQ", f.read(16))
            sections.append(TraceSection(size, offset))
        gridSize = header.gridDimX * header.gridDimY * header.gridDimZ
        blockSize = header.blockDimX * header.blockDimY * header.blockDimZ
        records: List[List[{probe_name}]] = []
        for i in range(gridSize):
            records.append([])
            for j in range(blockSize{warp_div}):
                {saved_reading} = struct.unpack("{format_string}", f.read({reading_bytes}))
                records[i].append({probe_name}({saved_reading}))
        return header, sections, records
"""


# NOTE following is just protocol, real implementation is platform-specific, 
# developers can extend theirs but please implement the following for integrity

def get_arch() -> str:
    """get architecture for assembler"""
    ...

def dump(workdir: str, name: str, suffix: str) -> str:
    """call objdump to extract assembly from binary"""
    ...

def prune(ptx: str, entry_name: str):
    """Prune Assembly to specific entry_name"""
    ...

def probing(asm: str, probes: List[Probe]):
    """Probe the probes into asm"""
    ...

def assemble(workdir: str, name: str):
    """call assembler to turn assembly to machine code"""
    ...


def write_kernel_info(name: str, params, probe_mem_sizes: List[int], 
    workdir: str, analyze_hook: str = "", file_name: str = "kernel.info"): 
    """write kernel info for hook driver to read back"""
    ...