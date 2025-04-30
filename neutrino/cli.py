"""CLI Entry for NEUTRINO: Fine-grained GPU Kernel Profiling via Programmable Probing"""

import subprocess
import os
import sys
import toml
import argparse


# Main Function, need a func to be referred by setup.py build system
def main(): 
    # NOTE READ CONFIG.TOML FOR DEFAULT SYSTEM CONFIGURATION
    # directory of this python file and other toolkits
    CURDIR = os.path.dirname(os.path.realpath(__file__))
    # directory of the neutrino internals
    NEUTRINO_BUILD_DIR: str = os.path.join(CURDIR, "build")
    NEUTRINO_PROBE_DIR: str = os.path.join(CURDIR, "probe")
    NEUTRINO_TOOLS_DIR: str = os.path.join(CURDIR, "tools")
    # load system configuration, generated in building
    config = toml.load(os.path.join(NEUTRINO_BUILD_DIR, "config.toml"))["system"] 
    # default configurations, can be overwritten by CLI parameters
    NEUTRINO_HOOK_DRIVER_NAME: str = config["NEUTRINO_HOOK_DRIVER_LIB_NAME"]
    NEUTRINO_REAL_DRIVER_DIR : str = config["NEUTRINO_REAL_DRIVER_LIB_DIR"]
    NEUTRINO_REAL_DRIVER_NAME: str = config["NEUTRINO_REAL_DRIVER_LIB_NAME"]
    NEUTRINO_MODE            : str = config["NEUTRINO_MODE"]
    # available built-in tools
    NEUTRINO_TOOLS = {tool[:-5] : tool for tool in os.listdir(NEUTRINO_TOOLS_DIR) if tool.endswith(".toml")}
    
    parser = argparse.ArgumentParser(
        prog='neutrino', usage='%(prog)s [options] command',
        description=f"""NOTE: Probes must be given via -p (--probe) option. Buit-in tools: {tuple(NEUTRINO_TOOLS.keys())}""", 
        epilog="Examples: `neutrino -t gmem_bytes python test/zero_.py`. Open issue(s) in https://github.com/neutrino-gpu/neutrino if encountered problems", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-p', '--probe', required=True, 
                        help="probe in form of path to the file")
    parser.add_argument('--tracedir', default="./trace", 
                        help="parent folder of traces")
    parser.add_argument('--driver', default=os.path.join(NEUTRINO_REAL_DRIVER_DIR, NEUTRINO_REAL_DRIVER_NAME),
                        help='path to the real cuda/hip driver shared library')
    parser.add_argument("--python", default=sys.executable, 
                        help='path to python executable used')
    parser.add_argument('--filter', 
                        help='filter OUT buggy kernels by (part of) name, split by :')
    parser.add_argument('-k', '--kernel', 
                        help='filter the kernel by (part of) name, split by :')
    parser.add_argument('--benchmark', action='store_true', 
                        help="enable benchmark mode to evaluate overhead w.r.t. the original kernel")
    parser.add_argument('--memusage', action='store_true', 
                        help="prevent the profiling and only measure the memory usage")
    # put command at the end of command
    parser.add_argument("command", nargs=argparse.REMAINDER) 
    # parse the arguments
    args = parser.parse_args()
    
    # same as this executable
    NEUTRINO_PYTHON: str = args.python # default to be this executable
    # directory to put the trace
    NEUTRINO_TRACEDIR: str = args.tracedir
    # filter of kernel
    NEUTRINO_FILTER: str = args.filter if args.filter is not None else ""
    NEUTRINO_KERNEL: str = args.kernel if args.kernel is not None else ""
    # Benchmark mode, will include an additional launch after the trace kernel
    # Used to measure the kernel-level slowdown of Neutrino, disabled by default
    NEUTRINO_BENCHMARK: str = str(int(args.benchmark))
    NEUTRINO_MEMUSAGE: str = str(int(args.memusage))
    # Path to the real driver
    NEUTRINO_REAL_DRIVER: str = args.driver
    # command to be executed
    command: str = args.command
    assert len(command) > 0, "Command must be specified"
    
    # Parse the PROBE
    NEUTRINO_PROBE: str = args.probe
    # No suffix := use built-in tools
    if not NEUTRINO_PROBE.endswith(".toml"): # No suffix := use built-in tools
        if  NEUTRINO_PROBE not in NEUTRINO_TOOLS:
            print(f"{NEUTRINO_PROBE} not in tools: {NEUTRINO_TOOLS}", file=sys.stderr)
            exit(-1)
        else:
            NEUTRINO_PROBE = os.path.join(NEUTRINO_TOOLS_DIR, NEUTRINO_TOOLS[NEUTRINO_PROBE])
    NEUTRINO_PROBE = toml.load(NEUTRINO_PROBE)
    # NOTE check if dynamic is True, shall have a specific keyword in top-level of probe
    NEUTRINO_DYNAMIC = "dynamic" in NEUTRINO_PROBE and NEUTRINO_PROBE["dynamic"] is True

    # a copied environment variables
    env = os.environ.copy()
    # configure Neutrino related environment variables
    env["NEUTRINO_REAL_DRIVER"]  = NEUTRINO_REAL_DRIVER
    env["NEUTRINO_DRIVER_NAME"]  = NEUTRINO_HOOK_DRIVER_NAME
    env["NEUTRINO_HOOK_DRIVER"]  = os.path.join(NEUTRINO_BUILD_DIR, NEUTRINO_HOOK_DRIVER_NAME)
    env["NEUTRINO_PYTHON"]       = NEUTRINO_PYTHON
    env["NEUTRINO_PROBING_PY"]   = os.path.join(NEUTRINO_BUILD_DIR, "process.py")
    env["NEUTRINO_FILTER"]       = NEUTRINO_FILTER
    env["NEUTRINO_KERNEL"]       = NEUTRINO_KERNEL
    env["NEUTRINO_TRACEDIR"]     = NEUTRINO_TRACEDIR
    env["NEUTRINO_PROBES"]       = toml.dumps(NEUTRINO_PROBE) # dump it to string
    # GNU LD_PRELOAD to overwrite dlopen, https://man7.org/linux/man-pages/man8/ld.so.8.html
    env["LD_PRELOAD"]            = os.path.join(NEUTRINO_BUILD_DIR, "preload.so")
    # Add to the LD_LIBRARY_PATH, this would overwrite ldconfig
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"]   = NEUTRINO_BUILD_DIR + ":" + env["LD_LIBRARY_PATH"]
    else:
        env["LD_LIBRARY_PATH"]   = NEUTRINO_BUILD_DIR
    # An Environmental Variable to enable the trace
    # NOTE some bugs here -> still working on
    env["NEUTRINO_ENABLE"] = "1"
    # An Environmental Variable to enable the benchmark mode
    env["NEUTRINO_BENCHMARK"] = NEUTRINO_BENCHMARK
    env["NEUTRINO_MEMUSAGE"]  = NEUTRINO_MEMUSAGE
    # An Environmental Variables to enable the debug mode -> more messages
    # env["NEUTRINO_VERBOSE"] = "1"
    if NEUTRINO_DYNAMIC:
        env["NEUTRINO_DYNAMIC"] = "1"

    # FIX for Triton
    if NEUTRINO_MODE == "CUDA":
        env["TRITON_LIBCUDA_PATH"] = NEUTRINO_BUILD_DIR
        env["NEUTRINO_PROBING_PY"] = os.path.join(NEUTRINO_PROBE_DIR, "cuda.py")
    elif NEUTRINO_MODE == "HIP":
        # NOTE There's a bug in Triton's impl here, for path we refer to the 
        # directory for ld.so to search, instead of spcific file name ...
        env["TRITON_LIBHIP_PATH"]  = os.path.join(NEUTRINO_BUILD_DIR, "libamdhip64.so") 
        env["NEUTRINO_PROBING_PY"] = os.path.join(NEUTRINO_PROBE_DIR, "hip.py")

    # start the program with new environment
    if len(command) > 0:
        proc = subprocess.Popen(command, env=env)
        proc.wait()

if __name__ == "__main__":
    main()