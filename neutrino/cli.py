"""System Entry for NEUTRINO: Fine-grained GPU Kernel Profiling via Programmable Probing

The system interface is similar to `valgrind` like: neutrino -t/-p <command>

TODO update to argparser.REMAINDER"""

import subprocess, os, sys, toml, psutil

def main():
    # directory of this python file and other toolkits
    CURDIR = os.path.dirname(os.path.realpath(__file__))
    # directory of the neutrino internals
    NEUTRINO_BUILD_DIR: str = os.path.join(CURDIR, "build")
    NEUTRINO_PROBE_DIR: str = os.path.join(CURDIR, "probe")
    NEUTRINO_TOOLS_DIR: str = os.path.join(CURDIR, "tools")
    # load system configuration, generated in building
    config = toml.load(os.path.join(NEUTRINO_BUILD_DIR, "config.toml")) # read the config.toml
    # default configurations, can be overwritten by CLI parameters
    NEUTRINO_HOOK_DRIVER_NAME: str = config["system"]["NEUTRINO_HOOK_DRIVER_LIB_NAME"]
    NEUTRINO_REAL_DRIVER_NAME: str = config["system"]["NEUTRINO_REAL_DRIVER_LIB_NAME"] 
    # Directory of the real cuda/hip lib.so
    NEUTRINO_REAL_DRIVER_DIR:  str = config["system"]["NEUTRINO_REAL_DRIVER_LIB_DIR"]
    NEUTRINO_MODE: str = config["system"]["NEUTRINO_MODE"]
    # same as this executable
    NEUTRINO_PYTHON: str = sys.executable # default to be this executable
    # directory to put the trace
    NEUTRINO_TRACEDIR: str = "./trace"
    # filter of kernel
    NEUTRINO_FILTER: str= ""
    # available built-in tools
    NEUTRINO_TOOLS = {tool[:-5] : tool for tool in os.listdir(NEUTRINO_TOOLS_DIR) if tool.endswith(".toml")}
    # Benchmark mode, will include an additional launch after the trace kernel
    # Used to measure the kernel-level slowdown of Neutrino, disabled by default
    NEUTRINO_BENCHMARK: str = "0"

    # info string to print if failed
    NEUTRINO_INFO_STRING = f"""usage: neutrino [-t/-p] command

    note:
      one of --tool (-t) or --probe (-p) must be given

    options:
      --tool -t   tool of neutrino                   required, available: {",".join(NEUTRINO_TOOLS.keys())}
      --probe -p  probe for neutrino in toml         required, file name to toml
      --trace-dir place to put trace,                default: {NEUTRINO_TRACEDIR} (inside workdir)
      --driver    path to real cuda/hip driver,      default: {os.path.join(NEUTRINO_REAL_DRIVER_DIR, NEUTRINO_REAL_DRIVER_NAME)} (detected)
      --python    path to python executable          default: {NEUTRINO_PYTHON} (detected)
      --filter    filter out buggy kernels by name   default: {NEUTRINO_FILTER}
      --benchmark run original kernel to measure overhead, default: disabled
      --help      print help message"""

    if len(sys.argv) == 1 or sys.argv[1] == "--help":
        print(NEUTRINO_INFO_STRING)
        exit(0)

    # command to be executed
    command: str = ""
    # one of following is required
    neutrino_tool = ""
    neutrino_probe_file = ""
    NEUTRINO_REAL_DRIVER = ""

    # a manual argparser as we want all following argument integrated
    i: int = 1
    while i < len(sys.argv): # 1st argument is name of script, ignored
        if sys.argv[i] == "-p" or sys.argv[i] == "--probe":
            neutrino_probe_file = sys.argv[i + 1]
            i += 2
        if sys.argv[i] == "-t" or sys.argv[i] == "--tool":
            neutrino_tool = sys.argv[i + 1]
            i += 2
        if sys.argv[i] == "--trace-dir":
            NEUTRINO_TRACEDIR = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--driver":
            NEUTRINO_REAL_DRIVER = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--python":
            NEUTRINO_PYTHON = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--filter":
            NEUTRINO_FILTER = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--benchmark": 
            NEUTRINO_BENCHMARK = "1"
            i += 1
        else:
            command = sys.argv[i:]
            break

    # try to load the probe from file or available toolkit
    if len(neutrino_probe_file) == 0:
        # check if neutrino_tool is given
        if len(neutrino_tool) == 0 or neutrino_tool not in NEUTRINO_TOOLS:
            print(NEUTRINO_INFO_STRING)
            exit(0)
        else:
            neutrino_probe_file = os.path.join(NEUTRINO_TOOLS_DIR, NEUTRINO_TOOLS[neutrino_tool])
    neutrino_probe = toml.load(neutrino_probe_file)

    # default configuration
    if NEUTRINO_REAL_DRIVER == "":
        NEUTRINO_REAL_DRIVER = os.path.join(NEUTRINO_REAL_DRIVER_DIR, NEUTRINO_REAL_DRIVER_NAME)

    # a copied environment variables
    env = os.environ.copy()
    # configure Neutrino related environment variables
    env["NEUTRINO_REAL_DRIVER"]  = NEUTRINO_REAL_DRIVER
    env["NEUTRINO_DRIVER_NAME"]  = NEUTRINO_HOOK_DRIVER_NAME
    env["NEUTRINO_HOOK_DRIVER"]  = os.path.join(NEUTRINO_BUILD_DIR, NEUTRINO_HOOK_DRIVER_NAME)
    env["NEUTRINO_PYTHON"]       = NEUTRINO_PYTHON
    env["NEUTRINO_PROBING_PY"]   = os.path.join(NEUTRINO_BUILD_DIR, "process.py")
    env["NEUTRINO_FILTER"]       = NEUTRINO_FILTER
    env["NEUTRINO_TRACEDIR"]     = NEUTRINO_TRACEDIR
    env["NEUTRINO_PROBES"]       = toml.dumps(neutrino_probe) # dump it to string
    # GNU LD_PRELOAD to overwrite dlopen, https://man7.org/linux/man-pages/man8/ld.so.8.html
    env["LD_PRELOAD"]          = os.path.join(NEUTRINO_BUILD_DIR, "preload.so")
    # Add to the LD_LIBRARY_PATH, this would overwrite ldconfig
    env["LD_LIBRARY_PATH"]     = NEUTRINO_BUILD_DIR + ":" + env["LD_LIBRARY_PATH"]
    # An Environmental Variable to enable the trace
    # NOTE some bugs here -> still working on
    env["NEUTRINO_ENABLE"] = "1"
    # An Environmental Variable to enable the benchmark mode
    env["NEUTRINO_BENCHMARK"] = NEUTRINO_BENCHMARK
    # An Environmental Variables to enable the debug mode -> more messages
    # env["NEUTRINO_VERBOSE"] = "1"
    # NOTE a fix for Triton
    if NEUTRINO_MODE == "CUDA":
        env["TRITON_LIBCUDA_PATH"] = NEUTRINO_BUILD_DIR
        env["NEUTRINO_PROBING_PY"]   = os.path.join(NEUTRINO_PROBE_DIR, "cuda.py")
    elif NEUTRINO_MODE == "HIP":
        env["TRITON_LIBHIP_PATH"] = NEUTRINO_BUILD_DIR
        env["NEUTRINO_PROBING_PY"]   = os.path.join(NEUTRINO_PROBE_DIR, "hip.py")

    # start the program with new environment
    if len(command) > 0:
        proc = subprocess.Popen(command, env=env)
        proc.wait()
    
if __name__ == "__main__":
    main()