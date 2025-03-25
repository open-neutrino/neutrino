import os
import sys
import subprocess
from pprint import pprint
import toml

CURDIR = os.path.dirname(os.path.realpath(__file__))

def check_command(cmd: str):
    try:
        _ = subprocess.run([cmd], stdout=subprocess.PIPE,  
            stderr=subprocess.PIPE, text=True, check=True)
        return True
    except FileNotFoundError:
        return False

# Use rocm-smi or nvidia-smi to detect the platform
if check_command("rocm-smi"):
    NEUTRINO_MODE = "HIP"
    NEUTRINO_DRIVER_HEADER_NAME = "hip/hip_runtime_api.h"
    NEUTRINO_IMPL_SRC = "hip.c"
    NEUTRINO_HOOK_DRIVER_LIB_NAME = "libamdhip64.so.6"
    NEUTRINO_DRIVER_HEADER_SEARCH_PATH = [
        "/opt/rocm/include/", # AFAIK, add if new path is met
    ]
elif check_command("nvidia-smi"):
    NEUTRINO_MODE = "CUDA"
    NEUTRINO_DRIVER_HEADER_NAME = "cuda.h"
    NEUTRINO_IMPL_SRC = "cuda.c"
    NEUTRINO_HOOK_DRIVER_LIB_NAME = "libcuda.so.1"
    NEUTRINO_DRIVER_HEADER_SEARCH_PATH = [
        "/usr/local/cuda/targets/x86_64-linux/include/", # for x86
        "/usr/local/cuda/targets/aarch64-linux/include/", # for ARM
        # add if missed
    ]
else:
    raise RuntimeError("ONLY SUPPORT CUDA and HIP(AMD-ONLY)")

# Internal Configurations
SRC_DIR = os.path.join(CURDIR, "src")
BUILD_DIR = os.path.join(CURDIR, "build")
CC = "cc" # NOTE don't use nvcc or hipcc, need gcc or clang
PY = sys.executable

for dir_ in NEUTRINO_DRIVER_HEADER_SEARCH_PATH:
    try:
        if NEUTRINO_DRIVER_HEADER_NAME in os.listdir(dir_):
          break
    except:
        pass
# NOTE this will be written in config.toml
NEUTRINO_DRIVER_HEADER_DIR = dir_

# NOTE Locate Driver Shared Library
# inspired by: https://github.com/triton-lang/triton/commit/58c54455ffa691be64f90f4e856501162373572c#diff-3d1f29795218f61553ab953426c15fa1e4162b224405b85529022293054da57aR25
# but we need to further locate the real driver library
libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
locs = [line.split()[-1] for line in libs.splitlines() if NEUTRINO_HOOK_DRIVER_LIB_NAME in line]
env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
if env_ld_library_path and not locs:
    locs = [os.path.join(dir_, NEUTRINO_HOOK_DRIVER_LIB_NAME) for dir_ in env_ld_library_path.split(":")
            if os.path.exists(os.path.join(dir_, NEUTRINO_HOOK_DRIVER_LIB_NAME))]

# try to locate the pointed path
NEUTRINO_REAL_DRIVER_LIB_NAME = ""
NEUTRINO_REAL_DRIVER_LIB_DIR = ""
real_libs = []
for loc in locs:
    real_lib = os.readlink(loc)
    if "lib32" not in loc and "lib32" not in real_lib: # NOTE fix lib32 bug
        if not real_lib.startswith("/"):
            NEUTRINO_REAL_DRIVER_LIB_DIR = os.path.dirname(loc)
            NEUTRINO_REAL_DRIVER_LIB_NAME = real_lib
        else:
            NEUTRINO_REAL_DRIVER_LIB_DIR = os.path.dirname(loc)
            NEUTRINO_REAL_DRIVER_LIB_NAME = os.path.basename(loc)

print(NEUTRINO_REAL_DRIVER_LIB_DIR, NEUTRINO_REAL_DRIVER_LIB_NAME, file=sys.stderr)

# NOTE call parse.py
cmd = [PY, os.path.join(SRC_DIR, "parse.py"), 
    os.path.join(NEUTRINO_DRIVER_HEADER_DIR, NEUTRINO_DRIVER_HEADER_NAME), 
    os.path.join(NEUTRINO_REAL_DRIVER_LIB_DIR, NEUTRINO_REAL_DRIVER_LIB_NAME)]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE compile cuda.c
cmd = [CC, os.path.join(SRC_DIR, NEUTRINO_IMPL_SRC), "-fPIC", "-shared", "-ldl", "-O3", 
        "-I", NEUTRINO_DRIVER_HEADER_DIR, "-o", os.path.join(BUILD_DIR, NEUTRINO_HOOK_DRIVER_LIB_NAME)]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE compile preload.c
cmd = [CC, os.path.join(SRC_DIR, "preload.c"), "-fPIC", "-shared", "-O3", 
       "-o", os.path.join(BUILD_DIR, "preload.so")]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE create a symbolic link like libcuda.so -> libcuda.so.1
# TODO verify if this is need 
cmd = ["ln", "-sf", NEUTRINO_HOOK_DRIVER_LIB_NAME, 
       os.path.join(BUILD_DIR, NEUTRINO_HOOK_DRIVER_LIB_NAME[:NEUTRINO_HOOK_DRIVER_LIB_NAME.index("so") + 2])]
print(" ".join(cmd), file=sys.stderr)
subprocess.check_output(cmd)

# NOTE dump system configuration for CLI usage
config = {}
config["system"] = {
    "NEUTRINO_MODE"                 : NEUTRINO_MODE,
    "NEUTRINO_DRIVER_HEADER_NAME"   : NEUTRINO_DRIVER_HEADER_NAME,
    "NEUTRINO_DRIVER_HEADER_DIR"    : NEUTRINO_DRIVER_HEADER_DIR,
    "NEUTRINO_HOOK_DRIVER_LIB_NAME" : NEUTRINO_HOOK_DRIVER_LIB_NAME,
    "NEUTRINO_REAL_DRIVER_LIB_NAME" : NEUTRINO_REAL_DRIVER_LIB_NAME,
    "NEUTRINO_REAL_DRIVER_LIB_DIR"  : NEUTRINO_REAL_DRIVER_LIB_DIR,
}
toml.dump(config, open(os.path.join(BUILD_DIR, "config.toml"), "w"))

print("Build Success, Configuration")
print("============================")
pprint(config)