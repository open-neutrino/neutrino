"""Code Generator for Unmodified Driver Functions and Symbols 

NOTE Support CUDA / ROCm

How it works?
All Driver symbols are exposed via <cuda.h> with signatures like:
CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev);

By parsing this symbol allow us to get and generate a mask driver like:
```c
CUresult (*real_cuDeviceGetName)(char*, int, CUdevice) = NULL;

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    if (shared_lib == NULL) { init(); }
    CUresult ret = real_cuDeviceGetName(name, len, dev);
    return ret;
}
```

But CUDA Symbols might be versioned slightly, like cuMemAlloc now has two
symbol cuMemAlloc (actually a macro) and cuMemAlloc_v2(real but not in cuda.h)
so we need to check libcuda.so to compromise the missed symbol (assume the
signature identical to unversioned.

NOTE You may see warnings functions such as lacking symbols for 
cuGL, cudbg, cuEGL, cuMem, cuProfiler, cuVDP, but they can be ignored mostly
"""
from typing import NamedTuple, List, Tuple, Dict
import sys
import os
import subprocess

LIB_PATH: str = None
HEADER_PATH: str = None
UNMODIFIED_C_NAME = "unmodified.c"
SIGNATURE_C_NAME  = "signature.c"

# NOTE list of modified symbols -> handled by modified.c unless code generation
MODIFIED_FUNCTIONS: Dict[str, List[str]] = {
    "cu": ["cuMemAlloc_v2", "cuMemFree_v2", "cuModuleLoadData", "cuModuleGetFunction", 
        "cuKernelGetFunction", "cuLibraryGetKernel", "cuLibraryGetModule",
        "cuLibraryLoadData", "cuLaunchKernel", "cuGetProcAddress_v2",  "cuGetProcAddress", 
        "cuModuleLoadDataEx", "cuModuleLoad", "cuModuleLoadFatBinary", "cuLaunchKernelEx"],
    "hip": ["hipModuleLoadData", "hipModuleLoadDataEx", "hipModuleGetFunction", 
        "hipModuleLaunchKernel", "hipMalloc", "hipFree", "hipModuleLoad", "hipKernelNameRef",
        "hipEventRecord", "hipGetErrorName", "hipGetErrorString", "hipApiName", 
        "hipKernelNameRefByPtr", "hip_init", "hipStreamGetCaptureInfo_v2_spt"] 
}

CODEGEN_TEMPLATE: Dict[str, str] = {
    "cu": """
CUresult {func_name}({param_list}) {{
    if (shared_lib == NULL)  {{ init(); }}
    CUresult err = real_{func_name}({param_val_list}); // call the real
    if (VERBOSE)  {{ fprintf(event_log, "[info] {func_name} %d\\n", err); }}
    return err;
}}""",
    "hip": """
hipError_t {func_name}({param_list}) {{
    if (shared_lib == NULL)  {{ init(); }} 
    hipError_t err = real_{func_name}({param_val_list}); // call the real
    if (VERBOSE) {{ fprintf(event_log, "[info] {func_name} %d\\n", err); }}
    return err;
}}"""
}

SIGNATURE_TEMPLATE = {
    "cu": 'CUresult (*real_{func_name})({param_list}) = NULL;',
    "hip":  'hipError_t (*real_{func_name})({param_list}) = NULL;'
}

INIT_TEMPLATE = '    real_{func_name} = dlsym(shared_lib, "{func_name}");'

IDENTIFIERS = {
    "cu": "CUresult CUDAAPI",
    "hip":  "hipError_t"
}

class Parameter(NamedTuple):
    type_name: str
    var_name: str

class Signature(NamedTuple):
    func_name: str
    params: List[Parameter]

class VersionedSymbol(NamedTuple):
    name: str
    version: str

def parse_parameter(param: str) -> Parameter:
    # Split the parameter into type and name
    param_parts = param.rsplit(' ', 1)
    # FIX some macro like __dparm
    if len(param_parts) == 2 and param_parts[1].startswith("__"):
        param_parts = param_parts[0].strip().rsplit(' ', 1)
    if len(param_parts) == 2:
        type_name = param_parts[0].strip()
        var_name: str = param_parts[1].strip()
        if var_name.startswith("*"): # avoid ptr * at variable side
            num_star = var_name.rfind("*") + 1
            type_name = type_name + "*" * num_star
            var_name = var_name[num_star:]
    else:
        type_name = param_parts[0].strip()
        var_name = ''
    
    return Parameter(type_name, var_name)

def parse_function_signature(signature: str) -> Signature:
    # Remove the trailing semicolon
    signature = signature.strip().rstrip(';\n')
    
    # Find the opening parenthesis for parameters
    paren_index = signature.find('(')
    func_name = signature[:paren_index].strip()
    space_index = func_name.rfind(' ')
    func_name = func_name[space_index + 1:]
    if "\n" in func_name:
        space_index = func_name.rfind('\n')
        func_name = func_name[space_index + 1:]

    # Extract parameters
    params_str = signature[paren_index + 1:].strip()
    params_str = params_str[:-1].strip()  # Remove closing parenthesis

    # Parse parameters
    param_list: List[Parameter] = []
    if params_str:
        # Split by commas, considering pointer types
        param_parts = []
        current_param = ''
        depth = 0
        
        for char in params_str:
            if char == ',' and depth == 0:
                param_parts.append(current_param.strip())
                current_param = ''
            else:
                current_param += char
                if char == '<':
                    depth += 1
                elif char == '>':
                    depth -= 1
            
        # Add the last parameter
        if current_param:
            param_parts.append(current_param.strip())

        for param in param_parts:
            param = param.strip()
            if param:
                param_list.append(parse_parameter(param))

    return Signature(func_name, param_list)

def parse_symbol(nm_line: str) -> str:
    if len(nm_line.strip()) != 0:
        symbol = nm_line.rsplit(" ", 1)[1]
        if "@" in symbol: # NOTE remove version tag @
            symbol = symbol.split("@")[0]
        return symbol
    else:
        return ""
    
def parse_version_symbol(symbol: str) -> Tuple[str, str]:
    if symbol[0] != "_" and "_" in symbol: # FIX __hip
        name, version = symbol.split("_", 1)
        return name, "_"+version
    else:
        return symbol, ""

def gencode(signature: Signature, template: str) -> str:
    param_list = []
    param_type_list = []
    param_val_list = []
    for param in signature.params:
        param_type_list.append(param.type_name)
        if "__" in param.var_name:
            print(param)
        param_val_list.append(param.var_name)
        param_list.append(param.type_name + " " + param.var_name)
    return template.format(
        func_name = signature.func_name,
        param_list = ", ".join(param_list),
        param_type_list = ", ".join(param_type_list),
        param_val_list = ", ".join(param_val_list)
    )

def gensignature(signature: Signature, template: str) -> str:
    param_list = []
    param_type_list = []
    param_val_list = []
    for param in signature.params:
        param_type_list.append(param.type_name)
        param_val_list.append(param.var_name)
        param_list.append(param.type_name + " " + param.var_name)
    return template.format(
        func_name = signature.func_name,
        param_list = ", ".join(param_list)
    )

def geninit(signature: Signature) -> str:
    return INIT_TEMPLATE.format(func_name = signature.func_name)

if __name__ == "__main__":
    # parse cli param if given, usage is python parse.py CUDA_HEADER_PATH, CUDA_LIB_PATH
    if len(sys.argv) >= 3:
        HEADER_PATH = sys.argv[1]
        LIB_PATH = sys.argv[2]
    else:
        print("Usage: python parse.py <HEADER_PATH> <LIB_PATH>")
        exit(1)

    unmodified_c = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), UNMODIFIED_C_NAME), "w")
    signature_c  = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), SIGNATURE_C_NAME),  "w")

    print(f"[INFO] use {HEADER_PATH} and {LIB_PATH}", file=sys.stderr)

    if HEADER_PATH.endswith("cuda.h") and "libcuda.so" in LIB_PATH: 
        target = "cu"
    elif HEADER_PATH.endswith("hip_runtime_api.h") and "libamdhip64.so" in LIB_PATH: 
        target= "hip"
    else: # can add more target for the future
        raise ValueError(f"[error] {LIB_PATH} is not supported")

    signatures: List[Signature] = []

    # parse cuda.h to extract cuda headers
    with open(HEADER_PATH, "r") as header_file:
        headers = header_file.readlines()
        idx = 0
        start_idx, ending_idx = 0, 0
        
        if target == "cu":
            for idx in range(len(headers)):
                # BUG FIX weird definition in cuda.h
                if "#define CUDAAPI" in headers[idx]:
                    start_idx = idx
                elif "CUDA API versioning support" in headers[idx]:
                    ending_idx = idx
                    break
        elif target == "hip":
            for idx in range(len(headers)):
                if " *  @defgroup API HIP API" in headers[idx]:
                    start_idx = idx
                elif '} /* extern "c" */' in headers[idx]:
                    ending_idx = idx
                    break
        idx = start_idx
        # print(f"start: {start_idx}, end: {ending_idx}", file=sys.stderr)
        identifier = IDENTIFIERS[target]
        while idx < ending_idx:
            if identifier in headers[idx] and "typedef" not in headers[idx]:
                end_idx = idx + 1
                if ";" in headers[idx]: # a full signature
                    parsed_signature = parse_function_signature(headers[idx])
                    signatures.append(parsed_signature)
                else:
                    while ";" not in headers[end_idx]:
                        end_idx += 1
                    parsed_signature = parse_function_signature("".join(headers[idx:end_idx+1]))
                    signatures.append(parsed_signature)
                idx = end_idx
            else:
                idx += 1


    # extract missing symbols from libcuda.so
    symbols_so: List[str] = []
    result = subprocess.run(["nm", "-D", LIB_PATH], stdout=subprocess.PIPE, text=True)
    so_log = result.stdout.split("\n")
    for line in so_log:
        symbol = parse_symbol(line)
        if symbol.startswith(target): # target is also prefix of API name :)
            symbols_so.append(symbol)
    
    # get the symbols missed in our cuda lib
    parsed_symbols = {signature.func_name: signature for signature in signatures}
    missed_symbols = [symbol for symbol in symbols_so if symbol not in parsed_symbols]
    print(f"[INFO] Extract {len(signatures)} Symbols from {HEADER_PATH}", file=sys.stderr)

    for symbol in missed_symbols:
        # try to extract symbol and version
        raw_symbol_name, version = parse_version_symbol(symbol)
        # check if raw_symbol in parsed_symbols
        if raw_symbol_name in parsed_symbols:
            # versioned symbol share the same parameter list
            raw_symbol = parsed_symbols[raw_symbol_name]
            signatures.append(Signature(func_name=symbol, params=raw_symbol.params))
        else:
            print(f"[WARNING] can't resolve {symbol}", file=sys.stderr)
    # remove redundent symbols, don't know why
    unique_names = set()
    unique_signatures = []
    for signature in signatures:
        if signature.func_name not in unique_names:
            unique_names.add(signature.func_name)
            unique_signatures.append(signature)

    print(f"[INFO] Resolved {len(signatures)} Symbols from {len(symbols_so)} Symbols in {LIB_PATH}", file=sys.stderr)

    print("// auto-generated by parse.py, used with modified.c", file=unmodified_c)
    print("// auto-generated by parse.py, used with modified.c", file=signature_c)
    inits = []
    for signature in unique_signatures:
        if signature.func_name not in MODIFIED_FUNCTIONS[target]: # REMOVE MODIFIED
            code = gencode(signature, CODEGEN_TEMPLATE[target])
            print(code, file=unmodified_c)
            signautre_ = gensignature(signature, SIGNATURE_TEMPLATE[target])
            print(signautre_, file=signature_c)
            init_ = geninit(signature)
            inits.append(init_)
    
    print("\nstatic void init_unmodified(void) {", file=signature_c)
    print("\n".join(inits), file=signature_c)
    print("}", file=signature_c)
