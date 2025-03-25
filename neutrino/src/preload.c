/**
 * A customized preload shared library to redirect driver (libcuda.so)
 * 1. redirect driver interaction so don't need to modify host env (/usr/lib/...)
 * 2. filter out proprietary product such as cuBLAS to conform updated NVIDIA EULA
 * 
 * @note Proprietary NVIDIA Softwares includes:
 * cublas/curand/cufft/cusparse/cusolver/optix/...
 * but at most case, only cuBLAS if you used PyTorch or other AIML workload
 */
 #define _GNU_SOURCE
 #include <dlfcn.h>     // for dynamic library
 #include <stdio.h>     // for I/O
 #include <string.h>    // for strcmp
 #include <execinfo.h>  // for backtrace and backtrace_symbols
 #include <stdlib.h>    // for malloc and free
 #include <time.h>      // for timing terms
 
 #ifndef STACK_TRACE_SIZE
 #define STACK_TRACE_SIZE 5
 #endif
 
 #ifndef DL_VERBOSE
 #define DL_VERBOSE 0
 #endif
 
 // Pointer to GLIBC dlopen function, by dlsym(RTLD_NEXT, "dlopen")
 static void* (*real_dlopen)(const char *filename, int flags) = NULL;
 
 static char* NEUTRINO_REAL_DRIVER = NULL;
 static char* NEUTRINO_HOOK_DRIVER = NULL;
 static char* NEUTRINO_DRIVER_NAME = NULL;
 
 /**
  * Provides a hook on both statically or dynamically loading shared library
  * by overwriting dlopen with the same signature as GLIBC dlopen
  * @cite https://man7.org/linux/man-pages/man3/dlopen.3.html
  * 
  * This will leads to 2 dlopen function in the search space of executable:
  * 1. our dlopen as follows, will be chosen automatically as LD_PRELOAD
  * 2. standard c library's dlopen, will be masked but still can be referred
  *    if RTLD_NEXT flag is specified
  * 
  * This ensure FULL COVERAGE because dlopen must be linked statically to 
  * enable dynamic linking (via dlopen) -> a puzzle in UNIX-like OS
  */
 void* dlopen(const char *filename, int flags) {
     // original (GLIBC) dlopen still exists in search space 
     // but is less prefered as LD_PRELOAD mask it
     // using dlsym with RTLD_NEXT we can extract GLIBC dlopen.
     if (!real_dlopen) 
         real_dlopen = dlsym(RTLD_NEXT, "dlopen");
     
     if (!NEUTRINO_DRIVER_NAME) {
         NEUTRINO_DRIVER_NAME = getenv("NEUTRINO_DRIVER_NAME");
         fprintf(stderr, "[info] NEUTRINO_DRIVER_NAME: %s\n", NEUTRINO_DRIVER_NAME);
     }   
 
     if (filename != NULL && (strstr(filename, NEUTRINO_DRIVER_NAME) != NULL)) {
         // Check if it's libcublas.so backtrace
         // @see https://man7.org/linux/man-pages/man3/backtrace.3.html
         void* array[STACK_TRACE_SIZE];
         int size       = backtrace(array, STACK_TRACE_SIZE);
         char** strings = backtrace_symbols(array, size);
         int call_from_cublas = 0;
         if (strings != NULL){
             for (int i = 0; i < size; i++) {
                 // we will add ALL Nvidia Propietray Product here
                 if (strstr(strings[i], "libcublas") != NULL) {
                     call_from_cublas = 1;
                     break;
                 }
             }
         }
         free(strings);
         void* ptr;
         if (call_from_cublas) {
             if (NEUTRINO_REAL_DRIVER == NULL) {
                 NEUTRINO_REAL_DRIVER = getenv("NEUTRINO_REAL_DRIVER");
                 if (NEUTRINO_REAL_DRIVER == NULL) { // fault
                     fprintf(stderr, "[error] NEUTRINO_REAL_DRIVER not set\n");
                     exit(1);
                 }
             }
             ptr = real_dlopen(NEUTRINO_REAL_DRIVER, flags);
             struct timespec ts;
             clock_gettime(CLOCK_REALTIME, &ts);
             long long time = ts.tv_nsec + ts.tv_sec * 1e9;
             printf("[info] %lld cublas use real: %s %p %d\n", time, NEUTRINO_REAL_DRIVER, ptr, flags);
             fflush(stdout);
         } else {
             char* NEUTRINO_HOOK_DRIVER = getenv("NEUTRINO_HOOK_DRIVER");
             if (NEUTRINO_HOOK_DRIVER == NULL) {
                 fprintf(stderr, "[error] NEUTRINO_HOOK_DRIVER not set\n");
                 ptr = real_dlopen(filename, flags); // try to backup
             }
             ptr = real_dlopen(NEUTRINO_HOOK_DRIVER, flags);
             if (DL_VERBOSE) {
                 struct timespec ts;
                 clock_gettime(CLOCK_REALTIME, &ts);
                 long long time = ts.tv_nsec + ts.tv_sec * 1e9;
                 printf("[info] %lld use hooked: %s %p %d\n", time, NEUTRINO_HOOK_DRIVER, ptr, flags);
                 fflush(stdout);
             }
         }
         return ptr;
     } else { // not interested, just let them go via loading the correct
         // Call the original dlopen
         void* ptr = real_dlopen(filename, flags);
         // Print the name of the module being loaded
         if (DL_VERBOSE) {
             struct timespec ts;
             clock_gettime(CLOCK_REALTIME, &ts);
             long long time = ts.tv_nsec + ts.tv_sec * 1e9;
             printf("[info] %lld Loading: %s %p %d\n", time, filename, ptr, flags);
             fflush(stdout);
         }
         return ptr;
     }
 }