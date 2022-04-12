# digHolo <img src="https://github.com/joelacarpenter/digHolo/blob/doc/DigHoloLogo_64x64.png" height="32" width="32"><br>

High-speed library for off-axis digital holography and Hermite-Gaussian decomposition
# User Guide
If this library was useful to you, please cite the [User Guide](https://arxiv.org/abs/2204.02348) [arXiv:2204.02348] <br>
Full function documentation, benchmarks and background information available in the User Guide.
<br>
# Video Tutorial
[digHolo library introduction (YouTube)](https://youtu.be/rQMfWO2MQJQ)<br>
[off-axis digital holography tutorial (YouTube)](https://youtu.be/9hgQyx1He_U)<br>
# Installation and setup
## System Requirements
CPU : x86-64 with AVX2 and FMA3 support (processors after ~2015) 
## Dependencies
 ### FFTW 3 
https://www.fftw.org/ (Used for all FFTs and DCTs).
### Intel MKL (or other BLAS/LAPACK implementation) 
Specifically, the functions cgesvd, sgels, cgemv, cgemm.
 Linking against openBLAS https://www.openblas.net/ has also been tested. Comment out "#define MKL_ENABLE" in digHolo.cpp  
### SVML (optional) 
Trigonometric functions will be implemented using SVML if available (e.g. intel compiler and Microsoft compiler > VS2019). Otherwise, will default to hand-coded vectorised fast-math implementations. 

## Compilation
The library can be compiled either as a dll/shared object, or as an executable.
The executable can be called as 'digHolo.exe {settingsFile}', where {SettingsFile} is the filename of a tab-delimited text file containing configuration information. 
The examples given link against FFTW and Intel MKL.
 digHolo is written in C++11 and requires support for AVX2 and FMA3 instruction sets.

**Linux (gcc or icc)** <br>
 *Shared Object*<br>
  g++ -std=c++11 -O3 -mavx2 -mfma -fPIC -shared digHolo.cpp -o libdigholo.so<br>
 
*Executable*<br>
  g++ -std=c++11 -O3 -mavx2 -mfma digHolo.cpp -o digHolo.exe -lfftw3f -lfftw3f_threads -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl <br>

**Windows** <br>
 Example Visual Studio solution files are provided.

### Notes on linking
*FFTW*  <br>
Warning : always link FFTW3 first. Intel MKL also supports an FFTW interface, but does not support DCTs.<br>
When linking statically with FFTW3, you'll have to comment out the line "#define FFTW_DLL" If you're dynamically linking, you'll have to include "#define FFTW_DLL"<br>
 Because Intel MKL also includes an FFTW compatible interface. Make sure you include the FFTW library before  MKL.<br>
 e.g. <br>
 libfftw3f-3.lib;mkl_rt.lib <br>
    NOT<br>
  mkl_rt.lib;libfftw3f-3.lib;<br>
Otherwise, when you link, you'll actually link with the MKL FFTs, not the FFTW FFTs. Which would be fine, except for the fact that real-to-real transforms <br>are not implemented in MKL. Real-to-real transforms are used during the 'AutoAlign' routine.<br>
<br>
*Intel MKL*<br>
Consult 'MKL Link link advisor' for assistance selecting the correct .libs<br>
https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html <br>
Windows examples <br>
e.g. static linked...<br>
 mkl_intel_lp64.lib;mkl_intel_thread.lib;mkl_core.lib;libiomp5md.lib <br>
or if you're having issues with libiomp5md.lib (openMP threads), link with the sequential version <br>
 mkl_intel_lp64.lib;mkl_sequential.lib;mkl_core.lib <br>
e.g. dynamic linked... <br>
 mkl_rt.lib <br>
 Most of your dlls will likely be in this folder, or something similar <br>
  C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64\..., if dynamically linked, you'll need to copy those into the same folder as your executable. <br>
 If linking against openMP threads library (libiomp5md.dll), don't forget the relevant dll will be in a different folder to most of the MKL dlls <br>
  C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\redist\intel64_win\compiler\libiomp5md.dll <br>

# Command-line execution
Although it is expected that this library would typically be compiled as a shared object/DLL, it is also possible to compile as an executable and process frames from the command-line. In this mode of operation, the executable is called with one or more arguments specifying the location of a tab-delimited text file which specifies the relevant configuration settings. <br>
For example, <br>
 digHolo.exe digholoSettings.txt <br>
An example digHoloSettings.txt file is provided for reference. Camera frame data is fed in as a binary file, from a filename specified within the tab-delimited text file. <br>
For the most part, the text file specifies which 'Set' routines to run and what values to use. <br>
For example, to set the 'fftWindowSizeX' property to 128 pixels, the following tab-delimited line would be included. <br>
 fftWindowSizeX 128 <br>
Which would in turn would mean the following function call is invoked before the digHolo processing pipeline is run... <br>
 digHoloConfigSetfftWindowSizeX(handleIdx, 128); <br>
Similarly, to specify a BeamCentreX for two different polarisation components of 100e-6 and 200e-6 respectively <br>
  BeamCentreX 100e-6 200e-6 <br>
Output files can also be specified, which will either be plain text or binary files depending on the type of output <br>
For example, <br>
  OutputFileSummary summary.txt <br>
  OutputFilenameFields fields.bin <br>
See the 'main' and 'digHoloRunBatchFromConfigFile' routines for the code itself which processes the command-line usage. <br>
An example digHoloSettings.txt file is provided for reference. <br>

