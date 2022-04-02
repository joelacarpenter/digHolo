//Defining this will make calls to the BLAS library using the CBLAS interface, rather than the FORTRAN interface
//e.g. cblas_cgesvd rather than cgesvd
#define CBLAS_ENABLE

//Defining this will compile against the mkl_lapack.h and mkl_blas.h or mkl_cblas.h headers
//If not defined, it will compile against the lapack.h and blas.h or cblas.h headers (e.g. openBLAS)
#define MKL_ENABLE

//Defining this line will enable LAPACK/BLAS functions. The program won't work without them, but it can help debugging.
#define LAPACKBLAS_ENABLE

//Defining this line will enable the Intel SVML library, used for sincos and exp. If SVML is not available, a custom fast sincos, exp will be used instead.
//_MSC_VER>=1920 should support SVML, as should Intel Compiler
#if (_MSC_VER>=1920) || defined(__INTEL_COMPILER)
#define SVML_ENABLE
#endif

#ifdef _MSC_VER
	//Microsoft compiler complains about functions like fopen
#define _CRT_SECURE_NO_WARNINGS
//Disable Arithmetic overflow errors (e.g. pointless warnings about adding two ints and then multiplying by a size_t)
#pragma warning (disable: 26451)
#endif

#include <cstdint>//required for int64_t
#include <cstring>
#include <iostream>
#include <sstream> 
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <functional>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <thread>//std::thread
#include <float.h>//FLT_MAX
#include <immintrin.h>//SIMD AVX intrisics

//When linking statically with FFTW3, you'll have to comment out the line #define FFTW_DLL
//If you're dynamically linking, you'll have to make sure #define FFTW_DLL
//Because Intel MKL also includes an FFTW compatible interface. Make sure you include the FFTW library _before_ MKL.
//e.g. libfftw3f-3.lib;mkl_rt.lib NOT mkl_rt.lib;libfftw3f-3.lib;
//Otherwise, when you link, you'll actually link with the MKL FFTs, not the FFTW FFTs. Which would be fine, except for the fact that real-to-real transforms are not implemented in MKL.
//Real-to-real transforms are used during the 'AutoAlign' routine.
#include <fftw3.h>

#ifdef LAPACKBLAS_ENABLE	
#ifdef MKL_ENABLE
	//Intel MKL library
	//Console 'MKL Link link advisor' for assistance selecting the correct .libs
	//https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html
	//e.g. static linked :  mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib
		//or if you're having issues with libiomp5md.lib (openMP threads), link with the sequential version
		//mkl_intel_lp64.lib;mkl_sequential.lib;mkl_core.lib
	//e.g. dynamic linked :  mkl_rt.lib

	//If using Intel MKL...
	//Don't forget C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\redist\intel64_win\compiler\libiomp5md.dll
	//C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\redist\intel64_win\compiler\libiomp5md.dll
	//Most of your dlls will be in folder C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64\..., you'll need to copy those into the same folder as your executable.

#include <mkl_lapack.h>//cgesvd, sgels
#ifdef CBLAS_ENABLE
#include <mkl_cblas.h> //cgemv, cgemm
#else //Fortran interface
#include <mkl_blas.h> //cgemv, cgemm
#endif
//Define the complex type so functions don't complain that complex isn't what they expect, even if it's bitwise compatible with std::complex<float>
#define BLAS_COMPLEXTYPE MKL_Complex8
#else
	//openBLAS (https://www.openblas.net/)
	//Link against libopenblas.lib (libopenblas.dll)
#include <lapack.h>
#ifdef CBLAS_ENABLE
#include <cblas.h> //cgemv, cgemm
#else //Fortran interface
#include <blas.h> //cgemv, cgemm
#endif
#define BLAS_COMPLEXTYPE _Complex float
#define cgesvd cgesvd_
#define sgels sgels_
#endif
#endif

//Header file containing some routine for allocated memory aligned (for AVX) multi-dimensional arrays.
//#include "memoryAllocation.h"
//Header file for the functions and defines used by this code
#include "digHolo.h"

#define CEILINT(a,b) ((a+b-1)/b)

const float pi = (float)3.14159265358979323846264338327950288;

//File or stdout used for printing to console
FILE* consoleOut = stdout;

//Pixel dimensions must be a multiple of this for simplicity of SIMD processing (16xint16 = AVX2)
#define DIGHOLO_PIXEL_QUANTA 16
//The maximum number of polarisations
#define DIGHOLO_POLCOUNTMAX 2
//The maximum number of axes (x and y)
#define DIGHOLO_AXISCOUNTMAX 2
//The maximum number of planes (Image plane, Fourier plane)
#define DIGHOLO_PLANECOUNTMAX 2
//Extra 2 planes used in WoI variables for the frame calibration.
#define DIGHOLO_PLANECOUNTCAL 2

//A default wavelength to use to initialise variables or when the user doesn't specify a wavelength
#define DIGHOLO_WAVELENGTH_DEFAULT 1565e-9

//[sqrt(2) / (3 + sqrt(2))]. The fraction of the maximum spatial frequency usable as a Fourier window radius without wrapping
#define DIGHOLO_WMAXRATIO 0.32037724101704079249230971981887705624103546142578125

//MEMORY ALLOCATION ROUTINES
//The memory boundary alignment. For AVX, it should be a multiple of 32. I've chosen 64 (x86 cache boundary)
#define ALIGN 64
//There can be some compiler dependent issues using aligned allocation routines.
//_mm_malloc has been around a long time and is well supported. Alternatives are _aligned_malloc, posix_memalign or aligned_alloc (C11 or C++17)
#define alignedAllocate _mm_malloc
#define alignedFree _mm_free

template<typename T>
void free1D(T*& a)
{
	if (a)
	{
		alignedFree(a);

	}
	a = 0;
}

template<typename T>
void free2D(T**& a)
{
	if (a)
	{
		//Free the data itself
		alignedFree(&a[0][0]);
		//Free the array of pointers
		alignedFree(a);

	}
	a = 0;
}

template<typename T>
void free3D(T***& arr3D)
{
	if (arr3D)
	{
		size_t i = 0;
		//Free the array data  itself
		alignedFree(&arr3D[0][0][0]);
		while (arr3D[i] != 0)
		{
			//Free the array of pointers to pointers
			alignedFree(arr3D[i]);
			i++;
		}
		//Free the array of pointers
		alignedFree(arr3D);

	}
	arr3D = 0;
}

template<typename T>
void free4D(T****& arr4D)
{
	if (arr4D)
	{
		//Free the array data
		alignedFree(&arr4D[0][0][0][0]);

		size_t i = 0;
		while (arr4D[i] != 0)
		{
			size_t j = 0;
			while (arr4D[i][j] != 0)
			{
				//Free the array of pointers to pointers to pointers
				alignedFree(arr4D[i][j]);
				j++;
			}
			//Free the array of pointers to pointers
			alignedFree(arr4D[i]);
			i++;
		}

		alignedFree(arr4D);

	}
	arr4D = 0;

}

template<typename T>
int allocate1D(size_t n, T*& a)
{
	if (n > 0)
	{
		if (a)
		{
			free1D<T>(a);
		}


		a = (T*)alignedAllocate(sizeof(T) * n, ALIGN);// _aligned_malloc(sizeof(T) * n, ALIGN);

		if (!a)
		{
			std::cout << "Allocate1D failed " << n << " (" << (n * sizeof(T) / 1073741824.0) << " GB)" << "\n\r";
			return DIGHOLO_ERROR_MEMORYALLOCATION;
		}
		return DIGHOLO_ERROR_SUCCESS;
	}
	else
	{
		return DIGHOLO_ERROR_MEMORYALLOCATION;
	}

}

template<typename T>
int allocate2D(size_t nx, size_t ny, T**& a)
{
	if (nx > 0 && ny > 0)
	{
		if (a)
		{
			free2D<T>(a);
		}


		a = (T**)alignedAllocate(sizeof(T*) * (nx + 1), ALIGN);
		T* a1D = (T*)alignedAllocate(sizeof(T) * nx * ny, ALIGN);

		if (a && a1D)
		{
			/* now allocate the actual rows */
			for (size_t i = 0; i < nx; i++)
			{
				a[i] = &a1D[i * ny];
			}
			a[nx] = 0;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			std::cout << "Allocate2D failed " << nx << "x" << ny << " (" << (nx * ny * sizeof(T) / 1073741824.0) << " GB)" << "\n\r";
			return DIGHOLO_ERROR_MEMORYALLOCATION;
		}
	}
	else
	{
		return DIGHOLO_ERROR_MEMORYALLOCATION;
	}
}


template<typename T>
size_t allocate2DSparseBlock(size_t groupCount, T**& a)
{
	size_t totalElements = 0;
	if (groupCount > 0)
	{
		if (a)
		{
			free2D<T>(a);
		}

		for (size_t i = 1; i <= groupCount; i++)
		{
			totalElements += (i * i);
		}

		a = (T**)alignedAllocate(sizeof(T*) * (groupCount + 1), ALIGN);
		T* a1D = (T*)alignedAllocate(sizeof(T) * totalElements, ALIGN);

		if (a && a1D)
		{
			size_t idx = 0;
			/* now allocate the actual rows */
			for (size_t i = 0; i < groupCount; i++)
			{
				a[i] = &a1D[idx];
				idx += ((i + 1) * (i + 1));
			}
			a[groupCount] = 0;
		}
		else
		{
			std::cout << "Allocate2DSparse failed " << groupCount << "	(" << (totalElements * sizeof(T) / 1073741824.0) << " GB)" << "\n\r";
			totalElements = 0;
		}
	}
	return totalElements;
}

template<typename T>
int allocate3D(size_t nx, size_t ny, size_t nz, T***& a)
{
	if (nx > 0 && ny > 0 && nz > 0)
	{
		if (a)
		{
			free3D<T>(a);
		}

		a = (T***)alignedAllocate(sizeof(T**) * (nx + 1), ALIGN); /* one extra for sentinel */
		T* a1D = (T*)alignedAllocate(sizeof(T) * (nx * ny * nz), ALIGN);

		if (a && a1D)
		{
			for (size_t i = 0; i < nx; i++)
			{
				a[i] = (T**)alignedAllocate(sizeof(T*) * (ny + 1), ALIGN);
				for (size_t j = 0; j < ny; j++)
				{
					//a[i][j] = (T*)_aligned_malloc(sizeof(T)*nz, ALIGN);
					a[i][j] = &a1D[i * ny * nz + j * nz];
				}
				a[i][ny] = 0;
			}
			a[nx] = 0;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			std::cout << "Allocate3D failed " << nx << "x" << ny << "x" << nz << " (" << (nx * ny * nz * sizeof(T) / 1073741824.0) << " GB)" << "\n\r";
			return DIGHOLO_ERROR_MEMORYALLOCATION;
		}
	}
	else
	{
		return DIGHOLO_ERROR_MEMORYALLOCATION;
	}
}

template<typename T>
int allocate4D(size_t nx, size_t ny, size_t nz, size_t nw, T****& a)
{
	if (nx > 0 && ny > 0 && nz > 0 && nw > 0)
	{
		if (a)
		{
			free4D<T>(a);
		}

		a = (T****)alignedAllocate(sizeof(T***) * (nx + 1), ALIGN); // one extra for sentinel
		T* a1D = (T*)alignedAllocate(sizeof(T) * (nx * ny * nz * nw), ALIGN);

		if (a && a1D)
		{
			// now allocate the actual rows
			for (size_t i = 0; i < nx; i++)
			{
				a[i] = (T***)alignedAllocate(sizeof(T**) * (ny + 1), ALIGN);
				for (size_t j = 0; j < ny; j++)
				{
					a[i][j] = (T**)alignedAllocate(sizeof(T*) * (nz + 1), ALIGN);
					for (size_t k = 0; k < nz; k++)
					{
						a[i][j][k] = &a1D[i * ny * nz * nw + j * nz * nw + k * nw];
					}
					a[i][j][nz] = 0;
				}
				a[i][ny] = 0;
			}
			a[nx] = 0;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			std::cout << "Allocate4D failed " << nx << "x" << ny << "x" << nz << "x" << nw << " (" << (nx * ny * nz * nw * sizeof(T) / 1073741824.0) << " GB)" << "\n\r";
			return DIGHOLO_ERROR_MEMORYALLOCATION;
		}
	}
	else
	{
		return DIGHOLO_ERROR_MEMORYALLOCATION;
	}
	
}
//END MEMORY ALLOCATION ROUTINES
/*
struct BmpHeader {
	char bitmapSignatureBytes[2] = { 'B', 'M' };
	uint32_t sizeOfBitmapFile = 54 + 786432; // total size of bitmap file
	uint32_t reservedBytes = 0;
	uint32_t pixelDataOffset = 54;
} bmpHeader;

struct BmpInfoHeader {
	uint32_t sizeOfThisHeader = 40;
	int32_t width = 512; // in pixels
	int32_t height = 512; // in pixels
	uint16_t numberOfColorPlanes = 1; // must be 1
	uint16_t colorDepth = 24;
	uint32_t compressionMethod = 0;
	uint32_t rawBitmapDataSize = 0; // generally ignored
	int32_t horizontalResolution = 3780; // in pixel per meter
	int32_t verticalResolution = 3780; // in pixel per meter
	uint32_t colorTableEntries = 0;
	uint32_t importantColors = 0;
} bmpInfoHeader;

struct Pixel {
	uint8_t blue = 255;
	uint8_t green = 255;
	uint8_t red = 0;
} pixel;

*/

//https://dev.to/muiz6/c-how-to-write-a-bitmap-image-from-scratch-1k6m
int writeToBitmap(unsigned char* pixelArrayRGB, int width, int height, const char *filename)
{
	const int widthPad = (int)(4.0 * ceil(width*3.0 / 4.0));

	//BmpHeader
	char bitmapSignatureBytes[2] = { 'B', 'M' };
	uint32_t sizeOfBitmapFile = 54 + 786432; // total size of bitmap file
	uint32_t reservedBytes = 0;
	uint32_t pixelDataOffset = 54;

	//BmpInfoHeader
	uint32_t sizeOfThisHeader = 40;
	int32_t h = height; // in pixels
	int32_t w = width; // in pixels
	uint16_t numberOfColorPlanes = 1; // must be 1
	uint16_t colorDepth = 24;
	uint32_t compressionMethod = 0;
	uint32_t rawBitmapDataSize = 0; // generally ignored
	int32_t horizontalResolution = 3780; // in pixel per meter
	int32_t verticalResolution = 3780; // in pixel per meter
	uint32_t colorTableEntries = 0;
	uint32_t importantColors = 0;

	std::ofstream fout(filename, std::ios::binary);
	fout.write((char*)&bitmapSignatureBytes, 2);
	fout.write((char*)&sizeOfBitmapFile, sizeof(uint32_t));
	fout.write((char*)&reservedBytes, sizeof(uint32_t));
	fout.write((char*)&pixelDataOffset, sizeof(uint32_t));

	fout.write((char*)&sizeOfThisHeader, sizeof(uint32_t));
	fout.write((char*)&w, sizeof(int32_t));
	fout.write((char*)&h, sizeof(int32_t));
	fout.write((char*)&numberOfColorPlanes, sizeof(uint16_t));
	fout.write((char*)&colorDepth, sizeof(uint16_t));
	fout.write((char*)&compressionMethod, sizeof(uint32_t));
	fout.write((char*)&rawBitmapDataSize, sizeof(uint32_t));
	fout.write((char*)&horizontalResolution, sizeof(int32_t));
	fout.write((char*)&verticalResolution, sizeof(int32_t));
	fout.write((char*)&colorTableEntries, sizeof(uint32_t));
	fout.write((char*)&importantColors, sizeof(uint32_t));


	//Memory rearrangement (flip RGB to BGR and enforce 4 byte memory alignment on width dimension, 'rows')
	const size_t totalSize = ((size_t)widthPad) * ((size_t)height);

	unsigned char* tempBuffer = 0;
	allocate1D(totalSize, tempBuffer);

	for (size_t yIdx = 0; yIdx < height; yIdx++)
	{
		for (size_t xIdx = 0; xIdx < width; xIdx++)
		{
			size_t idxIn = (yIdx * width   + xIdx)*3;
			size_t idxOut = (yIdx * widthPad + xIdx*3);
			tempBuffer[idxOut + 2] = pixelArrayRGB[idxIn + 0];
			tempBuffer[idxOut + 1] = pixelArrayRGB[idxIn + 1];
			tempBuffer[idxOut + 0] = pixelArrayRGB[idxIn + 2];
		}
	}

	fout.write((char*)tempBuffer, sizeof(unsigned char)*totalSize);
	fout.close();

	return 1;
}

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

//Struct for reading info about what features the CPU supports
struct cpuINFO
{
	char brand[0x40];
	char brandHex[0x81];
	int avx = 0;
	int avx2 = 0;
	int fma3 = 0;
	int avx512f = 0;
};

//Get info about the CPU such as name, brand and instruction sets supported (e.g. avx, avx2, avx512)
int cpuInfoGet(cpuINFO* c)
{

	for (int i = 0; i < 0x40; i++)
	{
		c[0].brand[i] = 0;
	}
	for (int i = 0; i < 0x81; i++)
	{
		c[0].brandHex[i] = 0;
	}

	int a = 0x80000000;
	int fma3 = 0;
	int avx = 0;
	int avx2 = 0;
	int avx512f = 0;

	//Unforunately this part is platform dependent. Calls the cpuID instruction to find out what is supported.
	//https://en.wikipedia.org/wiki/CPUID
#ifdef _WIN32
	int cpuInfo[4];

	__cpuid((int*)&cpuInfo[0], a);

	if (cpuInfo[0] >= 0x80000004)
	{
		__cpuid((int*)&c[0].brand[0], 0x80000002);
		__cpuid((int*)&c[0].brand[16], 0x80000003);
		__cpuid((int*)&c[0].brand[32], 0x80000004);
	}
	a = 1;
	__cpuidex((int*)&cpuInfo[0], a, 0);
	fma3 = cpuInfo[2] >> 12 & 1;
	avx = cpuInfo[2] >> 28 & 1;
	a = 7;
	__cpuidex((int*)&cpuInfo[0], a, 0);
	avx2 = cpuInfo[1] >> 5 & 1;
	avx512f = cpuInfo[1] >> 16 & 1;
#else
	unsigned int cpuInfo[4];

	__get_cpuid(a, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);

	if (cpuInfo[0] >= 0x80000004)
	{
		//cpuidGet(0x80000002,((cpuidType)&brand[0]));
		__get_cpuid(0x80000002, (unsigned int*)&c[0].brand[0], (unsigned int*)&c[0].brand[4], (unsigned int*)&c[0].brand[8], (unsigned int*)&c[0].brand[12]);
		__get_cpuid(0x80000003, (unsigned int*)&c[0].brand[16], (unsigned int*)&c[0].brand[20], (unsigned int*)&c[0].brand[24], (unsigned int*)&c[0].brand[28]);
		__get_cpuid(0x80000004, (unsigned int*)&c[0].brand[32], (unsigned int*)&c[0].brand[36], (unsigned int*)&c[0].brand[40], (unsigned int*)&c[0].brand[44]);
	}
	a = 1;
	//__get_cpuid(a, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
	__cpuid_count(a, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
	fma3 = cpuInfo[2] >> 12 & 1;
	avx = cpuInfo[2] >> 28 & 1;
	a = 7;
	//__get_cpuid(a, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
	__cpuid_count(a, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
	avx2 = cpuInfo[1] >> 5 & 1;
	avx512f = cpuInfo[1] >> 16 & 1;
#endif

	//The brand of the CPU as hexadecimal (rather than ascii). This hex string version will be used as a 'unique' identifier for things like writing the FFTW wisdom to file.
	for (int i = 0; i < 0x40; i++)
	{
		sprintf(&c[0].brandHex[2 * i], "%02x", c[0].brand[i]);
	}

	c[0].avx = avx;
	c[0].fma3 = fma3;
	c[0].avx2 = avx2;
	c[0].avx512f = avx512f;

	return 1;
}

//If SVML isn't available, implement vectorised functions like exp, sincos approximations.
//Otherwise we'll just use SVML
#ifndef SVML_ENABLE
//https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
__m256 exp_ps(__m256 x)
{
	const __m256   exp_hi = _mm256_set1_ps(88.3762626647949f);
	const __m256   exp_lo = _mm256_set1_ps(-88.3762626647949f);

	const __m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f);
	//const __m256   inv_LOG2EF = _mm256_set1_ps(0.693147180559945f);
	const __m256   cephes_exp_C1 = _mm256_set1_ps(0.693359375f);
	const __m256   cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4f);

	const __m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4f);
	const __m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3f);
	const __m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3f);
	const __m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2f);
	const __m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1f);
	const __m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1f);
	__m256   tmp = _mm256_setzero_ps(), fx;
	__m256i  imm0;
	__m256   one = _mm256_set1_ps(1.0f);

	x = _mm256_min_ps(x, exp_hi);
	x = _mm256_max_ps(x, exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	//Relative error 2.53575e-7 @ -0.346707
	fx = _mm256_mul_ps(x, cephes_LOG2EF);
	fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
	tmp = _mm256_floor_ps(fx);
	__m256  mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
	mask = _mm256_and_ps(mask, one);
	fx = _mm256_sub_ps(tmp, mask);
	tmp = _mm256_mul_ps(fx, cephes_exp_C1);
	__m256  z = _mm256_mul_ps(fx, cephes_exp_C2);
	x = _mm256_sub_ps(x, tmp);
	x = _mm256_sub_ps(x, z);
	z = _mm256_mul_ps(x, x);


	//Relative erro 4.23278e-6 @ 0.346556
	/* express exp(x) as exp(g + n*log(2)) */
   /* fx = _mm256_mul_ps(x, cephes_LOG2EF);
	fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	__m256  z = _mm256_mul_ps(fx, inv_LOG2EF);
	x = _mm256_sub_ps(x, z);
	z = _mm256_mul_ps(x, x);*/


	__m256 y = _mm256_fmadd_ps(cephes_exp_p0, x, cephes_exp_p1);
	y = _mm256_fmadd_ps(y, x, cephes_exp_p2);
	y = _mm256_fmadd_ps(y, x, cephes_exp_p3);
	y = _mm256_fmadd_ps(y, x, cephes_exp_p4);
	y = _mm256_fmadd_ps(y, x, cephes_exp_p5);
	y = _mm256_fmadd_ps(y, z, x);
	y = _mm256_add_ps(y, one);

	/* build 2^n */
	imm0 = _mm256_cvttps_epi32(fx);
	imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
	imm0 = _mm256_slli_epi32(imm0, 23);
	__m256  pow2n = _mm256_castsi256_ps(imm0);
	y = _mm256_mul_ps(y, pow2n);
	return y;
}

//This version is basically identical to the float32 version exp_ps, not a true float64 implementation
__m256d exp_pd(__m256d x)
{
	const __m256d   exp_hi = _mm256_set1_pd(88.3762626647949f);
	const __m256d   exp_lo = _mm256_set1_pd(-88.3762626647949f);

	const __m256d   cephes_LOG2EF = _mm256_set1_pd(1.442695040888963387004650940070860087871551513671875);
	// const __m256d   inv_LOG2EF = _mm256_set1_pd(0.693147180559945f);
	const __m256d   cephes_exp_C1 = _mm256_set1_pd(0.693359375);
	const __m256d   cephes_exp_C2 = _mm256_set1_pd(-2.121944400546905827679E-4);

	const __m256d   cephes_exp_p0 = _mm256_set1_pd(1.9875691500E-4);
	const __m256d   cephes_exp_p1 = _mm256_set1_pd(1.3981999507E-3);
	const __m256d   cephes_exp_p2 = _mm256_set1_pd(8.3334519073E-3);
	const __m256d   cephes_exp_p3 = _mm256_set1_pd(4.1665795894E-2);
	const __m256d   cephes_exp_p4 = _mm256_set1_pd(1.6666665459E-1);
	const __m256d   cephes_exp_p5 = _mm256_set1_pd(5.0000001201E-1);
	__m256d   tmp = _mm256_setzero_pd(), fx;
	__m128i  imm0;
	__m256d   one = _mm256_set1_pd(1.0f);

	x = _mm256_min_pd(x, exp_hi);
	x = _mm256_max_pd(x, exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	//Relative error 2.53575e-7 @ -0.346707
	fx = _mm256_mul_pd(x, cephes_LOG2EF);
	fx = _mm256_add_pd(fx, _mm256_set1_pd(0.5));
	tmp = _mm256_floor_pd(fx);
	__m256d  mask = _mm256_cmp_pd(tmp, fx, _CMP_GT_OS);
	mask = _mm256_and_pd(mask, one);
	fx = _mm256_sub_pd(tmp, mask);
	tmp = _mm256_mul_pd(fx, cephes_exp_C1);
	__m256d  z = _mm256_mul_pd(fx, cephes_exp_C2);
	x = _mm256_sub_pd(x, tmp);
	x = _mm256_sub_pd(x, z);
	z = _mm256_mul_pd(x, x);


	//Relative erro 4.23278e-6 @ 0.346556
	/* express exp(x) as exp(g + n*log(2)) */
	/*fx = _mm256_mul_pd(x, cephes_LOG2EF);
	fx = _mm256_round_pd(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	__m256d  z = _mm256_mul_pd(fx, inv_LOG2EF);
	x = _mm256_sub_pd(x, z);
	z = _mm256_mul_pd(x, x);*/


	__m256d y = _mm256_fmadd_pd(cephes_exp_p0, x, cephes_exp_p1);
	y = _mm256_fmadd_pd(y, x, cephes_exp_p2);
	y = _mm256_fmadd_pd(y, x, cephes_exp_p3);
	y = _mm256_fmadd_pd(y, x, cephes_exp_p4);
	y = _mm256_fmadd_pd(y, x, cephes_exp_p5);
	y = _mm256_fmadd_pd(y, z, x);
	y = _mm256_add_pd(y, one);

	/* build 2^n */
	//This bit is a bit dodgy, converting to float32, but there's not pd_epi64 conversion in AVX2, it's an AVX-512 command
	imm0 = _mm256_cvtpd_epi32(fx);// _mm_cvtps_epi32(_mm256_cvtpd_ps(fx));
	imm0 = _mm_add_epi32(imm0, _mm_set1_epi32(0x7f));
	imm0 = _mm_slli_epi32(imm0, 23);
	__m128  pow2n = _mm_castsi128_ps(imm0);
	y = _mm256_mul_pd(y, _mm256_cvtps_pd(pow2n));
	return y;
}

//This sincos function will fail for large angles, e.g. >~1 million pi
//To improve performance, a better range reduction of x0 onto the period +/- pi/4 would be necessary
//gcc alternative? https://stackoverflow.com/questions/40475140/mathematical-functions-for-simd-registers _ZGVdN8vvv_sincosf
__m256 sincos_ps(__m256* c, __m256 x0)
{
	//I don't know where these constants come from. Double-precision uses slightly different values
	//Payne Hanek?
	//Cephes library code above
	//https ://github-wiki-see.page/m/david-c14/SubmarineFree/wiki/Examining-the-Fast-Sine-Approximations
	//Dekker arithmetic ?
	//Agner Fog's vector library was also an inspiration for this routine, it also uses these same constants.
	//https://github.com/vectorclass/version2/blob/master/vectormath_trig.h
	const __m256 DP1F = _mm256_set1_ps(-0.78515625f * 2.0f);
	const __m256 DP2F = _mm256_set1_ps(-2.4187564849853515625E-4f * 2.0f);
	const __m256 DP3F = _mm256_set1_ps(-3.77489497744594108E-8f * 2.0f);
	const __m256 twoOnPi = _mm256_set1_ps(0.63661977236758138243288840385503135621547698974609375f);//2/pi
	//Few constant values
	const __m256 one256 = _mm256_set1_ps(1.0);
	const __m256i one = _mm256_set1_epi32(1);
	const __m256 negOne = _mm256_set1_ps(-1);
	const __m256 zero = _mm256_set1_ps(0);

	//Polynomial terms used for Taylor series expansion of sine and cos.
	//Adding more terms does nothing for float32.
	__m256 Psinf[4];
	__m256 Pcosf[4];

	Psinf[0] = _mm256_set1_ps(-0.1666666666666666574148081281236954964697360992431640625f);//1/factorial(3)
	Psinf[1] = _mm256_set1_ps(8.33333333333333321768510160154619370587170124053955078125e-03f);//1/factorial(5)
	Psinf[2] = _mm256_set1_ps(-1.984126984126984125263171154784913596813566982746124267578125e-04f);//1/factorial(7)
	Psinf[3] = _mm256_set1_ps(2.755731922398589251095059327045788677423843182623386383056640625e-06f);//1/factorial(9)

	Pcosf[0] = _mm256_set1_ps(-0.5f);//1/factorial(2)
	Pcosf[1] = _mm256_set1_ps(4.1666666666666664353702032030923874117434024810791015625e-02f);//1\factorial(4)
	Pcosf[2] = _mm256_set1_ps(-1.38888888888888894189432843262466121814213693141937255859375e-03f);//1\factorial(6)
	Pcosf[3] = _mm256_set1_ps(2.48015873015873015657896394348114199601695872843265533447265625e-05f);//1\factorial(8)

	//Anding with this will remove sign bit of a float
	const unsigned int signMaskInt = 0x7FFFFFFF;
	const float* signMaskF = (float*)&signMaskInt;
	const __m256 signMask = _mm256_set1_ps(signMaskF[0]);

	//absolute value of x0 (sign bit masked out)
	__m256 xa = _mm256_and_ps(x0, signMask);

	//Find the quadrant
	__m256 y = _mm256_round_ps(_mm256_mul_ps(xa, twoOnPi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	__m256i q = _mm256_cvtps_epi32(y);

	//This could be worth reading to improve accuracy
	/// "ARGUMENT REDUCTION FOR HUGE ARGUMENTS: Good to the Last Bit"
	// K. C. Ng et al, March 24, 1992
	// https://www.csee.umbc.edu/~phatak/645/supl/Ng-ArgReduction.pdf

	//Wraps everything between + / -pi / 4
	// Range reduction
	// x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
	__m256 x = _mm256_fmadd_ps(y, DP1F, xa);
	x = _mm256_fmadd_ps(y, DP2F, x);
	x = _mm256_fmadd_ps(y, DP3F, x);

	__m256 x2 = _mm256_mul_ps(x, x);
	__m256 x3 = _mm256_mul_ps(x2, x);

	//Horner method style polynomial evaluation
	__m256 sinV = _mm256_mul_ps(Psinf[3], x2);
	sinV = _mm256_fmadd_ps(sinV, x2, Psinf[2]);
	sinV = _mm256_fmadd_ps(sinV, x2, Psinf[1]);
	sinV = _mm256_fmadd_ps(sinV, x2, Psinf[0]);
	sinV = _mm256_fmadd_ps(sinV, x3, x);

	__m256 cosV = _mm256_mul_ps(Pcosf[3], x2);
	cosV = _mm256_fmadd_ps(cosV, x, Pcosf[2]);
	cosV = _mm256_fmadd_ps(cosV, x2, Pcosf[1]);
	cosV = _mm256_fmadd_ps(cosV, x2, Pcosf[0]);
	cosV = _mm256_fmadd_ps(cosV, x2, one256);

	//Now a whole bunch of swapping between sine and cos, plus some sign flips will be employed to construct full sine and cos functions
	//From the interval +/-pi/4. 
	//These variables are used as the conditional for 3 different 'if' statements, that will swap sin/cos and sign bits based on which quadrant we're in.
	const __m256 qShift2 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(one, _mm256_and_si256(q, one)));
	const __m256 qShift3 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(one, _mm256_and_si256(_mm256_srli_epi32(q, 1), one)));
	const __m256 qShift4 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(one, _mm256_and_si256(_mm256_srli_epi32(_mm256_add_epi32(q, one), 1), one)));

	//Flip sin and cos values if necessary
	__m256 sinV_ = _mm256_or_ps(_mm256_and_ps(qShift2, cosV), _mm256_andnot_ps(qShift2, sinV));
	cosV = _mm256_or_ps(_mm256_and_ps(qShift2, sinV), _mm256_andnot_ps(qShift2, cosV));
	sinV = sinV_;

	__m256 negSin = _mm256_mul_ps(sinV, negOne);
	__m256 ltZero = _mm256_cmp_ps(x0, zero, _CMP_LT_OS);

	//Flip the sign of sine if x<0
	sinV = _mm256_or_ps(_mm256_andnot_ps(ltZero, sinV), _mm256_and_ps(ltZero, negSin));
	negSin = _mm256_mul_ps(sinV, negOne);

	//Select +sine or -sine depending on the quadrant
	sinV = _mm256_or_ps(_mm256_andnot_ps(qShift3, sinV), _mm256_and_ps(qShift3, negSin));
	//Select +cos or -cos depending on the quadrant
	__m256 negCos = _mm256_mul_ps(cosV, negOne);
	cosV = _mm256_or_ps(_mm256_andnot_ps(qShift4, cosV), _mm256_and_ps(qShift4, negCos));

	c[0] = cosV;
	return sinV;
}

//Inverse sinc function, Taylor expansion of x*csc(x)
__m256 xcsc_ps(__m256 x0)
{
	const float B[] = { 1.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e+00,
1.6666666666666665741480812812369549646973609924316406250000000000000000000000000000000000000000000000e-01,
1.9444444444444444752839729062543483451008796691894531250000000000000000000000000000000000000000000000e-02,
2.0502645502645500640015452376019311486743390560150146484375000000000000000000000000000000000000000000e-03,
2.0998677248677249788412491060540787657373584806919097900390625000000000000000000000000000000000000000e-04,
2.1336045641601196259842104785420247026195283979177474975585937500000000000000000000000000000000000000e-05,
2.1633474427786596446595009102242812559779849834740161895751953125000000000000000000000000000000000000e-06,
2.1923271344567642348143804952032009936147005646489560604095458984375000000000000000000000000000000000e-07,
2.2213930853920414129458539066221123281508198488154448568820953369140625000000000000000000000000000000e-08,
2.2507674795567867192556934851982095024958141493698349222540855407714843750000000000000000000000000000e-09,
2.2805107707218210504466873082498766175940652090048388345167040824890136718750000000000000000000000000e-10,
2.3106421580996967056883462652674466643321071757100071408785879611968994140625000000000000000000000000e-11,
2.3411704028931946895665814109782041111881834005714608792914077639579772949218750000000000000000000000e-12,
2.3721016693292249581388281702987705818223493348106956091214669868350028991699218750000000000000000000e-13,
2.4034415154237361473283816007884783312538489255527629495645669521763920783996582031250000000000000000e-14,
2.4351953983824322819707983818800718319934437306600871764317162160295993089675903320312500000000000000e-15,
2.4673688033682493186494982782386660623426708426574049948243327889940701425075531005859375000000000000e-16,
2.4999672768310467871805584707818666930635055628681304162874710073083406314253807067871093750000000000e-17,
2.5329964356669153147321962640172335832500557852866272801062308417385793291032314300537109375000000000e-18,
2.5664619702639559155055581896210584700410144846440300362561126590321691764984279870986938476562500000e-19,
2.6003696460089974176554143570182781389488296763507902255394693691314955685811582952737808227539062500e-20,
2.6347253044141823749536611738059263017676864156427437061672863771732977511419449001550674438476562500e-21,
2.6695348641570913023487672463169811744713668590495819354387442914888772804715699749067425727844238281e-22,
2.7048043221089548426023479799726085284884239838165518558183776122449959444793421425856649875640869141e-23,
2.7405397543699319392597053166464580615825908026532700879383534406895372992352122309966944158077239990e-24,
2.7767473173164390618600292900242213453938062979192225338592870097784506833171747075539315119385719299e-25,
2.8134332486618780841645484013765617263573358808544755942032976204749067719590449598854320356622338295e-26,
2.8506038685312913829026949868459556162129993133469813901295681151211924495014748970334039768204092979e-27,
2.8882655805501746464131310942060315361662500130438441297155885191305599052780800350959111710835713893e-28,
2.9264248729476754654074784087816524756714457238524401424785766874722768476403157722476322533111670054e-29,
2.9650883196743632236438561588299363780959880024982907537930431943087746421810189413614811115849079215e-30 };

	const int orderCount = 12;//12 terms is enough for -pi/2->pi/2. 28 terms for (3pi/4)

	__m256 x2 = _mm256_mul_ps(x0, x0);
	__m256 y0 = _mm256_set1_ps(B[orderCount]);

	for (int k = (orderCount - 1); k >= 0; k--)
	{
		y0 = _mm256_fmadd_ps(y0, x2, _mm256_set1_ps(B[k]));
	}

	return y0;
}
#else
__m256 exp_ps(__m256 x) { return _mm256_exp_ps(x); }
__m256d exp_pd(__m256d x) { return _mm256_exp_pd(x); }
__m256 sincos_ps(__m256* c, __m256 x0) { return _mm256_sincos_ps(c, x0); }

//inverse sinc function. Used for calibrating the effect of finite pixel fill factor in the Fourier plane.
__m256 xcsc_ps(__m256 x) 
{ 
	const __m256 zero = _mm256_set1_ps(0.0);
	const __m256 one = _mm256_set1_ps(1.0);
	const __m256 isZero = _mm256_cmp_ps(x, zero, _CMP_EQ_OS);
	__m256 v = _mm256_div_ps(x, _mm256_sin_ps(x));
	v = _mm256_or_ps(_mm256_and_ps(isZero, one), _mm256_andnot_ps(isZero, v));
	return v;
}
#endif

//AVX complex multiply
static inline __m256 cmul(const __m256& ab, const __m256& xy) noexcept
{
	const __m256 aa = _mm256_shuffle_ps(ab, ab, 0xA0);//0b10100000
	const __m256 bb = _mm256_shuffle_ps(ab, ab, 0xF5);//0b11110101
	const __m256 yx = _mm256_shuffle_ps(xy, xy, 0xB1);//0b10110001
	//return _mm256_addsub_ps(_mm256_mul_ps(aa, xy), _mm256_mul_ps(bb, yx));
	return _mm256_fmaddsub_ps(aa, xy, _mm256_mul_ps(bb, yx));
}

//Sums up the elements of a SIMD block.
static inline float reduce(__m256& a) noexcept
{
	//2 hadds, 1 add, 1 cross-lane permute (2x7+1x4+1x3)=21 latency [2x2 1x0.5 1x1] = 5.5 CPI
	/*a = _mm256_hadd_ps(a, a);
	a = _mm256_hadd_ps(a, a);
	a = _mm256_add_ps(a, _mm256_permute2f128_ps(a, a, 1));
	*/

	 //Alternative (3 adds, 2 in-lane permutes, 1 cross-lane permute) (3x4+2x1+1x3)=18 latency [3x0.5+ 2x1+ 1x1] = 4.5 CPI
		a = _mm256_add_ps(a, _mm256_permute_ps(a, 0xB1));//0b10110001
		a = _mm256_add_ps(a, _mm256_permute_ps(a, 0x4E));//0b01001110
		a = _mm256_add_ps(a, _mm256_permute2f128_ps(a, a, 1));
	
	float* aPtr = (float*)&a;
	return aPtr[0];
}

static inline double reduce(__m256d& a) noexcept
{
	a = _mm256_add_pd(a, _mm256_permute_pd(a, 0x05));//0b0101
	a = _mm256_add_pd(a, _mm256_permute2f128_pd(a, a, 1));
	double* aPtr = (double*)&a;
	return aPtr[0];
}

//Takes in 8 complex numbers at the pointer V, and returns the abs(V)^2
static inline __m256 cabs2(complex64* V) noexcept
{
	const __m256i shufflePattern = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

	__m256 Vout[2];
	Vout[0] = _mm256_loadu_ps(&V[0][0]);
	Vout[1] = _mm256_loadu_ps(&V[4][0]);

	Vout[0] = _mm256_mul_ps(Vout[0], Vout[0]);
	Vout[1] = _mm256_mul_ps(Vout[1], Vout[1]);

	Vout[0] = _mm256_add_ps(Vout[0], _mm256_permute_ps(Vout[0], 0xB1));//0b10110001
	Vout[1] = _mm256_add_ps(Vout[1], _mm256_permute_ps(Vout[1], 0xB1));

	Vout[0] = _mm256_permutevar8x32_ps(Vout[0], shufflePattern);
	Vout[1] = _mm256_permutevar8x32_ps(Vout[1], shufflePattern);

	Vout[0] = _mm256_blend_ps(Vout[0], Vout[1], 0xF0); //0b11110000

	return Vout[0];
}

/*
//Takes two sets of 8 float32s and converts to a set of 16 int16
//Not currently used anywhere
static inline __m256i cvtps_epi16(__m256 A, __m256 B)
{
	__m256i x = _mm256_packs_epi32(_mm256_cvtps_epi32(A), _mm256_cvtps_epi32(B));
	return _mm256_permute4x64_epi64(x, 0xD8);
}
*/

//Takes two sets of 8 float32s and converts to a set of 16 uint16
static inline __m256i cvtps_epu16(__m256 A, __m256 B)
{
	__m256i x = _mm256_packus_epi32(_mm256_cvtps_epi32(A), _mm256_cvtps_epi32(B));
	return _mm256_permute4x64_epi64(x, 0xD8);
}
/**
* @brief Converts and input complex number array (fieldf) into a complex colourmap bitmap (RGB) based on a HSV representation of phase and lightness representation of amplitude.
* https://codereview.stackexchange.com/questions/174617/avx-assembly-for-fast-atan2-approximation
* Might no longer be necessary, potentially use _mm256_atan2_ps?
* @param[out] pixelBuffer : The output RGB pixel array into which the complex colourmapped representation of the field will be written.
* @param[in] fieldf : Complex number array to be converted to a complex colourmap bitmap
* @param[in] pixelCount : The number of elements (pixels) in the input complex number array, and the output complex colourmapped bitmap.
* @param[in] maxValue : The maximum magnitude value within the fieldf array. This sets the scaling of the colourmap
* @return errorCode : [DIGHOLO_ERROR_SUCCESS, DIGHOLO_ERROR_INVALIDHANDLE, DIGHOLO_ERROR_INVALIDDIMENSION, DIGHOLO_ERROR_FILENOTFOUND, DIGHOLO_ERROR_MEMORYALLOCATION]
*/
void complexColormapConvert(unsigned char* pixelBuffer, complex64* fieldf, size_t pixelCount, float maxValue)
{
	__m128* field = (__m128*)fieldf;
	size_t idx = 0;
	size_t blockSize = 8;

	__m256 norm = _mm256_set1_ps((float)(1.0 / maxValue));

	float hsvScale = (float)(1.0 / 360.0);

	__m256i zeroInt16 = _mm256_set1_epi16(0);
	__m256i zeroInt32 = _mm256_set1_epi32(0);

	__m256 hsvSlopeDownStart[3];
	__m256 hsvSlopeDownStop[3];


	__m256 hsvSlopeUpStart[3];
	__m256 hsvSlopeUpStop[3];

	__m256 hsvSlopeZeroStart[3];
	__m256 hsvSlopeZeroStop[3];

	//RED
	hsvSlopeDownStart[0] = _mm256_set1_ps(60 * hsvScale);
	hsvSlopeDownStop[0] = _mm256_set1_ps(120 * hsvScale);

	hsvSlopeUpStart[0] = _mm256_set1_ps(240 * hsvScale);
	hsvSlopeUpStop[0] = _mm256_set1_ps(300 * hsvScale);

	hsvSlopeZeroStart[0] = _mm256_set1_ps(120 * hsvScale);
	hsvSlopeZeroStop[0] = _mm256_set1_ps(240 * hsvScale);

	//GREEN
	hsvSlopeDownStart[1] = _mm256_set1_ps(180 * hsvScale);
	hsvSlopeDownStop[1] = _mm256_set1_ps(240 * hsvScale);

	hsvSlopeUpStart[1] = _mm256_set1_ps(0);
	hsvSlopeUpStop[1] = _mm256_set1_ps(60 * hsvScale);

	hsvSlopeZeroStart[1] = _mm256_set1_ps(240 * hsvScale);
	hsvSlopeZeroStop[1] = _mm256_set1_ps(360 * hsvScale);

	//BLUE
	hsvSlopeDownStart[2] = _mm256_set1_ps(300 * hsvScale);
	hsvSlopeDownStop[2] = _mm256_set1_ps(360 * hsvScale);

	hsvSlopeUpStart[2] = _mm256_set1_ps(120 * hsvScale);
	hsvSlopeUpStop[2] = _mm256_set1_ps(180 * hsvScale);

	hsvSlopeZeroStart[2] = _mm256_set1_ps(0);
	hsvSlopeZeroStop[2] = _mm256_set1_ps(120 * hsvScale);

	__m256 hsvSlope = _mm256_set1_ps((float)(1.0 / (60.0 * hsvScale)));
	__m256 hsvNegSlope = _mm256_set1_ps((float)(-1.0 / (60.0 * hsvScale)));
	__m256 one = _mm256_set1_ps(1.0f);
	__m256 oneMask = _mm256_cmp_ps(one, one, _CMP_EQ_OS);//may god have mercy on your soul

	size_t iters = pixelCount / 8;

	size_t r8;
	const float* rax;
	//	float * rcx;
	__m128 xmm0, xmm1, xmm2;
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
	__m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
	__m256 twopi256 = _mm256_set1_ps((float)(2 * pi));
	__m256 pi256 = _mm256_set1_ps((float)pi);

	unsigned char tempBuffer[32];

	const __m256 twoFiftyFive = _mm256_set1_ps(255);


	// load constants
	ymm8 = _mm256_setzero_ps();
	ymm9 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)); // posnan
	ymm15 = _mm256_set1_ps(-0.0464964733f); // coefa
	ymm14 = _mm256_set1_ps(0.159314215f); // coefb
	ymm13 = _mm256_set1_ps(-0.327622771f); // coefc
	ymm12 = _mm256_set1_ps(1); // ones
	ymm11 = _mm256_set1_ps(1.57079637f); // mpi_2
	ymm10 = _mm256_set1_ps(3.14159274f); // mpi

	// setup indices, pointers
	rax = (float*)field;
	//rcx = out;
	r8 = 0;

	do {
		// load bottom part of ymm0 and ymm1
		xmm0 = _mm_loadu_ps(rax);
		xmm1 = _mm_loadu_ps(rax + 8);
		r8 += 1;
		rax += 16;
		//		rcx += 8;

				// load top part
		ymm0 = _mm256_castps128_ps256(xmm0);
		ymm1 = _mm256_castps128_ps256(xmm1);
		ymm0 = _mm256_insertf128_ps(ymm0, _mm_loadu_ps(rax - 12), 1);
		ymm1 = _mm256_insertf128_ps(ymm1, _mm_loadu_ps(rax - 4), 1);

		// de-interleave x,y pairs into separate registers
		ymm3 = _mm256_shuffle_ps(ymm0, ymm1, 0x88);
		ymm0 = _mm256_shuffle_ps(ymm0, ymm1, 0xdd);
		ymm2 = _mm256_permute2f128_ps(ymm3, ymm3, 0x03);
		ymm1 = _mm256_permute2f128_ps(ymm0, ymm0, 0x03);
		ymm4 = _mm256_shuffle_ps(ymm3, ymm2, 0x44);
		ymm2 = _mm256_shuffle_ps(ymm3, ymm2, 0xee);
		ymm3 = _mm256_shuffle_ps(ymm0, ymm1, 0x44);
		ymm1 = _mm256_shuffle_ps(ymm0, ymm1, 0xee);
		xmm1 = _mm256_castps256_ps128(ymm1);
		xmm2 = _mm256_castps256_ps128(ymm2);
		ymm2 = _mm256_insertf128_ps(ymm4, xmm2, 1);
		ymm3 = _mm256_insertf128_ps(ymm3, xmm1, 1);
		__m256 absV = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(ymm2, ymm2), _mm256_mul_ps(ymm3, ymm3)));
		absV = _mm256_mul_ps(absV, norm);

		// absolute values and zero check
		ymm4 = _mm256_and_ps(ymm2, ymm9);
		ymm0 = _mm256_cmp_ps(ymm2, ymm8, 0); // eq
		ymm6 = _mm256_and_ps(ymm3, ymm9);
		ymm1 = _mm256_cmp_ps(ymm3, ymm8, 0); // eq

		// compute argument a to polynomial
		ymm5 = _mm256_max_ps(ymm6, ymm4);
		ymm1 = _mm256_and_ps(ymm1, ymm0);
		ymm0 = _mm256_min_ps(ymm6, ymm4);
		ymm4 = _mm256_cmp_ps(ymm4, ymm6, 1); // lt
		ymm7 = _mm256_rcp_ps(ymm5);
		ymm5 = _mm256_mul_ps(ymm7, ymm5);
		ymm2 = _mm256_cmp_ps(ymm2, ymm8, 1); // lt

		// compute polynomial
		ymm5 = _mm256_mul_ps(ymm7, ymm5);
		ymm7 = _mm256_add_ps(ymm7, ymm7);
		ymm7 = _mm256_sub_ps(ymm7, ymm5);
		ymm5 = _mm256_mul_ps(ymm0, ymm7);
		ymm7 = _mm256_mul_ps(ymm5, ymm5);
		ymm0 = _mm256_mul_ps(ymm7, ymm15);
		ymm0 = _mm256_add_ps(ymm0, ymm14);
		ymm0 = _mm256_mul_ps(ymm0, ymm7);
		ymm0 = _mm256_add_ps(ymm0, ymm13);
		ymm0 = _mm256_mul_ps(ymm0, ymm7);

		// finish up
		ymm7 = _mm256_xor_ps(ymm1, _mm256_castsi256_ps(_mm256_set1_epi32(0xffffffff))); // negnan
		ymm0 = _mm256_add_ps(ymm0, ymm12);
		ymm4 = _mm256_and_ps(ymm7, ymm4);
		ymm2 = _mm256_and_ps(ymm7, ymm2);
		ymm0 = _mm256_mul_ps(ymm0, ymm5);
		ymm5 = _mm256_sub_ps(ymm11, ymm0);
		ymm0 = _mm256_blendv_ps(ymm0, ymm5, ymm4);
		ymm5 = _mm256_sub_ps(ymm10, ymm0);
		ymm0 = _mm256_blendv_ps(ymm0, ymm5, ymm2);
		ymm2 = _mm256_cmp_ps(ymm8, ymm3, 2); // le
		ymm4 = _mm256_xor_ps(ymm0, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000))); // signbit
		ymm3 = _mm256_cmp_ps(ymm3, ymm8, 1); // lt
		ymm2 = _mm256_and_ps(ymm7, ymm2);
		ymm7 = _mm256_and_ps(ymm7, ymm3);
		ymm1 = _mm256_blendv_ps(ymm4, ymm8, ymm1);
		ymm1 = _mm256_blendv_ps(ymm1, ymm4, ymm7);
		ymm1 = _mm256_blendv_ps(ymm1, ymm0, ymm2);


		ymm1 = _mm256_add_ps(ymm1, pi256);
		ymm1 = _mm256_div_ps(ymm1, twopi256);//A value between 0 and 1.0 representing angle

		for (int k = 0; k < 3; k++)
		{
			__m256 slopeDownSection = _mm256_and_ps(_mm256_cmp_ps(ymm1, hsvSlopeDownStart[k], _CMP_GE_OS), _mm256_cmp_ps(ymm1, hsvSlopeDownStop[k], _CMP_LE_OS));//greater than or equal to
			__m256 slopeZeroSection = _mm256_and_ps(_mm256_cmp_ps(ymm1, hsvSlopeZeroStart[k], _CMP_GE_OS), _mm256_cmp_ps(ymm1, hsvSlopeZeroStop[k], _CMP_LE_OS));
			__m256 slopeUpSection = _mm256_and_ps(_mm256_cmp_ps(ymm1, hsvSlopeUpStart[k], _CMP_GE_OS), _mm256_cmp_ps(ymm1, hsvSlopeUpStop[k], _CMP_LE_OS));
			__m256 slopeOneSection = oneMask;
			slopeOneSection = _mm256_andnot_ps(slopeDownSection, slopeOneSection);
			slopeOneSection = _mm256_andnot_ps(slopeUpSection, slopeOneSection);
			slopeOneSection = _mm256_andnot_ps(slopeZeroSection, slopeOneSection);

			slopeDownSection = _mm256_and_ps(slopeDownSection, one);
			slopeUpSection = _mm256_and_ps(slopeUpSection, one);
			slopeOneSection = _mm256_and_ps(slopeOneSection, one);

			__m256 up = _mm256_mul_ps(hsvSlope, _mm256_sub_ps(ymm1, hsvSlopeUpStart[k]));
			__m256 down = _mm256_add_ps(_mm256_mul_ps(hsvNegSlope, _mm256_sub_ps(ymm1, hsvSlopeDownStart[k])), one);

			__m256 r = _mm256_add_ps(slopeOneSection, _mm256_mul_ps(slopeDownSection, down));

			r = _mm256_mul_ps(twoFiftyFive, _mm256_add_ps(r, _mm256_mul_ps(up, slopeUpSection)));//0...255

			r = _mm256_mul_ps(r, absV);


			__m256i argIdxes = _mm256_cvtps_epi32(r);
			argIdxes = _mm256_packs_epi32(argIdxes, zeroInt32);
			argIdxes = _mm256_permute4x64_epi64(argIdxes, 0xD8);
			argIdxes = _mm256_packus_epi16(argIdxes, zeroInt16);

			unsigned char* argi = (unsigned char*)&argIdxes;

			unsigned char* buff = &tempBuffer[k];

			buff[0 * 3] = argi[0];
			buff[1 * 3] = argi[1];
			buff[2 * 3] = argi[2];
			buff[3 * 3] = argi[3];
			buff[4 * 3] = argi[4];
			buff[5 * 3] = argi[5];
			buff[6 * 3] = argi[6];
			buff[7 * 3] = argi[7];

		}
		memcpy(&pixelBuffer[idx], &tempBuffer[0], 24);
		idx += 3 * blockSize;

	} while (r8 < iters);

}

//This is unsafe for pixel dimensions that aren't a multiple of 8

/**
* @brief Converts an uint16 array to a float32 array, with the ability to transpose the array in the process if desired
*
* @param[in] p : Pointer to the input uint16 array
* @param[out] pF : Pointer to the output float32 array
* @param[in] NxStart : The starting index along the x-axis to convert
* @param[in] NxStop : The stop (<NxStop) index along the x-axis to convert. The NxStart and NxStop range is mostly for threading purposes, so that each thread can have it's own range to convert.
* @param[in] Nx : The dimension of the input array along the x-axis.
* @param[in] Ny : The dimension of the input array along the y-axis.
* @param[in] transpose : When set, the pF float32 array will be written transposed relative to the input uint16 array.
*/
void convert_int16tofloat32(unsigned short* p, float* pF, int NxStart, int NxStop, int Nx, int Ny, unsigned char transpose)
{
	const size_t blockSize = 8;
	__m256* pF256 = (__m256*) & pF[0];

	if (transpose)
	{
		const __m256i idx2 = _mm256_set_epi32(7 * Nx, 6 * Nx, 5 * Nx, 4 * Nx, 3 * Nx, 2 * Nx, 1 * Nx, 0 * Nx);
		const __m256i shuffleBits = _mm256_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
		const int scale = 2;
		for (size_t i = NxStart; i < NxStop; i += 2)
		{
			for (size_t j = 0; j < Ny; j += 8)
			{
				const size_t idx0 = j * Nx + i;
				const int* ptr = (int*)&p[idx0];
				const __m256i a = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(_mm256_i32gather_epi32(ptr, idx2, scale), shuffleBits), 0xDB);//0b11011000

				const __m256 row0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extractf128_si256(a, 0)));
				const __m256 row1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extractf128_si256(a, 1)));

				const size_t idx2 = ((i)*Ny + j) / blockSize;
				pF256[idx2] = row0;

				const size_t idx3 = ((i + 1) * Ny + j) / blockSize;
				pF256[idx3] = row1;
			}
		}

	}
	else
	{
		//No transpose. Just converts epu16 to epi32 to ps (uint16-->int32-->float32)
		const size_t startIdx = (size_t)NxStart * (size_t)Ny;
		const size_t stopIdx = (size_t)NxStop * (size_t)Ny;
		for (size_t i = startIdx; i < stopIdx; i += 8)
		{
			__m128i pin = _mm_loadu_si128((__m128i*) & p[i]);
			__m256 pout = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(pin));
			_mm256_storeu_ps(&pF[i], pout);
		}
	}
}

int digHoloGenerateHGindices(int maxMG, int* M, int* N, int** MN)
{
	if (maxMG == 0)
	{
		maxMG = 1;
	}
	int idx = 0;
	for (int mgIdx = 0; mgIdx < maxMG; mgIdx++)
	{
		for (int modeIdx = 0; modeIdx <= mgIdx; modeIdx++)
		{
			int m = mgIdx - modeIdx;
			int n = mgIdx - m;
			M[idx] = m;
			N[idx] = n;
			MN[m][n] = idx;
			idx++;
		}
	}
	return idx;
}
//The main trick applied here to make the function work for very high order modes, is keep extracting the exponent of float64 and store it in another variable. This enables us to keep track of very large exponents, in effect creating a float that has 32-bits of exponent, instead of 11. So it's more like a float85 instead of float64
//The important part, is that instead of naively calculating HG*exp(-x^2/2), which involve very large numbers in HG being multiplied by very small numbers in the Gaussian exp(x^2/2), 
//instead, we evaluated A*exp(-x^2/2+B), where A and B represent our HG polynomial factor, but split up into significand and exponent in such a way that precision is maintained.
void HGbasisXY1D(float* hgScales, float w0, float* axisF, int axisCount, int modeCount, __m128i** modes)
{
	//The size of a single pixel along the axis (assumes a linear axis). Takes first and last element in the axis and assumes equal spacing inbetween.
	const double dX = abs((1.0 * axisF[axisCount - 1] - 1.0 * axisF[0]) / (axisCount - 1));

	//The beam waist (as float64)
	const double W = w0;
	//Scaling factor to normalise the axis in terms of beam waist rather than units of metres.
	const __m256d axisScale = _mm256_set1_pd(sqrt(2.0) / W);

	//A normalisation factor used to make sure the HG mode will have unity total power.
	const double normFactor = ((1.0 / sqrt(w0))) * sqrt(dX) / sqrt(sqrt(0.5));
	//Normalisation factor loaded 4-pack for SIMD
	const __m256d normFactor256 = _mm256_set1_pd(normFactor);

	//Variable used for incrementing the axis from one iteration to the next
	const __m256d axisStep = _mm256_set1_pd(4.0 * dX * (sqrt(2) / W));
	//The start of the axis. This uses the first 4 elements of the axisF array. This seems to be more accurate than just using axisF[0]+n*dX.
	const __m256d axis0 = _mm256_mul_pd(axisScale, _mm256_set_pd(axisF[3], axisF[2], axisF[1], axisF[0]));

	//We'll be processing 4 axis points per iteration of the loop (m256d=4xfloat64)
	const int count256 = axisCount / 4;
	//Later we'll be processing 8-axis points per iteration of the loop (m256=8xfloat32)
	const int halfCount256 = count256 / 2;

	//Just going to do 1 big memory allocation, and then point all the arrays we need to this allocation
	__m256d* modesD = 0;
	allocate1D(count256 * (3 + 1 + 1 + 1), modesD);// (__m256d*)_aligned_malloc(sizeof(__m256d) * count256 * (3 + 1 + 1 + 1), ALIGN);
	//Used for storing a scaling factor we'll be using to prevent float64 overflow in the exponent.
	__m256d* scale = &modesD[3 * count256];
	//We'll store the log scales as 32-bit integers. If we have exponents larger than 2^31, double precision couldn't handle it anyways.
	__m128i* log_scale = (__m128i*) & scale[count256];
	memset(&log_scale[0], 0, sizeof(__m128i) * count256 * 2);//Should only need to clear the first two modes.
	//This part of the memory will be used for converting the float64 mode to a float32 mode, before eventually being written out to the final int16 array
	__m128* modeF128 = (__m128*) & log_scale[2 * count256];


	const uint64_t signMaskInt = 0x7FFFFFFFFFFFFFFF;//Bit mask to pick off the sign bit of a double (used for comparing absolute values to find max value.
	const double* signMaskD = (double*)&signMaskInt;
	const __m256d signMask = _mm256_set1_pd(signMaskD[0]);

	//A mask that picks off just the 11 exponent bits of a float64
	const __m256i exponentMask = _mm256_set1_epi64x(0x7FF0000000000000);//0b0111111111110000000000000000000000000000000000000000000000000000//0b0 11111111111 0000000000000000000000000000000000000000000000000000
	//The exponent of a float64 is an unsigned int, offset by -1023. These values are used for applying the exponent bias (to convert it to an int) as well as the negating the exponent.
	const __m128i exponentNegBias = _mm_set1_epi32(2046);
	const __m128i exponentBias = _mm_set1_epi32(-1023);

	//Permute mask used to extract the lower 32-bits from a set of 4xint64, and pack them into the first lane as a set of 4xint32. Used for conversion of int64 to int32
	const __m256i cvtepi64_epi32ShuffleMask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

	//The loneliest number, made slightly less lonely by packing them together in a single m256d.
	const __m256d one = _mm256_set1_pd(1.0);
	//You know what -1/2 is.
	const __m256d minusHalf = _mm256_set1_pd(-0.5);
	//Used for converting between the base-2 used by float64, and the base-e we'll need when applying the Gaussian exp(-x^2/2+(stuff we've been keeping track of externally to prevent overflow))
	const __m256d log2scale = _mm256_set1_pd(log(2.0));

	//Pointers to current and previous modes
	//The previous mode
	__m256d* h1 = 0;
	//The mode before the previous mode
	__m256d* h0 = 0;
	//The current mode
	__m256d* mode = 0;

	//Pointers to arrays for keeping track of exponent factors. We'll be accumulating exponents in a separate array, so that the float64 doesn't overflow for very large orders.
	__m128i* logScalePreviousMode = &log_scale[0];
	__m128i* logScaleCurrentMode = &log_scale[count256];

	for (int modeIdx = 0; modeIdx < modeCount; modeIdx++)
	{
		__m256d x0 = axis0;

		//The fundamental mode. The first two modes are processed differently, as we need to setup the first two modes in order to bootstrap the recursive definition of the rest of the modes.
		if (modeIdx == 0)
		{
			const __m256d a0 = _mm256_set1_pd(0.7511255444649425072611848008818924427032470703125); // pi. ^ (-0.25);

			mode = &modesD[0];
			for (int i = 0; i < count256; i++)
			{
				mode[i] = a0;
				scale[i] = one;
			}
		}
		else
		{
			if (modeIdx == 1)
			{
				const __m256d a1 = _mm256_set1_pd(1.062251932027197032226695228018797934055328369140625); // sqrt(2).*pi. ^ (-0.25);
				mode = &modesD[count256];
				for (int i = 0; i < count256; i++)
				{
					mode[i] = _mm256_mul_pd(a1, x0);
					x0 = _mm256_add_pd(x0, axisStep);
				}
			}
			else //modeIdx>=2. All other modes are calculated using the previous two modes.
			{
				//Setup the pointers to the current and previous modes
				if (modeIdx == 2)
				{
					h0 = &modesD[0];
					h1 = &modesD[count256];
					mode = &modesD[2 * count256];
				}

				const __m256d kFactor1 = _mm256_set1_pd(sqrt(2.0 / (modeIdx)));
				const __m256d kFactor2 = _mm256_set1_pd(-sqrt((modeIdx - 1.0) / modeIdx));
				__m256d xk = _mm256_mul_pd(x0, kFactor1);
				const __m256d xkStep = _mm256_mul_pd(axisStep, kFactor1);

				//The formula being evaluated can be seen between Equation 5 and 6 of
				//https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2630232/#FN3
				for (int i = 0; i < count256; i++)
				{
					const __m256d hA = _mm256_mul_pd(xk, h1[i]);//kFactor1*x[i]*h1[i]
					const __m256d hB = _mm256_mul_pd(kFactor2, h0[i]);//+kFactor2 * h0[i];
					//The new hermite value
					__m256d h = _mm256_fmadd_pd(hB, scale[i], hA); //kFactor1* x[i] * h1[i]+ kFactor2* h0[i] * scale[i];

					//We're going to be constantly extracting out the exponent of the float64 and incrementing/decrementing it in a separate variable. Effectively making if behave like a float with 32-bits of exponent, and the standard 52-bits of significand.

					//These next few steps are doing floor(log2(h)), by extracting the 11 exponent bits of the float64.
					//Mask out only the exponent of the float64
					__m256d logscale = _mm256_and_pd(h, _mm256_castsi256_pd(exponentMask));
					//Cast it to an int (we'll be manipulating individual bits)
					__m256i logscaleInt = _mm256_castpd_si256(logscale);
					//Shift the exponent down to bit0.
					logscaleInt = _mm256_srli_epi64(logscaleInt, 52);

					//We'll convert to a int32 because we don't need 64-bits of exponent, even 32-bits is overkill. Could save us some memory bandwidth when we come to write out.
					__m128i logscale32 = _mm256_extractf128_si256(_mm256_permutevar8x32_epi32(logscaleInt, cvtepi64_epi32ShuffleMask), 0);
					//__m128i logscale32 = _mm256_cvtepi64_epi32(logscaleInt); //(AVX512 instruction)

					//Flip the sign of the exponent (the exponent of a float64 is just a biased uint11, rather than two's complement, so to flip the sign, you subtract it from 2046)
					__m128i logscale32Neg = _mm_sub_epi32(exponentNegBias, logscale32);
					__m256i logscale64Neg = _mm256_slli_epi64(_mm256_cvtepi32_epi64(logscale32Neg), 52);
					//Now this contains 2^(-exponent of h). i.e. 2^(-floor(log2(h))
					__m256d newScale = _mm256_castsi256_pd(logscale64Neg);
					//Add the exponent to the tally. Keep track of the exponent separately. The exponent bias is applied to convert it to a standard int32
					logScaleCurrentMode[i] = _mm_add_epi32(logScalePreviousMode[i], _mm_add_epi32(exponentBias, logscale32));

					mode[i] = _mm256_mul_pd(h, newScale);
					scale[i] = newScale;
					xk = _mm256_add_pd(xk, xkStep);
				}
			}
		}

		__m128i* logScale = logScaleCurrentMode;
		__m256d maxV = _mm256_set1_pd(0);
		x0 = axis0;

		for (int i = 0; i < count256; i++)
		{
			//x^2
			const __m256d x2D = _mm256_mul_pd(minusHalf, _mm256_mul_pd(x0, x0));
			//The bias we need to apply to the exponent of the exp() to reapply all the exponent scales we removed earlier.
			//Previously, we kept extracting the exponent bits from the float64 into a separate variable to increase precision. Now we need to put them back in.
			//log2scale is also applied to convert from base-2 to base-e
			const __m256d logScaleExp = _mm256_mul_pd(log2scale, _mm256_cvtepi32_pd(logScale[i]));
			//exp(-x^2+logscale).
			const __m256d expu2 = exp_pd(_mm256_add_pd(x2D, logScaleExp));
			const __m256d mode256 = _mm256_mul_pd(normFactor256, _mm256_mul_pd(mode[i], expu2));
			maxV = _mm256_max_pd(maxV, _mm256_and_pd(mode256, signMask));//mask out sign bit (absolute value) and then compare with running max value
			//At this point we could probably get away with converting to float32 before writing out.
			modeF128[i] = _mm256_cvtpd_ps(mode256);
			x0 = _mm256_add_pd(x0, axisStep);
		}

		//Shouldn't need to enforce sign mask still?
		maxV = _mm256_max_pd(maxV, _mm256_and_pd(_mm256_permute_pd(maxV, 1), signMask));//Compare to their neighbours
		maxV = _mm256_max_pd(maxV, _mm256_and_pd(_mm256_permute2f128_pd(maxV, maxV, 1), signMask));//all 4 elements filled with max absolute value

		//Now we're going to go through each mode, and convert it to int16 format, with an additional scaling factor which lets you convert the int16 to its float representation.
		//We know the max value for each mode, so we'll scale based on that
		const double* mxV = (double*)&maxV;
		const double scaleFactor = (32767) / mxV[0];//(2^15)-1
		const double scaleFactorInv = 1.0 / scaleFactor;
		const __m256 scaleFactor256 = _mm256_set1_ps((float)scaleFactor);
		//The multiplication factor you'll have to apply to the HG modes in int16 format, in order to get the true float32 value.
		hgScales[modeIdx] = (float)scaleFactorInv;// maxV / scaleFactor;

		//float32 mode inputs (from previous loop)
		const __m256* modeF256 = (__m256*) & modeF128[0];
		//int16 mode outputs
		__m256i* mode16 = (__m256i*) & modes[modeIdx][0];

		//Pointer we'll be using for packing two _m128i (8xint16) into a single output __m256i (16x16bit)
		__m256i modeInt32AB;

		for (int xIdx = 0; xIdx < (halfCount256); xIdx += 2)
		{
			//Apply the scaling factor to the modes, this should put them in a 16-bit range (+/-32767)
			const __m256 mode256A = _mm256_round_ps(_mm256_mul_ps(scaleFactor256, modeF256[xIdx]), _MM_FROUND_TO_NEAREST_INT);
			const __m256 mode256B = _mm256_round_ps(_mm256_mul_ps(scaleFactor256, modeF256[xIdx + 1]), _MM_FROUND_TO_NEAREST_INT);
			//Convert the float32-->int32-->int16
			modeInt32AB = _mm256_packs_epi32(_mm256_cvtps_epi32(mode256A), _mm256_cvtps_epi32(mode256B));
			modeInt32AB = _mm256_permute4x64_epi64(modeInt32AB, 0xD8);//0b11011000
			//Save to output array
			mode16[xIdx / 2] = modeInt32AB;
		}

		//Cyclic shift the memory pointers for the current mode, the previous mode, and the mode before the previous mode.
		__m256d* tempPtr = h0;
		h0 = h1;
		h1 = mode;
		mode = tempPtr;
		//Cyclic shift the memory pointers that accumulate the log scale for the current mode based on the previous mode.
		__m128i* tPtr = logScalePreviousMode;
		logScalePreviousMode = logScaleCurrentMode;
		logScaleCurrentMode = tPtr;
	}
	alignedFree(modesD);
}

float generateModeSeparablex(float** hgScaleX0, float** hgScaleY0, complex64* fieldx, short*** modesX0x, short*** modesY0, int pixelCountX_, int pixelCountY_, complex64* coefs, int* M, int* N, int modeCount, int polCount)
{
	//How many mm256 elements between the polarisations (if applicable)
	//How many mm256i elements are there in mode array
	const size_t pixelCountX = pixelCountX_;
	const size_t pixelCountY = pixelCountY_;
	__m256* field = (__m256*)fieldx;
	__m256i*** modesX0 = (__m256i***)modesX0x;

	const size_t pxCount = ((size_t)pixelCountX) / 16;

	const __m256i overlapPermute1 = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	const __m256i overlapPermute2 = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	//__m256 field256[8];
	__m256 maxV = _mm256_set1_ps(0);

	for (int polIdx = 0; polIdx < polCount; polIdx++)
	{
		float* hgScaleX = hgScaleX0[polIdx];
		float* hgScaleY = hgScaleY0[polIdx];
		__m256i** modesX = modesX0[polIdx];
		short** modesY = modesY0[polIdx];
		for (int modeIdx = 0; modeIdx < modeCount; modeIdx++)
		{
			const int mIdx = M[modeIdx];
			const int nIdx = N[modeIdx];

			const double xScale = hgScaleX[mIdx];
			const double yScale = hgScaleY[nIdx];
			//const __m256 xyScale = _mm256_set1_ps(xScale * yScale);

			const __m256i* modeX = modesX[mIdx];
			const short* modeY = modesY[nIdx];

			const float cR = (float)(xScale * yScale * coefs[modeIdx + polIdx * modeCount][0]);
			const float cI = (float)(xScale * yScale * coefs[modeIdx + polIdx * modeCount][1]);

			const float cPwr = cR * cR + cI * cI;
			if (cPwr > 0)
			{
				const __m256 coefR = _mm256_set1_ps(cR);
				const __m256 coefI = _mm256_set1_ps(cI);
				for (size_t pixelIdy = 0; pixelIdy < pixelCountY; pixelIdy++)
				{
					const short yV = modeY[pixelIdy];
					const __m256i yValue = _mm256_cvtepi16_epi32(_mm_set1_epi16(yV));
					const size_t yOffset = pixelIdy * (pixelCountX / 4) + polIdx * (pixelCountX * pixelCountY) / 4;

					if (yV != 0 || modeIdx == 0 || modeIdx == (modeCount - 1))
					{
						for (size_t pixelIdx = 0; pixelIdx < pxCount; pixelIdx++)
						{
							const __m256i xValue16 = (modeX[pixelIdx]);

							const int allZeros = _mm256_testz_si256(xValue16, xValue16);//const int allZeros = false;
							const size_t fieldIdx = yOffset + pixelIdx * 4;

							if (!allZeros || modeIdx == 0)
							{
								const __m256i xValueA = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xValue16, 0));
								const __m256i xValueB = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xValue16, 1));

								const __m256i value32A = _mm256_mullo_epi32(xValueA, yValue);
								const __m256i value32B = _mm256_mullo_epi32(xValueB, yValue);
								const __m256 valueA = _mm256_cvtepi32_ps(value32A);
								const __m256 valueB = _mm256_cvtepi32_ps(value32B);

								const __m256 valueAr = _mm256_mul_ps(valueA, coefR);
								const __m256 valueAi = _mm256_mul_ps(valueA, coefI);

								const __m256 valueBr = _mm256_mul_ps(valueB, coefR);
								const __m256 valueBi = _mm256_mul_ps(valueB, coefI);

								//__m256 A = _mm256_shuffle_ps(R, I, 0b01010000);
							/*	__m256 V0 = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueAr, overlapPermute1), _mm256_permutevar8x32_ps(valueAi, overlapPermute1), 0b1010101010);
								__m256 V1 = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueAr, overlapPermute2), _mm256_permutevar8x32_ps(valueAi, overlapPermute2), 0b1010101010);
								__m256 V2 = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueBr, overlapPermute1), _mm256_permutevar8x32_ps(valueBi, overlapPermute1), 0b1010101010);
								__m256 V3 = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueBr, overlapPermute2), _mm256_permutevar8x32_ps(valueBi, overlapPermute2), 0b1010101010);
								*/
								__m256 V0 = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueAr, overlapPermute1), _mm256_permutevar8x32_ps(valueAi, overlapPermute1), 0xAA);//0b10101010
								__m256 V1 = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueAr, overlapPermute2), _mm256_permutevar8x32_ps(valueAi, overlapPermute2), 0xAA);//0b10101010
								__m256 V2 = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueBr, overlapPermute1), _mm256_permutevar8x32_ps(valueBi, overlapPermute1), 0xAA);//0b10101010
								__m256 V3 = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueBr, overlapPermute2), _mm256_permutevar8x32_ps(valueBi, overlapPermute2), 0xAA);//0b10101010

								if (modeIdx == 0)
								{
									field[fieldIdx] = V0;
									field[fieldIdx + 1] = V1;
									field[fieldIdx + 2] = V2;
									field[fieldIdx + 3] = V3;
								}
								else
								{
									field[fieldIdx] = _mm256_add_ps(V0, field[fieldIdx]);
									field[fieldIdx + 1] = _mm256_add_ps(V1, field[fieldIdx + 1]);
									field[fieldIdx + 2] = _mm256_add_ps(V2, field[fieldIdx + 2]);
									field[fieldIdx + 3] = _mm256_add_ps(V3, field[fieldIdx + 3]);
								}

							}
							if (modeIdx == (modeCount - 1))
							{
								for (int l = 0; l < 4; l++)
								{
									__m256 V0 = field[fieldIdx + l];
									__m256 V0b = _mm256_permute_ps(V0, 0xB1);//0b10110001
									__m256 Vpwr = _mm256_add_ps(_mm256_mul_ps(V0, V0), _mm256_mul_ps(V0b, V0b));
									maxV = _mm256_max_ps(maxV, Vpwr);
								}
							}
						}
					}
				}
			}
		}
	}
	maxV = _mm256_max_ps(maxV, _mm256_permute_ps(maxV, 0xB1));//0b10110001//Compare to their neighbours (blocks of 1)
	maxV = _mm256_max_ps(maxV, _mm256_permute_ps(maxV, 0x4E));//0b01001110//Compare blocks of 2 to neighbouring blocks of 2
	maxV = _mm256_max_ps(maxV, _mm256_permute2f128_ps(maxV, maxV, 1));//Compare blocks of 4 to neighbouring blocks of 4
	float* maxVf = (float*)&maxV;
	return maxVf[0];
}

float generateModeSeparable(float** hgScaleX0, float** hgScaleY0, complex64* field, short*** modesX0, short*** modesY0, int pixelCountX_, int pixelCountY_, complex64* coefs, int* M, int* N, int modeCount, int polCount, int transpose, complex64**refX, complex64** refY, float* intensityOut)
{
	//How many mm256 elements between the polarisations (if applicable)
	//How many mm256i elements are there in mode array

	if (transpose)
	{
		//Swap x and y modal components
		float** hgScaleXtemp = hgScaleX0;
		hgScaleX0 = hgScaleY0;
		hgScaleY0 = hgScaleXtemp;

		short*** modesXtemp = modesX0;
		modesX0 = modesY0;
		modesY0 = modesXtemp;

		int pixelCountXtemp = pixelCountX_;
		pixelCountX_ = pixelCountY_;
		pixelCountY_ = pixelCountXtemp;

		complex64** refXtemp = refX;
		refX = refY;
		refY = refXtemp;
	}

	const size_t pixelCountX = pixelCountX_;
	const size_t pixelCountY = pixelCountY_;

	const size_t blockSize = 16;

	const __m256i overlapPermute1 = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	const __m256i overlapPermute2 = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	__m256 maxV = _mm256_set1_ps(0);

	for (int polIdx = 0; polIdx < polCount; polIdx++)
	{
		float* hgScaleX = hgScaleX0[polIdx];
		float* hgScaleY = hgScaleY0[polIdx];
		short** modesX = modesX0[polIdx];
		short** modesY = modesY0[polIdx];

		complex64* rfX = refX?refX[polIdx]:0;
		complex64* rfY = refY ? refY[polIdx] : 0;

		for (int modeIdx = 0; modeIdx < modeCount; modeIdx++)
		{
			const int mIdx = M[modeIdx];
			const int nIdx = N[modeIdx];

			const double xScale = hgScaleX[mIdx];
			const double yScale = hgScaleY[nIdx];

			const short* modeX = modesX[mIdx];
			const short* modeY = modesY[nIdx];

			const float cR = (float)(xScale * yScale * coefs[modeIdx + polIdx * modeCount][0]);
			const float cI = (float)(xScale * yScale * coefs[modeIdx + polIdx * modeCount][1]);

			const float cPwr = cR * cR + cI * cI;

			if (cPwr > 0 || modeIdx==0 || modeIdx == (modeCount - 1))
			{
				const __m256 coefR = _mm256_set1_ps(cR);
				const __m256 coefI = _mm256_set1_ps(cI);

				for (size_t pixelIdy = 0; pixelIdy < pixelCountY; pixelIdy++)
				{
					const short yV = modeY[pixelIdy];
					const __m256i yValue = _mm256_cvtepi16_epi32(_mm_set1_epi16(yV));

					const size_t yOffset = transpose?(pixelIdy*polCount*pixelCountX+polIdx*pixelCountX):pixelIdy * (pixelCountX) + polIdx * (pixelCountX * pixelCountY);

					if (yV != 0 || modeIdx == 0 || modeIdx == (modeCount - 1))
					{
						for (size_t pixelIdx = 0; pixelIdx < pixelCountX; pixelIdx+=blockSize)
						{
							const __m256i xValue16 = _mm256_loadu_si256((__m256i*)&modeX[pixelIdx]);//16 pixels worth

							const int allZeros = _mm256_testz_si256(xValue16, xValue16);//const int allZeros = false;
							
							const size_t fieldIdx = yOffset + pixelIdx;

							if (!allZeros || modeIdx == 0)
							{
								const __m256i xValueA = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xValue16, 0));
								const __m256i xValueB = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(xValue16, 1));

								const __m256i value32A = _mm256_mullo_epi32(xValueA, yValue);
								const __m256i value32B = _mm256_mullo_epi32(xValueB, yValue);

								//int32-->float32
								const __m256 valueA = _mm256_cvtepi32_ps(value32A);
								const __m256 valueB = _mm256_cvtepi32_ps(value32B);

								const __m256 valueAr = _mm256_mul_ps(valueA, coefR);
								const __m256 valueAi = _mm256_mul_ps(valueA, coefI);

								const __m256 valueBr = _mm256_mul_ps(valueB, coefR);
								const __m256 valueBi = _mm256_mul_ps(valueB, coefI);

								__m256 V[4];
								V[0] = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueAr, overlapPermute1), _mm256_permutevar8x32_ps(valueAi, overlapPermute1), 0xAA);//0b10101010
								V[1] = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueAr, overlapPermute2), _mm256_permutevar8x32_ps(valueAi, overlapPermute2), 0xAA);//0b10101010
								V[2] = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueBr, overlapPermute1), _mm256_permutevar8x32_ps(valueBi, overlapPermute1), 0xAA);//0b10101010
								V[3] = _mm256_blend_ps(_mm256_permutevar8x32_ps(valueBr, overlapPermute2), _mm256_permutevar8x32_ps(valueBi, overlapPermute2), 0xAA);//0b10101010

								if (modeIdx == 0)
								{
									if (rfX && rfY)
									{
										float rYr = rfY[pixelIdy][0];
										float rYi = rfY[pixelIdy][1];

										const __m256 rY = _mm256_set_ps(rYi, rYr, rYi, rYr, rYi, rYr, rYi, rYr);

										for (int l = 0; l < 4; l++)
										{
											__m256 rX = _mm256_loadu_ps(&rfX[pixelIdx + 4 * l][0]);
											__m256 ref = cmul(rX, rY);
											V[l] = _mm256_add_ps(ref, V[l]);
										}
									}
									_mm256_storeu_ps(&field[fieldIdx][0], V[0]);
									_mm256_storeu_ps(&field[fieldIdx + 4][0], V[1]);
									_mm256_storeu_ps(&field[fieldIdx + 8][0], V[2]);
									_mm256_storeu_ps(&field[fieldIdx + 12][0], V[3]);
								}
								else
								{
									for (int l = 0; l < 4; l++)
									{
										__m256 f = _mm256_loadu_ps(&field[fieldIdx + 4*l][0]);
										V[l] = _mm256_add_ps(V[l], f);
										_mm256_storeu_ps(&field[fieldIdx + 4*l][0], V[l]);
									}
								}
							}

							if (modeIdx == (modeCount - 1))
							{
								__m256 Vout[2];

								Vout[0] = cabs2(&field[fieldIdx + 0]);
								Vout[1] = cabs2(&field[fieldIdx + 8]);

								if (intensityOut)
								{
									_mm256_storeu_ps(&intensityOut[fieldIdx], Vout[0]);
									_mm256_storeu_ps(&intensityOut[fieldIdx + 8], Vout[1]);
								}

								maxV = _mm256_max_ps(maxV, Vout[0]);
								maxV = _mm256_max_ps(maxV, Vout[1]);
							}
						}
					}
				}
			}
		}
	}
	maxV = _mm256_max_ps(maxV, _mm256_permute_ps(maxV, 0xB1));//0b10110001//Compare to their neighbours (blocks of 1)
	maxV = _mm256_max_ps(maxV, _mm256_permute_ps(maxV, 0x4E));//0b01001110//Compare blocks of 2 to neighbouring blocks of 2
	maxV = _mm256_max_ps(maxV, _mm256_permute2f128_ps(maxV, maxV, 1));//Compare blocks of 4 to neighbouring blocks of 4
	float* maxVf = (float*)&maxV;
	return maxVf[0];
}



/**
* @brief Selects a window containing the off-axis term from the Fourier plane (fullPlane) and writes it/filters it, ready for the IFFT in the next step
*
* @param[in] SX : The starting x-coordinate of the corner of the window to select
* @param[in] SY : The starting y-coordinate of the corner of the window to select
* @param[in] windowWidth : The dimension of the window along the x-axis
* @param[in] windowHeight : The dimension of the window along the y-axis
* @param[in] fullWidth : The full width of the entire Fourier plane along the x-axis
* @param[in] fullHeight : The full height of the entire Fourier plane along the y-axis
* @param[out] windowOutput : A pointer to the array into which the off-axis term will be written, ready for the IFFT in the next step.
* @param[in] fullPlane : A pointer to the array containing the source array of the entire Fourier plane, from which the off-axis term will be selected.
* @param[in] lowResMode : When set to 1, the total dimension of windowOutput will be just the size of the window itself. Whereas when set to 0, windowOutput will have the same dimensions as fullPlane and hence we'll be doing a large zero-padded IFFT in the next step.
* @param[in] kxAxis : The spatial frequency (Fourier plane) x-axis in the fullPlane.
* @param[in] kyAxis : The spatial frequency (Fourier plane) y-axis in the fullPlane.
* @param[in] beamCentreX : The x-coordinate of the centre of the beam within the FFT window in the camera plane in metres. This function will be applying a tilt in Fourier space to recentre the beam in the output IFFT reconstructed field.
* @param[in] beamCentreY : The y-coordinate of the centre of the beam within the FFT window in the camera plane in metres. This function will be applying a tilt in Fourier space to recentre the beam in the output IFFT reconstructed field.
* @param[in] kx0 : The centre spatial frequency in the x-axis of the window (k0*sin(theta_x))
* @param[in] ky0 : The centre spatial frequency in the y-axis of the window (k0*sin(theta_y))
* @param[in] kr : The radius of the off-axis window as a spatial frequency. Spatial frequency components outside with radius will be zeroed. The window is circular in Fourier space.
* @param[in] avgIdx : If averaging multiple frames, this is the frame number within a group to be averaged. The avgIdx>0 frames will be added in-phase with the previous frames. avgIdx=0 can be written directly, whereas avgIdx>0 are accumulated in-phase with the previous frames.
* @param[in] avgCount : The total number of frames to be averaged. Used mostly for scaling the frames as they are added to give an average rather than a sum.
* @param[in] r2c : Indicates whether the fullPlane Fourier plane is the result of a real-to-complex transform. If it is, then only half the Fourier plane would have been stored and hence the indexing will be different.
* @param[in] norm0 : A normalisation factor applied to the values within the window. This is always 1.0 these days.
* @param[in] batchCal : (Optional) pointer to a complex64 containing a complex amplitude to apply to the entire window. Used for calibrating amplitude/phase on an individual frame basis. e.g. compensating for differing camera exposures or reference delay/phase drift.
* @param[in] pixelSize : The dimension of a camera pixel in metres.
* @param[in] fillFactorCorrection : Indicated whether the sinc envelope of a 100% pixel fill factor should be corrected or not.
* @return totalPower : The total power within the selected window.
*/
float digHoloCopyWindow(int SX, int SY, int windowWidth, int windowHeight, int fullWidth, int fullHeight, complex64* windowOutput, complex64* fullPlane, unsigned char lowResMode, 
	float* kxAxis, float* kyAxis, float beamCentreX, float beamCentreY, float kx0, float ky0, float kr, 
	int avgIdx, int avgCount,int r2c, float norm0, complex64* batchCal, float pixelSize, int fillFactorCorrection)
{

	size_t windowStride = windowWidth;

	const __m256 beamCX = _mm256_set1_ps(beamCentreX);
	const __m256 halfPixelSize = _mm256_set1_ps(0.5f*pixelSize);

	//Normalizes the FFT-IFFT. Only reason this matters is so that different window sizes field the same power values
	const double fftLength = 1.0 * fullWidth * fullHeight;
	double ifftLength = 1.0 * windowWidth * windowHeight;
	if (!lowResMode)
	{
		ifftLength = fftLength;
	}
	else
	{

	}
	const __m256 normFactor = _mm256_set1_ps((float)(norm0 / sqrt(fftLength * ifftLength)));
	__m256 batchFactor;
	if (batchCal)
	{
		const float batchFactorR = batchCal[0][0];
		const float batchFactorI = batchCal[0][1];
		batchFactor = _mm256_set_ps(batchFactorI, batchFactorR, batchFactorI, batchFactorR, batchFactorI, batchFactorR, batchFactorI, batchFactorR);
	}
	else
	{
		batchFactor = _mm256_set_ps(0,1,0,1,0,1,0,1);
	}

	//const int fftshiftX = (fullWidth / 2);
	const int fftshiftY = (fullHeight / 2);

	const int SX0 = (SX - fullWidth / 2);

	const size_t sourceStride = r2c ? (fullWidth / 2 + 1) : fullWidth;

	int DX = 0;//Why was this DX=1 fudge factor here for low-resolution mode? It causes wrapping on the colIdx (width) axis
	int DY = 0;

	int fftshiftDY = 0;
	//int applyFFTshift = !r2c;

	if (!lowResMode)
	{
		DY = SY;
		if (r2c)
		{
			DX = (SX + fullWidth / 2) % fullWidth;
		//	DY = (SY + fullHeight/ 2) % fullHeight;
		}
		else
		{
			DX = SX;
			//DX = (SX + fullWidth / 2) % fullWidth;
			DY = SY;// (SY + fullHeight / 2) % fullHeight;
		}
		//DX = SX;
		windowStride = fullWidth;
		fftshiftDY = fullHeight / 2;
	}

	//Constants for multiplying packed complex numbers
	const __m256i overlapPermute1 = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	const __m256i overlapPermute2 = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);
	const __m256 signFlips = _mm256_set_ps(+1, -1, +1, -1, +1, -1, +1, -1);
	const __m256 conjFlips = _mm256_set_ps(-1, +1, -1, +1, -1, +1, -1, +1);
	const __m256 kx00 = _mm256_set1_ps(kx0);

	const __m256 r02 = _mm256_set1_ps(kr * kr);
	__m256 totalPwr = _mm256_set1_ps(0);

	const float fullKyAxisShift = abs((kyAxis[1] - kyAxis[0]) * fullHeight);
	//If we're averaging, and this isn't the first item in the average, then do two passes. Otherwise, do 1 pass.
	const int passCount = ((avgCount > 1 && avgIdx > 0) + 1);

	const __m256 zero = _mm256_set1_ps(0.0);
	const __m256 one = _mm256_set1_ps(1.0);
	//const __m256 oneComplex = _mm256_set_ps(0, 1, 0, 1, 0, 1, 0, 1);
	const __m256 avgFactor = _mm256_set1_ps((float)(1.0 / avgCount));
	__m256 avgR = zero;
	__m256 avgI = zero;

	__m256 avgR0;
	__m256 avgR1;
	__m256 avgI0;
	__m256 avgI1;

	for (int passIdx = 0; passIdx < passCount; passIdx++)
	{
		//If this is te second pass.
		if (passIdx == 1)
		{
			//avg0 = _mm256_add_ps(avg0, avg1);

			//float* avgRF = (float*)&avgR;
			//float* avgIF = (float*)&avgI;
			//Why did this used to only sum up 0 to 3 instead of 0 to 7? Forget to update from SSE to AVX?
			//float dR = avgRF[7] + avgRF[6] + avgRF[5] + avgRF[4]+avgRF[3] + avgRF[2] + avgRF[1] + avgRF[0];
			//float dI = avgIF[7] + avgIF[6] + avgIF[5] + avgIF[4]+avgIF[3] + avgIF[2] + avgIF[1] + avgIF[0];
			reduce(avgR);
			reduce(avgI);
			__m256 avgPwr = _mm256_add_ps(_mm256_mul_ps(avgR, avgR), _mm256_mul_ps(avgI, avgI));
			//Could probably get away with _mm256_rsqrt_ps here instead, but this is a once-off calculation so no real speed benefit.
			__m256 avgMag = _mm256_sqrt_ps(avgPwr);
			avgR = _mm256_div_ps(avgR, avgMag);
			avgI = _mm256_div_ps(avgI, avgMag);
			//float avgPwr = sqrtf((dR * dR + dI * dI));
			avgR = _mm256_mul_ps(avgFactor, avgR);
			avgI = _mm256_mul_ps(avgFactor, avgI);

			avgR0 = _mm256_permutevar8x32_ps(avgR, overlapPermute1);
			avgR1 = _mm256_permutevar8x32_ps(avgR, overlapPermute2);

			avgI0 = _mm256_mul_ps(signFlips, _mm256_permutevar8x32_ps(avgI, overlapPermute1));
			avgI1 = _mm256_mul_ps(signFlips, _mm256_permutevar8x32_ps(avgI, overlapPermute2));
		}

		for (int j = 0; j < windowHeight; j++)
		{
			//Output address
			size_t yOffsetDest;
			if (lowResMode)
			{
				yOffsetDest = (j + DY) * windowStride;
				if (!r2c)
				{
					yOffsetDest = (((j + DY) + windowHeight/2) % windowHeight) * windowStride;
				}
			}
			else
			{
				if (!r2c)
				{
					yOffsetDest = (((j + DY) + fftshiftDY) % fullHeight) * windowStride;
				}
				else
				{
					yOffsetDest = (((j + DY) + fftshiftDY) % fullHeight) * windowStride;
					//yOffsetDest = (((j + DY)) % fullHeight) * windowStride;
				}
				
			}

			//Source with fftshift
			const int jSY = j + SY;
			const size_t rowIdx = ((jSY)+fftshiftY)% fullHeight;
			const size_t yOffsetSource = (rowIdx) *sourceStride;
			const float kyAxisValue = kyAxis[rowIdx];
			const __m256 ky = _mm256_set1_ps(kyAxisValue);

			const __m256 kyPixelSize = _mm256_mul_ps(ky, halfPixelSize);
			const __m256 invSincKY = xcsc_ps(kyPixelSize);

			float ky0shift = 0;

			if ((jSY) < 0)
			{
				ky0shift = fullKyAxisShift;
			}
			else
			{
				if ((jSY) > fullHeight)
				{
					ky0shift = -fullKyAxisShift;
				}
			}
			const __m256 kyY = _mm256_set1_ps(kyAxisValue * beamCentreY);
			const __m256 dky = _mm256_set1_ps((kyAxisValue - (ky0 + ky0shift)));
			const __m256 dky2 = _mm256_mul_ps(dky, dky);

			for (int i = 0; i < windowWidth; i += 8)
			{
				int colIdx = i + SX0;

				//If this is a c2c transform, then you can wrap-around on this axis.
				//If it's an r2c you can't wrap around because that half of k-space isn't stored.
				//This wrap-around could also be implemented for r2c on this axis.
				if (colIdx < 0 && !r2c)
				{
					colIdx = (colIdx + fullWidth);
				}
				/*if (applyFFTshift)
				{
					colIdx = ((colIdx)+fftshiftX) % fullWidth;
				}
				*/

				//A scenario can occur whereby colIdx would be negative if the FourierWindow size is set enormous.
				if (colIdx >= 0 && colIdx<sourceStride)
				{
					const size_t sourceOffset = (colIdx + yOffsetSource);

					__m256 a0 = _mm256_loadu_ps((float*)&fullPlane[sourceOffset]);
					__m256 a1 = _mm256_loadu_ps((float*)&fullPlane[sourceOffset + 4]);
					a0 = cmul(a0, batchFactor);
					a1 = cmul(a1, batchFactor);


					const __m256 kx = _mm256_loadu_ps((float*)&kxAxis[colIdx]);
					const __m256 kxX = _mm256_mul_ps(beamCX, kx);
					const __m256 dkx = _mm256_sub_ps(kx, kx00);
					const __m256 dkx2 = _mm256_mul_ps(dkx, dkx);
					const __m256 dr2 = _mm256_add_ps(dkx2, dky2);
										
					const __m256 sincEnv = fillFactorCorrection ? _mm256_mul_ps(xcsc_ps(_mm256_mul_ps(kx, halfPixelSize)), invSincKY) : one;

					const __m256 insideWindow = _mm256_cmp_ps(dr2, r02, _CMP_LE_OQ);//const __m256 insideWindow = _mm256_cmp_ps(r02, r02, _CMP_LE_OQ);
					//const __m256 insideWindow = _mm256_cmp_ps(r02, r02, _CMP_LE_OQ);//Ignore kspace filtering in window
					const __m256 normF = _mm256_mul_ps(sincEnv,_mm256_and_ps(normFactor, insideWindow));

					const __m256 arg = _mm256_add_ps(kxX, kyY);
					__m256 adjustR;
					const __m256 adjustI = _mm256_mul_ps(normF, sincos_ps(&adjustR, arg));
					adjustR =  _mm256_mul_ps(adjustR, normF);

					const __m256 adjustR0 = _mm256_permutevar8x32_ps(adjustR, overlapPermute1);
					const __m256 adjustR1 = _mm256_permutevar8x32_ps(adjustR, overlapPermute2);

					const __m256 adjustI0 = _mm256_mul_ps(signFlips, _mm256_permutevar8x32_ps(adjustI, overlapPermute1));
					const __m256 adjustI1 = _mm256_mul_ps(signFlips, _mm256_permutevar8x32_ps(adjustI, overlapPermute2));

					const __m256 a0swap = _mm256_permute_ps(a0, 0xB1);//0b10110001//Swap real and imaginary components
					const __m256 a1swap = _mm256_permute_ps(a1, 0xB1);//0b10110001

					const __m256 out0 = _mm256_fmadd_ps(a0, adjustR0, _mm256_mul_ps(a0swap, adjustI0));
					const __m256 out1 = _mm256_fmadd_ps(a1, adjustR1, _mm256_mul_ps(a1swap, adjustI1));

					size_t destOffset = 0;// (i + DX + yOffsetDest);

					if (!r2c)
					{
						//Has fftshift in it
						if (lowResMode)
						{
							destOffset = (i + windowWidth / 2) % windowWidth + yOffsetDest;
						 }
						else
						{
							destOffset = ((i + DX+fullWidth/2)%fullWidth + yOffsetDest);
						}
					}
					else
					{
						//%width prevents any attempts to address outside the edge of the window.
						if (lowResMode)
						{
							destOffset = (i + DX)% windowWidth + yOffsetDest;
						}
						else
						{
							destOffset = (i + DX)% fullWidth + yOffsetDest;
						}
						
					}
					
					//If this is the last pass
					if (passIdx == (passCount - 1))
					{
						//The last pass of two passes (averaging)
						if (passCount == 2)
						{
							const __m256 out0swap = _mm256_permute_ps(out0, 0xB1);//0b10110001//[out0R out0I]
							const __m256 out1swap = _mm256_permute_ps(out1, 0xB1);//0b10110001//[out1R out1I]

							__m256 o0 = _mm256_fmadd_ps(out0, avgR0, _mm256_mul_ps(out0swap, avgI0)); //[out0R*avgI out0I*avgI]+[out0I*avgR out0R*avgR]
							__m256 o1 = _mm256_fmadd_ps(out1, avgR1, _mm256_mul_ps(out1swap, avgI1));

							const __m256 windowOut0 = _mm256_loadu_ps(windowOutput[destOffset]);
							const __m256 windowOut1 = _mm256_loadu_ps(windowOutput[destOffset + 4]);

							_mm256_storeu_ps((float*)&windowOutput[destOffset], _mm256_add_ps(windowOut0, o0));
							_mm256_storeu_ps((float*)&windowOutput[destOffset + 4], _mm256_add_ps(windowOut1, o1));
						}
						else
						{ //If this is the last and only pass, then just write out the data.
							_mm256_storeu_ps((float*)&windowOutput[destOffset], _mm256_mul_ps(avgFactor, out0));
							_mm256_storeu_ps((float*)&windowOutput[destOffset + 4], _mm256_mul_ps(avgFactor, out1));

						}

						__m256 dPwr = _mm256_add_ps(_mm256_mul_ps(out0, out0), _mm256_mul_ps(out1, out1));
						totalPwr = _mm256_add_ps(totalPwr, dPwr);
					}
					else //If it's not the final pass
					{
						const __m256 w0 = _mm256_loadu_ps(windowOutput[destOffset]);//wI wR
						const __m256 w1 = _mm256_loadu_ps(windowOutput[destOffset + 4]);
						const __m256 w0Conj = _mm256_mul_ps(w0, conjFlips);//-wI wR
						const __m256 w1Conj = _mm256_mul_ps(w1, conjFlips);

						const __m256 w0swap = _mm256_permute_ps(w0Conj, 0xB1);//0b10110001//Swap real and imaginary components (wR wI)
						const __m256 w1swap = _mm256_permute_ps(w1Conj, 0xB1);//0b10110001

						__m256 dR = _mm256_mul_ps(w0, out0);//wI*oI wR*oR
						__m256 dI = _mm256_mul_ps(w0swap, out0);//wR*oI -wI*oR

						dR = _mm256_add_ps(dR, _mm256_mul_ps(w1, out1));//wI*oI wR*oR
						dI = _mm256_add_ps(dI, _mm256_mul_ps(w1swap, out1));//wR*oI -wI*oR

						avgR = _mm256_add_ps(avgR, dR);
						avgI = _mm256_sub_ps(avgI, dI);//Fixed typo where this used to be a _add_ps
					}
				}
			}
		}
	}
	//float* pwr = (float*)&totalPwr;
	//return (pwr[0] + pwr[1] + pwr[2] + pwr[3] + pwr[4] + pwr[5] + pwr[6] + pwr[7]);
	return reduce(totalPwr);
}

/**
* @brief Overlaps the field (fieldR, fieldI), with the modes (modesX/modesY) and stores the partial result to overlap Work
*
* Is mostly done with int16 math, overlap sum uses double precision float so that there's no loss in precision during the operations. Probably overkill, but keeps precision as we've already rounded the fields to 16-bit (although remember the experimental camera frame at the start was likely less than 16 bit).
* This gets a bit speed up from working with a separable basis. Field is overlapped with the x-component of the mode, and then that's used to work out the overlaps for all the y-components for that x-mode.
* @param[in] fieldR : Pointer to the real component of the field (int16 format). Both polarisations (if applicable)
* @param[in] fieldI : Pointer to the imaginary component of the field (int16 format). Both polarisations (if applicable)
* @param[in] fieldScales : A pointer to scaling factors (for each polarisation) to be applied to fieldR and fieldI in order to convert them to their proper float32 values.
* @param[in] modesX0 : Pointer to a polCount x modeCount int16 array containing the x-component of the HG mode basis.
* @param[in] modesY0 : Pointer to a polCount x modeCount int16 array containing the y-component of the HG mode basis.
* @param[in] modeScalesX0 : Pointer to a polCount x modeCount float32 array containg the scaling factor which should be applied to modesX0 to convert them to their true float32 value
* @param[in] modeScalesY0 : Pointer to a polCount x modeCount float32 array containg the scaling factor which should be applied to modesY0 to convert them to their true float32 value
* @param[in] pixelCountX : The dimension of the modes and fields along the x-axis.
* @param[in] pixelCountY : The dimension of the modes and fields along the y-axis.
* @param[in] overlapWork : Pointer to memory used for storing partially accumulated values for the overlap summation.
* @param[out] coefsOut : Pointer to the location the final overlap coefficients should be written to.
* @param[in] MN : A lookup table that maps the Cartesian indicies of the HG modes (m,n) to their corresponding enumeration in the coefsOut array. -1 indicates that HG(m,n) mode doesn't exist.
* @param[in] modeStart : The starting mode to process along the x-axis. Used for threading mostly.
* @param[in] modeStop : The stopping mode (modeIdx<modeStop) to process along the x-axis. Used for threading mostly.
* @param[in] maxMG : The number of mode groups currently supported. e.g. maxMG=1 means only the fundamental mode.
* @param[in] polCount : The number of polarisation components in the field.
* @param[in] modeCount : The total number of spatial modes per polarisation. 
* @param[in] polStart : The first polarisation component to process. Mostly used for threading.
* @param[in] polStop : The last-1 (polIdx<polStop) polarisation component to process. Mostly used for threading.
*/
void overlapFieldSeparable16(__m256i* fieldR, __m256i* fieldI, float *fieldScales, 
	__m256i*** modesX0, short*** modesY0, float **modeScalesX0, float **modeScalesY0, 
	int pixelCountX, int pixelCountY, 
	complex64* coefsOut, 
	int** MN, int modeStart, int modeStop, int maxMG, 
	int polCount, int modeCount, 
	int polStart, int polStop)
{
	//The number of int16s in a 256-bit block
	const size_t blockSizeInt16 = 16;
	
	//How many mm256 elements between the polarisations (if applicable)
	const size_t polOffset = (((size_t)pixelCountX) * pixelCountY) / blockSizeInt16;

	//How many mm256i elements are there in mode x array (256bit = 16xint16)
	const size_t pxCount = ((size_t)pixelCountX) / blockSizeInt16;

	//Used for resetting values to zero, precalculated.
	const size_t block = blockSizeInt16;

	const __m256d zero = _mm256_set1_pd(0.0);

	//For each assigned polarisation component.
	for (int polIdx = polStart; polIdx < polStop; polIdx++)
	{
		//Get pointers to the modal components for this polarisation and their corresponding scales
		__m256i** modesX = modesX0[polIdx];
		short** modesY = modesY0[polIdx];
		float* modeScalesX = modeScalesX0[polIdx];
		float* modeScalesY = modeScalesY0[polIdx];
		//The overlap workspace memory for this polarisation component.
		//__m256d* overlapWorkH = overlapWork[polIdx];

		//For every assigned x-mode component
		for (int mIdx = modeStart; mIdx < modeStop; mIdx++)
		{
			//Get the pointer to the x-mode
			const __m256i* modeX = modesX[mIdx];

			//Get the pointer to a list of the y-mode indices for this x-mode index
			const int* HG_N = MN[mIdx];

			//Memory used for accumulating the real and imaginary component of the overlap
			__m256d dR[block * 2];
			__m256d* dI = &dR[block];

			//For every pixel component along the y-axis
			for (int pixelIdy = 0; pixelIdy < pixelCountY; pixelIdy += block)
			{
				//zero dR and dI (note the block*2) because dR and dI are adjacent in memory
				memset(dR, 0, sizeof(__m256d) * block * 2);

				//For every block (256-bit) block of x-mode pixels
				for (size_t pixelIdx = 0; pixelIdx < pxCount; pixelIdx++)
				{
					//Get 16 pixels worth of x-component mode
					const __m256i xValue16 = modeX[pixelIdx];

					//If the mode is all zero, don't bother doing anything else.
					const int allZeros = _mm256_testz_si256(xValue16, xValue16);

					if (!allZeros)
					{
						//For every pixel along the y-axis within a SIMD block. Accessing in a cache blocking style for more efficient reading of fieldR/I
						//Taking advantage of the y-pixels in fieldR/I being adjacent in memory.
						for (size_t pxIdy = 0; pxIdy < block; pxIdy++)
						{
							//Read in the real and imaginary components of the field
							const size_t idx = (pixelIdy + pxIdy) * (pxCount)+pixelIdx + polIdx * polOffset;
							const __m256i fR16 = fieldR[idx];
							const __m256i fI16 = fieldI[idx];

							//A 16-bit to 32-bit fused-multiply add operation.
							//Multiplies 16-pixels worth of real-component of the field with the x-component of the mode. Then sums adjacent products to give 8 32-bit integers out.
							__m256i fR0 = _mm256_madd_epi16(fR16, xValue16);//FMA : Now we've got 8 int32s, shouldn't overflow max is (32767*32767)*2 and min is -(32767*32767)*2
							//Coverts to int32->float64 and adds the two AVX lanes (1/2 a block) together
							dR[pxIdy] = _mm256_add_pd(dR[pxIdy], _mm256_add_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(fR0, 0)), _mm256_cvtepi32_pd(_mm256_extracti128_si256(fR0, 1))));
							//Same as dR but for the imaginary component
							__m256i fI0 = _mm256_madd_epi16(fI16, xValue16);
							dI[pxIdy] = _mm256_add_pd(dI[pxIdy], _mm256_add_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(fI0, 0)), _mm256_cvtepi32_pd(_mm256_extracti128_si256(fI0, 1))));
						}
					}// all zeros
				}//x pixels

				//Horizontal sum within the blocks, and then rearrange and pack into new blocks that remove the redundancy created by the sum.
				double* dR0 = (double*)&dR[0];
				double* dI0 = (double*)&dI[0];
				for (size_t pxIdy = 0; pxIdy < block; pxIdy++)
				{
					//Reduce (sum) the elements witihn the SIMD block.
					dR[pxIdy] = _mm256_add_pd(dR[pxIdy], _mm256_permute_pd(dR[pxIdy], 0x05));//0b0101
					dR[pxIdy] = _mm256_add_pd(dR[pxIdy], _mm256_permute2f128_pd(dR[pxIdy], dR[pxIdy], 1));

					dI[pxIdy] = _mm256_add_pd(dI[pxIdy], _mm256_permute_pd(dI[pxIdy], 0x05));//0b0101
					dI[pxIdy] = _mm256_add_pd(dI[pxIdy], _mm256_permute2f128_pd(dI[pxIdy], dI[pxIdy], 1));

					//Now write these block sums into the output array. 
					dR0[pxIdy] = dR0[pxIdy * 4];
					dI0[pxIdy] = dI0[pxIdy * 4];
				}

				//At this point, the dR0 and dI0 arrays should contain an array of the overlaps between the field and the x-mode component for each of 16 adjacent y-pixels.

				//Now go through and update the overlap for each y-component for the x-component above.
				//We currently have the overlap calculated for a block of 16 y-pixels of the field for a given x-mode, now we need to multiply these by the value of the y-components to get the overlaps for each y-component.
				for (int nIdx = 0; nIdx < maxMG; nIdx++)
				{
					//Get the mode index of the mode with the x-index above, and this y-index (nIdx)
					const int modeIDX = HG_N[nIdx];
					//Multiplied by two because we'll be using this as a memory offset and real/imaginary are interlaced
					const int modeIdx = 2 * modeIDX;
					
					//If this is a valid mode (modes that don't exist have previously been initialised to indices of -1 so that we can check here)
					if (modeIdx >= 0)
					{
						//Get a pointer to the y-component
						const short* modeY = modesY[nIdx];

						//Load the current partial overlap sum for this x-y mode
						__m256d dRsum = zero;
						__m256d dIsum = zero;

						//Load a block of 16 y-component pixels
						const __m256i yValue16 = _mm256_loadu_si256((__m256i*)&modeY[pixelIdy]);

						//If all the pixels are zero, don't do anything
						const int allZeros = _mm256_testz_si256(yValue16, yValue16);
						if (!allZeros)
						{
							//Now expand those 16 y pixels out to float64 format
							__m128i* yV128 = (__m128i*) &yValue16;
							__m256i yV32[2];//2 x {8 x int32}
							__m128i* yV32_128 = (__m128i*) & yV32[0];//4 x {4 x int32}
							yV32[0] = _mm256_cvtepi16_epi32(yV128[0]);//Convert the first 8 x int16s to int32
							yV32[1] = _mm256_cvtepi16_epi32(yV128[1]);//Conver the second 8 x int16s to int32

							//Convert the int32s to float64s
							__m256d yV64[4]; //A block of 16 x float64, containing the 16 x int16s of yValue16
							yV64[0] = _mm256_cvtepi32_pd(yV32_128[0]);
							yV64[1] = _mm256_cvtepi32_pd(yV32_128[1]);
							yV64[2] = _mm256_cvtepi32_pd(yV32_128[2]);
							yV64[3] = _mm256_cvtepi32_pd(yV32_128[3]);

							//Multiply the y-components with the partial overlap of the field with the x-components calculated previously and accumulate the sum
							dRsum = _mm256_fmadd_pd(dR[0], yV64[0], dRsum);
							dRsum = _mm256_fmadd_pd(dR[1], yV64[1], dRsum);
							dRsum = _mm256_fmadd_pd(dR[2], yV64[2], dRsum);
							dRsum = _mm256_fmadd_pd(dR[3], yV64[3], dRsum);

							//imaginary component
							dIsum = _mm256_fmadd_pd(dI[0], yV64[0], dIsum);
							dIsum = _mm256_fmadd_pd(dI[1], yV64[1], dIsum);
							dIsum = _mm256_fmadd_pd(dI[2], yV64[2], dIsum);
							dIsum = _mm256_fmadd_pd(dI[3], yV64[3], dIsum);
							
						}

						//Reduce (sum) the elements within the block.
							const double ovR = reduce(dRsum);
							const double ovI = reduce(dIsum);

							//Accumulate the overlap to the coefficients array
							complex64* coef = &coefsOut[polIdx * modeCount];

							if (pixelIdy == 0)
							{
								coef[modeIDX][0] = (float)ovR;
								coef[modeIDX][1] = (float)ovI;
							}
							else
							{
								coef[modeIDX][0] += (float)ovR;
								coef[modeIDX][1] += (float)ovI;

								if (pixelIdy==(pixelCountY-block))
								{
									//Here right at the end, we'll apply the scales of the X-modes, the Y-modes and the fields that were needed to convert the int16 values to the true float32 values.
									const double modeScaleX = modeScalesX[mIdx];
									const double modeScaleY = modeScalesY[nIdx];
									const double fieldScale = fieldScales[polIdx];
									//This is the total scaling factor that should be applied to our calculated overlaps, now that we're moving from integer-like math, to the true floating point math
									const float modeScale = (float)(modeScaleX * modeScaleY * fieldScale);

									coef[modeIDX][0] *= modeScale;
									coef[modeIDX][1] *= modeScale;
								}
							}
					}//if modeIdx>0
				}//nIdx
			}//pixelIdy
		}//mIdx
	}
}


/**
* @brief This routine analyses the input field(s) and extracts parameters such as centre of mass, effective area, total power, maximum value and location of maximum value.
* 
* This is used by the AutoAlign routine as it attempts to estimate parameters, but it is also used by functions such as 'applyTilt16' for converting fields from float32 to int16 (i.e. it needs to know the maximum value so it can choose an effective scaling factor)
*
* If operating from a float32 frame buffer, this pointer will be the same as that set by digHoloSetFrameBuffer() or similar routines such as digHoloSetBatch()
* If operating from an uint16 frame buffer, this pointer will be an internally buffer allocated and populated soon before the internal FFT is invoked.
* @param[in] handleIdx : enumerated handle index of the digHoloObject
* @return pointer  : pointer to either an internal (uint16 mode), or external (float32 mode) buffer, or a null pointer if the handleIdx is invalid or the buffer is not set.
*/
void digHoloFieldAnalysis(float* Af, float* x, float* y, int Nx_, int Ny_, 
	int startIDX, int stopIDX, int batchCount, 
	float* cxOut, float* cyOut, 
	float* totalPowerOut, 
	float* maxAbsOut, int* maxIdxOut, 
	float* AeffOut, 
	float* sumPower, 
	size_t batchStride, 
	float windowCentreX, float windowCentreY, float windowRadius, 
	unsigned char removeZeroOrder1D, 
	unsigned char wrapYaxis, float* cyWrapOut)
{
	const size_t Nx = Nx_;
	const size_t Ny = Ny_;
	size_t startIdx = startIDX;
	size_t stopIdx = stopIDX;
	const __m256 normFactor = _mm256_set1_ps((float)(1.0 / (Nx * Ny)));//This norm factor can be important, particulary to keep E^4 under control within limits of float (10^38)
	const size_t blockSize = 8;
	const __m256* x256 = (__m256*) & x[0];
	const __m256i permuteMask1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);//move imaginary components to the second lane, real parts to the first lane.
	const __m256i permuteMask2 = _mm256_set_epi32(6, 4, 2, 0, 7, 5, 3, 1);//move imagingary components to first lane, real parts to second lane.

	const __m256i doublePadA = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	const __m256i doublePadB = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);
	float w2 = windowRadius * windowRadius;

	//If the window is negative or zero, set it to "infinite" size window (no window).
	if (windowRadius == 0)
	{
		w2 = 0;// FLT_MAX;
	}

	const __m256 cmpSign = _mm256_set1_ps((float)(2 * (windowRadius >= 0) - 1));

	const __m256 cx = _mm256_set1_ps(-windowCentreX);
	const __m256 cy = _mm256_set1_ps(-windowCentreY);
	const __m256 r2 = _mm256_mul_ps(cmpSign, _mm256_set1_ps(w2));

	__m256 comX[2];
	__m256 comY[2];
	__m256 comYwrap[2];
	__m256 sumPwr[2];
	__m256 Epow4[2];
	__m256 maxV[2];
	__m256i maxIdx[2];

	//const __m256 dA = _mm256_set1_ps(pixelSizeX * pixelSizeY);

	//Max value stuff
	const unsigned int signMaskInt = 0x7FFFFFFF;
	const float* signMaskF = (float*)&signMaskInt;
	const __m256 signMask = _mm256_set1_ps(signMaskF[0]);

	const __m256i neg1 = _mm256_set1_epi32(-1);
	const __m256 zero = _mm256_set1_ps(0);

	const __m256i increment = _mm256_set1_epi32(8);
	const __m256i incrementInit = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

	const size_t NxRoundDown = blockSize * (Nx / blockSize);

	if (stopIdx > (size_t)Ny)
	{
		stopIdx = Ny;
	}

	comX[1] = zero;
	comY[1] = zero;
	comYwrap[1] = zero;
	sumPwr[1] = zero;
	Epow4[1] = zero;
	maxV[1] = zero;
	maxIdx[1] = neg1;
	const __m256i maxX = _mm256_set1_epi32((int)Nx);

	const __m256i minX = _mm256_set1_epi32(-(!removeZeroOrder1D));

	if (removeZeroOrder1D && startIdx == 0)
	{
		startIdx++;
	}
	const float fullKyAxisShift = abs((y[1] - y[0]) * Ny);

	for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
	{
		const size_t batchOffset = batchIdx * batchStride;

		comX[0] = zero;
		comY[0] = zero;
		comYwrap[0] = zero;
		sumPwr[0] = zero;
		Epow4[0] = zero;
		maxV[0] = zero;
		maxIdx[0] = neg1;

		//For every y pixel
		for (size_t j = startIdx; j < stopIdx; j++)
		{
			if (!removeZeroOrder1D || (removeZeroOrder1D && j != Ny / 2))
			{
				//Load in a single y-position to the whole block
				const __m256 yj = _mm256_set1_ps(y[j] + wrapYaxis * (fullKyAxisShift * (y[j] < 0)));
				const __m256 yjWrapped = _mm256_set1_ps(y[j] + (fullKyAxisShift * (y[j] < 0)));
				const __m256 yjWindow = _mm256_add_ps(yj, cy);
				const __m256 yjWindow2 = _mm256_mul_ps(yjWindow, yjWindow);
				//The memory offset for this y-position
				const size_t yOffset = j * Nx;// / blockSize;

				//This will increment within the x-loop below to move through the memory in the array
				__m256i idx = _mm256_add_epi32(_mm256_set1_epi32((int)(2 * j * Nx)), incrementInit);//2*j*Nx+[7,6,5,4,3,2,1,0] index of each position
				__m256i xIdx = incrementInit;
				size_t IDX = 0;
				//For every x-position
				for (size_t i = 0; i < (NxRoundDown); i += blockSize) //WARNING : Could miss the edge cases here/over run
				{
					//Load a block of 8 x-positions
					const __m256 xi = x256[IDX];
					const __m256 xiWindow = _mm256_add_ps(xi, cx);
					const __m256 xiWindow2 = _mm256_mul_ps(xiWindow, xiWindow);
					const __m256 radius2 = _mm256_mul_ps(cmpSign, _mm256_add_ps(xiWindow2, yjWindow2));
					const __m256 validX1 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(maxX, xIdx));
					const __m256 validX2 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(xIdx, minX));
					const __m256 validX = _mm256_and_ps(validX1, validX2);

					const __m256 insideWindow = _mm256_and_ps(validX, _mm256_cmp_ps(radius2, r2, _CMP_GE_OS));
					const __m256 insideWindowA = _mm256_permutevar8x32_ps(insideWindow, doublePadA);
					const __m256 insideWindowB = _mm256_permutevar8x32_ps(insideWindow, doublePadB);

					//insideWindow = _mm256_and_ps(insideWindow, validX);
					//insideWindow = _mm256_and_ps(insideWindow, _mm256_cmp_ps(xi, xmax, _CMP_LT_OS));
					//Load the absolute value of the real and imaginary components of the field. Interlaced R/I at this stage
					//The absolute value is taken, because we'll be looking at the maximum absolute values in the field for setting the scale when we convert these fields to int16 later. In that case, it's the maximum absolute value which defines how we scale that 16-bit integer range.
					const size_t IDX0 = (yOffset + i + batchOffset);
					__m256 Va = _mm256_loadu_ps(&Af[2 * IDX0]);
					__m256 Vb = _mm256_loadu_ps(&Af[2 * IDX0 + blockSize]);

					//lambda not implemented yet
				/*	if (RefCalibration && RefCalibrationWavelengthCount > 0)
					{
						float* cal =(float*)&RefCalibration[0][0];
						const size_t calIdx = (yOffset + i);
						//const __m256 cal0 = _mm256_loadu_ps(&cal[2*calIdx]);
						//const __m256 cal1 = _mm256_loadu_ps(&cal[2 * calIdx+blockSize]);
						float scaleIt = 1.0;
						const __m256 cal0 = _mm256_set_ps(0,scaleIt,0, scaleIt,0, scaleIt,0, scaleIt);
						const __m256 cal1 = _mm256_set_ps(0, scaleIt, 0, scaleIt, 0, scaleIt, 0, scaleIt);
						Va = cmul(Va, cal0);
						Vb = cmul(Vb, cal1);
						_mm256_storeu_ps(&Af[2 * IDX0], Va);
						_mm256_storeu_ps(&Af[2 * IDX0+blockSize], Vb);
						
					}
					*/

					const __m256 V1 = _mm256_and_ps(insideWindowA, _mm256_and_ps(Va, signMask));
					const __m256 V2 = _mm256_and_ps(insideWindowB, _mm256_and_ps(Vb, signMask));
					//Rearrange V1 and V2 which have interleaved R/I components, into separate blocks for real and imaginary (Vr,Vi)
					__m256 Vr = _mm256_and_ps(insideWindow, _mm256_blend_ps(_mm256_permutevar8x32_ps(V1, permuteMask1), _mm256_permutevar8x32_ps(V2, permuteMask2), 0xF0));//0b11110000//Real part
					__m256 Vi = _mm256_and_ps(insideWindow, _mm256_blend_ps(_mm256_permutevar8x32_ps(V1, permuteMask2), _mm256_permutevar8x32_ps(V2, permuteMask1), 0xF0));//0b11110000//imag part


					//Get the intensity, abs^2, of the field
					const __m256 Vsqrd0 = _mm256_fmadd_ps(Vr, Vr, _mm256_mul_ps(Vi, Vi));
					const __m256 Vsqrd = _mm256_mul_ps(normFactor, Vsqrd0);

					sumPwr[0] = _mm256_add_ps(sumPwr[0], Vsqrd);
					comX[0] = _mm256_fmadd_ps(xi, Vsqrd, comX[0]);
					comY[0] = _mm256_fmadd_ps(yj, Vsqrd, comY[0]);
					comYwrap[0] = _mm256_fmadd_ps(yjWrapped, Vsqrd, comYwrap[0]);
					//Total intensity^2 (used for effective area calculation)
					Epow4[0] = _mm256_fmadd_ps(Vsqrd, Vsqrd, Epow4[0]);

					//Max value
					__m256 Acmp = _mm256_cmp_ps(Vsqrd0, maxV[0], _CMP_GE_OS);
					__m256i* AcmpInt = (__m256i*) & Acmp;
					//__m256i AcmpInt0 = _mm256_castps_si256(Acmp);
					//__m256i* AcmpInt = &AcmpInt0;
					maxIdx[0] = _mm256_or_si256(_mm256_andnot_si256(AcmpInt[0], maxIdx[0]), _mm256_and_si256(AcmpInt[0], idx));
					maxV[0] = _mm256_max_ps(Vsqrd0, maxV[0]);
					idx = _mm256_add_epi32(idx, increment);

					/*
					Acmp = _mm256_cmp_ps(V2, maxV[0], _CMP_GE_OS);
					maxIdx[0] = _mm256_or_si256(_mm256_andnot_si256(AcmpInt[0], maxIdx[0]), _mm256_and_si256(AcmpInt[0], idx));
					maxV[0] = _mm256_max_ps(V2, maxV[0]);
					idx = _mm256_add_epi32(idx, increment);
					*/


					__m256 sumPwr256;

					if (batchIdx == 0)
					{
						sumPwr256 = zero;
					}
					else
					{
						sumPwr256 = _mm256_loadu_ps(&sumPower[yOffset + i]);
					}
					const __m256 VsqrdSum = _mm256_add_ps(sumPwr256, Vsqrd);

					//This could be unwrapped
					if (batchIdx == (batchCount - 1))
					{
						__m256 Acmp = _mm256_cmp_ps(VsqrdSum, maxV[1], _CMP_GE_OS);
						__m256i* AcmpInt = (__m256i*) & Acmp;
						maxIdx[1] = _mm256_or_si256(_mm256_andnot_si256(AcmpInt[0], maxIdx[1]), _mm256_and_si256(AcmpInt[0], idx));
						maxV[1] = _mm256_max_ps(VsqrdSum, maxV[1]);

						sumPwr[1] = _mm256_add_ps(sumPwr[1], VsqrdSum);
						comX[1] = _mm256_fmadd_ps(xi, VsqrdSum, comX[1]);
						comY[1] = _mm256_fmadd_ps(yj, VsqrdSum, comY[1]);
						comYwrap[1] = _mm256_fmadd_ps(yjWrapped, VsqrdSum, comYwrap[1]);
						Epow4[1] = _mm256_fmadd_ps(VsqrdSum, VsqrdSum, Epow4[1]);
					}


					_mm256_storeu_ps(&sumPower[yOffset + i], VsqrdSum);
					xIdx = _mm256_add_epi32(xIdx, increment);
					IDX++;
				}
			}
		}

		int stopIDX = 1;// +(batchIdx == (batchCount - 1));
		if (batchIdx == (batchCount - 1))
		{
			stopIDX = 2;
		}
		
		for (int i = 0; i < stopIDX; i++)
		{
			//Sum within the block
			comX[i] = _mm256_add_ps(comX[i], _mm256_permute_ps(comX[i], 0xB1));//0b10110001
			comX[i] = _mm256_add_ps(comX[i], _mm256_permute_ps(comX[i], 0x4E));//0b01001110
			comX[i] = _mm256_add_ps(comX[i], _mm256_permute2f128_ps(comX[i], comX[i], 1));

			//Sum within the block
			comY[i] = _mm256_add_ps(comY[i], _mm256_permute_ps(comY[i], 0xB1));//0b10110001
			comY[i] = _mm256_add_ps(comY[i], _mm256_permute_ps(comY[i], 0x4E));//0b01001110
			comY[i] = _mm256_add_ps(comY[i], _mm256_permute2f128_ps(comY[i], comY[i], 1));

			comYwrap[i] = _mm256_add_ps(comYwrap[i], _mm256_permute_ps(comYwrap[i], 0xB1));//0b10110001
			comYwrap[i] = _mm256_add_ps(comYwrap[i], _mm256_permute_ps(comYwrap[i], 0x4E));//0b01001110
			comYwrap[i] = _mm256_add_ps(comYwrap[i], _mm256_permute2f128_ps(comYwrap[i], comYwrap[i], 1));

			//Sum within the block
			sumPwr[i] = _mm256_add_ps(sumPwr[i], _mm256_permute_ps(sumPwr[i], 0xB1));//0b10110001
			sumPwr[i] = _mm256_add_ps(sumPwr[i], _mm256_permute_ps(sumPwr[i], 0x4E));//0b01001110
			sumPwr[i] = _mm256_add_ps(sumPwr[i], _mm256_permute2f128_ps(sumPwr[i], sumPwr[i], 1));

			//Sum within the block
			Epow4[i] = _mm256_add_ps(Epow4[i], _mm256_permute_ps(Epow4[i], 0xB1));//0b10110001
			Epow4[i] = _mm256_add_ps(Epow4[i], _mm256_permute_ps(Epow4[i], 0x4E));//0b01001110
			Epow4[i] = _mm256_add_ps(Epow4[i], _mm256_permute2f128_ps(Epow4[i], Epow4[i], 1));

			//comX[i] = _mm256_div_ps(comX[i], sumPwr[i]);//Centre of mass in x
			//comY[i] = _mm256_div_ps(comY[i], sumPwr[i]);//Centre of mass in y

			//Epow4[i] = _mm256_div_ps(dA, Epow4[i]);
			//Aef[i] = _mm256_mul_ps(_mm256_mul_ps(sumPwr[i], sumPwr[i]), Epow4[i]);//Effective area (Petermann II)

			__m256 maxV1 = maxV[i];
			maxV[i] = _mm256_max_ps(maxV[i], _mm256_permute_ps(maxV[i], 0xB1));//0b10110001//Compare to their neighbours (blocks of 1)
			maxV[i] = _mm256_max_ps(maxV[i], _mm256_permute_ps(maxV[i], 0x4E));//0b01001110//Compare blocks of 2 to neighbouring blocks of 2
			maxV[i] = _mm256_max_ps(maxV[i], _mm256_permute2f128_ps(maxV[i], maxV[i], 1));//Compare blocks of 4 to neighbouring blocks of 4

			__m256 maxPosF = _mm256_cmp_ps(maxV[i], maxV1, _CMP_EQ_OS);
			__m256i* maxPos32 = (__m256i*) & maxPosF;

			//mask the values which have the maximum value. Replace the rest with indices of -1 (in case the max value happens to be index 0)
			maxIdx[i] = _mm256_or_si256(_mm256_andnot_si256(maxPos32[0], neg1), _mm256_and_si256(maxPos32[0], maxIdx[i]));
			//If multiple elements have the same value, this will take the one with the maximum index (could also do min). To do min you should change flag to _CMP_GT_OS
			maxIdx[i] = _mm256_max_epi32(maxIdx[i], _mm256_permutevar8x32_epi32(maxIdx[i], _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1)));
			maxIdx[i] = _mm256_max_epi32(maxIdx[i], _mm256_permutevar8x32_epi32(maxIdx[i], _mm256_set_epi32(4, 5, 6, 7, 0, 1, 2, 3)));
			maxIdx[i] = _mm256_max_epi32(maxIdx[i], _mm256_permute2f128_si256(maxIdx[i], maxIdx[i], 1));//Compare blocks of 4 to neighbouring blocks of 4

			//For i==0, outIdx will be batchIdx.
			//For i==1, this loop shouldn't have run at all unless batchIdx==batchCount-1, in which case batchIdx will then equal batchCount
			const size_t outIdx = batchIdx + i;

			float* cx_ = (float*)&comX[i];
			float* cy_ = (float*)&comY[i];
			float* cyWrap_ = (float*)&comYwrap[i];
			float* E4 = (float*)&Epow4[i];
			float* totalP = (float*)&sumPwr[i];

			int* maxV32 = (int*)&maxIdx[i];
			float* maxVf = (float*)&maxV[i];

			AeffOut[outIdx] = E4[0];
			cxOut[outIdx] = cx_[0];
			cyOut[outIdx] = cy_[0];
			cyWrapOut[outIdx] = cyWrap_[0];
			totalPowerOut[outIdx] = totalP[0];
			maxAbsOut[outIdx] = sqrtf(maxVf[0]);// = 32767.0 / maxAbsOut
			maxIdxOut[outIdx] = maxV32[0];
		}
		
	}
}

int WavelengthOrderingCalc(int idxIn, int lambdaCount, int batchCount, int orderingIn, int orderingOut, int& lambdaIdx, int& subBatchIdx)
{
	if (lambdaCount > 0)
	{
		int subBatchCount = batchCount / lambdaCount;

		//Wavelength fast axis
		if (!orderingIn)
		{
			lambdaIdx = idxIn % lambdaCount;
			subBatchIdx = idxIn / lambdaCount;
		} //
		else//Wavelength slow axis (i.e. wavelength 1, wavelength 1, wavelength 1, wavelength 2, wavelength 2, wavelength 2....
		{
			
			if (subBatchCount)
			{
				subBatchIdx = idxIn % subBatchCount;
				lambdaIdx = idxIn / subBatchCount;
			}
			else
			{
				lambdaIdx = 0;
				subBatchIdx = 0;
			}
		}

		if (!orderingOut)//Wavelength fast axis
		{
			return subBatchIdx * lambdaCount + lambdaIdx;
		}
		else
		{
			return lambdaIdx * subBatchCount + subBatchIdx;
		}

	}
	else //You should never get here (lambdaCount<=0)
	{
		lambdaIdx = 0;
		subBatchIdx = 0;
		return idxIn;
	}
}

void applyTilt16(float** scales, 
	int pixelStart_, int pixelStop_, int pixelCountX, int pixelCountY, 
	int polStart, int polStop, int polCount, int batchStart_, 
	int batchStop_, int batchCount_, 
	complex64* fieldIn, short* fieldOutR, short* fieldOutI, 
	int lambdaCount, 
	complex64* outArray, 
	short*** refWave, 
	int orderingIn, int orderingOut)
{
	//size_t batchCount = batchCount_;
	const size_t blockSize = 8;
	const size_t pixelStart = pixelStart_;
	const size_t pixelStop = pixelStop_;
	const size_t batchStart = batchStart_;
	const size_t batchStop = batchStop_;
	const size_t batchCount = batchCount_;
	//const __m256i zero16 = _mm256_set1_epi16(0);
	const size_t pxCount = pixelCountX / blockSize;

	const __m256i overlapPermute1 = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	const __m256i overlapPermute2 = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	const __m256i permuteMask1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);//move imaginary components to the second lane, real parts to the first lane.
	const __m256i permuteMask2 = _mm256_set_epi32(6, 4, 2, 0, 7, 5, 3, 1);//move imagingary components to first lane, real parts to second lane.
	const __m256 normF = _mm256_set1_ps(1);
	
	const size_t fieldStride = pixelCountX * pixelCountY;
	const size_t batchStride = fieldStride * polCount;
	const size_t stride = batchStride / blockSize;
	//const size_t frameCount = batchCount / lambdaCount;

	for (size_t polIdx = polStart; polIdx < polStop; polIdx++)
	{
		float* scale = scales[polIdx];

		__m256 *A = (__m256*)&fieldIn[polIdx * fieldStride];
		__m128i* AoutR = (__m128i*)&fieldOutR[polIdx * fieldStride];
		__m128i* AoutI = (__m128i*)&fieldOutI[polIdx * fieldStride];

		//__m256* cal = (__m256*)&RefCalibration[polIdx * fieldStride];

		__m256 *outArray32 = 0;
		if (outArray)
		{
			outArray32 = (__m256*)&outArray[polIdx * fieldStride];
		}

			const __m256 Int32toFloat32scale = _mm256_set1_ps((float)(1.0 / (16383.0 * 16383.0)));//15-bit so we don't overflow 32-bit when we go ([int16 x int16] + [int16 x int16])

			for (size_t batchIdx = batchStart; batchIdx < batchStop; batchIdx++)
			{
				
				//int lambdaIdx = batchIdx % lambdaCount;
				int lambdaIdx;
				int subBatchIdx;
				size_t batchIdxIn = batchIdx;
				WavelengthOrderingCalc((int)batchIdx, lambdaCount, (int)batchCount, orderingIn, orderingOut, lambdaIdx, subBatchIdx);
				size_t batchIdxOut = batchIdx;
				//scale[batchIdx] has to be rearranged as well!

				__m128i* adjustXR = (__m128i*) &refWave[lambdaIdx][polIdx][0];
				__m128i* adjustXI = (__m128i*) &refWave[lambdaIdx][polIdx][pixelCountX];
				short* adjustYR = &refWave[lambdaIdx][polIdx][2 * pixelCountX];
				short* adjustYI = &refWave[lambdaIdx][polIdx][2 * pixelCountX + pixelCountY];

				for (size_t pixelIdy = pixelStart; pixelIdy < pixelStop; pixelIdy++)
				{
					const size_t yOffset = pixelIdy * pxCount * 2;
					const __m256i refYR = _mm256_set1_epi32(adjustYR[pixelIdy]);
					const __m256i refYI = _mm256_set1_epi32(adjustYI[pixelIdy]);

					for (size_t pixelIdx = 0; pixelIdx < pxCount; pixelIdx++)
					{
						const __m256i refXR = _mm256_cvtepi16_epi32(adjustXR[pixelIdx]);
						const __m256i refXI = _mm256_cvtepi16_epi32(adjustXI[pixelIdx]);

						const __m256i refRa = _mm256_mullo_epi32(refYR, refXR);
						const __m256i refRb = _mm256_mullo_epi32(refYI, refXI);
						const __m256i refIa = _mm256_mullo_epi32(refYR, refXI);
						const __m256i refIb = _mm256_mullo_epi32(refYI, refXR);

						const __m256 adjustR = _mm256_mul_ps(Int32toFloat32scale, _mm256_cvtepi32_ps(_mm256_sub_epi32(refRa, refRb)));
						const __m256 adjustI = _mm256_mul_ps(Int32toFloat32scale, _mm256_cvtepi32_ps(_mm256_add_epi32(refIa, refIb)));

						const size_t idxIn = yOffset + (pixelIdx + batchIdxIn * stride) * 2;
						const size_t idxOut = yOffset + (pixelIdx + batchIdxOut * stride) * 2;

						//Read in complex fields (8 pixels worth, 16 elements)
						const __m256 a0 = _mm256_mul_ps(normF, A[idxIn]);//Complex field
						const __m256 a1 = _mm256_mul_ps(normF, A[idxIn + 1]);//Complex field
						//Separate into real and imaginary components
						const __m256 Ar = _mm256_blend_ps(_mm256_permutevar8x32_ps(a0, permuteMask1), _mm256_permutevar8x32_ps(a1, permuteMask2), 0xF0);//0b11110000//Real part
						const __m256 Ai = _mm256_blend_ps(_mm256_permutevar8x32_ps(a0, permuteMask2), _mm256_permutevar8x32_ps(a1, permuteMask1), 0xF0);//0b11110000//imag part
						//Multiply by tilt mask
						__m256 Vr = _mm256_sub_ps(_mm256_mul_ps(Ai, adjustI), _mm256_mul_ps(Ar, adjustR));
						__m256 Vi = _mm256_add_ps(_mm256_mul_ps(Ar, adjustI), _mm256_mul_ps(Ai, adjustR));

						//lambda not implemented yet
						/*if (RefCalibration && RefCalibrationWavelengthCount > 0)
						{
							size_t idxCal = yOffset + (pixelIdx + 0 * stride) * 2;;
							const __m256 cal0 = cal[idxCal];
							const __m256 cal1 = cal[idxCal + 1];
							const __m256 calr = _mm256_blend_ps(_mm256_permutevar8x32_ps(cal0, permuteMask1), _mm256_permutevar8x32_ps(cal1, permuteMask2), 0xF0);//0b11110000//Real part
							const __m256 cali = _mm256_blend_ps(_mm256_permutevar8x32_ps(cal0, permuteMask2), _mm256_permutevar8x32_ps(cal1, permuteMask1), 0xF0);//0b11110000//imag part

							__m256 vR = _mm256_sub_ps(_mm256_mul_ps(Vr, calr), _mm256_mul_ps(Vi, cali));
							__m256 vI = _mm256_add_ps(_mm256_mul_ps(Vr, cali), _mm256_mul_ps(Vi, calr));
							Vr = vR;
							Vi = vI;
						}*/

						if (outArray32)
						{
							outArray32[idxOut] = _mm256_blend_ps(_mm256_permutevar8x32_ps(Vr, overlapPermute1), _mm256_permutevar8x32_ps(Vi, overlapPermute1), 0xAA);//0b10101010
							outArray32[idxOut + 1] = _mm256_blend_ps(_mm256_permutevar8x32_ps(Vr, overlapPermute2), _mm256_permutevar8x32_ps(Vi, overlapPermute2), 0xAA);//0b10101010
						}

						//Scale and round to int (now should be between -32767 and +32767)
						const __m256 scaleF = _mm256_set1_ps(scale[batchIdx]);
						const __m256 aR = _mm256_round_ps(_mm256_mul_ps(scaleF, Vr), _MM_FROUND_TO_NEAREST_INT);
						const __m256 aI = _mm256_round_ps(_mm256_mul_ps(scaleF, Vi), _MM_FROUND_TO_NEAREST_INT);

						const __m256i aR32 = _mm256_cvtps_epi32(aR);//8 int32 (256 bits)
						const __m256i aI32 = _mm256_cvtps_epi32(aI);

						const __m128i aR16 = _mm256_extractf128_si256(_mm256_packs_epi32(aR32, _mm256_permute2f128_si256(aR32, aR32, 1)), 0);
						const __m128i aI16 = _mm256_extractf128_si256(_mm256_packs_epi32(aI32, _mm256_permute2f128_si256(aI32, aI32, 1)), 0);

						AoutR[idxOut / 2] = aR16;
						AoutI[idxOut / 2] = aI16;
					}
				}
			}
	}
}


template<typename T>
bool GenerateCoordinatesXY(int pixelCountX, int pixelCountY, T pixelSizeX, T pixelSizeY, T* xAxis, T* yAxis, int fftshift)
{
	if (fftshift)
	{
		for (int i = 0; i < pixelCountX; i++)
		{
			if (i <= (pixelCountX / 2))
			{
				xAxis[i] = pixelSizeX * (i);
			}
			else
			{
				xAxis[i] = pixelSizeX * (i - pixelCountX);
			}
		}

		for (int j = 0; j < pixelCountY; j++)
		{
			if (j <= (pixelCountY / 2))
			{
				yAxis[j] = pixelSizeY * j;
			}
			else
			{
				yAxis[j] = pixelSizeY * (j - pixelCountY);
			}
		}
	}
	else
	{
		for (int i = 0; i < pixelCountX; i++)
		{
			xAxis[i] = (i - pixelCountX / 2) * pixelSizeX;
		}

		for (int j = 0; j < pixelCountY; j++)
		{
			yAxis[j] = (j - pixelCountY / 2) * pixelSizeY;
		}
	}

	return false;
}

//Makes an estimate of the mode field diameter (beam radius x 2), based on the effective area, and the max mode group.
//The numbers in here were simulated and then fit to a quadratic dependence.
float AeffToMFD(float Aeff, int groupCount)
{
	if (groupCount)
	{
		float fudgeFactor = 1.00;
		float A = (float)(Aeff / (15.1059697707643 * groupCount + 6.65360122093789));
		float mfd = fudgeFactor * (float)(2 * sqrtf(A / pi) / 0.223008930683135);
		return mfd;
	}
	else
	{
		return 0;
	}
}

//Makes an estimate on what the radius of your window should be to completely enclose modes up to maxMG (maxMG=1=Fundamental)
//The numbers in here were found by simulation and fitting to quadratic.
float EstimateRequiredWindow(int maxMG, float waist)
{
	float x0 = 2 * waist;
	return sqrtf((float)(-1.3425626201e-04 * (x0 * x0) + 2.7718473619e-01 * x0 + 9.7044626301e-01));
}

//enumeration for 'resolution mode'. i.e. whether the reconstructed field has the same number of pixels as the original camera source image (full res) or low res, where the reconstructed field has the number of pixels of the window in the Fourier plane (low-res).
//Both modes contains the same information.
#define FULLRES_IDX 0
#define LOWRES_IDX 1

#define FFT_IDX 0
#define IFFT_IDX 1
#define FFTCAL_IDX 2
#define IFFTCAL_IDX 3

#define FFTW_HOWMANYMAX 4

//Alias names for the different aberration.
//Not really necessary here because the zernikes are calculated differently from other modules, but the numbering is kept the same for consistency.
//#define ZERN_MAX_ORDER 3
//#define PISTON 0
#define TILTX 1
#define TILTY 2
//#define ASTIGX 3
#define DEFOCUS 4
//#define ASTIGY 5
//#define TREFOILX 6
//#define COMAX 7
//#define COMAY 8
//#define TREFOILY 9
//#define SPHERICAL 12

//Used in autoalignment routine. Defines the order to optimise parameters, and how to store results in arrays.
#define AUTOALIGN_DEFOCUS 1
#define AUTOALIGN_WAIST 0
#define AUTOALIGN_TILTX 2
#define AUTOALIGN_CX 3
#define AUTOALIGN_TILTY 4
#define AUTOALIGN_CY 5

#define FFTW_PLANMODECOUNT 4
const int FFTW_PLANMODES[FFTW_PLANMODECOUNT] = { FFTW_ESTIMATE,FFTW_MEASURE,FFTW_PATIENT,FFTW_EXHAUSTIVE };

//#pragma once

class digHoloObject;

std::vector<digHoloObject*> digHoloObjects;
std::vector<int> digHoloObjectsIdx;
//The number of valid digHoloObjects in the digHoloObjects arrays. This can be different from the length of digHoloObjects if objects are deleted.
//The position of a given digHoloObject in the digHoloObjects vector never changes. Hence it's index in the vector can be used as a handle by external programs.
//If the object is deleted, the length of digHoloObjects will not be changed, but digHoloObjectsCount will decrement.
int digHoloObjectsCount = 0;

#define FFTW_WISDOM_FILENAME_MAXLENGTH 2048
char FFTW_WISDOM_FILENAME[FFTW_WISDOM_FILENAME_MAXLENGTH];
char MACHINENAME[FFTW_WISDOM_FILENAME_MAXLENGTH];
int fftwWisdomNew = 0;
int fftwInitialised = 0;
int fftwWisdomCustomFilename = 0;

//https://codereview.stackexchange.com/questions/75606/c11-class-similar-to-nets-manualresetevent-but-without-the-ability-to-rese
class ManualResetEvent
{
public:
	explicit ManualResetEvent(bool initial = false);

	void Set();
	void Reset();

	bool WaitOne();
	bool WaitNotOne();
private:
	ManualResetEvent(const ManualResetEvent&);
	ManualResetEvent& operator=(const ManualResetEvent&); // non-copyable
	bool flag_ = false;
	std::mutex protect_;
	std::condition_variable signal_;
};

ManualResetEvent::ManualResetEvent(bool initial)
	: flag_(initial)
{
}

void ManualResetEvent::Set()
{
	std::lock_guard<std::mutex> _(protect_);
	flag_ = true;
	signal_.notify_one();
}

void ManualResetEvent::Reset()
{
	std::lock_guard<std::mutex> _(protect_);
	flag_ = false;
}

bool ManualResetEvent::WaitOne()
{
	std::unique_lock<std::mutex> lk(protect_);
	while (!flag_) // prevent spurious wakeups from doing harm
		signal_.wait(lk);
	//flag_ = false; // waiting resets the flag
	return true;
}
bool ManualResetEvent::WaitNotOne()
{
	std::unique_lock<std::mutex> lk(protect_);
	while (flag_) // prevent spurious wakeups from doing harm
		signal_.wait(lk);
	//flag_ = false; // waiting resets the flag
	return true;
}

//Performance the SVD of 'a' and return the singular values 's'
int svdSingularValues(int M, int N, complex64* a, float* s)
{
	//M number of rows in A
	//N number of cols in A
	const int LDA = M;//Leading dimension of A (max(1,M)
	const int LDU = M;//Leading dimension of U (
	const int LDVT = N;

	int info = 0;//NULL;

	complex64* U = NULL;
	complex64* VT = NULL;

	const char jobu = 'N';
	const char jobv = 'N';
	const int LWORK = M * N;
	float* WORK = 0;
	allocate1D(2 * LWORK, WORK);
	float* RWORK = 0;
	allocate1D(LWORK, RWORK);

#ifdef LAPACKBLAS_ENABLE
	cgesvd(&jobu, &jobv, &M, &N, (BLAS_COMPLEXTYPE*)a, &LDA, s, (BLAS_COMPLEXTYPE*)U, &LDU, (BLAS_COMPLEXTYPE*)VT, &LDVT, (BLAS_COMPLEXTYPE*)WORK, &LWORK, RWORK, &info);
	//LAPACKE_cgesvd(LAPACK_COL_MAJOR,jobu, jobv, M, N, (BLAS_COMPLEXTYPE*)a, LDA, s, (BLAS_COMPLEXTYPE*)U, LDU, (BLAS_COMPLEXTYPE*)VT, LDVT, WORK);
/* int matrix_layout,
	char jobu,
		char jobvt,
		lapack_int m, lapack_int n,
		lapack_complex_float* a,
		lapack_int lda, float* s,
		lapack_complex_float* u,
		lapack_int ldu, lapack_complex_float* vt,
		lapack_int ldvt, float* superb );
		*/
#endif
	free1D(WORK);
	free1D(RWORK);
	return !(info > 0);
}

//Multiplies A by conjugate transpose of B and outputs to y
void UUconj(int Ny, int Nx, complex64* A, complex64* B, complex64* y)
{
	if (Ny >= Nx)
	{
		//A is Nx x Ny
		//B is Ny x Nx
		//C is Nx x Nx

		int M = Nx; //Number of rows of A and C
		int N = Nx; //Number of cols of B and C
		int K = Ny; //Number of columns of A and number of rows of B

		int LDA = Ny; //A (LDA,M) : When 'C' (LDAxM)
		int LDB = Ny; //B (LDB,N) : When 'N' (MxN)
		int LDC = Nx; //C (LDC,N)

		complex64 alpha;
		complex64 beta;

		beta[0] = 0;
		beta[1] = 0;

		alpha[0] = 1;
		alpha[1] = 0;


#ifdef LAPACKBLAS_ENABLE
#ifdef CBLAS_ENABLE
		cblas_cgemm(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasConjTrans, CBLAS_TRANSPOSE::CblasNoTrans, M, N, K, &alpha, A, LDA, B, LDB, &beta, y, LDC);
#else
		const char transA = 'C';
		const char transB = 'N';
		cgemm(&transA, &transB, &M, &N, &K, (BLAS_COMPLEXTYPE*)&alpha, (BLAS_COMPLEXTYPE*)A, &LDA, (BLAS_COMPLEXTYPE*)B, &LDB, (BLAS_COMPLEXTYPE*)&beta, (BLAS_COMPLEXTYPE*)y, &LDC);
#endif	
#endif
	}
	else
	{
		printf("Error : Attempted to find A*A' of a matrix that's wider than it is high. e.g. batchCount is larger than mode count\n");
	}
}

//Calculates the average singular value of the matrix A
float insertionLoss256(int Nx, int Ny, complex64* A)
{

	const int blockSize = 4;//elements per block (4 means 4 complex numbers, i.e. 8 float32)

	const int blockCount = Ny / blockSize;

	//Used to check whether some edge-case parts of the matrix need to be treated non-SIMD
	const int remStart = blockCount * blockSize;
	const int remStop = Ny;

	//The tallies used to sum up the elements of the matrix
	__m256 v256 = _mm256_set1_ps(0);
	float v = 0;

	for (int i = 0; i < Nx; i++)
	{
		const float* Aptr = &A[i * Ny][0];
		for (int j = 0; j < blockCount; j++)
		{
			const __m256 a = _mm256_loadu_ps(&Aptr[j * 2 * blockSize]);
			v256 = _mm256_fmadd_ps(a, a, v256);
		}

		for (int j = remStart; j < remStop; j++)
		{
			const float aR = Aptr[2 * j];
			const float aI = Aptr[2 * j + 1];

			v += (aR * aR + aI * aI);
		}

	}

	float* vPtr = (float*)&v256;
	v = v + vPtr[0] + vPtr[1] + vPtr[2] + vPtr[3] + vPtr[4] + vPtr[5] + vPtr[6] + vPtr[7];//lazy horizontal add
	v = v / Nx;
	return v;
}

//fits the function y at values x to a quadratic, starting guess parameters of B0 and final output polynomial values of B.
float fitToQuadratic(float* x, float* y, int sampleCount, float* B, float* B0, float tol, float maxIter)
{

	int N = sampleCount;
	unsigned int done = 0;
	unsigned int parameterCount = 3;

	float* Jr = 0;// (float*)malloc(sizeof(float) * N * parameterCount);

	float* r = 0;// (float*)malloc(sizeof(float) * N);
	float* work = 0;

	const int lwork = N * N * parameterCount * 2;


	allocate1D(lwork, work);
	allocate1D(N * parameterCount, Jr);
	allocate1D(N, r);

	float aa = B0[0];
	float bb = B0[1];
	float cc = B0[2];

	unsigned int iter = 0;
	float lastResid = 0;
	int i;
	int m = N;
	int n = parameterCount;
	int lda = N;
	int ldb = N;
	int info;

	int nrhs = 1;
	float* a = Jr;
	float* b = r;

	float resid = 0;
	unsigned char valid = 1;

	float maxPoint = 0;

	while (!done)
	{
		int k = 0;

		for (i = 0; i < N; i++)
		{
			Jr[k] = x[i] * x[i];

			float yy = aa * x[i] * x[i] + bb * x[i] + cc;
			r[i] = y[i] - yy;

			if (isnan(Jr[k]) || isnan(r[i]))
			{
				valid = 0;
			}
			k++;
		}
		if (valid)
		{
			for (i = 0; i < N; i++)
			{
				Jr[k] = x[i]; k++;
			}

			for (i = 0; i < N; i++)
			{
				Jr[k] = 1; k++;
			}

			for (i = 0; i < N; i++)
			{
				resid += r[i] * r[i];
			}

			//NOTE a=Jr and b=r
			const char trans = 'N';
#ifdef LAPACKBLAS_ENABLE
			sgels(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
			//LAPACKE_sgels(LAPACK_COL_MAJOR, 'N', m, n, nrhs, a, lda, b, ldb);
#endif

			aa = aa + b[0];
			bb = bb + b[1];
			cc = cc + b[2];

			B[0] = aa;
			B[1] = bb;
			B[2] = cc;
			//printf("	%10.10i	%10.10f	%10.10f	%10.10f\n",iter,AB,phiB,offsetB);
			iter++;
			//if (iter > maxIter)
			if (abs(resid - lastResid) < tol || iter > maxIter)
			{
				done = 1;
			}
			lastResid = resid;
			maxPoint = -B[1] / (2 * B[0]);
		}//valid
		else
		{
			done = 1;
			maxPoint = NAN;
		}
	}

	free1D(r);
	free1D(Jr);
	free1D(work);
	return maxPoint;
}

enum workFunction { NA, digHoloFFTBatchWorker, digHoloIFFTRoutineWorker, applyTiltWorker, fieldAnalysisWorker, overlapWorkerHG, powerMeter };


//Workpackage used for distributing work to threads.
//A bunch of variables that pass on information like indicies, pointers to relevant arrays, and ResetEvent to flag when the thread is finished.
//This could be better implemented using proper arrays instead of 'ptr1, ptr2...', that would also mean workPackage is created faster, and only for the necessary size.
struct workPackage
{
	int threadIdx = 0;
	int totalThreads = 0;
	size_t start = 0;
	size_t stop = 0;
	void* ptr1 = 0;
	void* ptr2 = 0;
	void* ptr3 = 0;
	void* ptr4 = 0;
	void* ptr5 = 0;
	void* ptr6 = 0;
	void* ptr7 = 0;
	void* ptr8 = 0;
	void* ptr9 = 0;
	void* ptr10 = 0;
	void* ptr11 = 0;
	void* ptr12 = 0;
	void* ptr13 = 0;
	void* ptr14 = 0;
	int flag1 = 0;
	int flag2 = 0;
	int flag3 = 0;
	int flag4 = 0;
	int flag5 = 0;
	int flag6 = 0;
	int flag7 = 0;
	int flag8 = 0;
	int flag9 = 0;
	int flag10 = 0;
	int flag11 = 0;
	int flag12 = 0;
	int flag13 = 0;
	int flag14 = 0;
	float var1 = 0;
	float var2 = 0;
	float var3 = 0;
	float var4 = 0;
	float var5 = 0;
	float var6 = 0;
	float var7 = 0;
	float var8 = 0;
	float var9 = 0;
	size_t idx1 = 0;
	size_t idx2 = 0;
	size_t idx3 = 0;
	size_t idx4 = 0;
	size_t idx5 = 0;
	//void (*callback)(workPackage &);
	workFunction callback = workFunction::NA;
	bool active = true;
	ManualResetEvent workNewEvent;
	ManualResetEvent workCompleteEvent;
};

//These parameters define the current configuration settings of the digital holography object. Typically these come from the GUI, and are in the same units as presented on the GUI.
//Internally, the digHoloObject will often convert these to SI units, and will often make it's own copy of the parameters at the relevant points in the digital holography processing.
//Keeping these internal copies 'prevents' the user from crashing the digital holography by changing the parameters in the middle of a processing run.
//Many parameters are read in at the start during the digHoloFFT routine, and then won't be read back in until another digHoloFFT is run.
class digHoloConfig
{
public:
	//The width/height of a camera frame
	int frameWidth = 320;
	int frameHeight = 256;

	//Physical size of the camera pixels
	float pixelSize = 20e-6f;
	//size of aperture (in Fourier space, as angle in degrees)
	float apertureSize = 0.4f;
	//Type of IFFT to perform in digital holography, full resolution (same dimensions as camera frame), or low-resolution (just the window)
	int resolutionIdx = 0;
	//Mode basis waist radius. At time of writing, even though this is an array (1 for each polarisation). This is currently not implemented.
	float* waist = 0;
	//The zernike polynomials in the plane of the camera. Really only tilt+defocus do anything. The rest are ignored, but the format is kept the same as the other modules. 2D polCount x zernCount
	float** zernCoefs = 0;
	//The length of the zernCoefs array along the zernike dimension
	const int zernCount = 6;
	//The centre of the beam on the camera for each polarisation
	float* BeamCentreX = 0;
	float* BeamCentreY = 0;
				
	bool isNull = true;

	//A new batch size that has been set by the user or elsewhere. Next time digHoloFFT is run, digHoloBatchCount will be updated from this value, and any required reallocation of memory etc will be performed.
	int batchCount = 0;

	//The amount of averaging of frames to perform. Total frames that are FFT'd is batchCount*avgCount. Total amount of frames that are IFFT'd is batchCount, as averaging is performed between the FFT and IFFT.
	int avgCount = 1;

	//The ordering of the averaging. Contigous frames (0), [A A A B B B C C C] or interlaced (1) [A B C A B C A B C]
	int avgMode = 0;

	//The centre wavelength. Not used in the many places, because there's also a wavelength array.
	//wavelengthCentre can be thought of as the alignment wavelength, whereas the wavelength array is for sweeps. The wavelengthCentre need not be in the wavelength array.
	float wavelengthCentre = (float)DIGHOLO_WAVELENGTH_DEFAULT;

	int wavelengthOrdering[2] = { 0,0 };

	//When these polarisation locks are set. The values for each polarisation are enforced to be the same.
	unsigned char PolLockTilt = false;
	unsigned char PolLockBasisWaist = true;
	unsigned char PolLockDefocus = false;

	//The polarisation mode this digital holography is operating in (e.g. single pol, or dual pol)
	int polCount = 1;

	//Specifies which parameters to optimise in the AutoAlignRoutine. These are made public, rather than method arguments to the AutoAlign routine, as leaves open the option for the user to turn them on/off while the AutoAlign procedure is running.
	unsigned char AutoAlignTilt = true;
	unsigned char AutoAlignDefocus = true;
	unsigned char AutoAlignCentre = true;
	unsigned char AutoAlignBasisWaist = true;
	unsigned char AutoAlignFourierWindowRadius = true;
	//Array for storing metric results of AutoAlign routine. e.g. an array containing IL, MDL, XTALK etc etc.
	//float* AutoAlignMetrics = 0;

	//Defines whether the polarisation components of the decomposition should be treated as part of a combined field (0) or independent fields (1).
	//For example, if 1 polarisation component never has any power. When set to 0, the MDL can still be perfect. Whereas when the polarisation components are treated as independent fields (1), the MDL would be infinite, as these are treated as entirely absent components.
	unsigned char AutoAlignPolIndependence = false;
	//When calculating parameters such as crosstalk, the spatial basis needs to be correct, in order that the resulting transfer matrix is ~diagonal. Calculation of crosstalk measures the power along the diagonal vs. off-diagonal.
	//However, for conveinience, a similar property can be calculated that doesn't depend on the spatial basis by multiplying the transfer matrix by it's conjugate transpose first, and then finding the crosstalk of the resulting matrix.
	//This produces something that's a bit like MDL, a bit like crosstalk, but doesn't require that you are measuring in the correct basis to give you a meaningful value.
	unsigned char AutoAlignBasisMulConjTrans = false;

	//If the AutoAlign goal metric changes by less than or equal to this amount in any iteration, the AutoAlign loop will terminate. Setting this higher, means faster response, but less accurate results.
	float AutoAlignTol = 0;
	//Setting this to false, means the AutoAlign routine does the 'snap' function, which starts the whole alignment from scratch, using no prior information other than the apertureSize, and the assumption that the two polarisations are mostly on the left vs. right side of the frame.
	//Setting this to true, means the AutoAlign will start from existing settings, and attempt to optimise them. This option should not be necessary, but allows the user to override the 'snap' autoalign function. Could be useful for very poor quality experimental data.
	unsigned char AutoAlignMode = false;

	int fillFactorCorrection = 1;

	//Specifies the search step to employ for each parameter in the auto align. These are set automatically these days, but it's still technically possible for the user to alter them.
	float AutoAlignTiltStep = 0;
	float AutoAlignDefocusStep = 0;
	float AutoAlignCentreStep = 0;
	float AutoAlignBasisWaistStep = 0;

	//Specifies the goal parameter for the auto align routine (e.g. IL, MDL, XTALK)
	int AutoAlignGoalIdx = 0;

	//The user desired window size for the FFT. May not match exactly what the FFT size ends up being (e.g. if a window size larger than the frame size is selected, also the number must be a multiple of 16 in practice for SIMD)
	int fftWindowSizeX = 0;// 128;
	int fftWindowSizeY = 0;// 128;

	//The FFTW plan mode. Defined as enumerated on the digHoloForm GUI, rather than the fftwf.h file, there's an additional array which maps the GUI combobox index to the FFTW int.
	int FFTW_PLANMODE = 0;

	//The number of supported mode groups (e.g. 9 groups = M+N<9 (0 to 8) --> 45 spatial modes total)
	int maxMG = 0;

	//The type of mode basis to decompose the field in (0:HG, 1:LG, 2: Custom transform supplied by user)
	int basisType = 0;

	//The amount of console info that will be printed for the user to see.
	//Level 0 : Nothing. Except perhaps hard errors coming from underlying libraries like MKL.
	//Level 1 : Basic info. Start/stop notices, parameter summaries etc.
	//Level 2 : Chatty. Mostly for debugging.
	int verbosity = 0;

	//The number of threads that will be used to perform the digital holography processing.
	int threadCount = std::thread::hardware_concurrency();

	//Default constructor
	digHoloConfig()
	{
		ConfigInit();
	}

	//A constructor that lets you make a new digHoloConfig object, with the same properties as an existing one.
	digHoloConfig(digHoloConfig& config)
	{
		//digHoloConfig(config.polCount, config.zernCount);
		ConfigInit();// digHoloConfig();
		CopySelf(config);//'this' pointer, wasn't working
	}
	void ConfigInit()
	{
		const int pCount = DIGHOLO_POLCOUNTMAX;
		const int zCount = zernCount;
		//The main alignment parameters. Setup as separate arrays, but could also have been implemented as 1, pCount x (zCount+3) 2D array that stores everything. In some ways that'd be better
		allocate1D(pCount, waist);
		allocate2D(pCount, zCount, zernCoefs);
		memset(&zernCoefs[0][0], 0, sizeof(float) * pCount * zCount);

		allocate1D(pCount, BeamCentreX);
		allocate1D(pCount, BeamCentreY);

		//Made up starting parameters
		if (pCount == 2)
		{
			BeamCentreX[0] = 0;
			BeamCentreX[1] = 0;

			BeamCentreY[0] = 0;
			BeamCentreY[1] = 0;
		}
		else
		{
			BeamCentreX[0] = 0;
			BeamCentreX[1] = 0;

			BeamCentreY[0] = 0;
			BeamCentreY[1] = 0;
		}

		for (int polIdx = 0; polIdx < pCount; polIdx++)
		{
			waist[polIdx] = 510;
			zernCoefs[polIdx][TILTX] = -0.75;
			zernCoefs[polIdx][TILTY] = 0.75;
		}

		//allocate1D(DIGHOLO_METRIC_COUNT, AutoAlignMetrics);
		isNull = false;
	}
	void Clear()
	{
		free1D(waist);
		free2D(zernCoefs);
		free1D(BeamCentreX);
		free1D(BeamCentreY);

		isNull = true;
	}
	//Destructor, frees the mallocs created in the constructor
	~digHoloConfig()
	{
		Clear();
	}

	//Creates a copy of an existing config object. Whenever a new property is added to the definition of what a digHoloConfig object is, this copy routine will have to be updated as well, to make sure it stays current.
	void CopySelf(digHoloConfig& sourceConfig)
	{
		frameWidth = sourceConfig.frameWidth;
		frameHeight = sourceConfig.frameHeight;

		pixelSize = sourceConfig.pixelSize;
		apertureSize = sourceConfig.apertureSize;

		memcpy(waist, sourceConfig.waist, sizeof(float) * polCount);
		memcpy(&zernCoefs[0][0], &sourceConfig.zernCoefs[0][0], sizeof(float) * polCount * zernCount);
		memcpy(BeamCentreX, sourceConfig.BeamCentreX, sizeof(float) * polCount);
		memcpy(BeamCentreY, sourceConfig.BeamCentreY, sizeof(float) * polCount);
		//memcpy(AutoAlignMetrics, sourceConfig.AutoAlignMetrics, sizeof(float) * DIGHOLO_METRIC_COUNT);
		batchCount = sourceConfig.batchCount;
		wavelengthCentre = sourceConfig.wavelengthCentre;
		wavelengthOrdering[0] = sourceConfig.wavelengthOrdering[0];
		wavelengthOrdering[1] = sourceConfig.wavelengthOrdering[1];

		PolLockTilt = sourceConfig.PolLockTilt;
		PolLockBasisWaist = sourceConfig.PolLockBasisWaist;
		PolLockDefocus = sourceConfig.PolLockDefocus;

		AutoAlignTilt = sourceConfig.AutoAlignTilt;
		AutoAlignDefocus = sourceConfig.AutoAlignDefocus;
		AutoAlignCentre = sourceConfig.AutoAlignCentre;
		AutoAlignBasisWaist = sourceConfig.AutoAlignBasisWaist;
		AutoAlignFourierWindowRadius = sourceConfig.AutoAlignFourierWindowRadius;
		AutoAlignGoalIdx = sourceConfig.AutoAlignGoalIdx;
		AutoAlignMode = sourceConfig.AutoAlignMode;
		AutoAlignTol = sourceConfig.AutoAlignTol;
		AutoAlignBasisMulConjTrans = sourceConfig.AutoAlignBasisMulConjTrans;
		AutoAlignPolIndependence = sourceConfig.AutoAlignPolIndependence;

		AutoAlignTiltStep = sourceConfig.AutoAlignTiltStep;
		AutoAlignDefocusStep = sourceConfig.AutoAlignDefocusStep;
		AutoAlignCentreStep = sourceConfig.AutoAlignCentreStep;
		AutoAlignBasisWaistStep = sourceConfig.AutoAlignBasisWaistStep;

		polCount = sourceConfig.polCount;

		fftWindowSizeX = sourceConfig.fftWindowSizeX;
		fftWindowSizeY = sourceConfig.fftWindowSizeY;

		basisType = sourceConfig.basisType;

		verbosity = sourceConfig.verbosity;
		avgCount = sourceConfig.avgCount;
		avgMode = sourceConfig.avgMode;

		threadCount = sourceConfig.threadCount;

		fillFactorCorrection = sourceConfig.fillFactorCorrection;
	}
	//Creates a copy of an existing config object. Whenever a new property is added to the definition of what a digHoloConfig object is, this copy routine will have to be updated as well, to make sure it stays current.
	void Copy(digHoloConfig& destConfig, digHoloConfig& sourceConfig)
	{
		destConfig.frameWidth = sourceConfig.frameWidth;
		destConfig.frameHeight = sourceConfig.frameHeight;

		destConfig.pixelSize = sourceConfig.pixelSize;
		destConfig.apertureSize = sourceConfig.apertureSize;

		memcpy(destConfig.waist, sourceConfig.waist, sizeof(float) * polCount);
		memcpy(&destConfig.zernCoefs[0][0], &sourceConfig.zernCoefs[0][0], sizeof(float) * polCount * zernCount);
		memcpy(destConfig.BeamCentreX, sourceConfig.BeamCentreX, sizeof(float) * polCount);
		memcpy(destConfig.BeamCentreY, sourceConfig.BeamCentreY, sizeof(float) * polCount);
		//memcpy(destConfig.AutoAlignMetrics, sourceConfig.AutoAlignMetrics, sizeof(float) * DIGHOLO_METRIC_COUNT);
		destConfig.batchCount = sourceConfig.batchCount;
		destConfig.wavelengthCentre = sourceConfig.wavelengthCentre;
		destConfig.wavelengthOrdering[0] = sourceConfig.wavelengthOrdering[0];
		destConfig.wavelengthOrdering[1] = sourceConfig.wavelengthOrdering[1];

		destConfig.PolLockTilt = sourceConfig.PolLockTilt;
		destConfig.PolLockBasisWaist = sourceConfig.PolLockBasisWaist;
		destConfig.PolLockDefocus = sourceConfig.PolLockDefocus;

		destConfig.AutoAlignTilt = sourceConfig.AutoAlignTilt;
		destConfig.AutoAlignDefocus = sourceConfig.AutoAlignDefocus;
		destConfig.AutoAlignCentre = sourceConfig.AutoAlignCentre;
		destConfig.AutoAlignBasisWaist = sourceConfig.AutoAlignBasisWaist;
		destConfig.AutoAlignFourierWindowRadius = sourceConfig.AutoAlignFourierWindowRadius;
		destConfig.AutoAlignGoalIdx = sourceConfig.AutoAlignGoalIdx;
		destConfig.AutoAlignTol = sourceConfig.AutoAlignTol;
		destConfig.AutoAlignBasisMulConjTrans = sourceConfig.AutoAlignBasisMulConjTrans;
		destConfig.AutoAlignPolIndependence = sourceConfig.AutoAlignPolIndependence;
		destConfig.AutoAlignMode = sourceConfig.AutoAlignMode;

		destConfig.AutoAlignTiltStep = sourceConfig.AutoAlignTiltStep;
		destConfig.AutoAlignDefocusStep = sourceConfig.AutoAlignDefocusStep;
		destConfig.AutoAlignCentreStep = sourceConfig.AutoAlignCentreStep;
		destConfig.AutoAlignBasisWaistStep = sourceConfig.AutoAlignBasisWaistStep;

		destConfig.polCount = sourceConfig.polCount;

		destConfig.fftWindowSizeX = sourceConfig.fftWindowSizeX;
		destConfig.fftWindowSizeY = sourceConfig.fftWindowSizeY;

		destConfig.basisType = sourceConfig.basisType;

		destConfig.verbosity = sourceConfig.verbosity;

		destConfig.avgCount = sourceConfig.avgCount;
		destConfig.avgMode = sourceConfig.avgMode;

		destConfig.threadCount = sourceConfig.threadCount;

		destConfig.fillFactorCorrection = sourceConfig.fillFactorCorrection;
	}

};

class digHoloObject
{

	//The current configuration of the digital holography object
public: digHoloConfig config;
	  //An additional config object containing the 'best' configuration. These are used for alignment purposes, as it allows the GUI to see the best config, whilst another config might be loaded (such as during auto alignment)
	  digHoloConfig configBackup;

	  //These are mostly deprecated flags that should be phased out. Kept here for temporary compatibility with legacy code.
	  //Specifies what to display in the openGL window for this digHoloObject. Debatable whether to put this in digHoloConfig, but it more belongs to the object itself. e.g. you might want to revert to a previous config by loading configBest, but not necessarily have the openGL window suddenly revert to whatever displaymode it was in when that configBest was found.
	  unsigned int DisplayMode = 0;

	  //This indicates whether this digHoloObject is enabled, it's really more a digHoloForm property, as it's not used inside the digHoloObject.
	  int Enabled = 0;

	  //Similarly, this is set/controlled by digHoloForm, digHoloObject does not use it internally.
	  unsigned char FrameThreadIsRunning = false;

	  //Triggers when the viewport has started, indicating that the digHoloObject is up and running.
	  ManualResetEvent InitialisingEvent;// = new ManualResetEvent(false);

private:

	//If you need this comment, you're in serious trouble. Turn back now.
	const int c0 = 299792458;
	const float pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286f;
	const double piD = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229;

	//Logical number of processors, as seen by the OS (can be different from physical processors, e.g. hyperthreading)
	const int CPU_COUNT = std::thread::hardware_concurrency();
	const int THREADCOUNT_MAX = 2 * CPU_COUNT;

	//A structure used to store information about the CPU. i.e. instruction sets supported, name, core count.
	cpuINFO cpuInfo;

	//The width/height of the full raw camera frames
	int digHoloFrameHeight = 0;
	int digHoloFrameWidth = 0;

	//The size of the window of interest in the camera plane (FFT, [0]) and in the Fourier plane/reconstructed plane (IFFT, [1])
	//In low resolution mode, the two elements of each of these width/height arrays are the dimensions of the FFT/IFFT respectively.
	//In high resolution mode, both FFT and IFFT are the same size, but the second element still instructs digHoloCopyWindow as to what dimensions the window in Fourier space is.
	//These are also enforced to be multiples of 16 for SIMD reasons.
	int* digHoloWoiPolWidth = 0;
	int* digHoloWoiPolHeight = 0;

	//These specify the corner points (Start X, Start Y) for selecting the FFT and IFFT windows.
	//2D array. fftIdx (camera or Fourier plane) x polIdx. The FFT and IFFTs both have their own windows of interest, as do the two polarisations.
	int** digHoloWoiSX = 0;
	int** digHoloWoiSY = 0;


	//The FFTW plans for the FFT and IFFTs. Planned out somewhat overkill as an array of size PROCESSOR_COUNT, so the threads assigned to FFT can be rapidly changed.
	//Would be better to just plan for the thread count scenarios actually used. e.g. if digHoloFFTPlan[threadCount]==0, then plan it out.
	fftwf_plan** digHoloFFTPlans = 0;
	fftwf_plan** digHoloIFFTPlans = 0;
	//A special case Real-to-Real transform, used in auto-alignment to correlate the Fourier plane, with the aperture window, to work out where the off-axis terms are in Fourier space.
	fftwf_plan digHoloDCTPlan = 0;
	fftwf_plan digHoloIDCTPlan = 0;
	complex64** digHoloFFTPlanWorkspace = 0;

	//Pointer to the camera pixels. Fed into the module from the outside, and used by digHoloFFT as the source of the digital holography processing
	float* digHoloPixelsCameraPlane = 0;
	//Pointer to the camera pixels in int16 format. i.e. raw off a camera.
	unsigned short* digHoloPixelsCameraPlaneUint16 = 0;
	//An internal buffer used for converting camera pixels passed in as int16s, that need to be converted to float32.
	float* digHoloPixelBuffer = 0;
	//Indicates the source of the pixel buffer. 0: float32 (directly from outside) 1:uint16 (must be processed into float32 form internally).
	int digHoloPixelBufferType = 0;
	//A flag used to indicate whether a uint16->float32 needs to be performed, when operating with uint16 input frames. 
	//Only used in AutoAlign, as it need not reconvert the frames during the AutoAlign process, as the frames can be assumed to be constant during the AutoAlign.
	//For other digHolo processing, we always have to reconvert, because we don't own the pixelBuffer being passed in from outside, and hence we have no way of knowing if the data has changed.
	//0: Conversion required.
	//1: Perform conversion this time, then set flag to -1 (meaning it won't run next time)
	//All other values : No conversion performed.
	int digHoloPixelBufferConversionFlag = 0;

	//Are the uint16 inputs frame buffer's pixels transposed?
	int digHoloPixelBufferUint16transposed = 0;

	//Arrays used for taking the input Reference Wave calibration supplied by the user, and calculating the corresponding calibration which should be applied to the final reconstructed field.
	complex64* digHoloRefCalibration = 0;
	complex64* digHoloRefCalibrationFourierPlane = 0;
	complex64* digHoloRefCalibrationFourierPlaneMasked = 0;
	complex64* digHoloRefCalibrationReconstructedPlane = 0;

	int digHoloRefCalibrationWavelengthCount = 0;
	int digHoloRefCalibrationWidth = 0;
	int digHoloRefCalibrationHeight = 0;
	int digHoloRefCalibrationEnabled = 0;

	int digHoloRefCalibrationFourierPlaneValid = 0;
	int digHoloRefCalibrationReconstructedPlaneValid = 0;


	//The length of the currently allocated digHoloPixelBuffer.
	size_t digHoloPixelBufferLength = 0;

	//The real-to-complex FFT corresponding with digHoloPixelsCameraPlane. The primary result of digHoloFFT.
	complex64* digHoloPixelsFourierPlane = 0;
	//The selected window of the Fourier plane. Pixels are copied from digHoloPixelsFourierPlane corresponding to the off-axis terms, and then IFFT'd by digHoloIFFT.
	complex64* digHoloPixelsFourierPlaneMasked = 0;

	//The IFFT of digHoloPixelsFourierPlaneMasked and the primary result of digHoloIFFT. This field will still have a tilt on it from the reference wave/window decentring in Fourier space. applyTilt16 removes this tilt/focus
	complex64* digHoloPixelsCameraPlaneReconstructedTilted = 0;
	//The final reconstructed field. Corresponds with digHoloPixelsCameraPlaneReconstructedTilted, with tilt/focus removed, and converted to 16bit integers.
	//The STAT_MAXABS elements of the digHoloPixelsStats array specify the scaling of these integers when wanting to convert them back to properly normalised/scaled floats.
	short* digHoloPixelsCameraPlaneReconstructed16 = 0;
	//The Real and imaginary components simply point to the master array digHoloPixelsCameraPlaneReconstructed16 above for conveinience. They are not allocated as separated arrays, they are just separate locations in 1 long array.
	short* digHoloPixelsCameraPlaneReconstructed16R = 0;
	short* digHoloPixelsCameraPlaneReconstructed16I = 0;

	//Array which stores properties about fields in the Fourier and reconstructed planes. Properties like total power, centre of mass, maximum absolute value, effective area.
	//There is one entry for each batch, plus an additional 1 which gives the properties over the whole batch. Primarily it is this final elements which are used for things like autoalign.
	//Dimensions are : (BatchCount+1) x 2 (one for the result of the FFT, and 1 for the result of the IFFT) x 2 polarisations x DIGHOLO_ANALYSISCount different parameters of interest
	float**** DIGHOLO_ANALYSIS = 0;//batch x fft x pol x statCount
	const int DIGHOLO_ANALYSISCount = 7;
	//This contains the sum of the intensity over all fields in a batch. 2D (fftPlaneIdx by pixels). There's no separate index for polarisation, it's just indexed as 1 double length array of pixels
	float*** DIGHOLO_ANALYSISBatchSum = 0;

	unsigned char* digHoloPlotPixels = 0;
	complex64* digHoloPlotPixelsComplex = 0;
	//const size_t digHoloPlotStringLength = 1024;
#define digHoloPlotStringLength 1024
	char digHoloPlotString[digHoloPlotStringLength];
	size_t digHoloPlotPixelsCount = 0;

	//Specifies whether updating the frame buffer (the source of the FFT, and set by the SetFrameBuffer routine) is allowed. 
	//It's typically disabled when a batch is set, so that the digHoloForm FramerGrabberRoutine doesn't overwrite it.
	//SetFrameBuffer does nothing if this is set to false.
	//Really though, you shouldn't be taking in frames from the camera if you've done a batch. You should have the camera paused.
	//unsigned char SetFrameBufferEnabled = true;

	//The current size of the batch for dig. holography processing. So for a given batch, there would be digHoloBatchCount x modeCount worth of coefficients
	int digHoloBatchCOUNT = 0;
	//The amount of averaging. The total raw frames fed into digHolo would be digHoloBatchCount*digHoloBatchAvgCount. These would in turn be averaged to become digHoloBatchCount output fields.
	int digHoloBatchAvgCount = 0;
	//The format in which the frames should be averaged. In contigous blocks (0) of digHoloBatchAvgCount [A A A B B B C C C..], or interlaced (1) [A B C A B C A B C]
	//int digHoloBatchAvgMode = 0;
	complex64** digHoloBatchCalibration = 0;
	int digHoloBatchCalibrationBatchCount = 0;
	int digHoloBatchCalibrationPolCount = 0;
	int digHoloBatchCalibrationEnabled = 0;


	//The current centre wavelength
	float digHoloWavelengthCentre = (float)(DIGHOLO_WAVELENGTH_DEFAULT);
	//int digHoloWavelengthOrderingIn = 0;
	//int digHoloWavelengthOrderingOut = 0;

	//An array of wavelengths used for batch processing. If there is only a single wavelength (digHoloWavelengthCount==1), then no wavelength-dependent calibration is applied.
	//If there is multiple wavelengths specified, then the tilt/focus when processing a batch is processed in a wavelength-dependent fashion.
	float* digHoloWavelength = 0;
	int digHoloWavelengthCount = 0;

	//This is used for updating the wavelength array above. The user can adjust/setup this array, then indicate 'digHoloWavelengthValid==false', and then next time digHoloIFFT is run, the new wavelength array will be copied across.
	//Extra copy is so that the user can't change the wavelength array while a digital holography process is potentially already running.
	float* digHoloWavelengthNew = 0;
	int digHoloWavelengthCountNew = 0;
	unsigned char digHoloWavelengthValid = 0;

	//Whenever a new fft or ifft is initialised, the parameters are stored. Then later, when the FFT/IFFT is run again, the current values can be checked against the initialised values.
	//e.g. if the size of the window has changed, you'll have to make new FFT plans, reallocate memory etc.
	//The FFT and IFFT have their own settings, which is why this is an array of length 2
	//An alternative way to do this would be to create two configs, e.g. configPrevious for FFT and one for IFFT, but these parameters map directly onto variables (e.g. digHoloWoiPolWidth) which don't map directly onto config. (digHoloWoiPolWidth depends on several config parameters)
	int* digHoloWoiPolWidth_Valid = 0;
	int* digHoloWoiPolHeight_Valid = 0;
	int* digHoloBatchCount_Valid = 0;
	int* digHoloBatchAvgCount_Valid = 0;
	int* digHoloPolCount_Valid = 0;
	float* digHoloPixelSize_Valid = 0;

	//The number of supported polarisations, updated from config->polCount at the start of digHoloFFT
	int digHoloPolCount = 0;
	//The pixel size, again updated from config->pixelSize
	float digHoloPixelSize = 0;
	//The type of IFFT processing to  perform, full resolution (reconstructed field has same dimensions as FFT), or just the sub-window (IFFT has smaller dimensions than FFT)
	int digHoloResolutionIdx = 0;
	//Size of the window in the Fourier plane. In units of degrees, same as config->apertureSize.
	float digHoloApertureSize = 0;

	//The x/y axes of the camera plane, and the reconstruction plane (camera-->FFT-->IFFT-->reconstruction plane).
	//It's a 2D array because the first dimension specifies whether it's the camera or reconstruction plane.
	float** digHoloXaxis = 0;
	float** digHoloYaxis = 0;

	//The x/y axes in the Fourier plane of the camera. The Xaxis/Yaxis and KXaxis/KYaxis arrays could all be consolidated into a single 2D array. However all these arrays are of different dimensions anyways as the number of pixels in different planes can be different.
	float* digHoloKXaxis = 0;
	float* digHoloKYaxis = 0;

	//Separable reference wave (e.g. tilt, focus). Stored as a 3D array of lambdaCount x polCount x (pixelCountX + pixelCountY). The x and y components are stored next to each other in the same array.
	short*** digHoloRef = 0;

	//A change in any of these properties will invalidate the reference wave
	float* digHoloRefTiltX_Valid = 0;
	float* digHoloRefTiltY_Valid = 0;

	float* digHoloRefTiltXoffset_Valid = 0;
	float* digHoloRefTiltYoffset_Valid = 0;

	float* digHoloRefDefocus_Valid = 0;
	float* digHoloRefCentreX_Valid = 0;
	float* digHoloRefCentreY_Valid = 0;
	float* digHoloRefWavelength_Valid = 0;
	int digHoloRefWavelengthCount_Valid = 0;
	int digHoloRefPolCount_Valid = 0;
	int digHoloRefPixelCountX_Valid = 0;
	int digHoloRefPixelCountY_Valid = 0;

	unsigned digHoloRefValid = false;

	//The number of supported mode groups
	int digHoloMaxMG = 9;
	//The number of total spatial modes corresponding with digHoloMaxMG (e.g. 9 MG = 45 spatial modes)
	int digHoloModeCount = 0;

	//HG basis modes
	//digHoloHG is the master array (the one which is actually allocated), however digHoloHGX and digHoloHGY point to locations in digHoloHG
	//How exactly things are setup/stored depends on where the x or y dimension is longer and whether the pixels are square (like on the camera) or rectangular (as they can be slightly in the reconstruction).
	//For reasons of keeping the basis small in memory for potential CPU caching, we don't allocate any more memory than we need to. This can help the overlap operation.
	short*** digHoloHG = 0;
	short*** digHoloHGX = 0;//mode x pixels
	short*** digHoloHGY = 0;

	//The scale used by the digHoloHGX/Y arrays. Multiplying HGscale[modeIdx] by digHoloHGx[modeIdx][...] gives the true normalised floating point value. digHoloHGX is like the significand, and HGscale is like the base-exponent of a traditional float point number
	float** digHoloHGscaleX = 0;
	float** digHoloHGscaleY = 0;

	//The X index of the HG modes, array is of size digHoloModeCount.
	int* HG_M = 0;
	//The Y index of the HG modes, array is of size digHoloModeCount.
	int* HG_N = 0;
	//2D lookup table HG_MN[mIdx][nIdx] returns the modeIdx (0...(digHoloModeCount-1)). Use when you know the x-order and y-order of the HG mode you're looking for, this tells you what modeIdx it's enumerated in the 1D list of modes, e.g. in digHoloOverlapCoefsHG.
	//If there is no mode with the requested indices, then the value is filled with -1. This lookup table is primarly used by the overlap routine, so it can iterate over the x-index, and then calculate all the y-index modes for that x-index mode. Rather than going sequentially from 0...(digHoloModeCount-1), which would mean each x-index mode would be read multiple times (instead of just once).
	int** HG_MN = 0;

	//The azimuthal (l) index of the LG mode basis, array is same size as HG_M and HG_N.
	int* LG_L = 0;
	//The radial (p) index of the LG mode basis, array is same size as HG_M and HG_N.
	int* LG_P = 0;
	//Tranformation matrix that converts HG coefficients, to LG coefficients.
	//This is actually a set of matrices, 1 matrix for each mode-group. That is, each mode group has it's own transformation matrix, rather than 1 big transform matrix, that would be mostly sparse.
	//i.e. LGtoHG[0]->1x1 matrix LGtoHG[1]->2x2, LGtoHG[2]->3x3, LGtoHG[3]->4x4...LGtoHG[N-1]->NxN
	complex64** LGtoHG = 0;
	int LGtoHGmaxMG = 0;

	//When false, this indicates that no valid mode basis has been calculated. This signifies that the mode basis should be recalculated.
	//Gets set to true when the dimensions or pixel size change in the FFT/IFFT, and hence modes must be recalculated.
	unsigned char digHoloBasisValid = false;

	//If the max mode group or waist changes, then the modes will have to be recalculated. These values store the maxMG and Waist for the last time the modes were calculated
	int digHoloBasisValid_maxMG = 0;
	float* digHoloBasisValid_Waist = 0;
	int digHoloBasisValid_PolCount = 0;

	//These are a shared array of workpackages, used by the threads of the digital holography processing. Consists of a bunch of flags, pointers and other parameters threads need to do their work.
   // array<workPackage> digHoloWorkPacks;
   // workPackage* digHoloWorkPacks = 0;

	//This is the length of the digHoloWorkPacks, and hence also the maximum number of threads that should be running simultaneously.
	const int digHoloThreadPoolSize = THREADCOUNT_MAX;

	int digHoloThreadCount = CPU_COUNT;

	//There is a threadpool always live, that's ready to take requests. The work pack will tell the thread what function to run, as well as provide all the other flags and pointers it needs to do that function.
	std::vector<workPackage*> digHoloWorkPacks;
	std::vector<std::thread> digHoloThreadPool;

	//std::thread digHoloViewportThread;
	//array used by the threads of the overlap routine...
	//totalThreads x digHoloBatchCount x digHoloPolCount x (digHoloModeCount* blockSize * 2 (for Real/Imag))
	//I made it a double, so that there's no loss in precision for the overlaps (16 bit x-mode, 16 bit y-mode, 16-bit field = 48 bits). Means there's not really any downside to using 16-bit ints over doing it 32-bit float in the end.
	//double*** digHoloOverlapWorkspace = 0;
	//double**** digHoloOverlapWorkspaceOut = 0;

	complex64*** digHoloPowerMeterWorkspace = 0;
	int digHoloPowerMeterWorkspace_PolCount = 0;
	int digHoloPowerMeterWorkspace_WavelengthCount = 0;
	size_t digHoloPowerMeterWorkspace_MemCount = 0;

	//If the batchcount, the mode count or the pol count change, then the arrays for the overlap routine will have to be reallocated. The current settings are compared against these values from the last time the overlap routine was run, to work out whether arrays need to be reallocated.
	int digHoloOverlapModesValid_BatchCount = 0;
	int digHoloOverlapModesValid_ModeCount = 0;
	int digHoloOverlapModesValid_PolCount = 0;
	int digHoloOverlapModesValid_ThreadCount = 0;

	//The overlap coefficients batch x (mode+pol). Modes for each polarisation are stored separately, rather than interlaced.
	complex64** digHoloOverlapCoefsHG = 0;

	//The overlap coefficients in the basis specified by the user. Derived from digHoloOverlapCoefsHG, through the digHoloBasisTransform matrix (or LGtoHG transform)
	complex64** digHoloOverlapCoefsCustom = 0;

	//A pointer that will either point to digHoloOverlapCoefsHG or digHoloOverlapCoefsCustom depending on the basis currently selected. Used to easily switch between bases.
	//This pointer allows sections of code to remain oblivious to exactly what type of basis we're currently working in.
	//This pointer is never directly memory allocated, it always points to an existing memory space.
	complex64** digHoloOverlapCoefsPtr = 0;
	//The mode count associated with digHoloOverlapCoefsPtr. This will either be digHoloModeCount or digHoloBasisTransformModeCountOut
	int digHoloModeCountOut = 0;

	//The matrix to apply to the HG coefficients to transform them into another basis.
	//There are two versions. The 'Full' version specified directly by the user. This may contain more modes than are currently being decomposed in the HG basis. 
	//In which case, only the section of the transform matrix that corresponds with the currently supported HG modes will be used (digHoloBasisTransform).
	//The idea here is, day-to-day, the user may wish to calculate a large transform matrix once, and use that same matrix regardless of how many HG modes they're working with on a given day.
	//Rather than recalculate a separate transform matrix each time they change the number of HG modes supported. For example, a big HG-to-LG transform that supports 100 mode groups, rather than calculating 100 separate transform matrices. (although HG-to-LG is a special case whereby the user need not calculate their own transform anyways).
	complex64* digHoloBasisTransformFull = 0;
	int digHoloBasisTransformFullModeCountIn = 0;
	int digHoloBasisTransformFullModeCountOut = 0;

	complex64* digHoloBasisTransform = 0;
	int digHoloBasisTransformModeCountIn = 0;
	int digHoloBasisTransformModeCountOut = 0;

	//complex float32 of the fields of the basis. Not used internally. Only used for export.
	complex64** digHoloBasisFields = 0;

	int digHoloBasisTransformIsValid = 0;

	int digHoloBasisType = 0;

	float** digHoloAutoAlignMetrics = 0;// [DIGHOLO_METRIC_COUNT] ;
	int digHoloAutoAlignMetricsWavelengthCount = 0;

public: digHoloObject()
{
	const int planeCount = DIGHOLO_PLANECOUNTMAX;
	const int polCount = DIGHOLO_POLCOUNTMAX;
	const int woiPlaneCount = planeCount + DIGHOLO_PLANECOUNTCAL;

	allocate2D(woiPlaneCount, polCount, digHoloWoiSX); memset(&digHoloWoiSX[0][0], 0, sizeof(int) * (woiPlaneCount * polCount));
	allocate2D(woiPlaneCount, polCount, digHoloWoiSY); memset(&digHoloWoiSY[0][0], 0, sizeof(int) * (woiPlaneCount * polCount));

	allocate1D(woiPlaneCount, digHoloWoiPolWidth); memset(&digHoloWoiPolWidth[0], 0, sizeof(int) * woiPlaneCount);
	allocate1D(woiPlaneCount, digHoloWoiPolHeight); memset(&digHoloWoiPolHeight[0], 0, sizeof(int) * woiPlaneCount);

	allocate1D(woiPlaneCount, digHoloWoiPolWidth_Valid); memset(&digHoloWoiPolWidth_Valid[0], 0, sizeof(int) * woiPlaneCount);
	allocate1D(woiPlaneCount, digHoloWoiPolHeight_Valid); memset(&digHoloWoiPolHeight_Valid[0], 0, sizeof(int) * woiPlaneCount);
	allocate1D(woiPlaneCount, digHoloBatchCount_Valid); memset(&digHoloBatchCount_Valid[0], 0, sizeof(int) * woiPlaneCount);
	allocate1D(planeCount, digHoloBatchAvgCount_Valid); memset(&digHoloBatchAvgCount_Valid[0], 0, sizeof(int) * planeCount);
	allocate1D(woiPlaneCount, digHoloPixelSize_Valid); memset(digHoloPixelSize_Valid, 0, sizeof(float) * woiPlaneCount);
	allocate1D(woiPlaneCount, digHoloPolCount_Valid); memset(digHoloPolCount_Valid, 0, sizeof(int) * woiPlaneCount);

	//Set these up as null pointers, they will be allocated with memory later in the digHoloFFT/digHoloIFFT init routines
	digHoloXaxis = (float**)alignedAllocate(sizeof(float*) * planeCount, ALIGN);//_aligned_malloc(sizeof(float*) * planeCount, ALIGN);
	digHoloYaxis = (float**)alignedAllocate(sizeof(float*) * planeCount, ALIGN);;//_aligned_malloc(sizeof(float*) * planeCount, ALIGN);

	if (digHoloXaxis && digHoloYaxis) //If this fails, you're in trouble no matter what
	{
		for (int i = 0; i < planeCount; i++)
		{
			digHoloXaxis[i] = 0;
			digHoloYaxis[i] = 0;
		}
	}
}
public: void Destroy()
{
	//Is this the proper way to cleans these up?
	config.Clear();
	configBackup.Clear();

	//delete &ViewportWaitHandle;
	//delete &ViewportNewFrameHandle;
	//delete &ViewportFinishedHandle;
	//delete& InitialisingEvent;

	free1D(digHoloWoiPolWidth);
	free1D(digHoloWoiPolHeight);

	free2D(digHoloWoiSX);
	free2D(digHoloWoiSY);

	//allocate2D(THREADCOUNT_MAX, FFTW_HOWMANYMAX, digHoloFFTPlans);
	const size_t dimX = (THREADCOUNT_MAX * DIGHOLO_PLANECOUNTCAL);
	for (int i = 0; i < dimX; i++)
	{
		for (int j = 0; j < FFTW_HOWMANYMAX; j++)
		{
			if (digHoloFFTPlans[i][j])
			{
				fftwf_destroy_plan(digHoloFFTPlans[i][j]); digHoloFFTPlans[i][j] = 0;
			}
			if (digHoloIFFTPlans[i][j])
			{
				fftwf_destroy_plan(digHoloIFFTPlans[i][j]); digHoloIFFTPlans[i][j] = 0;
			}
		}
	}
	free2D(digHoloFFTPlans);
	free2D(digHoloIFFTPlans);

	//A special case Real-to-Real transform, used in auto-alignment to correlate the Fourier plane, with the aperture window, to work out where the off-axis terms are in Fourier space.
	if (digHoloDCTPlan)
	{
		fftwf_destroy_plan(digHoloDCTPlan);
	}
	if (digHoloIDCTPlan)
	{
		fftwf_destroy_plan(digHoloIDCTPlan);
	}

	free2D(digHoloFFTPlanWorkspace);

	//The pixelBuffer of camera frames is fed in from outside. It doesn't belong to this object.
	//free1D(digHoloPixelsCameraPlane);
	free1D(digHoloPixelsCameraPlaneUint16);
	free1D(digHoloPixelBuffer);

	free1D(digHoloRefCalibration);
	free1D(digHoloRefCalibrationFourierPlane);
	free1D(digHoloRefCalibrationFourierPlaneMasked);
	free1D(digHoloRefCalibrationReconstructedPlane);


	free1D(digHoloPixelsFourierPlane);
	free1D(digHoloPixelsFourierPlaneMasked);

	free1D(digHoloPixelsCameraPlaneReconstructedTilted);
	free1D(digHoloPixelsCameraPlaneReconstructed16);//digHoloPixelsCameraPlaneReconstructed16R and digHoloPixelsCameraPlaneReconstructed16I are not allocated directly

	free4D(DIGHOLO_ANALYSIS);
	free3D(DIGHOLO_ANALYSISBatchSum);

	free1D(digHoloPlotPixels);
	free1D(digHoloPlotPixelsComplex);

	free2D(digHoloBatchCalibration);

	free1D(digHoloWavelength);
	free1D(digHoloWavelengthNew);

	free1D(digHoloWoiPolWidth_Valid);
	free1D(digHoloWoiPolHeight_Valid);
	free1D(digHoloBatchCount_Valid);
	free1D(digHoloBatchAvgCount_Valid);
	free1D(digHoloPolCount_Valid);
	free1D(digHoloPixelSize_Valid);

	//free2D(digHoloXaxis); 
	//free2D(digHoloYaxis);
	const int planeCount = DIGHOLO_PLANECOUNTMAX;

	for (int i = 0; i < planeCount; i++)
	{
		free1D(digHoloXaxis[i]);
		free1D(digHoloYaxis[i]);
	}
	free1D(digHoloXaxis);
	free1D(digHoloYaxis);

	free1D(digHoloKXaxis);
	free1D(digHoloKYaxis);

	//This used to be free2D. Did I forget to update it?
	free3D(digHoloHG);
	//These two are just pointers to positions in digHoloHG, so we just want to free the pointer array, not the memory itself (as free2D(digHoloHG) does that) 
	free2D(digHoloHGX);
	free2D(digHoloHGY);

	//The scale used by the digHoloHGX/Y arrays. Multiplying HGscale[modeIdx] by digHoloHGx[modeIdx][...] gives the true normalised floating point value. digHoloHGX is like the significand, and HGscale is like the base-exponent of a traditional float point number
	free2D(digHoloHGscaleX);
	free2D(digHoloHGscaleY);

	free1D(HG_M);
	free1D(HG_N);

	free2D(HG_MN);

	free1D(LG_L);
	free1D(LG_P);
	free2D(LGtoHG);

	free1D(digHoloBasisValid_Waist);

	threadPoolDestroy();

	free3D(digHoloPowerMeterWorkspace);

	free2D(digHoloOverlapCoefsHG);

	free2D(digHoloOverlapCoefsCustom);
	//digHoloOverlapCoefsPtr is not it's own allocated pointer

	free1D(digHoloBasisTransformFull);
	free1D(digHoloBasisTransform);

	free2D(digHoloBasisFields);

	free2D(digHoloAutoAlignMetrics);
}

	  //Memory leak because the arrays allocated in digHoloObject are not freed, but once digHoloObject is destroyed, the program is closing anyways
	 // public: digHoloObject::~digHoloObject()
public: ~digHoloObject()
{
	Destroy();
}//there used to be a ; here?

	  //The first method called by the digHoloForm to initialise the digHoloObject. Creates a window, and tells the digHoloObject the dimensions of the raw camera frames.
public: int digHoloInit()
{
	cpuInfoGet(&cpuInfo);

	for (int i = 0; i < digHoloThreadPoolSize; i++)
	{
		digHoloWorkPacks.push_back(new workPackage());

		workPackage* work = digHoloWorkPacks[i];
		work[0].workNewEvent.Reset();
		work[0].workCompleteEvent.Reset();
		work[0].active = true;
		work[0].callback = workFunction::NA;
		digHoloThreadPool.push_back(std::thread(&digHoloObject::digHoloThreadPoolRoutine, this, i));

	}
	memset(&digHoloPlotString[0], 0, digHoloPlotStringLength * sizeof(char));

	InitialisingEvent.Set();
	return 0;
}

	  void threadPoolDestroy()
	  {
		  for (int j = 0; j < digHoloThreadPoolSize; j++)
		  {
			  workPackage* workPack = digHoloWorkPacks[j];

			  //deactivate the thread
			  workPack[0].active = false;
			  workPack[0].callback = workFunction::NA;

			  //Reset event to Set when thread is complete
			  workPack[0].workCompleteEvent.Reset();
			  workPack[0].workNewEvent.Set();
		  }

		  //Wait for threads to finish
		  for (int j = 0; j < digHoloThreadPoolSize; j++)
		  {
			  workPackage* workPack = digHoloWorkPacks[j];
			  workPack[0].workCompleteEvent.WaitOne();
			  delete workPack;
			  //			  digHoloThreadPool[j].~thread();
		  }
	  }

	  //Changes the dimensions of the camera frame. The ramifications of changing the Frame dimensions are not really propagated through at the moment, so trying to change the camera dimensions will likely cause it to crash.
	  //It would need proper handling like digHoloFrameWidth_Valid checks in the digHoloFFTInit routine.
public: int SetFrameDimensions(int width, int height)
{
	if (width % DIGHOLO_PIXEL_QUANTA || height % DIGHOLO_PIXEL_QUANTA || width < 0 || height < 0)
	{
		if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
		{
			fprintf(consoleOut, "SetFrameDimensions Error : Width and height must both be multiples of %i.\n", DIGHOLO_PIXEL_QUANTA);
		}
		return DIGHOLO_ERROR_INVALIDDIMENSION;
	}
	else
	{
		config.frameWidth = width;
		config.frameHeight = height;
		return DIGHOLO_ERROR_SUCCESS;
	}
}
public: int SetFrameWidth(int width)
{
	if (width % 16)
	{
		if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
		{
			fprintf(consoleOut, "SetFrameDimensions Error : Width and height must both be multiples of 16.\n");
		}
		return 0;
	}
	else
	{
		config.frameWidth = width;
		return 1;
	}
}
public: int SetFrameHeight(int height)
{
	if (height % 16)
	{
		if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
		{
			fprintf(consoleOut, "SetFrameDimensions Error : Width and height must both be multiples of 16.\n");
		}
		return 0;
	}
	else
	{
		config.frameHeight = height;
		return 1;
	}
}

public: void GetFrameDimensions(int& width, int& height)
{
	width = config.frameWidth;
	height = config.frameHeight;
}
public: int GetFrameWidth()
{
	return config.frameWidth;
}

public: int GetFrameHeight()
{
	return config.frameHeight;
}
	  //Sets the size of the desired FFT window on the camera. 'desired' because if you set it to a number larger than the dimensions of the camera, the true window will end up being smaller (digHoloWoiPolWidth)
public: void WindowSetSize(int width, int height)
{
	config.fftWindowSizeX = width;
	config.fftWindowSizeY = height;
}
	  //Gets the size of the FFT window on the camera.
public: void WindowGetSize(int& width, int& height)
{
	width = config.fftWindowSizeX;
	height = config.fftWindowSizeY;
}



	  //Sets the frame buffer pointer which feeds the digHoloFFT routine
	  //Unless updating the pointer has been disabled, in which case this function does nothing.
public: int SetFrameBuffer(float* buffer)
{
	int errorCode = DIGHOLO_ERROR_NULLPOINTER;

	if (buffer)
	{
		//ViewportWaitHandle.Set();
		digHoloPixelBufferType = 0;
		digHoloPixelBufferConversionFlag = 0;
		digHoloPixelsCameraPlane = buffer;
		errorCode = DIGHOLO_ERROR_SUCCESS;
	}

	if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
	{
		fprintf(consoleOut, "FrameBuffer Pointer:	%p\n", (void*)buffer);
		fflush(consoleOut);
	}
	return errorCode;
}
public: int SetFrameBufferUint16(unsigned short* buffer, int transposed)
{
	int errorCode = 1;
	if (buffer)
	{
		digHoloPixelBufferType = 1;
		digHoloPixelBufferConversionFlag = 0;
		digHoloPixelBufferUint16transposed = transposed;
		digHoloPixelsCameraPlaneUint16 = buffer;
		errorCode = 0;

	}
	if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
	{
		fprintf(consoleOut, "FrameBuffer Pointer:	%p\n", (void*)buffer);
		fflush(consoleOut);
	}
	return errorCode;
}
public: int SetFrameBufferFromFile(const char* fname)
{
	int errorCode = DIGHOLO_ERROR_ERROR;

	if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
	{
		fprintf(consoleOut, "Loading uint16 frame buffer from file (%s)\n", fname);
		fflush(consoleOut);
	}
	//const char* fname = args[0].c_str();
	//frameBufferFile = fopen(fname, "r");
	std::ifstream frameBufferFile(fname, std::ios::binary);
	if (frameBufferFile)
	{
		//fseek(frameBufferFile, 0, SEEK_END);
		//int64_t lSizeBytes = ftell(frameBufferFile);

		frameBufferFile.seekg(0, std::ios::end);
		std::streampos file_size = frameBufferFile.tellg();
		frameBufferFile.seekg(0, std::ios::beg);
		size_t lSizeBytes = file_size;
		//rewind(frameBufferFile);
		//This file appears to be in the wrong format?
		if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
		{
			fprintf(consoleOut, "FrameBuffer file size %zu bytes\n", lSizeBytes); fflush(consoleOut);
		}
		if (lSizeBytes % 2)
		{
			if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
			{
				fprintf(consoleOut, "FrameBuffer file does not have an even number of bytes (%zu bytes). Frames should be in uint16 format.\n", lSizeBytes); fflush(consoleOut);
			}
			errorCode = DIGHOLO_ERROR_INVALIDDIMENSION;
		}
		else
		{
			int scale = sizeof(unsigned short) / sizeof(char);
			size_t frameBufferLength = (lSizeBytes / scale);
			unsigned short* frameBuffer16 = 0;
			int errCode = allocate1D(frameBufferLength, frameBuffer16);
			int errCode2 = allocate1D(frameBufferLength, digHoloPixelBuffer);

			if (errCode == DIGHOLO_ERROR_SUCCESS && errCode2 == DIGHOLO_ERROR_SUCCESS)
			{
				frameBufferFile.read((char*)frameBuffer16, sizeof(unsigned short) * frameBufferLength);

				//Could be sped up
				for (size_t pixelIdx = 0; pixelIdx < frameBufferLength; pixelIdx++)
				{
					digHoloPixelBuffer[pixelIdx] = frameBuffer16[pixelIdx];
				}

				size_t result = frameBufferFile.gcount();
				if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
				{
					fprintf(consoleOut, "File loaded. %zu bytes\n", result); fflush(consoleOut);
				}
				free1D(frameBuffer16);
				SetFrameBuffer(digHoloPixelBuffer);
			}
			else
			{
				frameBufferFile.close();
				return DIGHOLO_ERROR_MEMORYALLOCATION;
			}
		}
		frameBufferFile.close();
		errorCode = DIGHOLO_ERROR_SUCCESS;
	}
	else
	{
		fprintf(consoleOut, "FrameBuffer file could not be opened. %s\n", fname); fflush(consoleOut);
		errorCode = DIGHOLO_ERROR_FILENOTFOUND;
	}
	return errorCode;
}
	  //Returns a pointer to the current frame buffer being used by digHolo for processing.
public: float* GetFrameBuffer()
{
	return digHoloPixelsCameraPlane;
}
public: unsigned short* GetFrameBufferUint16(int& transposeMode)
{
	transposeMode = digHoloPixelBufferUint16transposed;
	return digHoloPixelsCameraPlaneUint16;
}

public: complex64* GetFourierPlaneFull(int& batchCount, int& polCount, int& width, int& height)
{
	if (digHoloPixelsFourierPlane)
	{
		width = digHoloWoiPolWidth_Valid[FFT_IDX];
		height = digHoloWoiPolHeight_Valid[FFT_IDX];
		polCount = digHoloPolCount_Valid[FFT_IDX];
		batchCount = digHoloBatchAvgCount_Valid[FFT_IDX] * digHoloBatchCount_Valid[FFT_IDX];
		return digHoloPixelsFourierPlane;
	}
	else
	{
		width = 0;
		height = 0;
		polCount = 0;
		batchCount = 0;
		return 0;
	}
}
public: complex64* GetFourierPlaneWindow(int& batchCount, int& polCount, int& width, int& height)
{
	if (digHoloPixelsFourierPlaneMasked)
	{
		width = digHoloWoiPolWidth_Valid[IFFT_IDX];
		height = digHoloWoiPolHeight_Valid[IFFT_IDX];
		polCount = digHoloPolCount_Valid[IFFT_IDX];
		batchCount = digHoloBatchCount_Valid[IFFT_IDX];
		return digHoloPixelsFourierPlaneMasked;
	}
	else
	{
		width = 0;
		height = 0;
		polCount = 0;
		batchCount = 0;
		return 0;
	}
}

public: int SetBatchUint16(int batchCount, unsigned short* buffer, int avgCount, int avgMode, int transpose)
{
	if (buffer)
	{
		//SetFrameBufferEnabled = false;
		digHoloPixelsCameraPlaneUint16 = buffer;
		digHoloPixelBufferType = 1;
		if (transpose) { transpose = 1; }
		digHoloPixelBufferUint16transposed = transpose;

		if (batchCount < 1)
		{
			batchCount = 1;
		}
		if (avgCount < 1)
		{
			avgCount = 1;
		}
		config.batchCount = batchCount;
		config.avgCount = avgCount;
		config.avgMode = avgMode;
		return DIGHOLO_ERROR_SUCCESS;
	}
	else
	{
		return DIGHOLO_ERROR_NULLPOINTER;
	}
}

	  //Setups up a batch of batchCount frames, starting at memory location buffer.
	  //Next time digHoloFFT is run, this batch will be processed.
public: int SetBatch(int batchCount, float* buffer, int avgCount, int avgMode)
{
	if (buffer)
	{
		digHoloPixelBufferType = 0;
		//SetFrameBufferEnabled = false;
		//Free the internal pixel buffer if it exists. We won't be using it, we'll be using the 'buffer' passed in from outside.
		free1D(digHoloPixelBuffer);
		digHoloPixelBufferLength = 0;

		digHoloPixelsCameraPlane = buffer;

		if (avgCount < 1)
		{
			avgCount = 1;
		}
		if (batchCount < 1)
		{
			batchCount = 1;
		}

		if (avgMode < 0 || avgMode>1)
		{
			avgMode = 0;
		}

		config.batchCount = batchCount;
		config.avgCount = avgCount;
		config.avgMode = avgMode;
		return DIGHOLO_ERROR_SUCCESS;
	}
	else
	{
		return DIGHOLO_ERROR_NULLPOINTER;
	}
}

public: int SetBatchCalibration(complex64* cal, int polCount, int batchCount)
{
	if (polCount > 0 && batchCount > 0 && cal)
	{
		//If the size of the array has changed.
		if (polCount != digHoloBatchCalibrationPolCount || batchCount != digHoloBatchCalibrationBatchCount)
		{
			allocate2D(polCount, batchCount, digHoloBatchCalibration);
		}

		memcpy(&digHoloBatchCalibration[0][0], &cal[0], sizeof(complex64) * polCount * batchCount);

		digHoloBatchCalibrationBatchCount = batchCount;
		digHoloBatchCalibrationPolCount = polCount;

		digHoloBatchCalibrationEnabled = true;
		return DIGHOLO_ERROR_SUCCESS;
	}
	else
	{
		//Should free here? or leave it to decrease memory access violations?
		digHoloBatchCalibrationEnabled = false;
		if (cal == 0)
		{
			return DIGHOLO_ERROR_NULLPOINTER;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}

public: complex64* GetBatchCalibration(int* polCount, int* batchCount)
{
	if (digHoloBatchCalibration)
	{
		polCount[0] = digHoloBatchCalibrationPolCount;
		batchCount[0] = digHoloBatchCalibrationBatchCount;
		return &digHoloBatchCalibration[0][0];
	}
	else
	{
		polCount[0] = 0;
		batchCount[0] = 0;
		return 0;
	}
}

public: int SetBatchCalibrationEnabled(int enabled)
{
	if (enabled == 0)
	{
		digHoloBatchCalibrationEnabled = 0;
	}
	else
	{
		digHoloBatchCalibrationEnabled = 1;
	}

	return DIGHOLO_ERROR_SUCCESS;
}
public: int GetBatchCalibrationEnabled()
{
	return digHoloBatchCalibrationEnabled;
}

public: int SetBatchCalibrationLoadFromFile(const char* fname, int polCount, int batchCount)
{
	std::ifstream frameBufferFile(fname, std::ios::binary);

	if (frameBufferFile)
	{
		//The expected size of a single frame.
		const size_t expectedTotalSize32 = polCount * batchCount * sizeof(complex64);

		frameBufferFile.seekg(0, std::ios::end);
		std::streampos file_size = frameBufferFile.tellg();
		frameBufferFile.seekg(0, std::ios::beg);
		const size_t fileSizeBytes = file_size;

		if (fileSizeBytes == expectedTotalSize32)
		{
			if (fileSizeBytes == expectedTotalSize32)
			{
				int scale = sizeof(complex64) / sizeof(char);
				size_t arrayLength = (fileSizeBytes / scale);

				complex64* cal = 0;
				allocate1D(arrayLength, cal);

				frameBufferFile.read((char*)cal, sizeof(complex64) * arrayLength);

				size_t result = frameBufferFile.gcount();

				if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
				{
					fprintf(consoleOut, "Batch calibration file loaded. %zu bytes\n", result); fflush(consoleOut);
				}

				SetBatchCalibration(cal, polCount, batchCount);

				free1D(cal);
			}

		}
		else
		{
			digHoloBatchCalibrationEnabled = false;
			return DIGHOLO_ERROR_INVALIDDIMENSION;
		}

		frameBufferFile.close();
		digHoloBatchCalibrationEnabled = true;
		return DIGHOLO_ERROR_SUCCESS;

	}
	else
	{
		digHoloBatchCalibrationEnabled = false;
		return DIGHOLO_ERROR_FILENOTFOUND;
	}
}

private: void stressTest()
{
	const int batchCount0 = digHoloBatchCOUNT;
	const int trialCount = 10000;
	srand(0);

	for (int trialIdx = 0; trialIdx < trialCount; trialIdx++)
	{
		int polCount = rand() % 2 + 1;
		int batchCount = rand() % batchCount0 + 1;
		int nx = rand() % 1024 + 1;
		int ny = rand() % 1024 + 1;

		int maxMG = rand() % 1000 + 1;

		float fourierWindowSize = (float)(1.0 * rand() / RAND_MAX);

		int resIdx = rand() % 2;

		config.polCount = polCount;
		config.fftWindowSizeX = nx;
		config.fftWindowSizeY = ny;
		config.batchCount = batchCount;
		config.maxMG = maxMG;
		config.apertureSize = fourierWindowSize;
		config.resolutionIdx = resIdx;

		int modeCount = 0;
		if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
		{
			fprintf(consoleOut, "DEBUG -> resIdx : %i	polCount : %i batchCount : %i nx : %i ny : %i maxMG : %i fourierWindow : %f\n\r", resIdx, polCount, batchCount, nx, ny, maxMG, fourierWindowSize);
			fflush(consoleOut);
		}
		ProcessBatch(batchCount, modeCount, polCount);

		//digHoloFFT();
		//digHoloIFFT();
		//digHoloApplyTilt();
		//digHoloOverlapModes();
	}
}

	   public: int GetViewportToFile(int displayMode, int forceProcessing, int& width, int& height, char*& windowString, const char* fname)
	   {

		   unsigned char* buffer = GetViewport(displayMode, forceProcessing, width, height, windowString);

		   if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
		   {
			   fprintf(consoleOut, "Viewport to file (%i x %i) {%s}\n\r",width,height,windowString);
			   fflush(consoleOut);
		   }

		   FILE* viewportFile = fopen(fname, "wb");

		   if (viewportFile)
		   {
			   std::string filename(fname);
			   std::string fileEx(".bmp");
			   size_t length = filename.length();
			   int posIdx = (int)filename.find(fileEx);

			   //If the user specified a bitmap file extension, assume they want a bitmap output
			   if (posIdx == (length - 4))
			   {
				   writeToBitmap(buffer, width, height, fname);
			   }
			   else
			   {
				   fwrite(buffer, sizeof(unsigned char), width * height * 3, viewportFile);
				   fclose(viewportFile);
			   }
			   return DIGHOLO_ERROR_SUCCESS;
		   }
		   else
		   {
			   return DIGHOLO_ERROR_FILENOTCREATED;
		   }
	   }

public: void debugRoutine()
{
	if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
	{
		fprintf(consoleOut, "<DEBUG>\n\r");
		fflush(consoleOut);
	}

	stressTest();
	
	if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
	{
		fprintf(consoleOut, "</DEBUG>\n\r");
		fflush(consoleOut);
	}
}


	  //Processes a batch, previously specified by SetBatch. It returns a pointer to the mode coefficients.
	  //However it also returns batchCount,modeCount and polCount, which are the dimensions of digHolOverlapCoefs
public: complex64** ProcessBatch(int& batchCount, int& modeCount, int& polCount)
{
	if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
	{
		fprintf(consoleOut, "digHoloFFT()\n\r");
		fflush(consoleOut);
	}
	digHoloFFT();
	if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
	{
		fprintf(consoleOut, "digHoloIFFT()\n\r");
		fflush(consoleOut);
	}
	digHoloIFFT();
	if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
	{
		fprintf(consoleOut, "digApplyTilt()\n\r");
		fflush(consoleOut);
	}
	digHoloApplyTilt();
	if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
	{
		fprintf(consoleOut, "digHoloOverlapModes()\n\r");
		fflush(consoleOut);
	}
	digHoloOverlapModes();
	if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
	{
		fprintf(consoleOut, "Complete.\n\r");
		fflush(consoleOut);
	}
	batchCount = digHoloBatchCOUNT;
	polCount = digHoloPolCount;
	modeCount = digHoloModeCountOut;

	return digHoloOverlapCoefsPtr;

}

public: int ProcessFFT()
{
	return digHoloFFT();
}

public: int ProcessIFFT()
{
	return digHoloIFFT();
}
public: int ProcessApplyTilt()
{
	return digHoloApplyTilt();
}
public: int ProcessOverlapModes()
{
	return digHoloOverlapModes();
}

public: int SetBasisTypeCustom(int modeCountIn, int modeCountOut, complex64* transform)
{
	allocate1D(modeCountIn * modeCountOut, digHoloBasisTransformFull);
	if (!digHoloBasisTransformFull)
	{
		return DIGHOLO_ERROR_MEMORYALLOCATION;
	}
	else
	{
		memcpy(&digHoloBasisTransformFull[0][0], &transform[0][0], sizeof(complex64) * modeCountIn * modeCountOut);
		digHoloBasisTransformFullModeCountIn = modeCountIn;
		digHoloBasisTransformFullModeCountOut = modeCountOut;
		digHoloBasisTransformIsValid = false;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

public: int GetBatchSummary(int planeIdx, int& parameterCount, int& batchCount, int& polCount, int& pixelCountX, int& pixelCountY, float*& parameters, float*& intensity, float*& x, float*& y)
{
	if (DIGHOLO_ANALYSIS)
	{
		batchCount = (digHoloBatchCOUNT * digHoloBatchAvgCount) + 1;
		polCount = digHoloPolCount;
		parameterCount = DIGHOLO_ANALYSISCount;
		int resIdx = digHoloResolutionIdx;

		if (planeIdx >= 0 && planeIdx < DIGHOLO_PLANECOUNTMAX)
		{
			if (planeIdx == 0)
			{
				pixelCountX = digHoloWoiPolWidth[0] / 2 + 1;
				pixelCountY = digHoloWoiPolHeight[0];
				x = &digHoloKXaxis[0];
				y = &digHoloKYaxis[0];
			}
			else
			{
				pixelCountX = digHoloWoiPolWidth[resIdx];
				pixelCountY = digHoloWoiPolHeight[resIdx];
				x = &digHoloXaxis[resIdx][0];
				y = &digHoloYaxis[resIdx][0];
			}
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDDIMENSION;
		}

		//2, DIGHOLO_ANALYSISCount, digHoloPolCount, (digHoloBatchCOUNT * digHoloBatchAvgCount) + 1
		parameters = &DIGHOLO_ANALYSIS[planeIdx][0][0][0];
		//allocate3D(6, digHoloPolCount, digHoloWoiPolWidth[FFT_IDX] * digHoloWoiPolHeight[FFT_IDX], DIGHOLO_ANALYSISBatchSum);
		intensity = &DIGHOLO_ANALYSISBatchSum[planeIdx][0][0];
		return DIGHOLO_ERROR_SUCCESS;
	}
	else
	{
		return DIGHOLO_ERROR_NULLPOINTER;
	}
}

public: int GetThreadCount()
{
	return config.threadCount;
}
public: int SetThreadCount(int threadCount)
{
	if (threadCount < 1)
	{
		threadCount = CPU_COUNT;
	}
	if (threadCount > THREADCOUNT_MAX)
	{
		threadCount = THREADCOUNT_MAX;
	}
	config.threadCount = threadCount;
	return DIGHOLO_ERROR_SUCCESS;
}
public: complex64** GetCoefs(int& batchCount, int& modeCount, int& polCount)
{
	batchCount = digHoloBatchCOUNT;
	polCount = digHoloPolCount;
	modeCount = digHoloModeCountOut;

	return digHoloOverlapCoefsPtr;

}

	  //This function does digHolo processing, and returns the resulting field in complex format.
	  //It overwrites the digHoloPixelsCameraPlaneReconstructedTilted array with the tilt/focus correction and returns a pointer to that array
	  //As opposed to normal processing, which does not write back to digHoloPixelsCameraPlaneReconstructedTilted, but only writes out to 16-bit ints
/*
public: complex64* GetField(float* pixelBuff, int& pixelCountX, int& pixelCountY)
{
	//ViewportPause();

	if (pixelBuff)
	{
		SetFrameBuffer(pixelBuff);
		digHoloFFT();
		digHoloIFFT();
		digHoloApplyTilt(-1, true);//-1 indicates both polarisations, true indicated overwrite an untilted version in 32-bit float format in addition to the 16-bit integer formats

		const int polCount = digHoloPolCount;
		const int batchIdx = 0;

		int fftIdx = IFFT_IDX;
		if (digHoloResolutionIdx == FULLRES_IDX)
		{
			fftIdx = FFT_IDX;
		}

		int width = digHoloWoiPolWidth[fftIdx];
		int height = digHoloWoiPolHeight[fftIdx];

		const size_t batchOffset = ((size_t)batchIdx) * width * height * polCount;

		pixelCountX = width;
		pixelCountY = height * polCount;

		return (complex64*)&digHoloPixelsCameraPlaneReconstructedTilted[batchOffset];
	}
	else
	{
		pixelCountX = 0;
		pixelCountY = 0;
		return 0;
	}
}
*/

public: int GetFields16(int& batchCount, int& polCount, short*& fieldR, short*& fieldI, float*& fieldScale, float*& x, float*& y, int& width, int& height)
{
	if (digHoloPixelsCameraPlaneReconstructed16R)
	{
		batchCount = digHoloBatchCount_Valid[IFFT_IDX];
		polCount = digHoloPolCount_Valid[IFFT_IDX];

		fieldR = digHoloPixelsCameraPlaneReconstructed16R;
		fieldI = digHoloPixelsCameraPlaneReconstructed16I;
		fieldScale = &DIGHOLO_ANALYSIS[IFFT_IDX][DIGHOLO_ANALYSIS_MAXABS][0][0];

		x = digHoloXaxis[IFFT_IDX];
		y = digHoloYaxis[IFFT_IDX];

		height = digHoloWoiPolHeight_Valid[IFFT_IDX];
		width = digHoloWoiPolWidth_Valid[IFFT_IDX];
		return DIGHOLO_ERROR_SUCCESS;
	}
	else
	{
		batchCount = 0;
		polCount = 0;
		fieldR = 0;
		fieldI = 0;
		fieldScale = 0;
		x = 0;
		y = 0;
		width = 0;
		height = 0;
		return DIGHOLO_ERROR_NULLPOINTER;
	}
}
	  public: complex64* GetRefCalibration(int& batchCount, int& polCount, float*& x, float*& y, int& width, int& height)
	  {
		  //const int planeIdx = IFFTCAL_IDX;
		  const int resIdx = digHoloResolutionIdx;
		  batchCount = digHoloBatchCount_Valid[IFFTCAL_IDX];
		  polCount = digHoloPolCount_Valid[FFT_IDX];
		  height = digHoloWoiPolHeight[resIdx];
		  width = digHoloWoiPolWidth[resIdx];
		  x = digHoloXaxis[IFFT_IDX];
		  y = digHoloYaxis[IFFT_IDX];
		  return digHoloRefCalibrationReconstructedPlane;
		 // return digHoloRefCalibrationFourierPlaneMasked;
		//  return digHoloPixelsFourierPlaneMasked;
		//  return digHoloPixelsCameraPlaneReconstructedTilted;
		//  return digHoloRefCalibrationFourierPlaneMasked;//digHoloPixelsFourierPlaneMasked;// 
		//  return digHoloPixelsFourierPlaneMasked;
	  }

public: complex64* GetFields(int& batchCount, int& polCount, float*& x, float*& y, int& width, int& height)
{
	if (digHoloPixelsCameraPlaneReconstructedTilted && digHoloPixelsCameraPlaneReconstructed16R)
	{
		batchCount = digHoloBatchCount_Valid[IFFT_IDX];
		polCount = digHoloPolCount_Valid[IFFT_IDX];
		height = digHoloWoiPolHeight_Valid[IFFT_IDX];
		width = digHoloWoiPolWidth_Valid[IFFT_IDX];
		x = digHoloXaxis[IFFT_IDX];
		y = digHoloYaxis[IFFT_IDX];

		//For each polarisation. Keep convert the 16-bit int fields to 32-bit floats, keep track of the total power and maximum absolute value.
		complex64* field = (complex64*)&digHoloPixelsCameraPlaneReconstructedTilted[0];
		const size_t pixelCount = (size_t)width * (size_t)height;
		const __m256i permuteIdxes = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
		const __m256i permuteIdxes16 = _mm256_set_epi8(31, 30, 27, 26, 29, 28, 25, 24, 23, 22, 19, 18, 21, 20, 17, 16, 15, 14, 11, 10, 13, 12, 9, 8, 7, 6, 3, 2, 5, 4, 1, 0);
		const __m256i zero = _mm256_set1_epi32(0);

		for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
		{
			const size_t batchOffset = batchIdx * pixelCount * polCount;
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				const size_t polOffset = polIdx * pixelCount;
				const float  fieldScale = (float)(1.0 / DIGHOLO_ANALYSIS[IFFT_IDX][DIGHOLO_ANALYSIS_MAXABS][polIdx][batchIdx]);
				const __m256 scale = _mm256_set1_ps(fieldScale);

				for (size_t pixelIdx = 0; pixelIdx < pixelCount; pixelIdx += 8)
				{
					short* fieldR = &digHoloPixelsCameraPlaneReconstructed16R[batchOffset + polOffset + pixelIdx];
					short* fieldI = &digHoloPixelsCameraPlaneReconstructed16I[batchOffset + polOffset + pixelIdx];

					//Load in 8xint16=128bits
					const __m128i fR16 = _mm_loadu_si128((__m128i*)fieldR);
					const __m128i fI16 = _mm_loadu_si128((__m128i*)fieldI);

					__m256i fRI32 = zero;
					fRI32 = _mm256_insertf128_si256(fRI32, fR16, 0);
					fRI32 = _mm256_insertf128_si256(fRI32, fI16, 1);

					fRI32 = _mm256_permutevar8x32_epi32(fRI32, permuteIdxes);//I7|I6|R7|R6|I5|I4|R5|R4||I3|I2|R3|R2|I1|I0|R1|R0
					fRI32 = _mm256_shuffle_epi8(fRI32, permuteIdxes16);//I|R|I|R...

					__m256 fRI32a = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(fRI32, 0)));
					__m256 fRI32b = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(fRI32, 1)));
					fRI32a = _mm256_mul_ps(fRI32a, scale);
					fRI32b = _mm256_mul_ps(fRI32b, scale);

					float* fieldOutA = (float*)field[batchOffset + polOffset + pixelIdx];
					float* fieldOutB = (float*)field[batchOffset + polOffset + pixelIdx + 4];
					_mm256_storeu_ps(fieldOutA, fRI32a);
					_mm256_storeu_ps(fieldOutB, fRI32b);
				}
			}
		}
		return field;
	}
	else
	{
		batchCount = 0;
		polCount = 0;
		height = 0;
		width = 0;
		x = 0;
		y = 0;
		return 0;
	}
}

public: complex64* HGtoLG(complex64** HGcoefs, complex64** LGcoefs, int batchCount, int modeCount, int polCount, int inverseTransform)
{
	complex64 alpha;
	alpha[0] = 1;
	alpha[1] = 0;
	complex64 beta;
	beta[0] = 0;
	beta[1] = 0;

	int incx = 1;
	int incy = 1;


	for (int polIdx = 0; polIdx < polCount; polIdx++)
	{
		size_t idx = polIdx * modeCount;
		for (int mgIdx = 0; mgIdx < digHoloMaxMG; mgIdx++)
		{
			complex64* U = LGtoHG[mgIdx];

			int mgIDX = mgIdx + 1;

			for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
			{
				complex64* coefsHG = &HGcoefs[batchIdx][idx];
				complex64* coefsLG = &LGcoefs[batchIdx][idx];

#ifdef LAPACKBLAS_ENABLE
#ifdef CBLAS_ENABLE
				auto tpose = inverseTransform ? CBLAS_TRANSPOSE::CblasConjTrans : CBLAS_TRANSPOSE::CblasNoTrans;
				cblas_cgemv(CBLAS_LAYOUT::CblasColMajor, tpose, mgIDX, mgIDX, &alpha, U, mgIDX, coefsHG, incx, &beta, coefsLG, incy);
#else
				const char trans = inverseTransform ? 'C' : 'N';
				cgemv(&trans, &mgIDX, &mgIDX, (BLAS_COMPLEXTYPE*)alpha, (BLAS_COMPLEXTYPE*)U, &mgIDX, (BLAS_COMPLEXTYPE*)coefsHG, &incx, (BLAS_COMPLEXTYPE*)beta, (BLAS_COMPLEXTYPE*)coefsLG, &incy);
#endif

#endif
			}
			idx += (mgIdx + 1);
		}

	}
	return (complex64*)&LGtoHG[0][0];
}

	  //Similar to ProcessBatch, except it processes the batch in a wavelength dependent way.
	  //A separate SetFrequencySweep type routine could also be employed.
public: complex64** ProcessBatchFrequencySweepLinear(int& batchCount, int& modeCount, int& polCount, float lambdaStart, float lambdaStop, int lambdaCount)
{
	//Set up a wavelength sweep array of length lambdaCount. The batchCount need not match the lambdaCount. e.g. multiple wavelength sweeps might be processed in a single batch (e.g. wavelength sweeps for N modes, meaning the batchCount is N*lambdaCount)
	SetWavelengthFrequencyLinear(lambdaStart, lambdaStop, lambdaCount);

	//The process the batch
	complex64** batchCoefs = ProcessBatch(batchCount, modeCount, polCount);
	return batchCoefs;
}

public: float* GetWavelengths(int *lambdaCount)
{
	if (digHoloWavelength)
	{
		digHoloWavelengthUpdate();
		lambdaCount[0] = digHoloWavelengthCount;
		return digHoloWavelength;
	}
	else
	{
		lambdaCount[0] = 0;
		return 0;
	}
}
public: complex64** ProcessBatchFrequencySweepArbitrary(int& batchCount, int& modeCount, int& polCount, float* wavelengths, int lambdaCount)
{
	//Set up a wavelength sweep array of length lambdaCount. The batchCount need not match the lambdaCount. e.g. multiple wavelength sweeps might be processed in a single batch (e.g. wavelength sweeps for N modes, meaning the batchCount is N*lambdaCount)
	SetWavelengthArbitrary(wavelengths, lambdaCount);

	//The process the batch
	complex64** batchCoefs = ProcessBatch(batchCount, modeCount, polCount);
	return batchCoefs;
}
	  //Pauses the viewport. The kind of thing you'd do before doing some batch processing etc., so that the digHolo routines in the viewport update, don't interfere with the processing.
	/*  void ViewportPause()
	  {
		  //Pause the viewport
		  ViewportWaitHandle.Reset();
		  //Wait for current viewport update to finish
		  ViewportFinishedHandle.WaitOne();
		  //Set the frame buffer to the current frame from the camera
	  }

	  //Restarts the viewport (if it's paused) and also indicates that there's a frame ready to be displayed in the viewport.
	  //Otherwise, just setting the ViewportWaitHandle->Set(), would mean the viewport doesn't update until the _next_ frame shows up. This indicates the new frame is already there, ready to be displayed.
	  void ViewportResumeNewFrame()
	  {
		  //New frame ready for the viewport
		  ViewportNewFrameHandle.Set();
		  //Restart the viewport
		  ViewportWaitHandle.Set();
	  }
	  */

public: unsigned char* GetViewport(int DisplayMode, int forceProcessing, int& width, int& height, char*& windowString)
{
	if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
	{
		fprintf(consoleOut, "width pointer : %p\n", (void*)&width);
		fprintf(consoleOut, "height pointer : %p\n", (void*)&height);
		fprintf(consoleOut, "windowString pointer : %p\n", (void*)&windowString);
		fflush(consoleOut);
	}
	windowString = digHoloPlotString;
	//This handle is usually used to pause the viewport, whilst some other processing is going on.
	if (DisplayMode >= 0)
	{
		if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		{
			fprintf(consoleOut, "ViewportWaitHandle waiting...\n");
			fflush(consoleOut);
		}
		//ViewportWaitHandle.WaitOne();
		if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		{
			fprintf(consoleOut, "ViewportWaitHandle continue.\n");
			fflush(consoleOut);
		}
	}

		//Check the size of the current camera frame
	digHoloFrameWidth = config.frameWidth;
	digHoloFrameHeight = config.frameHeight;
	const size_t plotSizeNew = digHoloFrameHeight * digHoloFrameWidth;
	if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
	{
		fprintf(consoleOut, "digHoloFrameHeight :	%i\n", digHoloFrameHeight);
		fprintf(consoleOut, "digHoloFrameWidth :	%i\n", digHoloFrameWidth);
		fflush(consoleOut);
	}
	//If the frame size has increased, you'll need to reallocate more memory.
	if (plotSizeNew > digHoloPlotPixelsCount)
	{
		//Allocate a potentially full camera frame worth of pixels
		allocate1D(plotSizeNew * 3 * 2, digHoloPlotPixels);
		allocate1D(plotSizeNew * 2, digHoloPlotPixelsComplex);
		memset(digHoloPlotPixels, 0, sizeof(unsigned char) * plotSizeNew * 3);
		memset(digHoloPlotPixelsComplex, 0, sizeof(complex64) * plotSizeNew);
		digHoloPlotPixelsCount = plotSizeNew;
	}
	else
	{
		if (DisplayMode != 0)
		{
			memset(digHoloPlotPixels, 0, sizeof(unsigned char) * plotSizeNew * 3);
			memset(digHoloPlotPixelsComplex, 0, sizeof(complex64) * plotSizeNew);
		}
	}

	//"Paused". Nothing to display
	if (DisplayMode == DIGHOLO_VIEWPORT_NONE)
	{
		//return 0;
	}
	//Plots raw camera pixels, with some additional annotations about where the FFT window is positioned.
	if (DisplayMode == DIGHOLO_VIEWPORT_CAMERAPLANE)
	{
		if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		{
			fprintf(consoleOut, "Viewport Raw.\n");
			fflush(consoleOut);
		}
		digHoloUpdateViewport_Raw(digHoloPlotPixels, forceProcessing, digHoloPlotPixelsComplex, width, height, windowString);
	}
	//FFT (Fourier space of camera)
	if (DisplayMode == DIGHOLO_VIEWPORT_FOURIERPLANE || DisplayMode == DIGHOLO_VIEWPORT_FOURIERPLANEDB)
	{
		if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		{
			fprintf(consoleOut, "Viewport FFT.\n");
			fflush(consoleOut);
		}
		digHoloUpdateViewport_FFTx(digHoloPlotPixels, forceProcessing, digHoloPlotPixelsComplex, width, height, windowString, (DisplayMode == DIGHOLO_VIEWPORT_FOURIERPLANEDB));
	}
	//FFT (Fourier space of camera, with just the parts inside the IFFT window indicated)
	if (DisplayMode == DIGHOLO_VIEWPORT_FOURIERWINDOW || DisplayMode == DIGHOLO_VIEWPORT_FOURIERWINDOWABS)
	{
		digHoloUpdateViewport_FFT_WindowOnly(digHoloPlotPixels, forceProcessing, digHoloPlotPixelsComplex, width, height, windowString, DisplayMode == DIGHOLO_VIEWPORT_FOURIERWINDOWABS);
	}

	//IFFT
	if (DisplayMode == DIGHOLO_VIEWPORT_FIELDPLANE || DisplayMode == DIGHOLO_VIEWPORT_FIELDPLANEABS)
	{

		digHoloUpdateViewport_IFFT(digHoloPlotPixels, forceProcessing, digHoloPlotPixelsComplex, width, height, windowString, (DisplayMode == DIGHOLO_VIEWPORT_FIELDPLANEABS));
	}

	//Mode reconstruction
	if (DisplayMode == DIGHOLO_VIEWPORT_FIELDPLANEMODE)
	{
		digHoloUpdateViewport_Modes(digHoloPlotPixels, forceProcessing, digHoloPlotPixelsComplex, width, height, windowString);
	}

	//A routine used for quickly checking how fast the code is running
	/*if (DisplayMode == 6)
	{
		benchmarkRoutine(0, 5, true);
	}
	if (DisplayMode == 7)
	{
		digHoloUpdateViewport_DCT(digHoloPlotPixels, digHoloPlotPixelsComplex, width, height, windowString);
	}*/
	if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
	{
		fprintf(consoleOut, "Viewport Complete.\n");
		fflush(consoleOut);
	}
	//ViewportFinishedHandle.Set();
	return digHoloPlotPixels;
}

	  //Sets the amount of console printout to perform. See config.verbosity for details.
public: int SetVerbosity(int verboseMode)
{
	if (verboseMode < 0)
	{
		verboseMode = 0;
	}
	config.verbosity = verboseMode;
	return DIGHOLO_ERROR_SUCCESS;
}

public: int GetVerbosity()
{
	return config.verbosity;
}

public: complex64* GetFieldsBasis(int& modeCountOut, int& polCount, float*& x, float*& y, int& width, int& height)
{
	polCount = digHoloPolCount;
	int modeCountIn = digHoloModeCount;

	int basisIdx = digHoloResolutionIdx;
	digHoloUpdateBasis(basisIdx, width, height);

	x = digHoloXaxis[IFFT_IDX];
	y = digHoloYaxis[IFFT_IDX];

	const size_t pixelCount = (size_t)width * (size_t)height * polCount;

	complex64** coefsHG = 0;
	allocate2D(modeCountIn, modeCountIn * polCount, coefsHG);
	memset(&coefsHG[0][0], 0, sizeof(complex64) * modeCountIn * modeCountIn * polCount);

	for (int polIdx = 0; polIdx < polCount; polIdx++)
	{
		for (int modeIdx = 0; modeIdx < modeCountIn; modeIdx++)
		{
			coefsHG[modeIdx][modeIdx + polIdx * modeCountIn][0] = 1;
		}
	}

	complex64** coefs = 0;
	//Generate a full HGtoLG transform matrix from the LGtoHG sparse version
	if (config.basisType == DIGHOLO_BASISTYPE_LG)
	{
		modeCountOut = modeCountIn;
		allocate2D(modeCountOut, modeCountIn * polCount, coefs);
		HGtoLG(coefsHG, coefs, modeCountOut, modeCountOut, polCount, true);
	}
	else
	{
		if (config.basisType == DIGHOLO_BASISTYPE_CUSTOM && digHoloBasisTransformIsValid)
		{
			modeCountOut = digHoloBasisTransformModeCountOut;
			allocate2D(modeCountOut, modeCountIn * polCount, coefs);
			for (int modeIdx = 0; modeIdx < modeCountOut; modeIdx++)
			{
				for (int polIdx = 0; polIdx < polCount; polIdx++)
				{
					for (int modeIdy = 0; modeIdy < modeCountIn; modeIdy++)
					{
						const int idx = modeIdy * modeCountOut + modeIdx;
						coefs[modeIdx][modeIdy + polIdx * modeCountIn][0] = digHoloBasisTransform[idx][0];
						coefs[modeIdx][modeIdy + polIdx * modeCountIn][1] = digHoloBasisTransform[idx][1];
					}
				}
			}
		}
		else
		{
			modeCountOut = modeCountIn;
			coefs = coefsHG;
		}
	}
	allocate2D(modeCountOut, pixelCount, digHoloBasisFields);
	for (int modeIdx = 0; modeIdx < modeCountOut; modeIdx++)
	{
		complex64* coefs0 = coefs[modeIdx];

		generateModeSeparable(digHoloHGscaleX, digHoloHGscaleY, &digHoloBasisFields[modeIdx][0], digHoloHGX, digHoloHGY, width, height, coefs0, HG_M, HG_N, modeCountIn, polCount,0,NULL,NULL,NULL);
	}

	if (coefsHG != coefs)
	{
		free2D(coefs);
	}

	free2D(coefsHG);
	return &digHoloBasisFields[0][0];
}


	  public: int RefCalibrationSetEnabled(int enabled)
	  {
		  digHoloRefCalibrationEnabled = enabled;
		  return DIGHOLO_ERROR_SUCCESS;
	  }

	  public: int RefCalibrationGetEnabled()
	  {
		  return digHoloRefCalibrationEnabled;
	  }

	  public: int RefCalibrationLoadFromFile(const char* fname, int width, int height, int lambdaCount)
	  {
		  std::ifstream frameBufferFile(fname, std::ios::binary);

		  if (frameBufferFile)
		  {
			  //The expected size of a single frame.
			  const size_t expectedFrameSize16 = width * height * sizeof(unsigned short);
			  const size_t expectedFrameSize32 = width * height * sizeof(complex64);

			  //The expected total size if there is a calibration for every wavelength
			  size_t expectedTotalSize16 = expectedFrameSize16 * lambdaCount;
			  size_t expectedTotalSize32 = expectedFrameSize32 * lambdaCount;

			  frameBufferFile.seekg(0, std::ios::end);
			  std::streampos file_size = frameBufferFile.tellg();
			  frameBufferFile.seekg(0, std::ios::beg);
			  const size_t fileSizeBytes = file_size;

			  if (fileSizeBytes == expectedFrameSize16)
			  {
				  lambdaCount = 1;
				  expectedTotalSize16 = expectedFrameSize16;
			  }
			  else
			  {
				  if (fileSizeBytes == expectedFrameSize32 && lambdaCount!=2)
				  {
					  lambdaCount = 1;
					  expectedTotalSize32 = expectedFrameSize32;
				  }
			  }

			  if (fileSizeBytes==expectedTotalSize16)
			  {

				  int scale = sizeof(unsigned short) / sizeof(char);
				  size_t arrayLength = (fileSizeBytes / scale);
				  
				  unsigned short* cal = 0;
				  allocate1D(arrayLength, cal);

				  frameBufferFile.read((char*)cal, sizeof(unsigned short) * arrayLength);

				  size_t result = frameBufferFile.gcount();

				  if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
				  {
					  fprintf(consoleOut, "Frame calibration (intensity, int16) file loaded. %zu bytes\n", result); fflush(consoleOut);
				  }

				  RefCalibrationSet16(cal, width, height, lambdaCount);

				  free1D(cal);
			  }
			  else
			  {
				  if (fileSizeBytes == expectedTotalSize32)
				  {
					  int scale = sizeof(complex64) / sizeof(char);
					  size_t arrayLength = (fileSizeBytes / scale);

					  complex64* cal = 0;
					  allocate1D(arrayLength, cal);

					  frameBufferFile.read((char*)cal, sizeof(complex64) * arrayLength);

					  size_t result = frameBufferFile.gcount();

					  if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
					  {
						  fprintf(consoleOut, "Frame calibration (field, complex64) file loaded. %zu bytes\n", result); fflush(consoleOut);
					  }

					  RefCalibrationSet32(cal, width, height, lambdaCount);

					  free1D(cal);
				  }
				  else
				  {
					  digHoloRefCalibrationEnabled = false;
					  return DIGHOLO_ERROR_INVALIDDIMENSION;
				  }
			  }

			  
			  frameBufferFile.close();
			 
			  return DIGHOLO_ERROR_SUCCESS;

		  }
		  else
		  {
			  digHoloRefCalibrationEnabled = false;
			  return DIGHOLO_ERROR_FILENOTFOUND;
		  }
	  }

			public: complex64* RefCalibrationGet(int* width, int* height, int* wavelengthCount)
			{
				return 0;
			}

	  public: int RefCalibrationSet16(unsigned short* cal, int w, int h, int wavelengthCount)
	  {
		  if (!cal)
		  {
			  digHoloRefCalibrationEnabled = false;
			  return DIGHOLO_ERROR_NULLPOINTER;
		  }
		  else
		  {
			  if (w > 0 && h > 0 && wavelengthCount > 0)
			  {
				  const size_t width = w;
				  const size_t height = h;
				  const size_t lambdaCount = wavelengthCount;
				  size_t length = width * height * lambdaCount;

				  //Technically don't need to reallocate if length is the same
				  allocate1D(length,digHoloRefCalibration);

				  digHoloRefCalibrationHeight = h;
				  digHoloRefCalibrationWidth = w;
				  digHoloRefCalibrationWavelengthCount = wavelengthCount;

				  int maxV = 0;

				  for (size_t idx = 0; idx < length; idx++)
				  {
					  digHoloRefCalibration[idx][0] = cal[idx];
					  digHoloRefCalibration[idx][1] = 0;
					  if (cal[idx] > maxV)
					  {
						  maxV = cal[idx];
					  }
				  }

				  //We'll autoscale the reference to be between 0 and 1.
				  float maxScale = 1.0f / maxV;

				  for (size_t idx = 0; idx < length; idx++)
				  {
					  float v = 1.0f/(digHoloRefCalibration[idx][0] * maxScale);

					  if (isnan(v) || isinf(v))
					  {
						  v = 0;
					  }
					  digHoloRefCalibration[idx][0] = sqrtf(v);
				  }

				  digHoloRefCalibrationInvalidate();
				  digHoloRefCalibrationEnabled = true;
				  return DIGHOLO_ERROR_SUCCESS;
			  }
			  else
			  {
				  digHoloRefCalibrationEnabled = false;
				  return DIGHOLO_ERROR_INVALIDARGUMENT;
			  }
		  }
	  }

	  public: int RefCalibrationSet32(complex64* cal, int w, int h, int wavelengthCount)
	  {
		  if (!cal)
		  {
			  digHoloRefCalibrationEnabled = false;
			  return DIGHOLO_ERROR_NULLPOINTER;
		  }
		  else
		  {
			  if (w > 0 && h > 0 && wavelengthCount > 0)
			  {
				  const size_t width = w;
				  const size_t height = h;
				  const size_t lambdaCount = wavelengthCount;
				  size_t length = width * height * lambdaCount;

				  //Technically don't need to reallocate if length is the same
				  allocate1D(length, digHoloRefCalibration);

				  digHoloRefCalibrationHeight = h;
				  digHoloRefCalibrationWidth = w;
				  digHoloRefCalibrationWavelengthCount = wavelengthCount;

				  float maxV = 0;

				  for (size_t idx = 0; idx < length; idx++)
				  {
					  float vR = cal[idx][0];
					  float vI = cal[idx][1];

					  digHoloRefCalibration[idx][0] = vR;
					  digHoloRefCalibration[idx][1] = vI;

					  float v2 = vR * vR + vI * vI;

					  if (v2 > maxV)
					  {
						  maxV = v2;
					  }
				  }

				  maxV = sqrtf(maxV);

				  //We'll autoscale the reference to be between 0 and 1.
				  float maxScale = 1.0f / maxV;

				  for (size_t idx = 0; idx < length; idx++)
				  {
					  //Normalize to a magnitude between 0 and 1, and then inverse and conjugate.
					  float vR = 1.0f / (digHoloRefCalibration[idx][0] * maxScale);
					  float vI = -1.0f / (digHoloRefCalibration[idx][1] * maxScale);

					  if (isnan(vR) || isinf(vR) || isnan(vI) || isinf(vI))
					  {
						  vR = 0;
						  vI = 0;
					  }

					  digHoloRefCalibration[idx][0] = vR;
					  digHoloRefCalibration[idx][1] = vI;
				  }

				  digHoloRefCalibrationInvalidate();
				  digHoloRefCalibrationEnabled = true;
				  return DIGHOLO_ERROR_SUCCESS;
			  }
			  else
			  {
				  digHoloRefCalibrationEnabled = false;
				  return DIGHOLO_ERROR_INVALIDARGUMENT;
			  }
		  }
	  }
private:

	//A thread pool is maintained, whereby digHoloThreadPoolSize threads are created, and constantly kept alive, ready to be dispatched to perform new functions (work.callback)
	//using flags and pointers etc specified in 'work'. Particularly for very small routines, the overhead of creating the thread can be the bottleneck.
	void digHoloThreadPoolRoutine(int i)
	{
		workPackage* work = digHoloWorkPacks[i];

		while (work[0].active)
		{

			//Wait for the ResetEvent to be reset
			work[0].workNewEvent.WaitOne();
			work[0].workNewEvent.Reset();
			//Function pointer/callback would be a more appropriate way to handle this.
			switch (work[0].callback)
			{
			case (int)workFunction::digHoloFFTBatchWorker:
			{
				digHoloFFTBatchWorker(work[0]);
				break;
			}
			case (int)workFunction::digHoloIFFTRoutineWorker:
			{
				digHoloIFFTRoutineWorker(work[0]);
				break;
			}
			case (int)workFunction::applyTiltWorker:
			{
				applyTiltWorker(work[0]);
				break;
			}
			case (int)workFunction::fieldAnalysisWorker:
			{
				fieldAnalysisWorker(work[0]);
				break;
			}
			case (int)workFunction::powerMeter:
			{
				digHoloPowerMeterRoutine(work[0]);
				break;
			}
			case (int)workFunction::overlapWorkerHG:
			{
				overlapWorkerHG(work[0]);
				break;
			}
			case (int)workFunction::NA:
				break;
			}
			work[0].workCompleteEvent.Set();
		}
	}


	//Plots camera frames, with annotations indicating where the FFT windows are.
	int digHoloUpdateViewport_Raw(unsigned char* digHoloPlotPixels, int forceProcessing, complex64* digHoloPlotPixelsComplex, int& plotWidth, int& plotHeight, char* windowName)
	{
		//Dimensions of a camera frame
		plotWidth = digHoloFrameWidth;
		plotHeight = digHoloFrameHeight;
		int plotPixelCount = plotWidth * plotHeight;
		size_t batchCount = config.batchCount * config.avgCount;
		if (batchCount < 1)
		{
			batchCount = 1;
		}
		//Variables which keep track of the power in each polarisation.
		double totalPowerRoI[2] = { 0,0 };

		//The maximum value in the frame, used for plotting purposes.
		float maxV = -FLT_MAX;

		//If this is the first run, we'll have to do at least 1 FFT to make sure the FFT windows are initialised.
		/*if (digHoloWoiPolHeight[FFT_IDX] == 0)
		{
			if (config.verbosity > 2)
			{
				fprintf(consoleOut, "digHoloFFT(pointer = %p)\n", (void*)digHoloPixelsCameraPlane);
				fflush(consoleOut);
			}
			digHoloFFT();

			if (config.verbosity > 2)
			{
				fprintf(consoleOut, "digHoloFFT Complete.\n");
				fflush(consoleOut);
			}
		}*/

		//Update the FFT window location/dimensions
		if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		{
			fprintf(consoleOut, "digHoloFFT_SetWindow....\n");
			fflush(consoleOut);
		}

		if (forceProcessing)
		{
			digHoloFFT_SetWindow();
		}

		if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		{
			fprintf(consoleOut, "digHoloFFT_SetWindow Complete.\n");
			fflush(consoleOut);
		}
		//The number of polarisations currently active
		const int polCount = digHoloPolCount;
		//The dimensions of the FFT
		const int woiHeight = digHoloWoiPolHeight[FFT_IDX];
		const int woiWidth = digHoloWoiPolWidth[FFT_IDX];

		//For each polarisation
		if (!digHoloPixelsCameraPlane || !digHoloPlotPixelsComplex || !digHoloPlotPixels)
		{
			std::snprintf(&windowName[0], DIGHOLO_VIEWPORT_NAMELENGTH, "No frame available.");
			if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			{
				fprintf(consoleOut, "No frame available.\n");
				fflush(consoleOut);
			}
			return 0;
		}
		float* cameraPlane = 0;
		complex64* plotPixelsComplex = 0;

		for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
		{
			//The offset in memory of this polarisation
			const size_t pixelOffset = batchIdx * plotPixelCount;
			//Pointer to the part of the camera frame corresponding with this polarisation
			cameraPlane = &digHoloPixelsCameraPlane[pixelOffset];
			//The corresponding position in the viewport plot field array. We'll be plotting the camera frames as if they're complex numbers, even though they're real-only.
			//Unlike the raw camera frames of the cameraModule, this display will always be auto-normalized to the peak value in the frame (weak frames will still display as full colour scale)
			const size_t pixelOffsetOut = 0;
			plotPixelsComplex = (complex64*)&digHoloPlotPixelsComplex[pixelOffsetOut][0];

			if (batchIdx == 0)
			{
				memset(plotPixelsComplex, 0, sizeof(complex64) * plotPixelCount);
				memset(digHoloPlotPixels, 0, sizeof(unsigned char) * 3 * plotPixelCount);
			}

			//Iterate through the camera frame, copy the values to the plotPixelsComplex array and keep track of total power, and the peak value (for setting colour scale later).
			for (int i = 0; i < plotWidth; i++)
			{
				for (int j = 0; j < plotHeight; j++)
				{
					const size_t idx = j * plotWidth + i;
					const float v = cameraPlane[idx];
					plotPixelsComplex[idx][0] += v;
					if (plotPixelsComplex[idx][0] > maxV)
						maxV = plotPixelsComplex[idx][0];
				}
			}
		}

		for (int polIdx = 0; polIdx < polCount; polIdx++)
		{
			//The x,y position of the corner of the FFT window.
			int SX = digHoloWoiSX[FFT_IDX][polIdx];
			int SY = digHoloWoiSY[FFT_IDX][polIdx];

			//Iterate through the FFT window, add up the power inside the window, but also add additional emphasis/scaling/offset to outside the region, so that it will be visible when plotted.

			for (int j = 0; j < plotHeight; j++)
			{
				for (int i = 0; i < plotWidth; i++)
				{
					const size_t idx = j * plotWidth + i;
					if (j >= SY && j < (SY + woiHeight) && i >= SX && i < (SX + woiWidth))
					{
						digHoloPlotPixels[idx] = 1;
						totalPowerRoI[polIdx] += plotPixelsComplex[idx][0];
					}
				}
			}
		}

		for (int j = 0; j < plotHeight; j++)
		{
			for (int i = 0; i < plotWidth; i++)
			{
				const size_t idx = j * plotWidth + i;
				if (!digHoloPlotPixels[idx] && woiHeight && woiWidth)
				{
					plotPixelsComplex[idx][0] = plotPixelsComplex[idx][0] / 2 + maxV / 2;
					//plotPixelsComplex[idx][1] = 0;
				}
			}
		}


		if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		{
			fprintf(consoleOut, "Creating plot RGB image...\n");
			fflush(consoleOut);
		}
		//Convert the plotPixelsComplex array into and RGB array for plotting.
		complexColormapConvert(digHoloPlotPixels, digHoloPlotPixelsComplex, plotPixelCount, maxV);
		if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		{
			fprintf(consoleOut, "Creating plot RGB image Complete.\n");
			fflush(consoleOut);
		}

		//Draw white lines on the plot indicating the centre lines of the window, and the waist size of the mode basis
		for (int polIdx = 0; polIdx < polCount; polIdx++)
		{
			//size_t pixelOffset = 3 * (polIdx * plotWidth * plotHeight / polCount);
			size_t pixelOffset = 0;// 3 * (polIdx * plotWidth * plotHeight / polCount);
			unsigned char* plotPixels = &digHoloPlotPixels[pixelOffset];
			int SX = digHoloWoiSX[FFT_IDX][polIdx];
			int SY = digHoloWoiSY[FFT_IDX][polIdx];

			int w = (int)round(config.waist[polIdx] / config.pixelSize);

			for (int j = SY; j < (SY + woiHeight); j++)
			{
				for (int i = SX; i < (SX + woiWidth); i++)
				{
					const size_t idx = (j * plotWidth + i) * 3;

					float dj = (float)(j - (SY + woiHeight / 2));
					float di = (float)(i - (SX + woiWidth / 2));
					int dr = (int)round(sqrt(di * di + dj * dj));

					if (i == SX || i == (SX + woiWidth - 1) || j == SY || j == (SY + woiHeight - 1) || (j == (SY + woiHeight / 2) || i == (SX + woiWidth / 2)) || dr == w)
					{
						plotPixels[idx] = 255;
						plotPixels[idx + 1] = 255;
						plotPixels[idx + 2] = 255;
					}
				}
			}
		}

		//Plot the image bitmap we've constructed.
		//openGL_SetPixelsRGB(digHoloViewportOpen[0], digHoloPlotPixels, plotWidth, plotHeight);

		//Write propeties of interest, such as power, to the titlebar of the window
		float tPower = (float)(totalPowerRoI[0] + totalPowerRoI[1]);
		float pwrDBM = (float)(10 * log10(tPower));
		float pwrDBMH = (float)(10 * log10(totalPowerRoI[0]));
		float pwrDBMV = (float)(10 * log10(totalPowerRoI[1]));

		if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		{
			fprintf(consoleOut, "Constructing window title...\n");
			fflush(consoleOut);
		}

		std::snprintf(&windowName[0], DIGHOLO_VIEWPORT_NAMELENGTH, "Total Power (dB) : %2.2f	(H: %2.2f V: %2.2f)", pwrDBM, pwrDBMH, pwrDBMV);

		if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		{
			fprintf(consoleOut, "Constructing window title Complete.\n");
			fflush(consoleOut);
		}
		return 1;
	}

	//Routine for plotting Fourier space to the viewport
	void digHoloUpdateViewport_FFT(unsigned char* digHoloPlotPixels, complex64* digHoloPlotPixelsComplex)
	{
		//Do the FFT of the camera frame
		digHoloFFT();
		//Get the window in Fourier space (but don't do an IFFT, just get the IFFT window setup)
		digHoloIFFT_SetWindow();

		//The maximum absolute value in Fourier space. Used for setting the colour scale when plotting.
		float maxV = -FLT_MAX;
		//The total power in Fourier space, shown in the window title.
		float totalPower = 0;

		//The number of polarisations enabled last time the FFT was run
		const int polCount = digHoloPolCount_Valid[FFT_IDX];
		//The dimensions of the last FFT initialised.
		const int width = digHoloWoiPolWidth_Valid[FFT_IDX];
		const int height = digHoloWoiPolHeight_Valid[FFT_IDX];

		//The total number of pixels in the full Fourier space per polarisation
		const size_t pixelCount = width * height;
		//As we use a real-to-complex Fourier transform, the Fourier space only has (width/2+1)*height pixels in it. The Hermitian conjugate side isn't calculated, nor plotted
		const int widthR2C = width / 2 + 1;

		//For each polarisation
		for (int polIdx = 0; polIdx < polCount; polIdx++)
		{

			//Get the pointer to this polarisation in the calculated Fourier space (R2C, (width/2+1)*height)
			const complex64* pixelsFourierPlane = (complex64*)&digHoloPixelsFourierPlane[polIdx * (width / 2 + 1) * height][0];
			//Get the pointer to this polarisation in the output plot (full Fourier space, width*height)
			const size_t pixelOffset = polIdx * pixelCount;
			complex64* plotPixelsComplex = (complex64*)&digHoloPlotPixelsComplex[pixelOffset][0];
			//Clear the plot pixels for this polarisation
			memset(plotPixelsComplex, 0, sizeof(complex64) * width * height);

			//Iterate through the calculated Fourier space, perform and FFTshift, and keep track of the total power and the maximum absolute value.
			for (int j = 0; j < height; j++)
			{
				const int jShift = (j + height / 2) % height;
				for (int i = 0; i < widthR2C; i++)
				{
					size_t idx = jShift * widthR2C + i;
					float vR = pixelsFourierPlane[idx][0];
					float vI = pixelsFourierPlane[idx][1];
					float pwr = vR * vR + vI * vI;

					size_t fftIdx = j * width + i + (width - widthR2C);
					plotPixelsComplex[fftIdx][0] = vR;
					plotPixelsComplex[fftIdx][1] = vI;
					totalPower += (pwr);

					//Ignore i==0, so that DC terms, at least along the x-axis, don't influence the colour scale.
					if (pwr > maxV && i != 0)
					{
						maxV = pwr;
					}
				}
			}
		}
		//Sqrt to convert intensity to field magnitude, and divide by 10 so that the colour scale isn't overwhelmed by the zero-order. This plot with saturate the zero-order region, and make the off-axis terms more visible.
		maxV = (float)(0.1 * sqrtf(maxV));
		complexColormapConvert(digHoloPlotPixels, digHoloPlotPixelsComplex, (width * height * polCount), maxV);

		//Iterate through each polarisation, and draw white boxes indicating where the IFFT windows are.
		//The actual window also has an elliptical spatial filter applied of radius digHoloApertureSize, but that's only visible in the WindowOnly plot option.
		for (int polIdx = 0; polIdx < polCount; polIdx++)
		{
			int SX = digHoloWoiSX[IFFT_IDX][polIdx];
			int SY = digHoloWoiSY[IFFT_IDX][polIdx];
			int EX = SX + digHoloWoiPolWidth[IFFT_IDX];
			int EY = SY + digHoloWoiPolHeight[IFFT_IDX];

			size_t pixelOffset = 3 * ((size_t)polIdx) * width * height;
			unsigned char* plotPixelsComplex = (unsigned char*)&digHoloPlotPixels[pixelOffset];

			for (int j = 0; j < height; j++)
			{
				int jShift = (j + height / 2) % height;
				for (int i = 0; i < width; i++)
				{
					int iShift = (i + width / 2) % width;

					unsigned char borderTop = iShift == SX && jShift >= SY && jShift < EY;
					unsigned char borderBottom = iShift == EX && jShift >= SY && jShift < EY;

					unsigned char borderLeft = jShift == SY && iShift >= SX && iShift < EX;
					unsigned char borderRight = jShift == EY && iShift >= SX && iShift < EX;

					unsigned char borderPol = jShift == 0;

					if (borderTop || borderBottom || borderLeft || borderRight || borderPol)
					{
						size_t idx = 3 * (jShift * width + iShift);
						plotPixelsComplex[idx] = 255;
						plotPixelsComplex[idx + 1] = 255;
						plotPixelsComplex[idx + 2] = 255;
					}
				}
			}
		}
		//Plot the bitmap
		//openGL_SetPixelsRGB(digHoloViewportOpen[0], digHoloPlotPixels, width, height * polCount);

		//Update the window title with information such as power, dimensions of FFT, dimensions of IFFT
		float pwrDBM = 10 * log10(totalPower);
		// String^ windowNameStr = "Pwr = " + pwrDBM.ToString("F2") + " [FFT : " + digHoloWoiPolWidth[FFT_IDX] + "x" + digHoloWoiPolHeight[FFT_IDX] + "] (IFFT : " + digHoloWoiPolWidth[IFFT_IDX] + "x" + digHoloWoiPolHeight[IFFT_IDX] + ")";

		const int maxLength = DIGHOLO_VIEWPORT_NAMELENGTH;
		char windowName[maxLength];
		wchar_t windowTitle[maxLength];
		memset(&windowTitle[0], 0, sizeof(wchar_t) * maxLength);
		memset(&windowName[0], 0, sizeof(char) * maxLength);

		std::snprintf(&windowName[0], DIGHOLO_VIEWPORT_NAMELENGTH, "Pwr (dB) : %2.2f	[FFT : %i x %i] (IFFT : %i x %i)", pwrDBM, digHoloWoiPolWidth[FFT_IDX], digHoloWoiPolHeight[FFT_IDX], digHoloWoiPolWidth[IFFT_IDX], digHoloWoiPolHeight[IFFT_IDX]);

		for (int i = 0; i < maxLength; i++)
		{
			windowTitle[i] = windowName[i];
		}

		// openGL_SetWindowText(digHoloViewportOpen[0], &windowTitle[0]);
	}

	//This plots the Fourier space, with the IFFT window applied.
	//Any areas that are exactly zero are indicated in white.
	int digHoloUpdateViewport_FFT_WindowOnly(unsigned char* digHoloPlotPixels, int forceProcessing, complex64* digHoloPlotPixelsComplex, int& plotWidth, int& plotHeight, char* windowName, int plotABS)
	{
		if (forceProcessing)
		{
			//Take the FFT of the camera image
			digHoloFFT();
			//Take the IFFT as well so that digHoloCopyWindow is run. We don't need the full IFFT, but it's done anyways because of the way digHoloIFFT is written as a single function.
			digHoloIFFT();
		}

		if (!digHoloPixelsFourierPlaneMasked || !digHoloWoiPolWidth_Valid || !digHoloWoiPolWidth_Valid || !digHoloWoiPolHeight_Valid)
		{
			return 0;
		}

		//The dimensions of the IFFT window
		const int width = digHoloWoiPolWidth_Valid[IFFT_IDX];
		const int height = digHoloWoiPolHeight_Valid[IFFT_IDX];
		const size_t pixelCount = (size_t)width * (size_t)height;

		int lowResMode = config.resolutionIdx;

		//Keeps track of the total power, for showing in the window title.
		float totalPower = 0;
		//Keeps track of the maximum absolute value, for setting the colour scale
		float maxV = -FLT_MAX;
		int batchCount = config.batchCount;

		for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
		{
			//For each polarisation...
			for (int polIdx = 0; polIdx < digHoloPolCount; polIdx++)
			{
				//Get pointers to the Fourier space, and the complex plotPixels field for this polarisation.
				//This FourierPlaneMasked field is not in R2C format, it's in full C2C format (width*height, not (width/2+1)*height)
				const complex64* pixelsFourierPlane = (complex64*)&digHoloPixelsFourierPlaneMasked[polIdx * pixelCount + batchIdx * digHoloPolCount * pixelCount];
				complex64* plotPixelsComplex = (complex64*)&digHoloPlotPixelsComplex[polIdx * pixelCount][0];

				if (batchCount == 1)
				{
					//For every pixel in the Fourier plane, copy it to the plotPixelsComplex array, and keep track of the total power, and the maximum absolute value.
					for (size_t j = 0; j < height; j++)
					{
						for (size_t i = 0; i < width; i++)
						{
							const int jShift = (j + height / 2) % height;
							const int iShift = (i + width / 2) % width;
							size_t IDX = j * width + i;
							size_t idx = lowResMode ? IDX : (jShift * width + iShift);
							float vR = pixelsFourierPlane[idx][0];
							float vI = pixelsFourierPlane[idx][1];
							float pwr = vR * vR + vI * vI;
							if (plotABS)
							{
								plotPixelsComplex[IDX][0] = sqrtf(pwr);
								plotPixelsComplex[IDX][1] = 0;
							}
							else
							{
								plotPixelsComplex[IDX][0] = vR;
								plotPixelsComplex[IDX][1] = vI;
							}
							totalPower += pwr;
							if (pwr > maxV && i > 0)
							{
								maxV = pwr;
							}
						}
					}
				}
				else
				{
					for (size_t j = 0; j < height; j++)
					{
						for (size_t i = 0; i < width; i++)
						{
							const int jShift = (j + height / 2) % height;
							const int iShift = (i + width / 2) % width;
							size_t IDX = j * width + i;
							size_t idx = lowResMode ? IDX : (jShift * width + iShift);

							float vR = pixelsFourierPlane[idx][0];
							float vI = pixelsFourierPlane[idx][1];
							float pwr = vR * vR + vI * vI;
							plotPixelsComplex[IDX][0] += pwr;
							totalPower += pwr;
						}
					}
					if (batchIdx == (batchCount - 1))
					{
						for (size_t i = 0; i < pixelCount; i++)
						{
							if (plotPixelsComplex[i][0] > maxV && i > 0)
							{
								maxV = plotPixelsComplex[i][0];
							}
							plotPixelsComplex[i][0] = sqrtf(plotPixelsComplex[i][0]);
						}
					}
				}
			}
		}
		//Convert the plotPixelsComplex array to an RGB bitmap
		maxV = sqrtf(maxV);
		complexColormapConvert(digHoloPlotPixels, digHoloPlotPixelsComplex, (int)(pixelCount * digHoloPolCount), maxV);

		//Iterate through and set any pixels in Fourier space that have exactly zero power, to white pixels.
		for (int polIdx = 0; polIdx < digHoloPolCount; polIdx++)
		{
			const complex64* pixelsFourierPlane = (complex64*)&digHoloPlotPixelsComplex[polIdx * pixelCount][0];// (complex64*)&digHoloPixelsFourierPlaneMasked[polIdx * pixelCount];

			for (size_t j = 0; j < height; j++)
			{
				for (size_t i = 0; i < width; i++)
				{
					size_t IDX = j * width + i;
					float vR = pixelsFourierPlane[IDX][0];
					float vI = pixelsFourierPlane[IDX][1];
					float pwr = vR * vR + vI * vI;
					if (pwr == 0)
					{
						const size_t idx = polIdx * pixelCount + IDX;
						if (j == 0 || i == 0 || i == (width - 1) || j == (height - 1) || i == (width / 2) || j == (width / 2))
						{
							digHoloPlotPixels[idx * 3] = 0;
							digHoloPlotPixels[idx * 3 + 1] = 0;
							digHoloPlotPixels[idx * 3 + 2] = 0;
						}
						else
						{
							digHoloPlotPixels[idx * 3] = 255;
							digHoloPlotPixels[idx * 3 + 1] = 255;
							digHoloPlotPixels[idx * 3 + 2] = 255;
						}
					}


				}
			}
		}

		plotWidth = width;
		plotHeight = height * digHoloPolCount;
		//Update the window title with information like the total power, the dimensions etc.
		float pwrDBM = 10 * log10(totalPower);
		std::snprintf(&windowName[0], DIGHOLO_VIEWPORT_NAMELENGTH, "Pwr (dB) : %2.2f	[FFT : %i x %i] (IFFT : %i x %i)", pwrDBM, digHoloWoiPolWidth[FFT_IDX], digHoloWoiPolHeight[FFT_IDX], digHoloWoiPolWidth[IFFT_IDX], digHoloWoiPolHeight[IFFT_IDX]);
		return 1;
	}

	//Plots the reconstructed plane (result of the IFFT) to the Viewport
	int digHoloUpdateViewport_IFFT(unsigned char* digHoloPlotPixels, int forceProcessing, complex64* digHoloPlotPixelsComplex, int& plotWidth, int& plotHeight, char* windowName, int plotABS)
	{
		//Process the camera frame using off-axis digital holography
		if (forceProcessing)
		{
			digHoloFFT();
			digHoloIFFT();
			digHoloApplyTilt();
		}

		if (!digHoloPolCount_Valid || !digHoloWoiPolWidth_Valid || !digHoloWoiPolHeight_Valid || !DIGHOLO_ANALYSIS || !digHoloPixelsCameraPlaneReconstructed16R || !digHoloPixelsCameraPlaneReconstructed16I)
		{
			return 0;
		}
		//Maximum absolute value for plotting purposes
		float maxV = -FLT_MAX;
		//The total power in each polarisation for displaying in window title
		float totalPower[2] = { 0,0 };

		//The polarisation count the last time the IFFT was initialised
		const int polCount = digHoloPolCount_Valid[IFFT_IDX];

		//The resolution mode we're operating in (full or low res)
		const int fftIdx = digHoloResolutionIdx;
		//If we're in low-res mode, then the dimensions are the size of the IFFT window. In full-res mode, the dimensions are the size of the FFT
		const int width = digHoloWoiPolWidth_Valid[fftIdx];
		int height = digHoloWoiPolHeight_Valid[fftIdx];
		const size_t pixelCount = width * height;

		//The batch item to plot.
		const size_t batchIdx = 0;
		const size_t batchOffset = batchIdx * pixelCount * polCount;
		const size_t batchCount = config.batchCount;
		//For each polarisation. Keep convert the 16-bit int fields to 32-bit floats, keep track of the total power and maximum absolute value.
		for (int polIdx = 0; polIdx < polCount; polIdx++)
		{
			const size_t polOffset = polIdx * pixelCount;
			//The scale required to convert the fields represented as int16, to the appropriate 32-bit floats
			float scale = (float)(1.0 / DIGHOLO_ANALYSIS[IFFT_IDX][DIGHOLO_ANALYSIS_MAXABS][polIdx][batchIdx]);
			for (size_t i = 0; i < pixelCount; i++)
			{
				float vR = 0;
				float vI = 0;

				if (batchCount > 1)
				{
					vR = sqrtf(DIGHOLO_ANALYSISBatchSum[IFFT_IDX][polIdx][batchOffset + i]);
					vI = 0;
				}
				else
				{
					vR = scale * digHoloPixelsCameraPlaneReconstructed16R[batchOffset + polOffset + i];
					vI = scale * digHoloPixelsCameraPlaneReconstructed16I[batchOffset + polOffset + i];
				}

				float pwr = vR * vR + vI * vI;

				totalPower[polIdx] += pwr;

				if (plotABS)
				{
					digHoloPlotPixelsComplex[polOffset + i][0] = sqrtf(pwr);
					digHoloPlotPixelsComplex[polOffset + i][1] = 0;
				}
				else
				{
					digHoloPlotPixelsComplex[polOffset + i][0] = vR;
					digHoloPlotPixelsComplex[polOffset + i][1] = vI;
				}


				if (pwr > maxV)
				{
					maxV = pwr;
				}
			}
		}
		//Convert field to RGB complex colourmap
		maxV = sqrtf(maxV);
		complexColormapConvert(digHoloPlotPixels, digHoloPlotPixelsComplex, pixelCount * polCount, maxV);

		height = height * polCount;
		//Drawing white lines on figure
		//Middle line dividing H/V
		for (int i = 0; i < width; i++)
		{
			int j = height / 2;
			int idx = j * width + i;

			digHoloPlotPixels[idx * 3] = 255;
			digHoloPlotPixels[idx * 3 + 1] = 255;
			digHoloPlotPixels[idx * 3 + 2] = 255;

			j = 3 * height / 4;
			idx = j * width + i;

			digHoloPlotPixels[idx * 3] = 255;
			digHoloPlotPixels[idx * 3 + 1] = 255;
			digHoloPlotPixels[idx * 3 + 2] = 255;

			j = height / 4;
			idx = j * width + i;

			digHoloPlotPixels[idx * 3] = 255;
			digHoloPlotPixels[idx * 3 + 1] = 255;
			digHoloPlotPixels[idx * 3 + 2] = 255;
		}

		//Through the middle top to bottom
		for (int i = 0; i < height; i++)
		{
			int j = width / 2;
			int idx = i * width + j;
			digHoloPlotPixels[idx * 3] = 255;
			digHoloPlotPixels[idx * 3 + 1] = 255;
			digHoloPlotPixels[idx * 3 + 2] = 255;
		}

		//openGL_SetPixelsRGB(digHoloViewportOpen[0], digHoloPlotPixels, width, height);
		plotWidth = width;
		plotHeight = height;
		//Update the window title with information about the power in H and V
		float pwrDBM = 10 * log10(totalPower[0] + totalPower[1]);
		float pwrDBMH = 10 * log10(totalPower[0]);
		float pwrDBMV = 10 * log10(totalPower[1]);

		std::snprintf(&windowName[0], DIGHOLO_VIEWPORT_NAMELENGTH, "Total Power (dB) : %2.2f	(H: %2.2f V: %2.2f)", pwrDBM, pwrDBMH, pwrDBMV);

		return 1;
	}

	//Plots a reconstruction of the field, by summing together the modes and their corresponding coefficients.
	int digHoloUpdateViewport_Modes(unsigned char* digHoloPlotPixels, int forceProcessing, complex64* digHoloPlotPixelsComplex, int& plotWidth, int& plotHeight, char* windowName)
	{

		if (digHoloIsValidPixelBuffer())
		{
			if (forceProcessing)
			{
				//Perform digital holography...
				digHoloFFT();
				digHoloIFFT();
				digHoloApplyTilt();
				//...and get the modal coefficients.
				digHoloOverlapModes();
			}

			if (!digHoloOverlapCoefsHG || !digHoloWoiPolWidth || !digHoloWoiPolHeight)
			{
				return 0;
			}

			//The number of polarisations during the last FFT process
			const int polCount = digHoloPolCount;

			//Get the dimensions of the IFFT and hence te modes, based on whether we're in full-res, or low-res mode.
			int basisIdx = digHoloResolutionIdx;
			const int width = digHoloWoiPolWidth[basisIdx];
			const int height = digHoloWoiPolHeight[basisIdx];

			//Which batch item to plot the coefficients for
			int batchCount = config.batchCount;
			
			complex64* plotPixelsComplex = 0;
			size_t pixelCount = width * height * polCount;
			if (batchCount > 1)
			{
				allocate1D(pixelCount, plotPixelsComplex);
			}
			else
			{
				plotPixelsComplex = digHoloPlotPixelsComplex;
			}
			float maxV = 0;
			for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
			{
				
				//maxV is the maximum absolute^2 value in the reconstruction. Only used for plotting purposes
				maxV = generateModeSeparable(digHoloHGscaleX, digHoloHGscaleY, plotPixelsComplex, digHoloHGX, digHoloHGY, width, height, &digHoloOverlapCoefsHG[batchIdx][0], HG_M, HG_N, digHoloModeCount, polCount,false,NULL,NULL,NULL);
				if (batchCount > 1)
				{
					for (size_t pixelIdx = 0; pixelIdx < pixelCount; pixelIdx++)
					{
						float vR = plotPixelsComplex[pixelIdx][0];
						float vI = plotPixelsComplex[pixelIdx][1];
						float pwr = vR * vR + vI * vI;
						digHoloPlotPixelsComplex[pixelIdx][0] += pwr;
						digHoloPlotPixelsComplex[pixelIdx][1] = 0;
					}
					if (batchIdx == (batchCount - 1))
					{
						for (size_t pixelIdx = 0; pixelIdx < pixelCount; pixelIdx++)
						{
							float pwr = digHoloPlotPixelsComplex[pixelIdx][0];
							if (pwr > maxV)
							{
								maxV = pwr;
							}
							digHoloPlotPixelsComplex[pixelIdx][0] = sqrtf(pwr);
							digHoloPlotPixelsComplex[pixelIdx][1] = 0;
						}
					}
				}
			}
			maxV = sqrtf(maxV);
			//Convert field to and RGB bitmap and plot.
			complexColormapConvert(digHoloPlotPixels, digHoloPlotPixelsComplex, width * height * polCount, maxV);
			
			plotWidth = width;
			plotHeight = polCount * height;
			return 1;
		}
		else
		{
			return 0;
		}
	}

	int digHoloUpdateViewport_FFTx(unsigned char* digHoloPlotPixels, int forceProcessing, complex64* digHoloPlotPixelsComplex, int& plotWidth, int& plotHeight, char* windowName, int dbMODE)
	{
		//Do the FFT of the camera frame
		if (digHoloIsValidPixelBuffer())
		{
			if (forceProcessing)
			{
				digHoloFFT();
				//Get the window in Fourier space (but don't do an IFFT, just get the IFFT window setup)
				digHoloIFFT_SetWindow();
			}
			//Check if there's valid data to display
			if (digHoloPolCount_Valid && digHoloWoiPolWidth_Valid && digHoloWoiPolHeight_Valid && DIGHOLO_ANALYSISBatchSum && digHoloPlotPixelsComplex && digHoloPlotPixels)
			{

			}
			else //if there isn't, abort.
			{
				std::snprintf(&windowName[0], DIGHOLO_VIEWPORT_NAMELENGTH, "No Fourier data available.");
				return 0;
			}
			//The maximum absolute value in Fourier space. Used for setting the colour scale when plotting.
			float maxV = -FLT_MAX;
			//The total power in Fourier space, shown in the window title.
			float totalPower = 0;

			//The number of polarisations enabled last time the FFT was run
			const int polCount = digHoloPolCount_Valid[FFT_IDX];
			//The dimensions of the last FFT initialised.
			const int width = digHoloWoiPolWidth_Valid[FFT_IDX];
			const int height = digHoloWoiPolHeight_Valid[FFT_IDX];

			if (polCount <= 0 || width <= 0 || height <= 0)
			{
				std::snprintf(&windowName[0], DIGHOLO_VIEWPORT_NAMELENGTH, "No Fourier data available.");
				return 0;
			}

			//The total number of pixels in the full Fourier space per polarisation
		//	const size_t pixelCount = width * height;
			//As we use a real-to-complex Fourier transform, the Fourier space only has (width/2+1)*height pixels in it. The Hermitian conjugate side isn't calculated, nor plotted
			const int widthR2C = width / 2 + 1;
			//For each polarisation
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{

				//Get the pointer to this polarisation in the calculated Fourier space (R2C, (width/2+1)*height)
				const float* pixelsFourierPlane = (float*)DIGHOLO_ANALYSISBatchSum[FFT_IDX][polIdx];
				//Get the pointer to this polarisation in the output plot (full Fourier space, width*height)
				const size_t pixelOffset = ((size_t)polIdx) * widthR2C * height;
				complex64* plotPixelsComplex = (complex64*)&digHoloPlotPixelsComplex[pixelOffset][0];
				//Clear the plot pixels for this polarisation
				//memset(plotPixelsComplex, 0, sizeof(complex64) * width * height);

				//Iterate through the calculated Fourier space, perform and FFTshift, and keep track of the total power and the maximum absolute value.
				for (int j = 0; j < height; j++)
				{
					const int jShift = (j + height / 2) % height;
					//const int jFFTshift = 
					for (int i = 0; i < widthR2C; i++)
					{
						size_t idx = jShift * widthR2C + i;
						float vR = pixelsFourierPlane[idx];
						//	float vI = 0;// pixelsFourierPlane[idx][1];
						float pwr = vR;// +vI * vI;

						size_t fftIdx = (j)*widthR2C + i;// j* width + i + (width - widthR2C);

						plotPixelsComplex[fftIdx][0] = sqrtf(vR);
						//plotPixelsComplex[fftIdx][1] = vI;
						totalPower += (pwr);

						//Ignore i==0, so that DC terms, at least along the x-axis, don't influence the colour scale.
						if (pwr > maxV && i != 0)
						{
							maxV = pwr;
						}
					}
				}
			}

			if (dbMODE)
			{
				size_t pixelCount = widthR2C * height * polCount;
				float maxV2 = sqrt(maxV);
				float minDB = -60;
				for (int i = 0; i < pixelCount; i++)
				{
					float dBV = minDB;
					if (digHoloPlotPixelsComplex[i][0])
					{
						dBV = 20 * log10(digHoloPlotPixelsComplex[i][0] / maxV2);
						if (dBV < minDB)
						{
							dBV = minDB + 0.0001f;
						}
						dBV = dBV - minDB;
						digHoloPlotPixelsComplex[i][0] = dBV;
					}


				}
				maxV = -minDB;
			}
			else
			{
				maxV = (float)sqrtf(maxV);
			}

			//Sqrt to convert intensity to field magnitude, and divide by 10 so that the colour scale isn't overwhelmed by the zero-order. This plot with saturate the zero-order region, and make the off-axis terms more visible.

			complexColormapConvert(digHoloPlotPixels, digHoloPlotPixelsComplex, ((width / 2 + 1) * height * polCount), maxV);

			//Iterate through each polarisation, and draw white boxes indicating where the IFFT windows are.
			//The actual window also has an elliptical spatial filter applied of radius digHoloApertureSize, but that's only visible in the WindowOnly plot option.
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				size_t polOffset = ((size_t)polIdx) * widthR2C * height;
				for (int j = 0; j < height; j++)
				{
					const int jShift = (j + height / 2) % height;
					for (int i = 0; i < widthR2C; i++)
					{
						size_t idx = polOffset + jShift * widthR2C + i;
						if (digHoloPlotPixelsComplex[idx][0] == 0 && digHoloPlotPixelsComplex[idx][1] == 0)
						{
							digHoloPlotPixels[3 * idx] = 255;
							digHoloPlotPixels[3 * idx + 1] = 255;
							digHoloPlotPixels[3 * idx + 2] = 255;
						}
					}
				}
			}
			//Plot the bitmap
			//openGL_SetPixelsRGB(digHoloViewportOpen[0], digHoloPlotPixels, width/2+1, height * polCount);
			plotWidth = width / 2 + 1;
			plotHeight = height * polCount;

			//Update the window title with information such as power, dimensions of FFT, dimensions of IFFT
			float pwrDBM = 10 * log10(totalPower);
			std::snprintf(&windowName[0], DIGHOLO_VIEWPORT_NAMELENGTH, "Pwr (dB) : %2.2f	[FFT : %i x %i] (IFFT : %i x %i) {BatchCount : %i}", pwrDBM, digHoloWoiPolWidth[FFT_IDX], digHoloWoiPolHeight[FFT_IDX], digHoloWoiPolWidth[IFFT_IDX], digHoloWoiPolHeight[IFFT_IDX], digHoloBatchCOUNT);
			return 1;
		}
		else
		{
			return 0;
		}
	}

	void digHoloUpdateViewport_DCT(unsigned char* digHoloPlotPixels, complex64* digHoloPlotPixelsComplex, int& plotWidth, int& plotHeight, char* windowName)
	{
		//Do the FFT of the camera frame
		//if (digHoloPixelsCameraPlane)
		{
			//digHoloFFT(digHoloPixelsCameraPlane);
			//Get the window in Fourier space (but don't do an IFFT, just get the IFFT window setup)
			//digHoloIFFT_SetWindow();

			//The maximum absolute value in Fourier space. Used for setting the colour scale when plotting.
			float maxV = -FLT_MAX;
			//The total power in Fourier space, shown in the window title.
			float totalPower = 0;

			//The number of polarisations enabled last time the FFT was run
			const int polCount = digHoloPolCount_Valid[FFT_IDX];
			//The dimensions of the last FFT initialised.
			const int width = digHoloWoiPolWidth_Valid[FFT_IDX];
			const int height = digHoloWoiPolHeight_Valid[FFT_IDX];

			//The total number of pixels in the full Fourier space per polarisation
			//const size_t pixelCount = width * height;
			//As we use a real-to-complex Fourier transform, the Fourier space only has (width/2+1)*height pixels in it. The Hermitian conjugate side isn't calculated, nor plotted
			const int widthR2C = width / 2 + 1;

			//For each polarisation
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{

				//Get the pointer to this polarisation in the calculated Fourier space (R2C, (width/2+1)*height)
				const float* pixelsFourierPlane = (float*)DIGHOLO_ANALYSISBatchSum[FFT_IDX + 4][polIdx];
				// const complex* pixelsFourierPlane = (complex*)&digHoloPixelsFourierPlane[0][0];
				 //Get the pointer to this polarisation in the output plot (full Fourier space, width*height)
				const size_t pixelOffset = ((size_t)polIdx) * widthR2C * height;
				complex64* plotPixelsComplex = (complex64*)&digHoloPlotPixelsComplex[pixelOffset][0];
				//Clear the plot pixels for this polarisation
				memset(plotPixelsComplex, 0, sizeof(complex64) * width * height);

				//Iterate through the calculated Fourier space, perform and FFTshift, and keep track of the total power and the maximum absolute value.
				for (int j = 0; j < height; j++)
				{
					const int jShift = (j + height / 2) % height;
					for (int i = 0; i < widthR2C; i++)
					{
						size_t idx = jShift * widthR2C + i;
						float vR = pixelsFourierPlane[idx];
						float vI = 0;// pixelsFourierPlane[idx][1];
						float pwr = vR * vR;// +vI * vI;

						size_t fftIdx = idx;// j* width + i + (width - widthR2C);
						plotPixelsComplex[fftIdx][0] = vR;
						plotPixelsComplex[fftIdx][1] = vI;
						totalPower += (pwr);

						//Ignore i==0, so that DC terms, at least along the x-axis, don't influence the colour scale.
						if (pwr > maxV && i != 0)
						{
							maxV = pwr;
						}
					}
				}
			}
			//Sqrt to convert intensity to field magnitude, and divide by 10 so that the colour scale isn't overwhelmed by the zero-order. This plot with saturate the zero-order region, and make the off-axis terms more visible.
			maxV = (float)(0.1 * sqrtf(maxV));
			complexColormapConvert(digHoloPlotPixels, digHoloPlotPixelsComplex, ((width / 2 + 1) * height * polCount), maxV);

			//Iterate through each polarisation, and draw white boxes indicating where the IFFT windows are.
			//The actual window also has an elliptical spatial filter applied of radius digHoloApertureSize, but that's only visible in the WindowOnly plot option.
			/*for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				int polOffset = polIdx * widthR2C * height;
				for (int j = 0; j < height; j++)
				{
					const int jShift = (j + height / 2) % height;
					for (int i = 0; i < widthR2C; i++)
					{
						size_t idx = polOffset + jShift * widthR2C + i;
						if (digHoloPlotPixelsComplex[idx][0] == 0 && digHoloPlotPixelsComplex[idx][1] == 0)
						{
							digHoloPlotPixels[3 * idx] = 255;
							digHoloPlotPixels[3 * idx + 1] = 255;
							digHoloPlotPixels[3 * idx + 2] = 255;
						}
					}
				}
			}*/
			//Plot the bitmap
			//openGL_SetPixelsRGB(digHoloViewportOpen[0], digHoloPlotPixels, width/2+1, height * polCount);
			plotWidth = width / 2 + 1;
			plotHeight = height * polCount;
			//Update the window title with information such as power, dimensions of FFT, dimensions of IFFT
			float pwrDBM = 10 * log10(totalPower);
			// const int maxLength = 1024;
			// char windowName[maxLength];
			 //wchar_t windowTitle[maxLength];
			std::snprintf(&windowName[0], DIGHOLO_VIEWPORT_NAMELENGTH, "Pwr (dB) : %2.2f	[FFT : %i x %i] (IFFT : %i x %i)", pwrDBM, digHoloWoiPolWidth[FFT_IDX], digHoloWoiPolHeight[FFT_IDX], digHoloWoiPolWidth[IFFT_IDX], digHoloWoiPolHeight[IFFT_IDX]);

		}
		//	else
		//	{

		//	}

			/* for (int i = 0; i < maxLength; i++)
			 {
				 windowTitle[i] = windowName[i];
			 }*/
			 // windowTitle[windowName->Length] = '\0';
			//  openGL_SetWindowText(digHoloViewportOpen[0], &windowTitle[0]);
	}



	//This will setup a New wavelength sweep array. It won't be used until digHoloFFT runs next.
	//This allows you to change the wavelength sweep parameters asyncronously whilst the digHolo is running.
public: int SetWavelengthFrequencyLinear(float lambdaStart, float lambdaStop, int lambdaCount)
{
	if (lambdaCount > 0)
	{
		//An array which lets you manually set which wavelength calibration to use for each image in the batch
		//e.g. for swept-laser measurements
		if (lambdaCount != digHoloWavelengthCountNew)
		{
			allocate1D(lambdaCount, digHoloWavelengthNew);
			digHoloWavelengthCountNew = lambdaCount;
		}

		//Lambda was specified in nm
		if (lambdaStart > 1)
		{
			lambdaStart = lambdaStart * DIGHOLO_UNIT_LAMBDA;
		}
		if (lambdaStop > 1)
		{
			lambdaStop = lambdaStop * DIGHOLO_UNIT_LAMBDA;
		}


		float fStart = c0 / (lambdaStart);
		float fStop = c0 / (lambdaStop);
		float df = (fStop - fStart) / (lambdaCount - 1);

		//We got a wise guy over here! You want a frequency sweep of 1 or less. That's not really a sweep is it now champ?
		if (lambdaCount <= 1)
		{
			//If the stop wavelength is invalid, use the start wavelength
			if (lambdaStop <= 0 && lambdaStart > 0)
			{
				fStart = c0 / ((lambdaStart) / 2);
			}
			else
			{
				//If the start wavelength is invalid, use the stop wavelength
				if (lambdaStart <= 0 && lambdaStop > 0)
				{
					fStart = c0 / ((lambdaStart) / 2);
				}
				else
				{
					//Both wavelengths are invalid. Just use the centre wavelength, you're clearly being trolled at this point.
					if (lambdaStart <= 0 && lambdaStop <= 0)
					{
						digHoloWavelengthCentre = (float)(config.wavelengthCentre * DIGHOLO_UNIT_LAMBDA);
						fStart = c0 / (digHoloWavelengthCentre);
					}
					else
					{
						//Otherwise, if both the start and stop wavelengths are valid, start at the average wavelength between the specified start/stop wavelengths
						fStart = c0 / ((lambdaStop + lambdaStart) / 2);
					}
				}
			}
			//Frequency step is zero because there's only 1 step
			df = 0;
		}

		//Create wavelength array, linearly spaced by df in frequency.
		for (int lambdaIdx = 0; lambdaIdx < lambdaCount; lambdaIdx++)
		{
			float f0 = fStart + df * lambdaIdx;
			float lambda0 = c0 / f0;

			digHoloWavelengthNew[lambdaIdx] = lambda0;;
		}

		//The wavelength array has been updated, the old digHoloWavelength array is now out of date, and should be updated next time digital holography is processed.
		digHoloWavelengthValid = false;
		return DIGHOLO_ERROR_SUCCESS;
	}
	else
	{
		return DIGHOLO_ERROR_INVALIDDIMENSION;
	}
}

public: int SetWavelengthArbitrary(float* wavelengths, int lambdaCount)
{
	if (lambdaCount > 0)
	{
		if (wavelengths)
		{
			if (lambdaCount != digHoloWavelengthCountNew)
			{
				allocate1D(lambdaCount, digHoloWavelengthNew);
				digHoloWavelengthCountNew = lambdaCount;
			}
			//An alternative would be to just set digHoloWavelengthNew = wavelengths, but this extra buffering level of buffering could be handy/safer.
			//It means that after this is called, the external user can do whatever they want with 'wavelengths' including freeing it.
			memcpy(digHoloWavelengthNew, wavelengths, sizeof(float) * lambdaCount);

			digHoloWavelengthValid = false;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_NULLPOINTER;
		}
	}
	else
	{
		return DIGHOLO_ERROR_INVALIDDIMENSION;
	}
}

	   int digHoloFFTInitFirstTime()
	   {
		   fftwf_init_threads();
		   strcpy(&MACHINENAME[0], cpuInfo.brandHex);
		   //One of the first routines run by the digHoloObject.
		   //Setups up an array of work packages for the digHolo processing threads, and starts the Viewport
		   memset(&FFTW_WISDOM_FILENAME[0], 0, sizeof(char) * FFTW_WISDOM_FILENAME_MAXLENGTH);

		   const char* FFTW_WISDOM_FN = "FFTW_WISDOM";
		   strcat(&FFTW_WISDOM_FILENAME[0], FFTW_WISDOM_FN);
		   strcat(&FFTW_WISDOM_FILENAME[0], MACHINENAME);
		   strcat(&FFTW_WISDOM_FILENAME[0], ".txt");

		   //const char vname[] = fftwf_cc();
		   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
		   {
			   fprintf(consoleOut, "<SYSTEM ENVIRONMENT>\n");
			   fprintf(consoleOut, "FFTW version : %s\n", fftwf_version);
			   fprintf(consoleOut, "CPU : %s\n", cpuInfo.brand);
			   fprintf(consoleOut, "	Count : %i\n", CPU_COUNT);
			   fprintf(consoleOut, "	AVX : %i\n", cpuInfo.avx);
			   fprintf(consoleOut, "	AVX2 : %i\n", cpuInfo.avx2);
			   fprintf(consoleOut, "	FMA3 : %i\n", cpuInfo.fma3);
			   fprintf(consoleOut, "	AVX512f : %i\n", cpuInfo.avx512f);
#ifdef SVML_ENABLE
			   fprintf(consoleOut, "SVML : %i\n", 1);
#endif
		   }
		   digHoloFFTWisdomLoad();
		   fftwInitialised = true;
		   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
		   {
			   fprintf(consoleOut, "</SYSTEM ENVIRONMENT>\n\n");
		   }
		   return 1;
	   }

	   int digHoloIsValidPixelBuffer()
	   {
		   return digHoloPixelsCameraPlane || (digHoloPixelsCameraPlaneUint16 && digHoloPixelBufferType);
	   }

	   int digHoloFFTCalibrationUpdate()
	   {
		   int updated = false;
		   if (digHoloRefCalibrationEnabled)
		   {
			   int windowMoved = false;
			   const int polCount = digHoloPolCount;
			   const int wavelengthCount = digHoloRefCalibrationWavelengthCount;
			   for (int polIdx = 0; polIdx < polCount; polIdx++)
			   {
				   windowMoved = windowMoved || (digHoloWoiSX[FFT_IDX][polIdx] != digHoloWoiSX[FFTCAL_IDX][polIdx]);
				   windowMoved = windowMoved || (digHoloWoiSY[FFT_IDX][polIdx] != digHoloWoiSY[FFTCAL_IDX][polIdx]);
			   }

			   const int windowChangedSize = (digHoloWoiPolWidth[FFT_IDX] != digHoloWoiPolWidth[FFTCAL_IDX]) || (digHoloWoiPolHeight[FFT_IDX] != digHoloWoiPolHeight[FFTCAL_IDX]);
			   const int wavelengthCountChanged = digHoloBatchCount_Valid[FFTCAL_IDX] != wavelengthCount;
			   const int polCountChanged = digHoloPolCount_Valid[FFTCAL_IDX] != polCount;
			   
			   //If the window size has changed, we'll have to redo the FFTW plans
			   if (windowChangedSize || wavelengthCountChanged || polCountChanged || !digHoloRefCalibrationFourierPlaneValid)
			   {
				   size_t length = ((size_t)(digHoloWoiPolWidth[FFT_IDX] * digHoloWoiPolHeight[FFT_IDX])) * wavelengthCount * polCount;
				   allocate1D(length, digHoloRefCalibrationFourierPlane);
				   digHoloRefCalibrationFourierPlaneValid = false;
			   }

			   if (!digHoloRefCalibrationFourierPlaneValid || windowMoved)
			   {
				   //Input source for the FFTs
				   complex64* bufferIn = digHoloRefCalibration;
				   //Output location for the FFTs
				   complex64* bufferOut = digHoloRefCalibrationFourierPlane;
				   const int batchCount = wavelengthCount;
				   fftwf_plan** fftPlans = &digHoloFFTPlans[THREADCOUNT_MAX];
				   const int r2c = 0;
				   //Perform digHoloBatchCount*digHoloPolCount FFTs, starting from this pixelsF pointer
				   digHoloFFTBatch(bufferIn, bufferOut, polCount, batchCount, fftPlans, r2c);

				   for (int polIdx = 0; polIdx < polCount; polIdx++)
				   {
					   digHoloWoiSX[FFTCAL_IDX][polIdx] = digHoloWoiSX[FFT_IDX][polIdx];
					   digHoloWoiSY[FFTCAL_IDX][polIdx] = digHoloWoiSY[FFT_IDX][polIdx];
				   }

				   digHoloWoiPolWidth[FFTCAL_IDX] = digHoloWoiPolWidth[FFT_IDX];
				   digHoloWoiPolHeight[FFTCAL_IDX] = digHoloWoiPolHeight[FFT_IDX];
				   digHoloBatchCount_Valid[FFTCAL_IDX] = wavelengthCount;

				   digHoloRefCalibrationFourierPlaneValid = true;
				   digHoloRefCalibrationReconstructedPlaneValid = false;
				   updated = true;
			   
			   }
		   }
		   return updated;
	   }

	   void digHoloRefCalibrationInvalidate()
	   {
		   digHoloRefCalibrationReconstructedPlaneValid = false;
		   digHoloRefCalibrationFourierPlaneValid = false;
	   }

	   int digHoloIFFTCalibrationUpdate()
	   {
		   int updated = false;
		   
		   if (digHoloRefCalibrationEnabled)
		   {
			   const int resolutionIdx = digHoloResolutionIdx;
			   const int polCount = digHoloPolCount;
			   const int wavelengthCount = digHoloRefCalibrationWavelengthCount;
			   const int windowChangedSize = (digHoloWoiPolWidth[IFFT_IDX] != digHoloWoiPolWidth[IFFTCAL_IDX]) || (digHoloWoiPolHeight[IFFT_IDX] != digHoloWoiPolHeight[IFFTCAL_IDX]);
			   const int wavelengthCountChanged = digHoloBatchCount_Valid[IFFTCAL_IDX] != wavelengthCount;
			   const int polCountChanged = digHoloPolCount_Valid[IFFTCAL_IDX] != polCount;

			   float** zernCoefs = 0;

			   float wavelengthStart = 0;
			   float wavelengthStop = 0;

			   if (digHoloRefCalibrationWavelengthCount == 1)
			   {
				   wavelengthStart = digHoloWavelengthCentre;
				   wavelengthStop = digHoloWavelengthCentre;
			   }
			   else
			   {
				   wavelengthStart = digHoloWavelength[0];
				   wavelengthStop = digHoloWavelength[digHoloWavelengthCount - 1];
			   }

			   const int r2c = 0;

			   const int windowMoved = digHoloIFFT_GetWindowBounds(zernCoefs, digHoloWoiPolWidth[IFFTCAL_IDX], digHoloWoiPolHeight[IFFTCAL_IDX], digHoloWoiSX[IFFTCAL_IDX], digHoloWoiSY[IFFTCAL_IDX], wavelengthStart, wavelengthStop,r2c);

			   //If the window size
			   if (windowChangedSize || wavelengthCountChanged || polCountChanged || !digHoloRefCalibrationReconstructedPlaneValid)
			   {
				   size_t length = ((size_t)(digHoloWoiPolWidth[resolutionIdx] * digHoloWoiPolHeight[resolutionIdx])) * wavelengthCount * polCount;
				   allocate1D(length, digHoloRefCalibrationReconstructedPlane);
				   allocate1D(length, digHoloRefCalibrationFourierPlaneMasked);
				   memset(digHoloRefCalibrationFourierPlaneMasked, 0, length * sizeof(complex64));
				   digHoloRefCalibrationReconstructedPlaneValid = false;
			   }

			   if (!digHoloRefCalibrationReconstructedPlaneValid || windowMoved)
			   {
				   digHoloWoiPolWidth[IFFTCAL_IDX] = digHoloWoiPolWidth[IFFT_IDX];
				   digHoloWoiPolHeight[IFFTCAL_IDX] = digHoloWoiPolHeight[IFFT_IDX];
				   digHoloBatchCount_Valid[IFFTCAL_IDX] = wavelengthCount;
				   digHoloPolCount_Valid[IFFTCAL_IDX] = polCount;
				   digHoloIFFTBatchCal();

				   digHoloRefCalibrationReconstructedPlaneValid = true;
				   updated = true;
			   }
			   
		   }
		   return updated;
		   
	   }
	   //Diagnostic timers for the digHoloFFT routine. Used for benchmarking how fast it runs
	   int64_t benchmarkFFTCounter = 0;
	   std::chrono::duration<double> benchmarkFFTTime = std::chrono::duration<double>(0);

	   int64_t benchmarkFFTAnalysisCounter = 0;
	   std::chrono::duration<double> benchmarkFFTAnalysisTime = std::chrono::duration<double>(0);

	   //The FFT, and start of the digital holography processing chain.
	   //FFTs a camera frame using a real-to-complex 2D FFT.
	   //Also runs digHoloFieldAnalysis to analyse properties of the Fourier space like centre of mass, peak positions, effective area etc.
	   int digHoloFFT()
	   {
		   //If it's a null pointer, don't process anything
		   if (digHoloIsValidPixelBuffer())
		   {
			   //Take the start time for benchmarking purposes
			   //const int64_t startTime = QPC(); 
			   std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

			   if (!fftwInitialised)
			   {
				   digHoloFFTInitFirstTime();
			   }

			   //Setup the FFT window position, and allocate memory/FFT plan if necessary.
			   //Properties are also read in from config inside this function
			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				   fprintf(consoleOut, "digHoloFFT_SetWindow()\n\r");
				   fflush(consoleOut);
			   }

			   digHoloFFT_SetWindow();

			   //Input source for the FFTs
			   float* bufferIn = digHoloPixelsCameraPlane;
			   //Output location for the FFTs
			   complex64* bufferOut = digHoloPixelsFourierPlane;
			   const int polCount = digHoloPolCount;
			   const int batchCount = digHoloBatchAvgCount * digHoloBatchCOUNT;
			   fftwf_plan** fftPlans = &digHoloFFTPlans[0];
			   const int r2c = 1;
			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				   fprintf(consoleOut, "digHoloFFTBatch()\n\r");
				   fflush(consoleOut);
			   }
			   //Perform digHoloBatchCount*digHoloPolCount FFTs, starting from this pixelsF pointer
			   digHoloFFTBatch(bufferIn,bufferOut,polCount, batchCount,fftPlans,r2c);
			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				   fprintf(consoleOut, "digHoloFFTAnalysis()\n\r");
				   fflush(consoleOut);
			   }
			   digHoloFFTAnalysis();

			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				   fprintf(consoleOut, "digHoloFFTCalibrationUpdate()\n\r");
				   fflush(consoleOut);
			   }
			   digHoloFFTCalibrationUpdate();

			   //Update benchmark timers
			   std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();
			   benchmarkFFTTime += (stopTime - startTime);
			   benchmarkFFTCounter++;
		   }
		   else
		   {
			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				   fprintf(consoleOut, "FFT : Frame pointer is NULL\n");
				   fflush(consoleOut);
			   }
			   return DIGHOLO_ERROR_NULLPOINTER;
		   }
		   return DIGHOLO_ERROR_SUCCESS;
	   }

	   void digHoloFFTAnalysis()
	   {
		   //Analyse the Fourier space to get properties of interest, like centre of mass, effective area, total power, peak position etc.
   //Mostly for alignment purposes. The DIGHOLO_ANALYSIS routine also sums together the intensities of all fields in a batch, and performs analysis on the summed, average value as well.
   //k0 for the centre wavelength
		   float k0 = 2 * pi / (digHoloWavelengthCentre);
		   //Select a region in k-space to exclude from the analysis (2 aperture radii worth centres on the zero-order)
		   float windowCX = 0;
		   float windowCY = 0;
		   float windowR = 2 * k0 * sinf((digHoloApertureSize));//Small-angle approximation (sin(x) = x)

		   //The x/y axes in Fourier-space
		   float* X = &digHoloKXaxis[0];
		   float* Y = &digHoloKYaxis[0];
		   //Analyse both polarisations
		   int polStart = 0;
		   int polStop = digHoloPolCount;
		   //This is a real-to-complex transform, so the Fourier space to be analysed has dimensions (width/2+1)*height
		   int pixelCountX = digHoloWoiPolWidth[FFT_IDX] / 2 + 1;
		   int pixelCountY = digHoloWoiPolHeight[FFT_IDX];
		   size_t polStride = ((size_t)(digHoloWoiPolWidth[FFT_IDX] / 2 + 1)) * ((size_t)(digHoloWoiPolHeight[FFT_IDX]));
		   size_t batchStride = polStride * digHoloPolCount;
		   //The number of batch elements to process (all of them, +1 extra for the analysis of the sum over all batch items)
		   int batchCount = digHoloBatchCOUNT * digHoloBatchAvgCount;
		   //How many threads to use in the processing along the y-axis. This processing is parallelised in polarisation and along the y-axis.
		   int pixelThreads = digHoloThreadCount / digHoloPolCount;
		   if (pixelThreads <= 0)
		   {
			   pixelThreads = 1;
		   }
		   //Pointer to the outcome of the FFT (R2C FFT)
		   complex64* field = (complex64*)&digHoloPixelsFourierPlane[0][0];
		   //Perform analysis

		 //  complex64* RefCalibration = 0;
		 //  int RefCalibrationWavelengthCount = 0;

		   std::chrono::steady_clock::time_point startTimeA = std::chrono::steady_clock::now();

		   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		   {
			   fprintf(consoleOut, "digHoloFieldAnalysisRoutine()\n");
			   fflush(consoleOut);
		   }

		   digHoloFieldAnalysisRoutine(X, Y, field, polStart, polStop, FFT_IDX, pixelCountX, pixelCountY, batchCount, pixelThreads, batchStride, polStride, windowCX, windowCY, windowR, true, false);

		   std::chrono::steady_clock::time_point stopTimeA = std::chrono::steady_clock::now();
		   benchmarkFFTAnalysisTime += (stopTimeA - startTimeA);
		   benchmarkFFTAnalysisCounter++;
	   }

	   //The routine for performing batch FFTs, called from digHoloFFT.
	   //This routine will launch threads. 1 for each batch item in each polarisation. Each FFT will be single-threaded.
	   //Unless there's only 1 polarisation and 1 batch item, in which case all CPU cores will be assigned to the only FFT.
	   void digHoloFFTBatch(void* bufferIn, complex64* bufferOut, int polCount, int batchCount, fftwf_plan** FFTPlans, int r2c)
	   {
		   //Number of polarisations and batch items
		   //const int polCount = digHoloPolCount;
		   //const int batchCount = digHoloBatchAvgCount * digHoloBatchCOUNT;

		   //We'll be distributing threads based on how many polarisations and batch items there are to process.
		   if (batchCount > 1 || polCount == 2)
		   {
			   //total threads launched = polThreads*batchThreads, with each thread performing an FFT using fftThreads
			   int fftThreads = 1;//The number of threads for each FFT
			   int polThreads = 1;//The thread dimension assigned to polarisation
			   int batchThreads = 1;//the thread dimension assigned to batch items

			   //Single frame, dual polarisation
			   //1 thread for each polarisation, and 1/2 the CPU cores for each individual FFT
			   if (batchCount == 1)
			   {
				   fftThreads = digHoloThreadCount / polCount;
				   polThreads = polCount;
				   batchThreads = 1;
			   }
			   else
			   { //If there's multiple batch items, then each FFT is single-threaded, with PROCESSOR_COUNT threads launched to evenly distribute between all the polCount*batchCount FFTs to be performed.
				   polThreads = polCount;
				   fftThreads = 1;
				   batchThreads = digHoloThreadCount / polCount;
			   }
			   //Check for impossible thread counts (zero or negative). Saturate to 1.
			   if (fftThreads < 1)
			   {
				   fftThreads = 1;
			   }
			   if (batchThreads < 1)
			   {
				   batchThreads = 1;
			   }
			   if (polThreads < 1)
			   {
				   polThreads = 1;
			   }
			   if (polThreads > polCount)
			   {
				   polThreads = polCount;
			   }
			   if (batchThreads > batchCount)
			   {
				   batchThreads = batchCount;
			   }
			   //The total number of threads is the number of threads assigned to handle polarisation by the number of threads assigned to handle the batch elements.
			   const int totalThreads = polThreads * batchThreads;
			   //Chunk of batch items assigned to each thread. ~evenly distributed
			   const int batchEach = (int)(ceil((1.0 * batchCount / (batchThreads))));

			   //Callback routine to do the FFTs

			   //Keep track of what thread we're up to
			   int threadIdx = 0;
			   //For each polarisation...
			   for (int polIdx = 0; polIdx < polThreads; polIdx++)
			   {
				   //Get a pointer to the start of the FFT window
				   const int cx = digHoloWoiSX[FFT_IDX][polIdx];//x-corner of window
				   const int cy = digHoloWoiSY[FFT_IDX][polIdx];//y-corner of window
				   const size_t idx = cy * digHoloFrameWidth + cx;//pixel position in camera frame of that x-y corner
				   
				   void* pxBuffer = 0;
				   if (r2c)
				   {
					   float* bufferInFloat32 = (float*)bufferIn;
					   pxBuffer = &bufferInFloat32[idx];
				   }
				   else
				   {
					   complex64* bufferInComplex32 = (complex64*)bufferIn;
					   pxBuffer = &bufferInComplex32[idx];
				   }
				  

				   //Launch batchThreads worth of threads to handle FFT processing in this polarisation (polIdx)
				   for (int batchIdx = 0; batchIdx < batchThreads; batchIdx++)
				   {
					   //Concise alias of threadIdx
					   const int j = threadIdx;

					   //First batch item this thread will process
					   const int startIdx = batchEach * batchIdx;
					   //batch items, <stopIdx will be processed
					   int stopIdx = batchEach * (batchIdx + 1);
					   //batchThreads might not be exactly divisible by batchCount, so check we don't go over the edge.
					   if (stopIdx > batchCount)
					   {
						   stopIdx = batchCount;
					   }


					   int batchStep = stopIdx - startIdx;

					   //The number of batch items is larger than the maximum 'howmany' FFTW plans we have available.
					   //Set the batchStep to the maximum 'howmany', and we'll call fftwf_execute multiple times in a loop, rather than 1 big fftw_execute that does all the batch items.
					   if (batchStep > FFTW_HOWMANYMAX)
					   {
						   batchStep = FFTW_HOWMANYMAX;
					   }
					   //For every batch item, starting at startIdx, and in steps of 'howmany' we'll be processing per fftwf_execute call.
					   for (int batchIdx = startIdx; batchIdx < stopIdx; batchIdx += batchStep)
					   {
						   //The number of FFTs to perform per fftwf_execute call.
						   int howmany = batchStep;
						   //Check we won't go over th edge. If the number of FFTs would go beyond the boundary of batch items assigned to this thread, then saturate to stopIdx.
						   if ((batchIdx + batchStep) > stopIdx)
						   {
							   howmany = stopIdx - batchIdx;
						   }

						   if (!FFTPlans[fftThreads - 1][howmany - 1])
						   {
							   digHoloFFTPlan(fftThreads - 1, howmany - 1,&FFTPlans[fftThreads - 1][howmany - 1],r2c);
						   }
					   }

					   workPackage* workPack = digHoloWorkPacks[j];
					   //Assign these values to the thread's work package
					   workPack[0].start = startIdx;
					   workPack[0].stop = stopIdx;

					   //Pointer to the source of the FFT
					   workPack[0].ptr1 = (void*)pxBuffer;
					   //Pointer to the result of the FFT
					   workPack[0].ptr2 = (void*)bufferOut;
					   workPack[0].ptr3 = (void*)FFTPlans;

					   //Tell the thread how many polarisations there are (so it knows how far apart the batch items are in memory)
					   workPack[0].flag1 = polCount;
					   //Which polarisation to process
					   workPack[0].flag2 = polIdx;
					   //How many FFT threads to use in the FFT routine
					   workPack[0].flag3 = fftThreads;
					   workPack[0].flag4 = r2c;

					   workPack[0].callback = workFunction::digHoloFFTBatchWorker;

					   //Reset event to Set when thread is complete
					   workPack[0].workCompleteEvent.Reset();
					   workPack[0].workNewEvent.Set();
					   //Launch the thread
					   //Increment the thread counter
					   threadIdx++;
				   }
			   }

			   //Wait for threads to finish
			   for (int j = 0; j < totalThreads; j++)
			   {
				   workPackage* workPack = digHoloWorkPacks[j];
				   workPack[0].workCompleteEvent.WaitOne();
			   }

		   }
		   else
		   {
			   //If there is only a single item to process (single pol. and single batch item). Then don't launch threads, just do an FFT using all the CPU cores
			   const int polIdx = 0;//If you got here there should only be a single polarisation
			   const int cx = digHoloWoiSX[FFT_IDX][polIdx];//x-corner of FFT window
			   const int cy = digHoloWoiSY[FFT_IDX][polIdx];//y-corner of FFT window
			   const size_t idx = cy * digHoloFrameWidth + cx;//pixelIdx of x-y corner of FFT window
			   void* pxBuffer = 0;
			   if (r2c)
			   {
				   float* bufferInFloat32 = (float*)bufferIn;
				   pxBuffer = &bufferInFloat32[idx];
			   }
			   else
			   {
				   complex64* bufferInComplex32 = (complex64*)bufferIn;
				   pxBuffer = &bufferInComplex32[idx];
			   }

			   //Perform the FFT (2D R2C), using PROCESSOR_COUNT threads[PROCESSOR_COUNT-1], and in a 'howmany' batch of 1, [0]
			   if (!FFTPlans[digHoloThreadCount - 1][0])
			   {
				   digHoloFFTPlan(digHoloThreadCount - 1, 0,&FFTPlans[digHoloThreadCount - 1][0],r2c);

			   }
			   if (r2c)
			   {
				   fftwf_execute_dft_r2c(FFTPlans[digHoloThreadCount - 1][0], (float*)pxBuffer, bufferOut);
			   }
			   else
			   {
				   fftwf_execute_dft(FFTPlans[digHoloThreadCount - 1][0], (complex64*)pxBuffer, bufferOut);
			   }

		   }
	   }

	   //The worker thread routine for FFT processing.
	   //The FFT of the camera image is performed in this routine, for whichever polarisation and batch items that have been assigned to this thread
	   void digHoloFFTBatchWorker(workPackage& e)
	   {
		   //pointer to the start of the camera frame for the processing
		  float* bufferIn = (float*)e.ptr1;
		  complex64* bufferInComplex = (complex64*)e.ptr1;

		   //pointer to location to store the result of the FFT
		   complex64* bufferOut = (complex64*)e.ptr2;

		   fftwf_plan** FFTPlans = (fftwf_plan**)e.ptr3;

		   //Total number of polarisations, used to work out how far apart the batch items are in memory
		   const int polCount = e.flag1;
		   //Which polarisation this thread should process
		   const int polIdx = e.flag2;
		   //How many threads the FFT routine should be launched using
		   const int fftThreads = e.flag3;
		   const int r2c = e.flag4;

		   //The start/stop of the batch item loop
		   const int startIdx = (int)e.start;
		   const int stopIdx = (int)e.stop;

		   //Size of a single camera frame (input to FFT)
		   const size_t strideFull = digHoloFrameWidth * digHoloFrameHeight;

		   //Size of the corresponding output of the FFT (R2C transform)
		   const size_t strideLow = r2c?((digHoloWoiPolWidth[FFT_IDX] / 2 + 1) * digHoloWoiPolHeight[FFT_IDX]): ((digHoloWoiPolWidth[FFT_IDX]) * digHoloWoiPolHeight[FFT_IDX]);

		   //The number of batch items to be processed in this thread.
		   //We'll be using the plan_many option of FFTW, so that multiple FFTs  are performed from a single fftwf_execute command.
		   //Unless more batch items need to be processed than we have FFT plans for, in which case we'll have to loop over the batch items as well.
		   int batchStep = stopIdx - startIdx;

		   //The number of batch items is larger than the maximum 'howmany' FFTW plans we have available.
		   //Set the batchStep to the maximum 'howmany', and we'll call fftwf_execute multiple times in a loop, rather than 1 big fftw_execute that does all the batch items.
		   if (batchStep > FFTW_HOWMANYMAX)
		   {
			   batchStep = FFTW_HOWMANYMAX;
		   }

		   //For ever batch item, starting at startIdx, and in steps of 'howmany' we'll be processing per fftwf_execute call.
		   for (int batchIdx = startIdx; batchIdx < stopIdx; batchIdx += batchStep)
		   {
			   //The number of FFTs to perform per fftwf_execute call.
			   int howmany = batchStep;
			   //Check we won't go over th edge. If the number of FFTs would go beyond the boundary of batch items assigned to this thread, then saturate to stopIdx.
			   if ((batchIdx + batchStep) > stopIdx)
			   {
				   howmany = stopIdx - batchIdx;
			   }

			   //Position of this batch item, in this polarisation in the camera frame buffer
			  // const size_t frameOffsetIn = (batchIdx * polCount + polIdx) * strideFull;
			   const size_t frameOffsetIn = (batchIdx)*strideFull;
			   //The corresponding position of the output of the FFT for this batch item in this polarisation
			   const size_t frameOffsetOut = (batchIdx * polCount + polIdx) * strideLow;

			   //Execute the FFT, using fftThreads, and in a batch of howmany FFTs
			   if (r2c)
			   {
				   fftwf_execute_dft_r2c(FFTPlans[fftThreads - 1][howmany - 1], &bufferIn[frameOffsetIn], &bufferOut[frameOffsetOut]);
			   }
			   else
			   {
				   fftwf_execute_dft(FFTPlans[fftThreads - 1][howmany - 1], (complex64*)&bufferInComplex[frameOffsetIn], &bufferOut[frameOffsetOut]);
			   }

			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				 //  fprintf(consoleOut, "FFT complete.\n");
				 //  fflush(consoleOut);
			   }
		   }

		   //Signal the thread is complete
		  // ManualResetEvent resetEvent = e.resetEvent;
		 //  e.resetEvent.Set();
	   }

	   //Routine which sets the boundaries/dimensions of the FFT window on the camera frame.
	   //Takes the user specified beam centres, and FFT window sizes, and then converts them to valid values that are multiples of 16 for SIMD purposes, and that don't go off the edge of the frame etc.
	   //Also reallocates memory etc if required, and returns 'true' if the size of the FFT changed.
	   bool digHoloFFT_SetWindow()
	   {
		   //These settings get read in from the user facing config object here.
		  //Helps prevent config changing by the user in the GUI mid-way through the process.
		   digHoloPolCount = config.polCount;//Number of polarisations to process
		   digHoloPixelSize = (float)(config.pixelSize * DIGHOLO_UNIT_PIXEL);//Pixel size on the camera
		   digHoloApertureSize = (float)(config.apertureSize * DIGHOLO_UNIT_ANGLE);//radius of the IFFT window in units of radians
		   digHoloResolutionIdx = config.resolutionIdx;//Full/low resolution mode
		   digHoloWavelengthCentre = (float)(config.wavelengthCentre * DIGHOLO_UNIT_LAMBDA);//the centre wavelength

		   digHoloFrameWidth = config.frameWidth;
		   digHoloFrameHeight = config.frameHeight;

		   //We'll also update the operational thread count here.
		   digHoloThreadCount = config.threadCount;

		   if (config.fftWindowSizeX <= 0)
		   {
			   config.fftWindowSizeX = digHoloFrameWidth / digHoloPolCount;
		   }
		   if (config.fftWindowSizeY <= 0)
		   {
			   config.fftWindowSizeY = digHoloFrameHeight;
		   }

		   //Size of the FFT window floored to multiple of 16
		   //16 is because AVX processes in 256 bit blocks, which means when we eventually convert the field to a 16-bit number, we'll have 16 x int16s per packed AVX block.
		   //Keeping it multiples of 16 means we can always iterate over the x and y dimensions using an exact number of 256-bit blocks, and don't have to worry about handling edge cases, or memory alignment issues
		   int windowSizeX = DIGHOLO_PIXEL_QUANTA * ((config.fftWindowSizeX) / DIGHOLO_PIXEL_QUANTA);
		   int windowSizeY = DIGHOLO_PIXEL_QUANTA * ((config.fftWindowSizeY) / DIGHOLO_PIXEL_QUANTA);
		   //Centre of the beams in the two polarisations on the camera.
		   const float BeamCentreX[2] = { (float)(config.BeamCentreX[0] * DIGHOLO_UNIT_PIXEL),(float)(config.BeamCentreX[1] * DIGHOLO_UNIT_PIXEL) };
		   const float BeamCentreY[2] = { (float)(config.BeamCentreY[0] * DIGHOLO_UNIT_PIXEL),(float)(config.BeamCentreY[1] * DIGHOLO_UNIT_PIXEL) };

		   //Concise alias of parameters (just for making code neater locally)
		   const float pixelSize = digHoloPixelSize;
		   const int frameWidth = digHoloFrameWidth;
		   const int frameHeight = digHoloFrameHeight;
		   const int polCount = digHoloPolCount;

		   //Memory allocated using allocate1D,2D etc, uses _aligned_malloc to a boundary of ALIGN.
		   //This converts the byte value of ALIGN to the number of floats per ALIGN.
		   //This is used to make sure we select a camer window for our FFT that aligns to a memory boundary for AVX.
		   //You don't need to memory align, you coul also use loadu_ps instead of load_ps, but load_ps is faster
		   const size_t blockSize = ALIGN / sizeof(float);//blocksize of 8 (2^3)

		   //Make sure the FFT window is always at least 16x16 for SIMD reasons.
		   if (windowSizeX < 16)
		   {
			   windowSizeX = 16;
		   }
		   if (windowSizeY < 16)
		   {
			   windowSizeY = 16;
		   }

		   //If the window widthrequested is larger than the width of the frame, set it to the width of the frame
		   if (windowSizeX > frameWidth)
		   {
			   windowSizeX = frameWidth;
		   }
		   //If the window height is larger than the height of the frame for this polarisation, set the window size to the maximum height of this polarisation
		   if (windowSizeY > frameHeight) //if (windowSizeY > frameHeight / polCount)
		   {
			   windowSizeY = frameHeight;// / polCount;
		   }

		   //For each polarisation, setup the FFT window, and check for compliance with valid values
		   for (int polIdx = 0; polIdx < polCount; polIdx++)
		   {
			   //concise alias for beam centre on camera
			   const float beamCX = BeamCentreX[polIdx];
			   const float beamCY = BeamCentreY[polIdx];
			   //Convert the beamCX/CY positions (in units of metres) to units of pixels, relative to the first pixel on the camera (corner pixel, first memory position, rather than centre of the camera)
			   const float CX = ((beamCX / pixelSize) + (frameWidth / 2));
			   //const float CY = ((beamCY / pixelSize) + (frameHeight / (2 * polCount)));
			   const float CY = ((beamCY / pixelSize) + (frameHeight / (2)));

			   //Get the far lower edge of a window centred at position CX/CY, and rounded to the nearest AVX compliant address
			   int cx = (int)round((CX - windowSizeX / 2) / blockSize) * blockSize;//Enforce that final memory address will be multiple of ALIGN (SSE/AVX memory alignment)
			   int cy = (int)round((CY - windowSizeY / 2));//Assuming that the Y dimension will always be aligned because digHoloFrameHeight/(2*polCount) is a multiple of 8

			   //Check the edge of the window isn't off the edge of the camera frame.
			   //If it is, set this edge to the edge of the camera frame.
			   if (cx < 0)
			   {
				   cx = 0;
			   }
			   if (cy < 0)
			   {
				   cy = 0;
			   }

			   //Check the other edge of the window isn't off the other edge of the camera frame.
			   //If it is, set this edge to the edge of the camera frame.
			   if ((cx + windowSizeX) > frameWidth)
			   {
				   cx = frameWidth - windowSizeX;
			   }
			   /*  if ((cy + windowSizeY) > frameHeight / polCount)
				 {
					 cy = frameHeight / polCount - windowSizeY;
				 }*/
			   if ((cy + windowSizeY) > frameHeight)
			   {
				   cy = frameHeight - windowSizeY;
			   }
			   //Store what  should now be a valid starting position (corner) of the FFT window.
			   //This cx/cy position indicates the first position in memory of the FFT window
			   digHoloWoiSX[FFT_IDX][polIdx] = cx;
			   digHoloWoiSY[FFT_IDX][polIdx] = cy;
		   }
		   //No longer used, but digHoloFFT_SetWindow, does return whether or not the FFT window size changed.
		   //What could be more useful is returning whether the size or position WoiSX/SY changed, because that indicates that the digHoloCopyWindow routine for the IFFT would need to be re-run.
		   const bool sizeChanged = (digHoloWoiPolWidth[FFT_IDX] != windowSizeX) || (digHoloWoiPolHeight[FFT_IDX] != windowSizeY);
		   //Update the size of the window
		   digHoloWoiPolWidth[FFT_IDX] = windowSizeX;
		   digHoloWoiPolHeight[FFT_IDX] = windowSizeY;

		   //Initialised the FFT memory and plans if required.
		   //This function checks for changes in FFT length, batch count, pixelSize etc and updates relevant parameters if needed.
		   digHoloFFTInit();

		   return sizeChanged;
	   }

	   int digHoloDCTGetPlan()
	   {
		   //Use whatever FFTW plan mode the user has specified, e.g. FFTW_ESTIMATE, FFTW_PATIENT etc.
		   const int planMode = FFTW_PLANMODES[config.FFTW_PLANMODE];
		   //Concise alias of FFT, pol and frame dimensions
		   const int height = digHoloWoiPolHeight[FFT_IDX];
		   const int width = digHoloWoiPolWidth[FFT_IDX];
		   const int onembed[2] = { height ,width / 2 + 1 };//Dimension of the output Fourier-space (Real-to-complex transform, means dimensions are (width/2+1)*height, because the Hermitian symmetric half isn't calculated.

		   complex64** tempCameraPlane = 0;
		   allocate2D(2, FFTW_HOWMANYMAX * width * height, tempCameraPlane);

		   float* in = (float*)tempCameraPlane[0];//Could also the camera frame pointer instead, as long as we use FFTW_PRESERVE_INPUT so we don't corrupt the camera frames
		   complex64* out = (complex64*)&tempCameraPlane[1][0];//Could also use digHoloPixelsFourierPlane

		   if (digHoloDCTPlan)
		   {
			   fftwf_destroy_plan(digHoloDCTPlan); digHoloDCTPlan = 0;
		   }
		   if (digHoloIDCTPlan)
		   {
			   fftwf_destroy_plan(digHoloIDCTPlan); digHoloIDCTPlan = 0;
		   }
		   digHoloDCTPlan = fftwf_plan_r2r_2d(onembed[0], onembed[1], in, (float*)out, FFTW_REDFT10, FFTW_REDFT10, planMode | FFTW_PRESERVE_INPUT);
		   digHoloIDCTPlan = fftwf_plan_r2r_2d(onembed[0], onembed[1], in, (float*)out, FFTW_REDFT01, FFTW_REDFT01, planMode | FFTW_PRESERVE_INPUT);

		   free2D(tempCameraPlane);
		   fftwWisdomNew++;
		   return (digHoloDCTPlan != 0);
	   }

	   int digHoloFFTWisdomLoad()
	   {
		   //Loads FFTW wisdom from file. FFTW uses 'wisdom' which means it optimises which FFT algorithms it uses, based on the machine it's running on.
		   //The filename has the MachineName appended to it, so if you run this code on a different machine, it'll make it's own wisdom file

		   int wisdomReturnCode = fftwf_import_wisdom_from_filename(FFTW_WISDOM_FILENAME);
		   return wisdomReturnCode;
	   }
	   int digHoloFFTWisdomSave()
	   {
		   int wisdomReturnCode = fftwf_export_wisdom_to_filename(FFTW_WISDOM_FILENAME);
		   fftwWisdomNew = 0;
		   return wisdomReturnCode;
	   }

	   void digHoloFFTPlan(int threadIdx, int fftIdx, fftwf_plan *outputPlan, int R2C)
	   {
		   //Concise alias of FFT, pol and frame dimensions
		   const int height = digHoloWoiPolHeight[FFT_IDX];
		   const int width = digHoloWoiPolWidth[FFT_IDX];
		   const int widthOut = R2C?(width / 2 + 1):width;
		   const int polCount = digHoloPolCount;//It needs to know the polCount so it knows the output distance (odist) between outputs.

		   const int heightFull = digHoloFrameHeight;// / polCount; If you divide by pol or not doesn't make a difference here, it's the width that's important due to the way it's set out in memory
		   const int widthFull = digHoloFrameWidth;

		   //FFTW properties
		   const int rank = 2;//2D Fourier transform
		   const int n[2] = { height,width };//dimensions of the Fourier transform

		   //Temporary workspace for planning FFTs
		   float* in = (float*)digHoloFFTPlanWorkspace[0];//Could also the camera frame pointer instead, as long as we use FFTW_PRESERVE_INPUT so we don't corrupt the camera frames
		   complex64* out = (complex64*)&digHoloFFTPlanWorkspace[1][0];//Could also use digHoloPixelsFourierPlane

		   //For picking off sub-window, embedded in a larger matrix (selecting a smaller window inside the camera frame).
		   const int inembed[2] = { heightFull,widthFull };//Dimension of the raw camera frame
		   const int onembed[2] = { height ,widthOut };//Dimension of the output Fourier-space (Real-to-complex transform, means dimensions are (width/2+1)*height, because the Hermitian symmetric half isn't calculated.
		   //Indicates that the input/output arrays are contigious in memory. e.g. as opposed to some matrix where the columns aren't stored in memory next to each other, or we're not making a matrix be selecting every Nth column of a larger matrix etc.
		   //Just a normal input/output matrix as 1 big chunk of memory.
		   const int istride = 1;
		   const int ostride = 1;

		   //Parameters for planning multiple FFTs using a single fftwf_execute call...
		   //The distance between input arrays (1 camera frame worth)
		   const int idist = (digHoloFrameHeight * digHoloFrameWidth);
		   //Distance between output arrays (R2C length x 2 for polarisation)
		   const int odist = (height * (widthOut) * polCount);

		   //Use whatever FFTW plan mode the user has specified, e.g. FFTW_ESTIMATE, FFTW_PATIENT etc.
		   const int planMode = FFTW_PLANMODES[config.FFTW_PLANMODE];

		   int howmany = fftIdx + 1;
		   //Can be helpful to print this to screen, so that the user knows there's an FFT plan pending.
			//When you first run an FFT on a new computer, especially a large one with FFTW_PATIENT or FFTW_EXHAUSTIVE active, it'll look like the code has crashed, however it's actually planning
		   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG) //It could potentially be useful to also output when the FFTW planning mode is >FFTW_PATIENT, in which case it might seem like the code is stuck/crashed, but it's actually just waiting for planning routine.
		   {
			   fprintf(consoleOut, "	Planning FFT (WoI)	:	%i x %i	(%i:%i)\n\r", digHoloWoiPolWidth[FFT_IDX], digHoloWoiPolHeight[FFT_IDX], (threadIdx + 1), howmany);
			   fflush(consoleOut);
		   }
		   fftwf_plan_with_nthreads(threadIdx + 1);

		   if (outputPlan[0])
		   {
			   fftwf_destroy_plan(outputPlan[0]); outputPlan[0] = 0;
			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				   fprintf(consoleOut, "	Destroyed plan.\n\r");
				   fflush(consoleOut);
			   }
		   }

		   if (R2C)
		   {
			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				   fprintf(consoleOut, "	R2C plan.\n\r");
				   fflush(consoleOut);
			   }
			   outputPlan[0] = fftwf_plan_many_dft_r2c(rank, &n[0], howmany, in, &inembed[0], istride, idist, out, &onembed[0], ostride, odist, planMode | FFTW_PRESERVE_INPUT);
			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				   fprintf(consoleOut, "	Complete.\n\r");
				   fflush(consoleOut);
			   }
		   }
		   else
		   {
			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				   fprintf(consoleOut, "	C2C plan.\n\r");
				   fflush(consoleOut);
			   }
			   outputPlan[0] = fftwf_plan_many_dft(rank, &n[0], howmany, (complex64*)in, &inembed[0], istride, idist, out, &onembed[0], ostride, odist, FFTW_FORWARD, planMode | FFTW_PRESERVE_INPUT);
			   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
			   {
				   fprintf(consoleOut, "	Complete.\n\r");
				   fflush(consoleOut);
			   }
		   }

		   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		   {
			   fprintf(consoleOut, "	Planning FFT complete.\n\r");
			   fflush(consoleOut);
		   }
		   //Indicates that new plans have been made, and hence FFTW wisdom may need rewriting to file
		   fftwWisdomNew++;
	   }

	   void digHoloIFFTPlan(int threadIdx, int fftIdx, fftwf_plan* outputPlan)
	   {
		   //Why do I use _Valid here and not use it in digHoloFFTPlan?
		   int width = digHoloWoiPolWidth_Valid[IFFT_IDX];
		   int height = digHoloWoiPolHeight_Valid[IFFT_IDX];

		   const int polCount = digHoloPolCount;
		   //Useful to tell the user that a plan is underway, as for FFTW_PATIENT mode, this planning process could take some time. Without notification, the user could think it's crashed/hung.

		   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
		   {
			   fprintf(consoleOut, "	Planning IFFT (WoI)	:	%i x %i	(%i:%i)\n\r", width, height, (threadIdx + 1), (fftIdx + 1));
			   fflush(consoleOut);
		   }

		   //The type of planning mode to employ. e.g. FFTW_PATIENT, FFTW_ESTIMATE etc.
		   const int planMode = FFTW_PLANMODES[config.FFTW_PLANMODE];

		   //FFTW 'plan many' parameters
		   const int rank = 2;//2D Fourier transform
		   const int n[2] = { height,width };//Dimensions of IFFT
		   //Input/output buffer arrays for planning
		   complex64* in = digHoloFFTPlanWorkspace[0];
		   complex64* out = digHoloFFTPlanWorkspace[1];

		   //These are values define picking a sub-array from a larger array. Relevant to the FFT, where we select a window of raw pixels from the camera using these settings.
		   //Not relevant to the IFFT, where the digHoloCopyWindow routine makes the window (and also applies a tilt in the Fourier plane to centre the beam in the reconstructed plane).
		   //Without that tilt, you just use these settings to select the IFFT window
		   const int inembed[2] = { height,width };
		   const int onembed[2] = { height,width };
		   const int istride = 1;
		   const int ostride = 1;

		   //Distance between FFTs in memory (the size of the FFT x 2 for polarisation if applicable). Same for input and output of FFT, they're all just normal contigious blocks of memory
		   const int idist = height * width * polCount;
		   const int odist = idist;

		   //Defines how many FFTs to do in a single call to 'fftwf_execute'.
		   const int howmany = fftIdx + 1;

		   fftwf_plan_with_nthreads(threadIdx + 1);
		   if (outputPlan[0])
		   {
			   fftwf_destroy_plan(outputPlan[0]);
			   outputPlan[0] = 0;
		   }
		   outputPlan[0] = fftwf_plan_many_dft(rank, &n[0], howmany, in, &inembed[0], istride, idist, out, &onembed[0], ostride, odist, FFTW_BACKWARD, planMode);

		   fftwWisdomNew++;
	   }

	   //Checks for changes in FFT properties, such as the size of the window, the pixel size of the camera, the number of polarisations, the batchCount etc.
	   //Then updates memory allocations and FFT plans accordingly
	   void digHoloFFTInit()
	   {
		   //Dimensions of the FFT window
		   const size_t width = digHoloWoiPolWidth[FFT_IDX];
		   const size_t height = digHoloWoiPolHeight[FFT_IDX];
		   //Number of polarisations supported
		   const size_t polCount = digHoloPolCount;

		   //Check for changes in FFT window dimensions
		   const unsigned char sizeChanged = ((digHoloWoiPolWidth[FFT_IDX] != digHoloWoiPolWidth_Valid[FFT_IDX]) || (digHoloWoiPolHeight[FFT_IDX] != digHoloWoiPolHeight_Valid[FFT_IDX]));
		   //If the camera pixel size was changed, that will change the axes in the Fourier plane
		   const unsigned char pixelSizeChanged = (digHoloPixelSize != digHoloPixelSize_Valid[FFT_IDX]);
		   //If the polarisation count has changed, some memory will need to be reallocated
		   const unsigned char polCountChanged = polCount != (size_t)digHoloPolCount_Valid[FFT_IDX];

		   //Check for changes in batch count
		   if (config.batchCount <= 0)
		   {
			   config.batchCount = 1;
		   }
		   //the batch count is now whatever the user specified, or defaulting to 1 if the user specified something shameful.
		   digHoloBatchCOUNT = config.batchCount;
		   digHoloBatchAvgCount = config.avgCount;

		   //Check if the batch count has changed
		   const unsigned char batchCountChanged = digHoloBatchCOUNT != digHoloBatchCount_Valid[FFT_IDX];
		   const unsigned char avgCountChanged = digHoloBatchAvgCount != digHoloBatchAvgCount_Valid[FFT_IDX];

		   //If the batch count or the pol count has changed, then the array that stores statistics about the Fourier plane and IFFT plane will have to be reallocated
		   if (batchCountChanged || polCountChanged || avgCountChanged)
		   {
			   //The extra +1 to digHoloBatchCount is because digHoloPixelsStats provides numbers for every batch element +1 element at the end which is the aggregate over the whole batch. i.e. the total power, or the average effective area/centre of mass.
			   //The second dimension of 2 is 1 for Fourier plane (result of FFT), and 1 for reconstructed  plane (result of IFFT)
			   allocate4D(2, DIGHOLO_ANALYSISCount, polCount, (digHoloBatchCOUNT * digHoloBatchAvgCount) + 1, DIGHOLO_ANALYSIS);
			   const size_t analysisLength = 2 * DIGHOLO_ANALYSISCount * polCount * ((digHoloBatchCOUNT * digHoloBatchAvgCount) + 1);
			   memset(&DIGHOLO_ANALYSIS[0][0][0][0], 0, sizeof(float) * analysisLength);
		   }



		   if (sizeChanged || batchCountChanged || polCountChanged || avgCountChanged)
		   {
			   if (batchCountChanged || avgCountChanged)
			   {
				   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
				   {
					   fprintf(consoleOut, "Batch Count : %i\n\r", digHoloBatchCOUNT);
					   if (digHoloBatchAvgCount > 1)
					   {
						   fprintf(consoleOut, "Batch Avg Count : %i\n\r", digHoloBatchAvgCount);
					   }
					   fflush(consoleOut);
				   }
			   }

			   const size_t pixelCountFFT_R2C = (digHoloBatchCOUNT * digHoloBatchAvgCount) * (width / 2 + 1) * height * polCount;
			   //FT of the camera image
			   allocate1D(pixelCountFFT_R2C, digHoloPixelsFourierPlane);
			   digHoloBatchCount_Valid[FFT_IDX] = digHoloBatchCOUNT;
			   digHoloBatchAvgCount_Valid[FFT_IDX] = digHoloBatchAvgCount;
			   digHoloPolCount_Valid[FFT_IDX] = digHoloPolCount;

		   }

		   if (sizeChanged || polCountChanged)
		   {
			   //This is an array where the sum of intensity over all batch elements is stored. x2 for FFT/IFFT. x2 again for FT of the sum.
				//That extra +4 is for doing extra Fourier processing to work out where the off-axis term is. i.e. convolve the Fourier plane with the aperture window to work out where the off-axis terms i.
				//Used by digHoloFieldAnalysis and digHoloAutoAlign
			   allocate3D(6, polCount, width * height, DIGHOLO_ANALYSISBatchSum);
			   memset(&DIGHOLO_ANALYSISBatchSum[0][0][0], 0, sizeof(float) * 6 * polCount * width * height);

			   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
			   {
				   fprintf(consoleOut, "FFT size changed\n");
				   fflush(consoleOut);
			   }

			   //Remember the dimensions of the last time an FFT was planned.
			   digHoloWoiPolWidth_Valid[FFT_IDX] = digHoloWoiPolWidth[FFT_IDX];
			   digHoloWoiPolHeight_Valid[FFT_IDX] = digHoloWoiPolHeight[FFT_IDX];
			   
			   const size_t dimX = (THREADCOUNT_MAX * DIGHOLO_PLANECOUNTCAL);
			   //If this is the first time FFT planning has been run, setup a 2D array of plans, that lets us select plans for various combinations of threads x howmany FFTs per execute
			   if (digHoloFFTPlans == 0)
			   {
				   allocate2D(dimX, FFTW_HOWMANYMAX, digHoloFFTPlans);
				   for (int fftIdx = 0; fftIdx < FFTW_HOWMANYMAX; fftIdx++)
				   {
					   for (int threadIdx = 0; threadIdx < dimX; threadIdx++)
					   {
						   digHoloFFTPlans[threadIdx][fftIdx] = 0;
					   }
				   }
				   
				   //Why was I doing this as individual items set to zero before? Some sort of issue with using memset for this?
				   //memset(&digHoloFFTPlans[0][0], 0, sizeof(fftwf_plan) * dimX * FFTW_HOWMANYMAX);
			   }

			   //A memory workspace used for FFT planning.
			   
			   //if (!digHoloFFTPlanWorkspace)
			   //{
			   //You've gotta do the full digHoloFrameWidth x digHoloFrameHeight because that's the spacing between frames in memory for the 'howmany' type plans.
				   allocate2D(2, FFTW_HOWMANYMAX * digHoloFrameWidth*digHoloFrameHeight, digHoloFFTPlanWorkspace);
			   //}

			   //Create FFTW plans for every combo of threads and howmany batch size.
			   for (int fftIdx = 0; fftIdx < FFTW_HOWMANYMAX; fftIdx++)
			   {
				   // int howmany = fftIdx + 1;
				   for (int threadIdx = 0; threadIdx < dimX; threadIdx++)
				   {
					   // fftwf_plan_with_nthreads(threadIdx + 1);
					   if (digHoloFFTPlans[threadIdx][fftIdx])
					   {
						   fftwf_destroy_plan(digHoloFFTPlans[threadIdx][fftIdx]);
					   }
					   digHoloFFTPlans[threadIdx][fftIdx] = 0;// fftwf_plan_many_dft_r2c(rank, &n[0], howmany, in, &inembed[0], istride, idist, out, &onembed[0], ostride, odist, planMode | FFTW_PRESERVE_INPUT);
				   }
			   }
		   }

		   //If the dimensions of the FFT window have changed, or the size of the camera pixels has changed, then the x-y axes in the camera plane, and Fourier plane will need to be updated.
		   if (sizeChanged || pixelSizeChanged)
		   {
			   //Concise alias of the FFT window dimensions
			   const int height = digHoloWoiPolHeight[FFT_IDX];
			   const int width = digHoloWoiPolWidth[FFT_IDX];

			   //If this FFT dimensions have changed, reallocate memory for the x/y (camera plane) and kx/ky (Fourier plane) axes.
			   if (sizeChanged)
			   {
				   allocate1D(width, digHoloKXaxis);
				   allocate1D(height, digHoloKYaxis);
				   allocate1D(width, digHoloXaxis[FULLRES_IDX]);
				   allocate1D(height, digHoloYaxis[FULLRES_IDX]);
			   }

			   //Dimensions of a single pixel in Fourier space along kx/ky axes.
			   const float pixelPeriod = (float)(2.0 * digHoloPixelSize);
			   const float kFactorX = (float)(2 * (2.0 * piD / pixelPeriod) / width);
			   const float kFactorY = (float)(2 * (2.0 * piD / pixelPeriod) / height);
			   //Recalculate Fourier-space kx/ky axes. The 'true' flag at the end indicates this axis should be calculated in FFTshifted form, so that the axes match the output of the FFT
			   //The full Fourier-space axis is calculated (not just the R2C part)
			   GenerateCoordinatesXY(width, height, kFactorX, kFactorY, digHoloKXaxis, digHoloKYaxis, true);
			   //Calculate the camera x/y axes. The 'false' flag means there is no FFT shifting 
			   GenerateCoordinatesXY(width, height, digHoloPixelSize, digHoloPixelSize, digHoloXaxis[FULLRES_IDX], digHoloYaxis[FULLRES_IDX], false);
			   //Remember the pixel size that was used last time the x/y, kx/ky axes were calculated, so we'll know if we need to update these
			   digHoloPixelSize_Valid[FFT_IDX] = digHoloPixelSize;
			   //As the FFT dimesions and/or the camera pixel size has changed, that means the mode basis is now out of date, and will need to be recalculated if we're doing mode overlaps.
			   digHoloBasisValid = false;
		   }

		   if (sizeChanged || pixelSizeChanged || polCountChanged)
		   {
			   digHoloRefValid = false;
		   }

		   if (digHoloPixelBufferType)
		   {
			   if (digHoloPixelBufferConversionFlag == 0 || digHoloPixelBufferConversionFlag == 1)
			   {
				   const size_t totalBatchCount = (size_t)digHoloBatchCOUNT * (size_t)digHoloBatchAvgCount;
				   const size_t framePixelCount = (size_t)digHoloFrameWidth * (size_t)digHoloFrameHeight;
				   const size_t totalBufferLength = totalBatchCount * framePixelCount;

				   //If the size has changed, allocate a new pixel buffer.
				   if (totalBufferLength != digHoloPixelBufferLength)
				   {
					   allocate1D(totalBufferLength, digHoloPixelBuffer);
					   digHoloPixelBufferLength = totalBufferLength;
				   }

				   const int nxStart = 0;
				   const int nxStop = digHoloFrameWidth;
				   const int transpose = digHoloPixelBufferUint16transposed;
				   for (size_t frameIdx = 0; frameIdx < totalBatchCount; frameIdx++)
				   {
					   const size_t pixelIdx0 = frameIdx * framePixelCount;

					   convert_int16tofloat32(&digHoloPixelsCameraPlaneUint16[pixelIdx0], &digHoloPixelBuffer[pixelIdx0], nxStart, nxStop, digHoloFrameWidth, digHoloFrameHeight, transpose);
				   }
				   digHoloPixelsCameraPlane = digHoloPixelBuffer;

				   if (digHoloPixelBufferConversionFlag == 1)
				   {
					   digHoloPixelBufferConversionFlag = -digHoloPixelBufferConversionFlag;
				   }
			   }
		   }
	   }

	   //Timers used for benchmarking the speed of the IFFT routine
	   std::chrono::duration<double> benchmarkIFFTTime = std::chrono::duration<double>(0);
	   int64_t benchmarkIFFTCounter = 0;

	   //The IFFT routine. Filters off a window in Fourier-space, applies a tilt to that window in Fourier space to centre the beam after the IFFT.
	   int digHoloIFFT()
	   {
		   if (digHoloPixelsFourierPlane)
		   {
			   //Benchmark performance timer
			   //const int64_t startTime = QPC();
			   std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

			   //Set the window to be IFFT'd. Checks for changes to window size, wavelength etc that will require the window to be recalculated, memory to be reallocated, and/or the IFFTs to be re-planned
			   digHoloIFFT_SetWindow();

			   //Launches threads to do the IFFT processing. 1 thread for each batch item in each polarisation.
			   //Also performs analysis on the results using the digHoloFieldAnalysis routine.
			   //Unlike the FFT, where the digHoloFieldAnalysis is effectively optional (although it's a neglible contribution to speed).
			   //For the IFFT, the results of digHoloFieldAnalysis will be important for converting the results of the IFFT to int16 format. 
			   //Do do so, we need to know the scaling factor that maps the 16-bit integers onto floats, and digHoloFieldAnalysis provides that information, it tells use the maximum absolute value of all the real/imaginary pixels in the field, which we can then use to scale accordingly.
			   //digHoloIFFTBatch(complex64* bufferInFull, complex64* bufferInMasked, complex64* bufferOut, int polCount, int batchCount, fftwf_plan** FFTPlans, int r2c, int avgCount, int avgMode)
			   
			   digHoloIFFTCalibrationUpdate();
			   digHoloIFFTBatch();// digHoloPixelsFourierPlane, digHoloPixelsFourierPlaneMasked, digHoloPixelsCameraPlaneReconstructedTilted, digHoloPolCount, digHoloBatchCOUNT, digHoloIFFTPlans, r2c, digHoloBatchAvgCount, config.avgMode);

			  
			   //Save wisdom if there's been new plans made.
			   if (fftwWisdomNew)
			   {
				   digHoloFFTWisdomSave();
			   }
			   //Performance timer increment
			   std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();
			   benchmarkIFFTTime += (stopTime - startTime);
			   benchmarkIFFTCounter++;
			   return DIGHOLO_ERROR_SUCCESS;
		   }
		   else
		   {
			   return DIGHOLO_ERROR_NULLPOINTER;
		   }
	   }

	   int digHoloWavelengthUpdate()
	   {
		   int updated = false;
		   //Check for changes in wavelength.
			//If the wavelength(s) has changed, that will mean the position of our window in Fourier space might also have changed.
		   if (!digHoloWavelengthValid && digHoloWavelengthCountNew > 0)//If the current wavelength array is invalid, and there's a new array available. Copy it.
		   {
			   if (digHoloWavelengthCount != digHoloWavelengthCountNew)
			   {
				   allocate1D(digHoloWavelengthCountNew, digHoloWavelength);
			   }
			   memcpy(digHoloWavelength, digHoloWavelengthNew, sizeof(float) * digHoloWavelengthCountNew);
			   digHoloWavelengthCount = digHoloWavelengthCountNew;
			   //Current wavelength array is now valid.
			   digHoloWavelengthValid = true;
			   updated = true;
		   }
		   else //wavelength array is valid, or the wavelength array is invalid, but no new wavelength array is available to reinitialise it
		   {
			   //If the wavelength array is unallocated
			   if (digHoloWavelength == 0)
			   {
				   //Setup a single-element wavelength array based on the user specified centre wavelength config->wavelengthCentre
				   digHoloWavelengthCount = 1;
				   allocate1D(digHoloWavelengthCount, digHoloWavelength);
				   digHoloWavelengthCentre = (float)(config.wavelengthCentre * DIGHOLO_UNIT_LAMBDA);
				   digHoloWavelength[0] = digHoloWavelengthCentre;
				   //Current wavelength array is now valid
				   digHoloWavelengthValid = true;
				   updated = true;
			   }
		   }
		   //There's a potential conflict here. If the user specifies a sweep of size 1 (which would be unusual) then it will just ignore it and use the wavelengthCentre
		   if (digHoloWavelengthCount == 1)
		   {
			   digHoloWavelengthCentre = (float)(config.wavelengthCentre * DIGHOLO_UNIT_LAMBDA);
			   if (digHoloWavelength[0] != digHoloWavelengthCentre)
			   {
				   updated = true;
			   }
			   digHoloWavelength[0] = digHoloWavelengthCentre;
		   }
		   return updated;
	   }

	   int digHoloIFFT_GetWindowBounds(float **zernCoefs, int& windowWidthOut, int& windowHeightOut, int *WoiSX, int* WoiSY, float lambdaStart, float lambdaStop, int r2c)
	   {
		   //This is the max pol count, not necessarily the current pol count. Just for memory allocation purposes
		   const int polCountMax = 2;

		   //Tilt x/y in units of radians
		   float tiltx[polCountMax];
		   float tilty[polCountMax];

		   const int polCount = digHoloPolCount;

		   //Read in the current tilt configuration from the config object
		  // float** zernCoefs = config.zernCoefs;
		   if (zernCoefs)
		   {
			   for (int polIdx = 0; polIdx < polCount; polIdx++)
			   {
				   int polIDX = polIdx;

				   //If polLock is enabled for tilt, then V polarisation will be set to the same value as the H polarisation
				   if (config.PolLockTilt)
				   {
					   polIDX = 0;
				   }
				   //Convert from degrees to radians and store in our temporary local array tiltx/y
				   tiltx[polIdx] = (float)(zernCoefs[polIDX][TILTX] * DIGHOLO_UNIT_ANGLE);
				   tilty[polIdx] = (float)(zernCoefs[polIDX][TILTY] * DIGHOLO_UNIT_ANGLE);
			   }
		   }
		   else
		   {
			   for (int polIdx = 0; polIdx < polCount; polIdx++)
			   {
				   //Convert from degrees to radians and store in our temporary local array tiltx/y
				   tiltx[polIdx] = 0;
				   tilty[polIdx] = 0;
			   }
		   }

		   //The extreme wavelength ends of the wavelength array will define the size of our IFFT window.
		   //We need to make a IFFT window big enough, such that for all wavelengths, the window captures the angle-space we're interested in.
		   //As the wavelength is swept (if doing multiple wavelengths), the order we're interested in will move in Fourier space (angle of the reference beam is fixed tiltx/tilty, which corresponds to different spatial frequency kx/ky depending on what the wavelength is).
		  // const float lambdaStart = digHoloWavelength[0];
		  // const float lambdaStop = digHoloWavelength[digHoloWavelengthCount - 1];
		   //Select the maximum and minimum wavenumbers. Assumes digHoloWavelength either increases or decreases monotonically. i.e. the the edges of the digHoloWavelength array are the extreme wavelengths, the max/min wavelength isn't in the middle of the array of something weird.
		   const float k0Max = 2 * pi / (fmax(lambdaStart, lambdaStop));
		   const float k0Min = 2 * pi / (fmin(lambdaStart, lambdaStop));
		   //Dimensions of a pixel in Fourier-space.
		   const float dkx = (digHoloKXaxis[1] - digHoloKXaxis[0]);
		   const float dky = (digHoloKYaxis[1] - digHoloKYaxis[0]);

		   //The radius of the IFFT window in radians (Fourier space). We'll be selecting a cone of angles centred on tiltx/y of radius r0
		   const float r0 = digHoloApertureSize;

		   //The dimensions of FFT window, which in turn results to the size of the Fourier space we'll be selecting a window from.
		   const int heightFull = digHoloWoiPolHeight[FFT_IDX];
		   const int widthFull = digHoloWoiPolWidth[FFT_IDX];

		   //Keeps track of the extreme window bounds in pixel coordinates, over both polarisations (if both exist).
		   float minSX[] = { FLT_MAX,FLT_MAX };
		   float maxEX[] = { -FLT_MAX,-FLT_MAX };
		   float minSY[] = { FLT_MAX,FLT_MAX };
		   float maxEY[] = { -FLT_MAX,-FLT_MAX };

		   //Arrays for storing the window centre and window width for each polarisation (if both exist)
		   int woiCX[polCountMax];
		   int woiCY[polCountMax];
		   int windowWidthPol[polCountMax];
		   int windowHeightPol[polCountMax];

		   //What will become the final IFFT window width and height
		   int windowWidth = 0;
		   int windowHeight = 0;

		   //the SIMD blocksize we want to enforce. AVX processes in 256-bit chunks, meaning if we end up working with int16, this corresponds with 16 pixels.
		   //We want our window to be a multiple of this blocksize, so that we don't have to worry about memory alignment, or edge cases
		   const int blockSize = 16;

		   //For each polarisation
		   for (int polIdx = 0; polIdx < polCount; polIdx++)
		   {
			   //Find the maximum pixelIdx for a window in Fourier space of radius r0, centred at tiltx/y, at the longest wavelength. This will be one the outer edges of our window
			   maxEX[polIdx] = (float)(((tiltx[polIdx] + r0) * k0Max) / dkx + widthFull / 2.0);
			   maxEY[polIdx] = (float)(((tilty[polIdx] + r0) * k0Max) / dky + heightFull / 2.0);
			   //Find the minimum pixelIdx for a window in Fourier space of radius r0, centred at tiltx/y, at the shortest wavelength. This will be the inner edge of our window.
			   minSX[polIdx] = (float)(((tiltx[polIdx] - r0) * k0Min) / dkx + widthFull / 2.0);
			   minSY[polIdx] = (float)(((tilty[polIdx] - r0) * k0Min) / dky + heightFull / 2.0);

			   //Calculate the middle of the window to the nearest pixel by averaging the extreme edges of the window.
			   //This has also been rounded to a multiple of 2 , hence the /4.0)*2, as odd numbers were causing issues with digHoloCopyWIndow
			   woiCX[polIdx] = (int)round((maxEX[polIdx] + minSX[polIdx]) / 4.0)*2;
			   woiCY[polIdx] = (int)round((maxEY[polIdx] + minSY[polIdx]) / 4.0)*2;

			   //Calculate a window width/height based on the boundaries of the aperture
			   windowWidthPol[polIdx] = (int)(maxEX[polIdx] - minSX[polIdx] + 1);
			   windowHeightPol[polIdx] = (int)(maxEY[polIdx] - minSY[polIdx] + 1);

			   //Round-up window dimensions to multiple of 16 pixels (SIMD)
			   windowWidthPol[polIdx] = (int)(blockSize * ceil((1.0f * windowWidthPol[polIdx]) / blockSize));
			   windowHeightPol[polIdx] = (int)(blockSize * ceil((1.0f * windowHeightPol[polIdx]) / blockSize));

			   //If this is the largest window width/height we've seen thus far, remember it.
			   //Windows for each polarisation don't have to be in the same position in Fourier space, but we are constraining them to have the same dimensions.
			   if (windowWidthPol[polIdx] > windowWidth)
			   {
				   windowWidth = windowWidthPol[polIdx];
			   }
			   if (windowHeightPol[polIdx] > windowHeight)
			   {
				   windowHeight = windowHeightPol[polIdx];
			   }
		   }
		   //If we managed to get a window with dimensions of zero (should be the only possible scenario that's less than blockSize), then set it to the blockSize.
		   if (windowHeight < blockSize)
		   {
			   windowHeight = blockSize;
		   }
		   if (windowWidth < blockSize)
		   {
			   windowWidth = blockSize;
		   }

		   //Keep track of whether the IFFT window has changed position, or changed size. If it has, there could be other things that need updating
		   bool itChanged = false;

		   //For each supported polarisation.
		   for (int polIdx = 0; polIdx < polCount; polIdx++)
		   {
			   //The indices of these start/stop pixels are defined in a full, _not_ FFTshifted Fourier space. (i.e. in the form you'd usually plot in, running from negative frequencies to positive frequencies)
			 //  int cx = (int)round((CX - windowSizeX / 2) / blockSize) * blockSize;
			   //Calculate the starting index of the window (x,y).
			   int newSX = (int)(woiCX[polIdx] - windowWidth / 2.0);
			   //If the window is starting at a negative spatial frequency along the x-axis, that's forbidden, because the R2C doesn't calculate those.
			   //Constrains the window to only select positive spatial frequencies along the kx-axis.
			   if (r2c)
			   {
				   if (newSX < widthFull / 2)
				   {
					   newSX = (widthFull / 2);
				   }
			   }
			   if (newSX < 0)
			   {
				   newSX = 0;
			   }
			   //Given a starting position for the window, if the other side of the window would be off the edge of the Fourier space. Correct this, such that 1 edge of the window is on the edge of the Fourier space, and the other inner edge is windowWidth away.
			   if ((newSX + windowWidth) > widthFull)
			   {
				   newSX = widthFull - windowWidth;
			   }

			   //Check if this new window starting point along the kx-axis has changed from it's previous value.
			   if (newSX != WoiSX[polIdx]) { itChanged = true; }
			   //Store the result
			  // digHoloWoiSX[IFFT_IDX][polIdx] = newSX;
			   WoiSX[polIdx] = newSX;

			   //Same as above, for the y-axis.
			   int newSY = (int)(woiCY[polIdx] - windowHeight / 2.0);
			   //ky-axis can have negative spatial frequencies, however it can't have frequencies so negative, they're outside the Fourier space (negative columns in memory)
			   if (newSY < 0)
			   {
				  //  newSY = 0;
			   }
			   if ((newSY + windowHeight) > heightFull)
			   {
				   // newSY = heightFull - windowHeight;
			   }
			   WoiSY[polIdx] = newSY;
			   if (newSY != WoiSY[polIdx]) { itChanged = true; }
		   }

		   //Check if the window dimensions in x/y have changed, and in either case, store the results as the new window dimensions digHoloWoiPolWidth
		   if (windowWidth != windowWidthOut) { itChanged = true; }
		  // digHoloWoiPolWidth[IFFT_IDX] = windowWidth;
		   windowWidthOut = windowWidth;
		   if (windowHeight != windowHeightOut) { itChanged = true; }
		   //digHoloWoiPolHeight[IFFT_IDX] = windowHeight;
		   windowHeightOut = windowHeight;
		   return itChanged;
	   }

	   //Select the IFFT window. This is the region of the Fourier plane which will be IFFT'd to get the reconstructed field
	   //Checks for changes in parameter such as window dimensions, pol count, batch count etc, and reallocates memory/ifft plans if required.
	   //Returns a bool indicating whether the IFFT window changed position or size.
	   unsigned char digHoloIFFT_SetWindow()
	   {

		   digHoloWavelengthUpdate();
		   int r2c = 1;
		   int itChanged = digHoloIFFT_GetWindowBounds(config.zernCoefs, digHoloWoiPolWidth[IFFT_IDX], digHoloWoiPolHeight[IFFT_IDX], digHoloWoiSX[IFFT_IDX], digHoloWoiSY[IFFT_IDX], digHoloWavelength[0], digHoloWavelength[digHoloWavelengthCount - 1],r2c);

		   //Reallocate memory, replan IFFTs etc, if the IFFT dimensions or other relevant parameters have changed, such as pixel sizes, batch counts etc.
		   //The digHoloIFFTInit routine itself checks most of these parameters itself, the only reason 'itChanged' is passed to digHoloIFFTInit, is so that digHoloPixelsFourierPlaneMasked can be zeroed. 
		   //Which is only an issue when doing full-res IFFTs. If the window moves to a new location, then the Fourier space will need to be zeroed, so that the old values arne't still in there. 
		   //Low-res mode doesn't need this because it always writes the whole window to be IFFTd during digHoloCopyWindow, whereas in full-res mode, we only write a small region corresponding to the window, and the rest is  assumed to already be zeros (so we don't re-zero the whole Fourier space every iteration, only when we need to).
		   digHoloIFFTInit(itChanged);
		   return itChanged;
	   }

	   //Similar to the digHoloFFTInit routine, but for the IFFT.
//This routine checks whether parameters like IFFT dimensions, pixels size, pol counts or batch counts have changed.
//If they have then memory will need to be reallocated, IFFTs re-planned etc.
	   void digHoloIFFTInit(unsigned char somethingChanged)
	   {
		   //An consise alias for the digHoloResolutionIdx.
		   //This only works because FULLRES_IDX=FFT_IDX and LOWRES_IDX=IFFT_IDX. They're really two separate concepts though. 
		   //FFT_IDX/IFFT_IDX is associated with one of the transforms, LOWRES_IDX/FULLRES_IDX is associated with the way the IFFT is performed.
		   const int resIdx = digHoloResolutionIdx;
		   const size_t polCount = digHoloPolCount;

		   //If we're in low-resolution mode, the dimensions of the IFFT are as per the window size, digHoloWoiPolWidth[IFFT_IDX]
		   //If we're in high-resolution mode, the dimensions of the IFFT are as per the FFT [FFT_IDX]
		   const int width = digHoloWoiPolWidth[resIdx];
		   const int height = digHoloWoiPolHeight[resIdx];

		   //Check if the size of the IFFT has changed. This will trip either if the window changes size, or the resolution mode changes
		   const unsigned char sizeChanged = ((width != digHoloWoiPolWidth_Valid[IFFT_IDX]) || (height != digHoloWoiPolHeight_Valid[IFFT_IDX]));
		   //Check if the batch count has changed
		   const unsigned char batchCountChanged = digHoloBatchCOUNT != digHoloBatchCount_Valid[IFFT_IDX];
		   //Check if the average count has changed
		  // const unsigned char batchAvgCountChanged = digHoloBatchAvgCount != digHoloBatchAvgCount_Valid[IFFT_IDX];
		   //If the camera pixel size has changed, that will mean the x/y axis in the reconstructed field plane after the IFFT will need to be updated.
		   const unsigned char pixelSizeChanged = (digHoloPixelSize_Valid[IFFT_IDX] != digHoloPixelSize);
		   //If the number of polarisations has changed, then memory will have to be reallocated.
		   const unsigned char polCountChanged = digHoloPolCount_Valid[IFFT_IDX] != polCount;

		   //The total number of pixels in a batch of IFFTs
		   const size_t pixelCountIFFT = digHoloBatchCOUNT * width * height * polCount;

		   //If the dimensions of the IFFT has changed, or the size of the batch has changed, we'll have to allocate a bunch of memory for these new parameters
		   if (batchCountChanged || sizeChanged || polCountChanged)
		   {
			   //Reallocate the int16 representation of the reconstructed field.
			   //Notice the x2, that's because we allocate a single double length array that contains both the real and imaginary components as two separate chunk of the same memory space (not interleaved)
			   allocate1D(pixelCountIFFT * 2, digHoloPixelsCameraPlaneReconstructed16);
			   digHoloPixelsCameraPlaneReconstructed16R = &digHoloPixelsCameraPlaneReconstructed16[0];
			   digHoloPixelsCameraPlaneReconstructed16I = &digHoloPixelsCameraPlaneReconstructed16[pixelCountIFFT];

			   //The Fourier-space, this is simialar to digHoloPixelsFourierPlane, except it is the full Fourier space that will be transformed (i.e. it has negative kx components, where as the R2C doesn't).
			   //This is the source array of the IFFT
			   
			   allocate1D(pixelCountIFFT, digHoloPixelsFourierPlaneMasked);
			   //This is the output of the IFFT
			   allocate1D(pixelCountIFFT, digHoloPixelsCameraPlaneReconstructedTilted);

			   //Remember the batch and pol counts last time this memory was allocated.
			   digHoloBatchCount_Valid[IFFT_IDX] = digHoloBatchCOUNT;
			   digHoloPolCount_Valid[IFFT_IDX] = (int)polCount;
			   // digHoloBatchAvgCount_Valid[IFFT_IDX] = digHoloBatchAvgCount;
		   }

		   //If the IFFT size has changed, we'll have to generate FFTW plans for these new IFFT sizes
		   if (sizeChanged || polCountChanged)
		   {
			   const size_t dimX = (THREADCOUNT_MAX * DIGHOLO_PLANECOUNTCAL);
			   //If this is the first run, allocate memory for the plans. The plans are a 2D array that allows us to select how many threads, and howmany FFTs we want to employ per fftwf_execute call.
			   if (digHoloIFFTPlans == 0)
			   {
				   allocate2D(dimX, FFTW_HOWMANYMAX, digHoloIFFTPlans);
				   //Go through and do the IFFT plans for every combination of howmany and thread count.
				   for (int fftIdx = 0; fftIdx < FFTW_HOWMANYMAX; fftIdx++)
				   {
					   //Defines how many FFTs to do in a single call to 'fftwf_execute'.
					 //  const int howmany = fftIdx + 1;
					   for (int threadIdx = 0; threadIdx < dimX; threadIdx++)
					   {
						   // fftwf_plan_with_nthreads(threadIdx + 1);
						   digHoloIFFTPlans[threadIdx][fftIdx] = 0;// fftwf_plan_many_dft(rank, &n[0], howmany, in, &inembed[0], istride, idist, out, &onembed[0], ostride, odist, FFTW_BACKWARD, planMode);
					   }
				   }
			   }

			   //Go through and do the IFFT plans for every combination of howmany and thread count.
			   for (int fftIdx = 0; fftIdx < FFTW_HOWMANYMAX; fftIdx++)
			   {
				   //Defines how many FFTs to do in a single call to 'fftwf_execute'.
				 //  const int howmany = fftIdx + 1;
				   for (int threadIdx = 0; threadIdx < dimX; threadIdx++)
				   {
					   // fftwf_plan_with_nthreads(threadIdx + 1);
					   if (digHoloIFFTPlans[threadIdx][fftIdx])
					   {
						   fftwf_destroy_plan(digHoloIFFTPlans[threadIdx][fftIdx]);
					   }
					   digHoloIFFTPlans[threadIdx][fftIdx] = 0;// fftwf_plan_many_dft(rank, &n[0], howmany, in, &inembed[0], istride, idist, out, &onembed[0], ostride, odist, FFTW_BACKWARD, planMode);
				   }
			   }


			   //Remember what the IFFT size was last time it was planned. Needed so we can recognise later if it's changed.
			   digHoloWoiPolWidth_Valid[IFFT_IDX] = width;
			   digHoloWoiPolHeight_Valid[IFFT_IDX] = height;

		   }

		   //If the size of the IFFT has changed, or the pixel size on the camera has changed, the x/y axes of the reconstructed field will also have to be updated.
		   if (sizeChanged || pixelSizeChanged)
		   {
			   //If the size  changed, reallocate memory fothe x-y axes in the reconstructed field plane.
			   if (sizeChanged)
			   {
				   allocate1D(width, digHoloXaxis[LOWRES_IDX]);
				   allocate1D(height, digHoloYaxis[LOWRES_IDX]);
			   }

			   //Dimensions of the FFT
			   const int widthFFT = digHoloWoiPolWidth[FFT_IDX];
			   const int heightFFT = digHoloWoiPolHeight[FFT_IDX];
			   //Dimensions of the IFFT
			   const int widthIFFT = width;
			   const int heightIFFT = height;

			   //Get the pixel dimension in the x and y directions of the reconstructed field.
			   //Even though the pixels on the camera are assumed to be square, the pixels in the reconstructed field may not be, depending on the dimensions of the FFT and IFFT selected.
			   //If the FFT and IFFT are both square, then the output pixels in the reconstructed field would also be square. Both otherwise, they may not be.
			   const float pixelSize = digHoloPixelSize;
			   const float pixelSizeX = ((pixelSize)*widthFFT) / widthIFFT;
			   const float pixelSizeY = ((pixelSize)*heightFFT) / heightIFFT;
			  
			   //Recalculate the x/y axes in the plane of the reconstructed field. The 'false' at the end, indicates that the axes should _not_ be FFTshifted.
			   GenerateCoordinatesXY<float>(widthIFFT, heightIFFT, pixelSizeX, pixelSizeY, digHoloXaxis[LOWRES_IDX], digHoloYaxis[LOWRES_IDX], false);
			   //Remember the camera pixel size that was used to calculate the reconstructed field axes for next time
			   digHoloPixelSize_Valid[IFFT_IDX] = digHoloPixelSize;

			   //Indicate that the mode basis will need to be recalculated, because the size of reconstructed field pixels has changed.
			   digHoloBasisValid = false;
		   }

		   if (sizeChanged || pixelSizeChanged || polCountChanged || somethingChanged)
		   {
			   digHoloRefValid = false;
		   }

		   //Zero the array used as the source for the IFFT. This is only relevant to full-resolution mode, as low-resolution mode, the full IFFT space is updated every time with digHoloCopyWindow.
		   //In Full resolution mode, only a small portion of the FOurier space to be IFFTd is updated, with the assumption being that anything else outside the window has been zeroed.
		   //Hence, basically if anything changes (dimensions, or position of IFFT window), this will need to be re-zeroed.
		   if (sizeChanged || polCountChanged || somethingChanged || batchCountChanged)
		   {
			   memset(digHoloPixelsFourierPlaneMasked, 0, sizeof(complex64) * pixelCountIFFT);
		   }

	   }

	   //Process a batch of IFFTs. There are essentially 3 parts to this.
	   //digHoloCopyWindow, prepares the Fourier space ready to be IFFTd, by selecting the relevant window in Fourier space.
	   //fftw_exectute, does the IFFT itself.
	   //digHoloFieldAnalysis, analyses the results of the IFFT to get parameters of interest like centre of mass, effective area etc. (useful for alignment), but more importantly, it gets the maximum absolute values of the fields over all real/imaginary components, which is used in the conversion from float32 to int16 for the reconstructed field in applyTilt
	   //void digHoloIFFTBatch()
	//   digHoloIFFTBatch(X,X,digHoloPolCount,digHoloBatchCOUNT,digHoloIFFTPlans,r2c,digHoloBatchAvgCount,config.avgMode,

	   void digHoloIFFTBatch()
	   {
		   int r2c = 1;
		   int *SXin = digHoloWoiSX[FFT_IDX];
		   int* SYin = digHoloWoiSY[FFT_IDX];
		   int *SXout = digHoloWoiSX[IFFT_IDX];
		   int* SYout = digHoloWoiSY[IFFT_IDX];

		   int WoiWidthIn = digHoloWoiPolWidth[FFT_IDX];
		   int WoiHeightIn = digHoloWoiPolHeight[FFT_IDX];

		   int WoiWidthOut = digHoloWoiPolWidth[IFFT_IDX];
		   int WoiHeightOut = digHoloWoiPolHeight[IFFT_IDX];

		   float* BeamCentreX = config.BeamCentreX;
		   float* BeamCentreY = config.BeamCentreY;

		   float** zernCoefs = config.zernCoefs;
		   int wavelengthCount = digHoloWavelengthCount;

		   complex64* RefCalibration = 0;
		   int RefCalibrationWavelengthCount = 0;

		   if (digHoloRefCalibrationEnabled && digHoloFrameWidth == digHoloRefCalibrationWidth && digHoloFrameHeight == digHoloRefCalibrationHeight && digHoloRefCalibrationReconstructedPlane)
		   {
			   RefCalibration = digHoloRefCalibrationReconstructedPlane;
			   RefCalibrationWavelengthCount = digHoloRefCalibrationWavelengthCount;
		   }
		   float normFactor = 1.0;

		   digHoloIFFTBatch(digHoloPixelsFourierPlane, digHoloPixelsFourierPlaneMasked, digHoloPixelsCameraPlaneReconstructedTilted, digHoloPolCount, digHoloBatchCOUNT, digHoloIFFTPlans, r2c, 
			   digHoloBatchAvgCount, config.avgMode, SXin, SYin,SXout,SYout,WoiWidthIn,WoiHeightIn,WoiWidthOut,WoiHeightOut, BeamCentreX, BeamCentreY, zernCoefs,wavelengthCount,RefCalibration,RefCalibrationWavelengthCount,normFactor);
	   }

	   void digHoloIFFTBatchCal()
	   {
		   int r2c = 0;
		   complex64* fourierPlane = digHoloRefCalibrationFourierPlane;
		   complex64* fourierPlaneMasked = digHoloRefCalibrationFourierPlaneMasked;
		   complex64* reconstructedPlane = digHoloRefCalibrationReconstructedPlane;
		   int polCount = digHoloPolCount;
		   int batchCount = digHoloRefCalibrationWavelengthCount;
		   fftwf_plan** FFTPlans = &digHoloIFFTPlans[THREADCOUNT_MAX];
		   int avgCount = 1;
		   int avgMode = 0;

		   int* SXin = digHoloWoiSX[FFTCAL_IDX];
		   int* SYin = digHoloWoiSY[FFTCAL_IDX];
		   int* SXout = digHoloWoiSX[IFFTCAL_IDX];
		   int* SYout = digHoloWoiSY[IFFTCAL_IDX];

		   int WoiWidthIn = digHoloWoiPolWidth[FFTCAL_IDX];
		   int WoiHeightIn = digHoloWoiPolHeight[FFTCAL_IDX];

		   int WoiWidthOut = digHoloWoiPolWidth[IFFTCAL_IDX];
		   int WoiHeightOut = digHoloWoiPolHeight[IFFTCAL_IDX];

		   float BeamCentre[DIGHOLO_POLCOUNTMAX*2];
		   memset(&BeamCentre[0], 0, sizeof(float) * 2 * DIGHOLO_POLCOUNTMAX);
		   float* BeamCentreX = &BeamCentre[0];// config.BeamCentreX;
		   float* BeamCentreY = &BeamCentre[DIGHOLO_POLCOUNTMAX];// config.BeamCentreY;

		   float** zernCoefs = 0;// config.zernCoefs;
		   int wavelengthCount = digHoloWavelengthCount;

		   complex64* RefCalibration = 0;
		   int RefCalibrationWavelengthCount = 0;

		   const int resIdx = digHoloResolutionIdx;
		   float normFactor = sqrtf((float)((1.0 * digHoloWoiPolWidth[resIdx] * digHoloWoiPolHeight[resIdx]) / (1.0 * digHoloWoiPolWidth[FFT_IDX] * digHoloWoiPolHeight[FFT_IDX])));

		   digHoloIFFTBatch(fourierPlane, fourierPlaneMasked, reconstructedPlane, polCount, batchCount, FFTPlans, r2c,
			   avgCount, avgMode, SXin, SYin, SXout, SYout, WoiWidthIn, WoiHeightIn, WoiWidthOut, WoiHeightOut, BeamCentreX, BeamCentreY, zernCoefs, wavelengthCount,RefCalibration,RefCalibrationWavelengthCount,normFactor);
		   
	   }

	   void digHoloIFFTBatch(complex64* bufferInFull, complex64* bufferInMasked, complex64* bufferOut, int polCount, int batchCount, fftwf_plan** FFTPlans, 
		   int r2c, int avgCount, int avgMode, int *SXin, int *SYin, int*SXout, int*SYout, int WoiWidthIn, int WoiHeightIn, int WoiWidthOut, int WoiHeightOut, float*BeamCentreX, float* BeamCentreY, float** zernCoefs, int wavelengthCount, complex64* RefCalibration, int RefCalibrationWavelengthCount,float normFactor)
	   {
		   //Concise aliases for the relevant parameters.
		   //However the IFFT is performed, depends on how many polarisations and batch items are active, as well as whether we're working in full-resolution mode (where the IFFT is the same dimensions as the FFT), or window-only low-resolution mode, where the IFFT is only the size of the actual part of Fourier space we're interested in.
		   //const int polCount = digHoloPolCount;
		  // const int batchCount = digHoloBatchCOUNT;
		   // const int resolutionIdx = digHoloResolutionIdx;
		  // const int avgCount = digHoloBatchAvgCount;
		  // const int avgMode = config.avgMode;//This should be the only place where the averaging mode needs to be read. It's only required so that the averaging routine knows in what order to address FourierPlanes that should be averaged together. Contiguous vs. interlaced. i.e. C vs. Fortran. In contiguous blocks (0) of digHoloBatchAvgCount [A A A B B B C C C..], or interlaced (1) [A B C A B C A B C]
		   const int resolutionMode = digHoloResolutionIdx;
		   //If we have more than 1 batched IFFT to perform, or if there's two polarisations, then we'll be launching threads to process these.
		   if (batchCount > 1 || polCount == 2)
		   {
			   //Total number of threads launched will be polThreads*batchThreads, with each thread executing an FFTW using fftThreads
			   int fftThreads = 1;//The number of thread per fftwf_execute call
			   int polThreads = 1;//the dimensions of the thread block in the polarisation direction
			   int batchThreads = 1;//the dimensions of the thread bloc in the batch direction.


			   //A single camera frame
			   if (batchCount == 1)
			   {
				   //Launch 1 thread for each polarisation
				   polThreads = polCount;
				   //No threads assigned to batch dimension, because there's only a single frame to process
				   batchThreads = 1;
				   //Each individual fftwf_execute should use all the CPU cores if there's only 1 polarisation, and 1/2 the cores if there's two polarisations
				   fftThreads = digHoloThreadCount / polCount;
			   }
			   else
			   {
				   //Launch threads for each polarisation, and how ever many batch threads as it takes to use up all the CPU cores.
				   polThreads = polCount;
				   batchThreads = digHoloThreadCount / polCount;
				   //Each individual fftwf_execute will the single threaded.
				   fftThreads = 1;
			   }

			   //Check for strange scenarios where there would be negative or zero threads assigned.
			   //Could happen if you've only got 1 CPU core and 2 polarisations (what year is this?)
			   if (fftThreads < 1)
			   {
				   fftThreads = 1;
			   }
			   if (batchThreads < 1)
			   {
				   batchThreads = 1;
			   }
			   if (polThreads < 1)
			   {
				   polThreads = 1;
			   }
			   if (polThreads > polCount)
			   {
				   polThreads = polCount;
			   }
			   if (batchThreads > batchCount)
			   {
				   batchThreads = batchCount;
			   }

			   //The total number of threads we'll be launching to handle all the polarisation and batch items.
			   const int totalThreads = polThreads * batchThreads;

			   //Approximately evenly distributed batch items per thread.
			   const int batchEach = (int)ceil((1.0 * batchCount / (batchThreads)));

			   //The current thread being launched.
			   int threadIdx = 0;
			   //For each polarisation...
			   for (int polIdx = 0; polIdx < polThreads; polIdx++)
			   {
				   //Launch batchThreads worth of threads
				   for (int batchIdx = 0; batchIdx < batchThreads; batchIdx++)
				   {
					   //Concise alias of threadIdx
					   const int j = threadIdx;

					   //Setup the start and stop batch items for this thread to process.
					   //Check for scenario where the stop index is out of bounds.
					   const int startIdx = batchEach * batchIdx;
					   int stopIdx = batchEach * (batchIdx + 1);
					   if (stopIdx > batchCount)
					   {
						   stopIdx = batchCount;
					   }

					   //This section double-checks that we have all the FFT plans we're going to need
					   //The total number of batch items to process
					   int batchStep = stopIdx - startIdx;
					   //Check if there's more batch items than we can handle with a single 'fftwf_execute' command.
					   //If there is, we'll have to call fftwf_execute multiple times in a loop.
					   //Realistically, the 'plan_many' type of fftwf_execute that does several FFT/IFFTs in a single execution is theoretically faster, but in practice seems to make little difference.
					   if (batchStep > FFTW_HOWMANYMAX)
					   {
						   batchStep = FFTW_HOWMANYMAX;
					   }

					   //For every batch item
					   for (int batchIdx = startIdx; batchIdx < stopIdx; batchIdx += batchStep)
					   {
						   //Do this many IFFTs per fftwf_execute call
						   int howmany = batchStep;
						   //Unless that's too many, that would go beyond the bounds specified by stopIdx, in which case do however many it takes to get to the boundary, but no further.
						   if ((batchIdx + batchStep) > stopIdx)
						   {
							   howmany = stopIdx - batchIdx;
						   }
						   if (howmany > 0) //How could that happen?
						   {
							   if (!FFTPlans[fftThreads - 1][howmany - 1])
							   {
								   digHoloIFFTPlan(fftThreads - 1, howmany - 1, &FFTPlans[fftThreads - 1][howmany - 1]);
							   }
						   }
					   }

					   workPackage* workPack = digHoloWorkPacks[j];

					   workPack[0].start = startIdx;
					   workPack[0].stop = stopIdx;

					   //Tell the thread which polarisation component to do for each batch item, and how many threads to use per fftwf_execute.
					   workPack[0].flag1 = polIdx;
					   workPack[0].flag2 = fftThreads;
					   workPack[0].flag3 = avgCount;
					   workPack[0].flag4 = avgMode;

					   workPack[0].flag5 = batchCount;
					   workPack[0].flag6 = wavelengthCount;

					   workPack[0].ptr1 = bufferInFull;
					   workPack[0].ptr2 = bufferInMasked;

					   workPack[0].ptr3 = bufferOut;

					   workPack[0].ptr4 = zernCoefs;

					   workPack[0].ptr5 = FFTPlans;

					   workPack[0].ptr6 = SXin;
					   workPack[0].ptr7 = SYin;

					   workPack[0].ptr8 = SXout;
					   workPack[0].ptr9 = SYout;

					   workPack[0].ptr10 = RefCalibration;
					   complex64* batchCalibration = 0;
					   if (digHoloBatchCalibration && digHoloBatchCalibrationEnabled)
					   {
						   batchCalibration = digHoloBatchCalibration[polIdx % digHoloBatchCalibrationPolCount];
					   }
					   workPack[0].ptr11 = batchCalibration;
					   workPack[0].idx1 = digHoloBatchCalibrationBatchCount;

					   workPack[0].flag7 = WoiWidthIn;
					   workPack[0].flag8 = WoiHeightIn;

					   workPack[0].flag9 = WoiWidthOut;
					   workPack[0].flag10 = WoiHeightOut;

					   workPack[0].flag11 = polCount;
					   workPack[0].flag12 = resolutionMode;

					   workPack[0].flag13 = r2c;
					   workPack[0].flag14 = RefCalibrationWavelengthCount;
					  

						workPack[0].var1 = BeamCentreX[polIdx] * DIGHOLO_UNIT_PIXEL;// (float)(config.BeamCentreX[polIdx] * DIGHOLO_UNIT_PIXEL);
						workPack[0].var2 = BeamCentreY[polIdx] * DIGHOLO_UNIT_PIXEL; //(float)(config.BeamCentreY[polIdx] * DIGHOLO_UNIT_PIXEL);
						workPack[0].var3 = normFactor;

					   workPack[0].callback = workFunction::digHoloIFFTRoutineWorker;
					   //Reset the "I'm done" handle event of the thread and launch.
					   workPack[0].workCompleteEvent.Reset();
					   workPack[0].workNewEvent.Set();
					   // ThreadPool::QueueUserWorkItem(slmCallback, workPack[j]);

					   threadIdx++;
				   }
			   }
			   //Wait for threads to finish
			   for (int j = 0; j < totalThreads; j++)
			   {
				   workPackage* workPack = digHoloWorkPacks[j];
				   workPack[0].workCompleteEvent.WaitOne();
			   }
		   }
		   else
		   {
			   if (!FFTPlans[digHoloThreadCount - 1][0])
			   {
				   digHoloIFFTPlan(digHoloThreadCount - 1, 0,&FFTPlans[digHoloThreadCount - 1][0]);
			   }
			   //If there's only a single polarisation, and a single batch item, then don't launch any threads, just do the IFFT routine.
			  // digHoloIFFTRoutine(digHoloThreadCount, 0, 1, 0, avgCount, avgMode);
			   const int polIdx = 0;

			   complex64* batchCalibration = 0;
			   if (digHoloBatchCalibration && digHoloBatchCalibrationEnabled)
			   {
				   batchCalibration = digHoloBatchCalibration[polIdx % digHoloBatchCalibrationPolCount];
			   }

			   digHoloIFFTRoutine(digHoloThreadCount, 0, 1, polIdx, avgCount, avgMode, batchCount, wavelengthCount, bufferInFull, bufferInMasked, bufferOut,
				   BeamCentreX[polIdx], BeamCentreY[polIdx], polCount,
				   SXin, SYin, SXout, SYout, WoiWidthOut, WoiHeightOut, WoiWidthIn, WoiHeightIn, resolutionMode, zernCoefs, FFTPlans, r2c,RefCalibration,RefCalibrationWavelengthCount,normFactor, batchCalibration,digHoloBatchCalibrationBatchCount);
		   }
	   }

	   //This is essentially an overloaded version of digHoloIFFTRoutine, that inputs a workpackage for use as a thread, rather than individual method arguments for single-threaded operation.
	   void digHoloIFFTRoutineWorker(workPackage& e)
	   {
		   //workPackage^ e = (workPackage^)o;
		   const int startIdx = (int)e.start;
		   const int stopIdx = (int)e.stop;

		   const int polIdx = e.flag1;
		   const int fftThreads = e.flag2;
		   const int avgCount = (int)e.flag3;
		   const int avgMode = (int)e.flag4;

		   const int batchCount = e.flag5;// digHoloBatchCOUNT
		   const int lambdaCount = e.flag6;//digHoloWavelengthCount
		   complex64* fourierPlane = (complex64*)e.ptr1;// digHoloPixelsFourierPlane
		   complex64* fourierPlaneMasked = (complex64*)e.ptr2;//digHoloPixelsFourierPlaneMasked
		   complex64* cameraPlaneReconstructedTilted = (complex64*)e.ptr3;//digHoloPixelsCameraPlaneReconstructedTilted

		   float** zernCoefs = (float**)e.ptr4;
		   fftwf_plan** fftPlans = (fftwf_plan**)e.ptr5;

		   int* WoiSXin = (int*)e.ptr6;//digHoloWoiSX[FFT_IDX]
		   int* WoiSYin = (int*)e.ptr7;

		   int* WoiSXout = (int*)e.ptr8;//digHoloWoiSX[IFFT_IDX]
		   int* WoiSYout = (int*)e.ptr9;

		   complex64* RefCalibration = (complex64*)e.ptr10;
		   complex64* batchCalibration = (complex64*)e.ptr11;
		   int batchCalibrationCount = (int)e.idx1;

		   const int WoiWidthIn = e.flag7;//digHoloWoiPolWidth[FFT_IDX]
		   const int WoiHeightIn = e.flag8;
		   const int WoiWidthOut = e.flag9;//digHoloWoiPolWidth[IFFT_IDX]
		   const int WoiHeightOut = e.flag10;
		   const int polCount = e.flag11;
		   const int resolutionMode = e.flag12;//digHoloResolutionIdx
		   const int r2c = e.flag13;
		   const int RefCalibrationWavelengthCount = e.flag14;

		   float beamCX = e.var1;//(float)(config.BeamCentreX[polIdx] * DIGHOLO_UNIT_PIXEL)
		   float beamCY = e.var2;
		   float norm0 = e.var3;

		   digHoloIFFTRoutine(fftThreads, startIdx, stopIdx, polIdx, avgCount, avgMode, batchCount, lambdaCount, 
			   fourierPlane, fourierPlaneMasked, cameraPlaneReconstructedTilted,
			   beamCX, beamCY, polCount,
			   WoiSXin, WoiSYin, WoiSXout, WoiSYout, WoiWidthOut, WoiHeightOut, WoiWidthIn, WoiHeightIn, resolutionMode, zernCoefs, fftPlans, r2c, RefCalibration, RefCalibrationWavelengthCount,norm0,batchCalibration,batchCalibrationCount);

	   }

	   //The main worker routine for the IFFT. Given a polarisation, a range of batch items, and the number of FFT threads to use per fftwf_execute, this routine performs IFFTs
	   void digHoloIFFTRoutine(int fftThreads, int batchStartIdx, int batchStopIdx, int polIdx, int avgCount, int avgMode, int batchCount, int wavelengthCount, 
		   complex64* fourierPlane, complex64* fourierPlaneMasked, complex64* cameraPlaneReconstructedTilted,
		   float beamCX, float beamCY, int polCount, 
		   int *WoiSXin, int *WoiSYin, int *WoiSXout, int *WoiSYout, int WoiWidthOut, int WoiHeightOut, int WoiWidthIn, int WoiHeightIn, int resolutionMode, float **zernCoefs, fftwf_plan** FFTPlans, int r2c, 
		   complex64* RefCalibration, int RefCalibrationWavelengthCount,float normFactor, complex64* batchCalibration,int batchCalibrationCount)
	   {
		   //The start/stop batch items to process [start->(stopIdx-1)]
		   const int startIdx = batchStartIdx;
		   const int stopIdx = batchStopIdx;
		   //The total number of batch elements. The routine only needs to know this if avgMode=1, in which case it need to know where the end of batches [A B C A B C] are in memory.
		   //const int batchCount = digHoloBatchCOUNT;
		   //const int lambdaCount = digHoloWavelengthCount;
		   //The output of the FFT (Fourier plane, Real-to-complex format, not IFFT window applied)
		  // complex64* fourierPlane = digHoloPixelsFourierPlane;
		   //The source of the IFFT (Fourier plane, full complex format, IFFT window applied/fftshifted)
		  // complex64* fourierPlaneMasked = digHoloPixelsFourierPlaneMasked;
		   //The destination of the IFFT (Reconstructed field, with the tilt/defocus of the reference arm on-top {will be removed later})
		  // complex64* cameraPlaneReconstructedTilted = digHoloPixelsCameraPlaneReconstructedTilted;

		   const size_t strideIn = r2c? ((WoiWidthIn / 2 + 1) * WoiHeightIn): ((WoiWidthIn) * WoiHeightIn);

		   size_t strideOut = (resolutionMode==FULLRES_IDX)?(WoiWidthIn*WoiHeightIn):(WoiWidthOut*WoiHeightOut);

		   //These memory strides will be used for copying and manipulating (tilt+fftshift) the relevant window of the Fourier plane from the output of the FFT to the input of the IFFT
		   //Length of individual FFT outputs (R2C format).
		   //const size_t strideIn = (digHoloWoiPolWidth[FFT_IDX] / 2 + 1) * digHoloWoiPolHeight[FFT_IDX];
		   //Length of individual IFFT input/outputs (full complex format). Size will depend on whether we're in full-resolution or low-resolution mode.
		   //const size_t strideOut = digHoloWoiPolWidth[digHoloResolutionIdx] * digHoloWoiPolHeight[digHoloResolutionIdx];

		   //The centre of the beam on the camera. This is defined in absolute terms on the camera, rather than relative to the FFT window
		   //We'll need to know where the centre of the beam is relative to the centre of the FFT window, in order to know how much tilt to apply in the Fourier domain, to centre the beam in the IFFT reconstructed plane.
		  // const float beamCX = (float)(config.BeamCentreX[polIdx] * DIGHOLO_UNIT_PIXEL);
		  // const float beamCY = (float)(config.BeamCentreY[polIdx] * DIGHOLO_UNIT_PIXEL);

		   //FFT window start/stop corner position, and the width/height of the window
		   const int cx = WoiSXin[polIdx];
		   const int cy = WoiSYin[polIdx];
		   const int windowSizeX = WoiWidthIn;// digHoloWoiPolWidth[FFT_IDX];
		   const int windowSizeY = WoiHeightIn;// digHoloWoiPolHeight[FFT_IDX];

		   //The total size of the camera frames
		   const int frameWidth = digHoloFrameWidth;
		   const int frameHeight = digHoloFrameHeight;

		   //Work out where the centre of the FFT window is as a pixel coordinate.
		   int cx0 = (cx + windowSizeX / 2) - frameWidth / 2;
		   // int cy0 = (cy + windowSizeY / 2) - frameHeight / (2 * digHoloPolCount);
		   int cy0 = (cy + windowSizeY / 2) - frameHeight / (2);
		   
		   //Calculate the beam centre in metres relative to the FFT window itself. When we apply a tilt in the Fourier plane, we don't need to centre the beam relative to the centre of the camera, we need to centre it relative to the FFT window.
		   const float pixelSize = digHoloPixelSize;
		   //The absence of zernCoefs here has also been used to signal that no beam centering will be applied. i.e. when we're doing field reconstruction, we want to centre the output. But if we're FFT/IFFT the Reference wave calibration, we do not want to recentre.
		   const float beamCentreX = zernCoefs?(beamCX - (cx0)*pixelSize):0;
		   const float beamCentreY = zernCoefs ?(beamCY - (cy0)*pixelSize):0;

		   //The tilt of the reference wave, and the aperture size.
		   //We'll be applying a filter in Fourier-space to select only the spatial frequencies corresponding to the cone of digHoloAperture angle around the tiltx/y position.
		   //The digHoloIFFT_SetWindow provides a rectangular aperture, that's been rounded/shifted slightly to comply with AVX considerations. This additional elliptical filtering ensures we only select the relevant spatial frequencies, a remove some noise.
		   //It makes little difference versus just using a raw rectangular aperture really. Could be more useful for very broad wavelength sweeps, as it means the IFFT window can be fixed, and relatively large to support the wavelength sweep, but only the relevant window in angle-space is selected for each wavelength.
		   //The rectangular window doesn't move with wavelength, the elliptical filter of angles does.
		   //float** zernCoefs = config.zernCoefs;
		   const float tiltX = zernCoefs ? (float)(zernCoefs[polIdx][TILTX] * DIGHOLO_UNIT_ANGLE) : 0;
		   const float tiltY = zernCoefs ? (float)(zernCoefs[polIdx][TILTY] * DIGHOLO_UNIT_ANGLE) : 0;

		   const float apertureSize = digHoloApertureSize;
		   
		   const float sinTiltX = sinf(tiltX);
		   const float sinTiltY = sinf(tiltY);
		   const float sinApertureSize = sinf(apertureSize);

		   //Specifies how the wavelength axis is organised in the input data, and how it should be organise when written out
		   const int orderingIn = config.wavelengthOrdering[DIGHOLO_WAVELENGTHORDER_INPUT];
		   const int orderingOut = config.wavelengthOrdering[DIGHOLO_WAVELENGTHORDER_OUTPUT];

		   if (apertureSize > 0)
		   {
			   int noPowerWarning = 0;
			   //For every assigned batch item.
			   for (int batchIdx = startIdx; batchIdx < stopIdx; batchIdx++)
			   {
				   complex64* batchCal = 0;
				   if (batchCalibration && batchCalibrationCount > 0)
				   {
					   batchCal = &batchCalibration[batchIdx % batchCalibrationCount];
				   }
				   for (int avgIdx = 0; avgIdx < avgCount; avgIdx++)
				   {
					   //Get the wavelength assigned to this batch item.
					   //const int lambdaIdx = batchIdx % digHoloWavelengthCount;//If theres more batch items than wavelengths, this will wrap around (e.g. if you're doing sweeps for multiple input modes as a single batch).
					   int lambdaIdx;
					   int subBatchIdx;
					   int batchIdxOut = WavelengthOrderingCalc(batchIdx, wavelengthCount, batchCount, orderingIn, orderingOut, lambdaIdx, subBatchIdx);
					   //Get corresponding wavenumber.
					   const float k0 = 2 * pi / digHoloWavelength[lambdaIdx];
					   //Get the centre of the IFFT window as a kx/ky value (small-angle approximation, sin(x) = x)
					   const float kx0 = k0 * sinTiltX;
					   const float ky0 = k0 * sinTiltY;
					   //Get the radius of the IFFT window as a k-space value (again small-angle approximation)
					   const float kr = k0 * sinApertureSize;

					   //The position in the source array (the output of the FFT, R2C format), we'll be copying the window from.
					   //The avgMode defines whether we'll be averaging Fourier windows in contiguous blocks (0), [A A A B B B C C C] or interlaced (1), [A B C A B C A B C]
					   size_t frameOffsetIn = 0;//const size_t frameOffsetIn = (avgMode == 0) ? ((batchIdx * avgCount + avgIdx) * digHoloPolCount + polIdx) * strideIn : (((batchIdx * digHoloPolCount + polIdx) + (avgIdx * batchCount * digHoloPolCount)) * strideIn);
					   switch (avgMode)
					   {
					   case DIGHOLO_AVGMODE_SEQUENTIAL: //Frames to be averaged are adjacent [A A A B B B C C C]
						   frameOffsetIn = ((batchIdx * avgCount + avgIdx) * polCount + polIdx) * strideIn;
						   break;
					   case DIGHOLO_AVGMODE_INTERLACED: //Frames to be averaged are batchCount apart (interlaced) [A B C A B C A B C]. i.e. the batch is duplicated as a whole multiple times in adjacent memory
						   frameOffsetIn = (((batchIdx * polCount + polIdx) + (avgIdx * batchCount * polCount)) * strideIn);
						   break;
					   case DIGHOLO_AVGMODE_SEQUENTIALSWEEP: //Similar to 0, except rather than individual frames being adjacent in memory, it's entire wavelength sweeps. [{A} {A} {A} {B} {B} {B} {C} {C} {C}]. If lambdaCount==1, then this mode is the same as 0.
						   frameOffsetIn = (((batchIdx * polCount + polIdx) + (avgIdx * wavelengthCount * polCount)) * strideIn);
						   break;
					   }
					   //The position in the destination array (the input to the IFFT, full-complex format), we'll be copying the IFFT window into this array.
					   size_t frameOffsetOut = (batchIdxOut * polCount + polIdx) * strideOut;

					   //The x/y axis in Fourier space
					   float* kxAxis = &digHoloKXaxis[0];
					   float* kyAxis = &digHoloKYaxis[0];

					   int fillFactorCorrection = config.fillFactorCorrection;

					   //Copy the relevant portion of the Fourier space from the output of the FFT (digHoloPixelsFourierPlane) to the input of the IFFT (digHoloPixelsCameraPlaneReconstructedTilted)
					   //In the process, beam centring (tilt in the Fourier plane) will be applied in a wavelength-dependent fashion. As will an elliptical spatial filter, to zero any angles outside the specified digHoloApertureSize
					   //This process occurs slightly differently whether we're operating in low-resolution mode, or full-resolution mode. As the destination in memory that the result must be written to is different for the two scenarios (low resolution writes to the start of the array, full-resolution writes somewhere else, and there's an FFTshift in there)
					   //float pwr = digHoloCopyWindow(digHoloWoiSX[IFFT_IDX][polIdx], digHoloWoiSY[IFFT_IDX][polIdx], digHoloWoiPolWidth[IFFT_IDX], digHoloWoiPolHeight[IFFT_IDX], digHoloWoiPolWidth[FFT_IDX], digHoloWoiPolHeight[FFT_IDX], (complex64*)&fourierPlaneMasked[frameOffsetOut][0], (complex64*)&fourierPlane[frameOffsetIn][0], (digHoloResolutionIdx == LOWRES_IDX), kxAxis, kyAxis, beamCentreX, beamCentreY, kx0, ky0, kr, avgIdx, avgCount);
					   float pwr = digHoloCopyWindow(WoiSXout[polIdx], WoiSYout[polIdx], WoiWidthOut, WoiHeightOut, WoiWidthIn, WoiHeightIn, (complex64*)&fourierPlaneMasked[frameOffsetOut][0], (complex64*)&fourierPlane[frameOffsetIn][0], (resolutionMode == LOWRES_IDX), kxAxis, kyAxis, beamCentreX, beamCentreY, kx0, ky0, kr, avgIdx, avgCount, r2c, normFactor, batchCal, pixelSize, fillFactorCorrection);

					   //A check to see if there's actually any power in the IFFT window we just created. This should also trigger if there' NaN.
					   //This should never occur, if it does, something's gone wrong. It's printed here for debugging purposes.
					   if (!(pwr > 0))
					   {
						   // System::Diagnostics::Debug::WriteLine("Warning : no power in IFFT window!	"+batchIdx + "	" + pwr);
						   noPowerWarning++;
					   }
				   }
			   }

			   if (noPowerWarning && (config.verbosity >= DIGHOLO_VERBOSITY_BASIC))
			   {
				   fprintf(consoleOut, "Warning : no power in %i IFFT windows. Tilt(x,y) = %f,%f\n\r", noPowerWarning, (tiltX / (DIGHOLO_UNIT_ANGLE)), (tiltY / (DIGHOLO_UNIT_ANGLE)));
				   fflush(consoleOut);
			   }
		   }
		   //Now that the input to the IFFT has been set up with the digHoloCopyWindow routine, we can now do the IFFTs

		   //The total number of batch items to process
		   int batchStep = stopIdx - startIdx;
		   //Check if there's more batch items than we can handle with a single 'fftwf_execute' command.
		   //If there is, we'll have to call fftwf_execute multiple times in a loop.
		   //Realistically, the 'plan_many' type of fftwf_execute that does several FFT/IFFTs in a single execution is theoretically faster, but in practice seems to make little difference.
		   if (batchStep > FFTW_HOWMANYMAX)
		   {
			   batchStep = FFTW_HOWMANYMAX;
		   }

		   //For every batch item
		   for (int batchIdx = startIdx; batchIdx < stopIdx; batchIdx += batchStep)
		   {
			   //Do this many IFFTs per fftwf_execute call
			   int howmany = batchStep;
			   //Unless that's too many, that would go beyond the bounds specified by stopIdx, in which case do however many it takes to get to the boundary, but no further.
			   if ((batchIdx + batchStep) > stopIdx)
			   {
				   howmany = stopIdx - batchIdx;
			   }

			   //The IFFT plan corresponding to the specified number of IFFT threads, and the number of IFFTs per execute call.
			   const fftwf_plan fftPlan = FFTPlans[fftThreads - 1][howmany - 1];

			   //The input/output location corresponding to this batch item in this polarisation.
			   const size_t frameOffsetOut = (batchIdx * polCount + polIdx) * strideOut;

			   //Perform the IFFT
			   fftwf_execute_dft(fftPlan, &fourierPlaneMasked[frameOffsetOut], &cameraPlaneReconstructedTilted[frameOffsetOut]);

			   if (RefCalibration && RefCalibrationWavelengthCount > 0)
			   {
				   for (int bIdx = 0; (bIdx < howmany); bIdx++)
				   {
					   const size_t batchIDX = batchIdx + bIdx;

					   const size_t idx = ((batchIDX) * polCount + polIdx) * strideOut;
					   complex64* field = &cameraPlaneReconstructedTilted[idx];

					   int lambdaIdx;
					   int subBatchIdx;
					   //int batchIdxOut = 
						WavelengthOrderingCalc(batchIdx, wavelengthCount, batchCount, orderingIn, orderingOut, lambdaIdx, subBatchIdx);

					   lambdaIdx = lambdaIdx % digHoloRefCalibrationWavelengthCount;

					   const size_t calIdx = ((lambdaIdx) * polCount + polIdx) * strideOut;
					   complex64* cal = &RefCalibration[calIdx];

					   const size_t length = strideOut;
					   digHoloApplyRefCalibration(field, cal, length);
				   }
			   }
		   }
	   }

	   //Timers used for benchmarking the ApplyTiltRoutine.
	   int64_t benchmarkApplyTiltCounter = 0;
	   std::chrono::duration<double> benchmarkApplyTiltTime = std::chrono::duration<double>(0);

	   int64_t benchmarkIFFTAnalysisCounter = 0;
	   std::chrono::duration<double> benchmarkIFFTAnalysisTime = std::chrono::duration<double>(0);


	   void digHoloApplyRefCalibration(complex64* field, complex64* calibration, size_t length)
	   {
		   const size_t blockSize = 4;
		 //  const int resIdx = digHoloResolutionIdx;
		 //  float normFactor = sqrtf((1.0*digHoloWoiPolWidth[resIdx] * digHoloWoiPolHeight[resIdx]) / (1.0*digHoloWoiPolWidth[FFT_IDX] * digHoloWoiPolHeight[FFT_IDX]));
		 //  __m256 normF = _mm256_set1_ps(normFactor);
		   for (size_t idx = 0; idx < length; idx += blockSize)
		   {
			   
			   __m256 f = _mm256_loadu_ps(&field[idx][0]);
			   __m256 c = _mm256_loadu_ps(&calibration[idx][0]);
			   f = cmul(f, c);
			   _mm256_storeu_ps(&field[idx][0], f);
		   }
	   }

	   //The digHoloApplyTilt routine removes the tilt/focus of the reference wave from the reconstructed field, and also converts the field to an int16
	   //This is the default overloaded version of the method, which does both polarisation components (-1), and does not overwrite the source array with an 32-bit float version of the field (only outputs the 16-bit versions)
	   int digHoloApplyTilt()
	   {
		   return digHoloApplyTilt(-1, 0);
	   }

	   //As above, removes tilt/focus from the reconstructed field, in a wavelength dependent fashion, as well as converting the field to int16 format.
	   //Input arguments specify which polarisation to process (-1 means process both) and if overwrite it set, the method will also overwrite the source array with a 32-bit float complex version of the field with the tilt/focus removed.
	   //Specifying outArray32 will enable a float32 version of the field to be written (instead of just the int16 version of the field)
	   int digHoloApplyTilt(int polIdx0, complex64* outArray32)
	   {
		   if (digHoloPixelsCameraPlaneReconstructedTilted)
		   {
			   //Timer used for benchmarking performance of the digHoloApplyTilt routine
			  // const int64_t startTime = QPC();
			   std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

			   //If we're in full-resolution mode, the dimensions of the field will be that of the FFT
			   //If we're in low-resolution mode, the dimensions of the field with be that of the IFFT (window-only size)
			   const int fftIdx = digHoloResolutionIdx;
			   //Dimensions in x and y for a single frame in a single pol
			   const int pixelCountX = digHoloWoiPolWidth[fftIdx];
			   const int pixelCountY = digHoloWoiPolHeight[fftIdx];

			   //Setup the start/stop polarisation indices base on which polarisation(s) we'll be processing.
			   //Default is to process both polarisations
			   int polStart = 0;
			   int polStop = digHoloPolCount;
			   int polCount = digHoloPolCount;

			   //Just process the H polarisation
			   if (polIdx0 == 0)
			   {
				   polStart = 0;
				   polStop = 1;
				   polCount = 1;
			   }
			   else
			   {
				   //Just process the V polarisation
				   if (polIdx0 == 1)
				   {
					   polStart = 1;
					   polStop = 2;
					   polCount = 1;
				   }
			   }

			   //The X and Y axes of the reconstructed field
			   float* X = &digHoloXaxis[IFFT_IDX][0];
			   float* Y = &digHoloYaxis[IFFT_IDX][0];

			   //The number of batch items, and the distance between them in memory
			   const int batchCount = digHoloBatchCOUNT;
			   const size_t batchStride = ((size_t)pixelCountX) * pixelCountY * polCount;

			   //Approximate number of threads to launch total
			   const int threadCountDefault = digHoloThreadCount;

			   //The number of threads to launch along the polarisation dimension
			   int polThreads = polCount;

			   //The number of thread to launch along the pixel dimension
			   int workThreads = (threadCountDefault) / polCount;

			   //Double-check that we'll be launching at least 1 thread (could happen on single-core machine for dual-pol)
			   if (polThreads < 1)
			   {
				   polThreads = 1;
			   }
			   if (workThreads < 1)
			   {
				   workThreads = 1;
			   }

			   //Analyse the full untilted field (windowR=0, means analysis the whole field, i.e. analyses everything outside an aperture of radius zero, at position cx=0,cy=0)
		
				   const float windowCX = 0;
				   const float windowCY = 0;
				   const float windowR = 0;
				   const size_t polStride = ((size_t)pixelCountX) * ((size_t)pixelCountY);
				   
				   std::chrono::steady_clock::time_point startTimeA = std::chrono::steady_clock::now();
				   digHoloFieldAnalysisRoutine(X, Y, &digHoloPixelsCameraPlaneReconstructedTilted[0], polStart, polStop, IFFT_IDX, pixelCountX, pixelCountY, batchCount, workThreads, batchStride, polStride, windowCX, windowCY, windowR, false, false);
				   std::chrono::steady_clock::time_point stopTimeA = std::chrono::steady_clock::now();
				   benchmarkIFFTAnalysisTime += (stopTimeA - startTimeA);
				   benchmarkIFFTAnalysisCounter++;
			   
			   //Approximately how many pixels to assign to each thread along the y-axis (rounded up to a value of 8, for SIMD/AVX reasons)
			   const int pixelsEach = (int)(8 * ceil((1.0 * pixelCountY / workThreads) / 8.0));
			   const int batchsEach = (int)ceil(1.0 * digHoloBatchCOUNT / workThreads);

			   //The current thread index.
			   int j = 0;


				   complex64* out32 = 0;

				   if (outArray32)
					   out32 = &outArray32[0];

				   digHoloUpdateReferenceWave(polIdx0);
				   for (int polThreadIdx = 0; polThreadIdx < polThreads; polThreadIdx++)
				   {
					   //For every thread working on a different part of the y-axis
					   for (int threadIdx = 0; threadIdx < workThreads; threadIdx++)
					   {
						   int batchStart = 0;
						   int batchStop = digHoloBatchCOUNT;
						   int pixelStart = 0;
						   int pixelStop = pixelCountY;

						   if (batchsEach >= pixelsEach)
						   {
							   //The starting y-position for this thread
							   batchStart = batchsEach * threadIdx;
							   //The stop index for iterating over the y-axis
							   batchStop = batchsEach * (threadIdx + 1);

							   //Check we don't go beyond the length of the y-axis
							   if (batchStop > digHoloBatchCOUNT)
							   {
								   batchStop = digHoloBatchCOUNT;
							   }
						   }
						   else
						   {
							   pixelStart = pixelsEach * threadIdx;
							   pixelStop = pixelsEach * (threadIdx + 1);

							   if (pixelStop > pixelCountY)
							   {
								   pixelStop = pixelCountY;
							   }
						   }
						   workPackage* workPack = digHoloWorkPacks[j];

						   //The input field (untilted, raw from the IFFT). This will be overwritten with the tilt/focus applied if overwrite=true
						   workPack[0].ptr1 = (void*)digHoloPixelsCameraPlaneReconstructedTilted;
						   //The output field in int16 format, with tilt/focus applied in a wavelength-dependent fashion
						   workPack[0].ptr2 = (void*)digHoloPixelsCameraPlaneReconstructed16R;
						   workPack[0].ptr3 = (void*)digHoloPixelsCameraPlaneReconstructed16I;

						   workPack[0].ptr4 = (void*)out32;
						   workPack[0].ptr5 = digHoloRef;

						   //Length of the x-axis
						   workPack[0].flag1 = pixelCountX;
						   workPack[0].flag2 = pixelCountY;
						   //the polarisation assigned to this thread
						   workPack[0].flag3 = polThreadIdx;
						   workPack[0].flag4 = polThreadIdx+1;
						   workPack[0].flag5 = digHoloPolCount;
						   workPack[0].flag6 = batchCount;
						   //The total length of the digHoloWavelength array. Tilts/focus are applied in a wavelength dependent fashion, depending on how many batch elements there are, and how many wavelengths there are.
						   workPack[0].flag7 = digHoloWavelengthCount;
						   workPack[0].flag9 = batchStart;
						   workPack[0].flag10 = batchStop;
						   workPack[0].flag11 = pixelStart;
						   workPack[0].flag12 = pixelStop;
						   
						   //Reset the thread finished handle and launch the thread
						   workPack[0].callback = workFunction::applyTiltWorker;
						   workPack[0].workCompleteEvent.Reset();
						   workPack[0].workNewEvent.Set();

						   // ThreadPool::QueueUserWorkItem(slmCallback, workPack[j]);
						   j++;
					   }
				   }

			   //Wait for all threads to complete
			   const int threadCount = j;
			   for (int j = 0; j < threadCount; j++)
			   {
				   workPackage* workPack = digHoloWorkPacks[j];
				   workPack[0].workCompleteEvent.WaitOne();
			   }
			   
			   //Keep track of how long this operation took for benchmarking purposes.
			   std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();
			   benchmarkApplyTiltTime += (stopTime - startTime);
			   benchmarkApplyTiltCounter++;
			   return DIGHOLO_ERROR_SUCCESS;
		   }
		   else
		   {
			   return DIGHOLO_ERROR_NULLPOINTER;
		   }
	   }


	   //A routine which takes a set of field (corresponding to various polarisations and batch items) and outputs important properties of those fields. 
	   //e.g. centre of mass, total power, effective area, and importantly, the maximum absolute value of the real/imaginary components. Which is important for the conversion of fields from float32 to int16
	   //This routine also creates the total intensity over all batch items per polarisation, and provides the statistics for that aggregate field/intensity as well.
	   //The user can also specify a spatial filter (as a centre position and a radius). Which specifies the region in which to perform the calculations.
	   //A negative radius for the spatial filter will cause only the values inside that window to be counted. (e.g. if you want to just look inside a IFFT window).
	   //A positive radius for the spatial filter will cause values inside the window to be ignored (e.g. if you want to mask out the zero-order to get an estimate of where the off-axis term in is Fourier space).
	   //Properties like centre of mass can be used to estimate tilt in Fourier space or beam centre in the camera plane.
	   //Properties like effective area can be used to estimate focus/beam waist
	   //This routine also provides the statistics aggregated over all batch items. e.g. the total power over the whole batch, or the average centre of mass over the whole batch.
	   //This is a multi-threaded operation, that works per polarisation, but otherwise parallelised along the y-axis. 
	   //It's easier to parallelise along the y-axis, because the x-axis is contigious in memory, and because parallelising by batch item is more difficult, as we'll be adding together the batch items to get aggregate values. Hence batches aren't independent in that sense, as they're combined together.
	   void digHoloFieldAnalysisRoutine(float* X, float* Y, complex64* field, int polStart, int polStop, int fftIdx, int pixelCountX, int pixelCountY, int batchCount, int pixelThreads, size_t batchStride, size_t polStride, float windowCX, float windowCY, float windowR, unsigned char removeZeroOrder1D, unsigned char wrapYaxis)
	   {
		   //The total number of threads we'll be launching (1 per polarisation, and then a bunch of others to parallise the pixel calculations along the y-axis)
		  // const int threadCount = (polStop - polStart) * pixelThreads;
		   const size_t polCount = digHoloPolCount;
		   //Shared work packages used by the digital holography processing threads. Contains information the threads need to know, in order to know what they need to calculate.
		  // workPackage* workPack = digHoloWorkPacks;
		   //The routine that each thread will be running
		  // WaitCallback^ statsCallback = gcnew WaitCallback(this, &digHoloObject::fieldAnalysisWorker);
		   //Approximately how many pixels along the y-axis should each thread be handling. Evently distributed, and enforced as a multiple of 8 (SIMD, AVX, 8x32bit floats per instruction)
		   const int pixelsEach = (int)(8 * ceil((1.0 * pixelCountY / pixelThreads) / 8.0));

		   //A chunk of memory the threads can use to store partial results. This could potentially be allocated globally as a persistent chunk of memory.
		   float**** workspace = 0;
		   //Every thread, for every polarisation, for every parameter to be calculated, for every batch item +1 extra batch item to store the aggregate parameters over the whole batch
		   allocate4D(pixelThreads, polCount, DIGHOLO_ANALYSISCount, (batchCount + 1), workspace);
		   // memset(&workspace[0][0][0][0], 0, pixelThreads* digHoloPolCount* digHoloPixelsStatsCount*(batchCount + 1));

			//The current thread index.
		   int j = 0;
		   int pixelThreadsActual = 0;
		   //For each polarisation we're processing...
		   for (int polIdx = polStart; polIdx < polStop; polIdx++)
		   {
			   if (polIdx >= 0 && polIdx < polCount) //How could that happen?
			   {

				   //The pointer to the first batch item in this polarisation
				   complex64* fieldIn = &field[polIdx * polStride];

				   /*   complex64* calIn = 0;

					  if (RefCalibration)
					  {
						  calIn = &RefCalibration[polIdx * polStride];
					  }
					  */

					  //Tells us where to store the sum of intensities over the whole batch.
					  //The FFT and IFFT each have their own chunks of memory for storing batch sum intensities. Useful for alignment mostly, but also for viewing by the user.
				   float* batchSum = DIGHOLO_ANALYSISBatchSum[fftIdx][polIdx];

				   pixelThreadsActual = 0;
				   //For every thread along the y-axis
				   for (int pixelThreadIdx = 0; pixelThreadIdx < pixelThreads; pixelThreadIdx++)
				   {

					   //The first y pixel assigned to this thread
					   int startIdx = pixelsEach * pixelThreadIdx;
					   //The final-1 pixel assigned to this thread
					   int stopIdx = pixelsEach * (pixelThreadIdx + 1);
					   //Check we haven't gone over the boundary.
					   if (stopIdx > pixelCountY)
					   {
						   stopIdx = pixelCountY;
					   }

					   if (stopIdx > startIdx)
					   {
						   workPackage* workPack = digHoloWorkPacks[j];
						   //The bounds of the y-axis processing loop start/stop pixels along the y-axis
						   workPack[0].start = startIdx;
						   workPack[0].stop = stopIdx;

						   //The pointer to the first batch item in this polarisation.
						   workPack[0].ptr1 = (void*)fieldIn;
						   //The x/y axis
						   workPack[0].ptr2 = (void*)X;
						   workPack[0].ptr3 = (void*)Y;
						   //Location to store the sum of intensities over all batch items
						   workPack[0].ptr4 = (void*)batchSum;
						   //Thread workspace to store partial results
						   workPack[0].ptr5 = (void*)workspace[pixelThreadsActual][polIdx];
						   //  workPack[0].ptr6 = (void*)calIn;

							 //The x/y dimensions of the field
						   workPack[0].flag1 = pixelCountX;
						   workPack[0].flag2 = pixelCountY;
						   //The number of pixels between batch items (pixelCountX*pixelCountY*polCount)
						   workPack[0].idx3 = batchStride;
						   //The total number of batch items
						   workPack[0].flag4 = batchCount;
						   //If set, does not include any regions along the x=0 or y=0 axes. Calibration errors on the camera often show up here in the Fourier plane, so it's best to filter them out.
						   workPack[0].flag5 = removeZeroOrder1D;
						   workPack[0].flag6 = wrapYaxis;
						   //  workPack[0].flag7 = RefCalibrationWavelengthCount;

							 //The centre position and radius of the area of the spatial filter to include/ignore from the calculation
						   workPack[0].var1 = windowCX;
						   workPack[0].var2 = windowCY;
						   workPack[0].var3 = windowR;

						   //Reset the thread finished event handle, and launch the thread
						   workPack[0].callback = workFunction::fieldAnalysisWorker;
						   workPack[0].workCompleteEvent.Reset();
						   workPack[0].workNewEvent.Set();
						   // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
							//ThreadPool::QueueUserWorkItem(statsCallback, workPack[j]);
						   j++;
						   pixelThreadsActual++;
					   }
				   }
			   }
		   }
		   //Foolishly used 'j' for both loops. Alias to 'totalThreads'
		   const int totalThreads = j;

		   //Wait for all threads to finish.
		   for (int j = 0; j < totalThreads; j++)
		   {
			   workPackage* workPack = digHoloWorkPacks[j];
			   workPack[0].workCompleteEvent.WaitOne();
		   }

			 //Once the threads have finished, go through a do final pass of processing to calculate the final values.

			 //Calculate the area of a single pixel. This is required for calculating effective area (Petermann II)
		   const float pixelSizeX = fabs(X[2] - X[1]);
		   const float pixelSizeY = fabs(Y[2] - Y[1]);
		   const float dA = pixelSizeX * pixelSizeY;

		   //For each relevant polarisation
		   for (int polIdx = polStart; polIdx < polStop; polIdx++)
		   {
			   if (polIdx >= 0 && polIdx < polCount)
			   {
				   //For all batch items, plus the extra one for the aggregate values over the whole batch
				   for (int batchIdx = 0; batchIdx < (batchCount + 1); batchIdx++)
				   {
					   //Centre of mass calculation
					   float comx = 0;
					   float comy = 0;
					   float comyWrap = 0;
					   //Total power calculation
					   float totalPwr = 0;
					   //Intensity squared calculation (for effective area)
					   float E4 = 0;
					   //The maximum absolute value. Will ultimately be used for converting fields from float32 format, to int16 format.
					   float maxAbs = -FLT_MAX;
					   //The index of that maximum value (indexed as real/imaginary, so the index treats the field as float instead of complex. i.e. the largest possible index is twice the length of the field).
					   //The maximum value correspond to a particular real or imaginary component within a complex number. This index is not used by anything at this stage.
					   int maxAbsIdx = -1;

					   //Over all y-axis pixel threads
					   for (int pixelThreadIdx = 0; pixelThreadIdx < pixelThreadsActual; pixelThreadIdx++)
					   {
						   //Get the results of this thread for each of the relevant parameters to be calculated.
						   float dPwr = workspace[pixelThreadIdx][polIdx][DIGHOLO_ANALYSIS_TOTALPOWER][batchIdx];
						   float dComX = workspace[pixelThreadIdx][polIdx][DIGHOLO_ANALYSIS_COMX][batchIdx];
						   float dComY = workspace[pixelThreadIdx][polIdx][DIGHOLO_ANALYSIS_COMY][batchIdx];
						   float dComYwrap = workspace[pixelThreadIdx][polIdx][DIGHOLO_ANALYSIS_COMYWRAP][batchIdx];
						   float dE4 = workspace[pixelThreadIdx][polIdx][DIGHOLO_ANALYSIS_AEFF][batchIdx];

						   //Properties like centre of mass (partial at this stage), total power and total intensity^2 are summed over all pixels.
						   comx += dComX;
						   comy += dComY;
						   comyWrap += dComYwrap;
						   totalPwr += dPwr;
						   E4 += dE4;

						   //If the maximum absolute value seen by this thread is the biggest so far. Remember that value, and remember the corresponding index position.
						   if (workspace[pixelThreadIdx][polIdx][DIGHOLO_ANALYSIS_MAXABS][batchIdx] > maxAbs)
						   {
							   //The maximum absolute value.
							   maxAbs = workspace[pixelThreadIdx][polIdx][DIGHOLO_ANALYSIS_MAXABS][batchIdx];
							   //This is some crazy-ass shit. The index is an int32, but it's being stored as a float because they're the same memory length.
							   //An alternative would be to make digHoloPixelsStats a double instead of a float, or to allocate a separate array for STAT_MAXABSIDX that's declared as an int.
							   //He's a mad man...a MAD MAAAAANNN!!!
							   int* idx = (int*)&workspace[pixelThreadIdx][polIdx][DIGHOLO_ANALYSIS_MAXABSIDX][batchIdx];
							   maxAbsIdx = idx[0];
						   }
					   }

					   //Final processing of the values aggregated over all threads
					   //Total power
					   DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_TOTALPOWER][polIdx][batchIdx] = totalPwr;
					   //Final centre of mass calculation
					   DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_COMX][polIdx][batchIdx] = comx / totalPwr;
					   DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_COMY][polIdx][batchIdx] = comy / totalPwr;
					   DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_COMYWRAP][polIdx][batchIdx] = comyWrap / totalPwr;
					   //Effective area
					   DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_AEFF][polIdx][batchIdx] = dA * (totalPwr * totalPwr / E4);//Petermann II
					   //The inverse of the maximum absolute value multiplied by 2^15-1. Hence if we multiply the field by this number, we'll get a value from -32767 to 32767, which is the range of a int16
					   //This value here can be used to convert a reconstructed field stored as a int16, to the correct float32 complex number
					   DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_MAXABS][polIdx][batchIdx] = (float)(32767.0 / maxAbs);
					   //Again, tricking C into storing an int32 in a float32 memory slot.
					   float* maxAbsIdxf = (float*)&maxAbsIdx;
					   DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_MAXABSIDX][polIdx][batchIdx] = maxAbsIdxf[0];
				   }
			   }
		   }
		   if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
		   {
			   fprintf(consoleOut, "Freeing memory...\n\r");
			   fflush(consoleOut);
		   }
		   //Free the thread workspace for this routine.
		   free4D(workspace);
	   }

	   //The worker routine for the digHoloFieldAnalysisRoutine. Each thread does this operation
	   void fieldAnalysisWorker(workPackage& e)
	   {
		   //workPackage^ e = (workPackage^)o;

		   //Start/stop indices for processing along the pixel y-axis
		   const int startIdx = (int)e.start;
		   const int stopIdx = (int)e.stop;
		   //A pointer to the first batch item in the relevant polarisation
		   complex64* fields = (complex64*)e.ptr1;
		   //the x and y axes corresponding with tiltedField
		   float* X = (float*)e.ptr2;
		   float* Y = (float*)e.ptr3;
		   //The location to store the sum of intensities over the whole batch
		   float* sumPower = (float*)e.ptr4;
		   //Workspace for the thread to store partially calculated results. These values will be further processed once all threads are finished.
		   float** workspace = (float**)e.ptr5;

		   //The x/y dimension of the field
		   const int pixelCountX = e.flag1;
		   const int pixelCountY = e.flag2;
		   //How many pixels apart are batch items
		   const size_t batchStride = e.idx3;
		   //How many batch items are there total
		   const int batchCount = e.flag4;
		   //discard regions along the x=0 and y=0 axes (camera calibration errors often show up here in the Fourier plane)
		   const int filter1D = e.flag5;
		   const int wrapYaxis = e.flag6;

		   //Defines the postion (x,y) and radius (R) of a region to include/ignore from the calculation of the relevant parameters.
		   //Used for discarding regions like the zero-order, or only calculating parameters within a certain window of interest.
		   const float windowCX = e.var1;
		   const float windowCY = e.var2;
		   const float windowR = e.var3;

		   //Memory locations for storing the partial results for each parameter. These are 1D arrays of length batchCount+1.
		   //Parameters are calculate for each batch item +1 extra for the calculation of the aggregate parameters over the whole batch. e.g. total power, effective area etc.
		   float* cxOut = &workspace[DIGHOLO_ANALYSIS_COMX][0];//Centre of mass (x)
		   float* cyOut = &workspace[DIGHOLO_ANALYSIS_COMY][0];//Centre of mass (y)
		   float* cyWrapOut = &workspace[DIGHOLO_ANALYSIS_COMYWRAP][0];//Centre of mass (y)
		   float* totalPowerOut = &workspace[DIGHOLO_ANALYSIS_TOTALPOWER][0];//Total power in the frame
		   int* maxIdxOut = (int*)&workspace[DIGHOLO_ANALYSIS_MAXABSIDX][0];//Tells you in 1D index of the position with the maximum absolute value. This could be casting problem (int (32-bit) to float (24-bit significand)). Feature currently unused anyway. Better to cast memory as int
		   float* AeffOut = &workspace[DIGHOLO_ANALYSIS_AEFF][0];//The effective area (Petermann II) of the intensity.
		   float* maxAbsOut = &workspace[DIGHOLO_ANALYSIS_MAXABS][0]; //The scale of the 16-bit ints the field has now been stored as. You can think of this as like the exponent bits of a float.

		  // complex64* calibration = (complex64*)e.ptr6;
		   //int calibrationWavelengthCount = e.flag7;

		   //The routine itself that does the analysis
		   digHoloFieldAnalysis((float*)&fields[0], X, Y, pixelCountX, pixelCountY, startIdx, stopIdx, batchCount, cxOut, cyOut, totalPowerOut, maxAbsOut, maxIdxOut, AeffOut, sumPower, batchStride, windowCX, windowCY, windowR, filter1D, wrapYaxis, cyWrapOut);
		   
	   }

	   //The worker thread for the digHoloApplyTilt() routine.
	   //Applies a wavelength dependent tilt/focus to the field coming from the IFFT
	   void applyTiltWorker(workPackage& e)
	   {

		   /*						   //The input field (untilted, raw from the IFFT). This will be overwritten with the tilt/focus applied if overwrite=true
						   workPack[0].ptr1 = (void*)digHoloPixelsCameraPlaneReconstructedTilted;
						   //The output field in int16 format, with tilt/focus applied in a wavelength-dependent fashion
						   workPack[0].ptr2 = (void*)digHoloPixelsCameraPlaneReconstructed16R;
						   workPack[0].ptr3 = (void*)digHoloPixelsCameraPlaneReconstructed16I;

						   workPack[0].ptr4 = (void*)out32;
						   workPack[0].ptr5 = digHoloRef;

						   //Length of the x-axis
						   workPack[0].flag1 = pixelCountX;
						   workPack[0].flag2 = pixelCountY;
						   //the polarisation assigned to this thread
						   workPack[0].flag3 = polThreadIdx;
						   workPack[0].flag4 = polThreadIdx+1;
						   workPack[0].flag5 = digHoloPolCount;
						   //The total length of the digHoloWavelength array. Tilts/focus are applied in a wavelength dependent fashion, depending on how many batch elements there are, and how many wavelengths there are.
						   workPack[0].flag7 = digHoloWavelengthCount;
						   //doOverlap flag defines which routine to run.
						   workPack[0].flag8 = doOverlap;

						   workPack[0].flag9 = batchStart;
						   workPack[0].flag10 = batchStop;
						   workPack[0].flag11 = pixelStart;
						   workPack[0].flag12 = pixelStop;


							   workPack[0].flag13 = digHoloMaxMG;
							   workPack[0].flag14 = digHoloModeCount;

							   workPack[0].ptr6 = digHoloHGX;
							   workPack[0].ptr7 = digHoloHGY;
							   workPack[0].ptr8 = digHoloHGscaleX;
							   workPack[0].ptr9 = digHoloHGscaleY;
							   workPack[0].ptr10 = digHoloOverlapWorkspace[j];
							   workPack[0].ptr11 = HG_MN;
							   workPack[0].ptr12 = digHoloOverlapCoefsHG;
							   &*/

		   //The source field to have the tilt/focus removed from (and also an output destination if overwrite = true)
		   complex64* tiltedField = (complex64*)e.ptr1;
		   //The destination field with the tilt/focus removed, in int16 format.
		   short* untiltedFieldR = (short*)e.ptr2;
		   short* untiltedFieldI = (short*)e.ptr3;

		   complex64* outArray32 = (complex64*)e.ptr4;
		   short*** refWave = (short***)e.ptr5;

		   //Array of wavelengths to tilt/focus to be applied in a wavelength dependent fashion if applicable.
		   //float* lambdas = (float*)e.ptr6;
		   

		   


		   //Length of the x-axis
		   const int pixelCountX = e.flag1;
		   const int pixelCountY = e.flag2;
		   //The polarisation component assigned to this thread
		   const int polStart = e.flag3;
		   const int polStop = e.flag4;
		   const int polCount = e.flag5;
		   //The total number of batch items
		   const int batchCount = e.flag6;
		   //The length of the wavelength array (float *lambda)
		   const int lambdaCount = e.flag7;



		   //The scale that should be applied to each batch item to convert it from a float32 to an int16. This has been previously calculated using the digHoloFieldAnalysis routine.
		   //This should probably be fed in using the work package rather than written here
		   float** scales = DIGHOLO_ANALYSIS[IFFT_IDX][DIGHOLO_ANALYSIS_MAXABS];

		   //Apply the tilt
		   //const int doOverlap = e.flag8;
		   int batchStart = e.flag9;
		   int batchStop = e.flag10;
		   int pixelStart = e.flag11;
		   int pixelStop = e.flag12;
		   
		   //They're both output ordering, because the data rearrangement has already occured and need not be applied here again.
		   int orderingIn = config.wavelengthOrdering[DIGHOLO_WAVELENGTHORDER_OUTPUT];
		   int orderingOut = config.wavelengthOrdering[DIGHOLO_WAVELENGTHORDER_OUTPUT];

		/*   complex64* RefCalibration = 0;
		   int RefCalibrationWavelengthCount = 0;
		   if (digHoloRefCalibrationEnabled)
		   {
			   RefCalibration = digHoloRefCalibrationReconstructedPlane;
			   RefCalibrationWavelengthCount = digHoloRefCalibrationWavelengthCount;
		   }*/

			   applyTilt16(scales, pixelStart, pixelStop, pixelCountX, pixelCountY, polStart, polStop, polCount, batchStart, batchStop, batchCount,tiltedField, untiltedFieldR, untiltedFieldI, lambdaCount, outArray32,refWave,orderingIn, orderingOut);
	   }

	   //Benchmark timers use to calculate how fast the overlap routine is running.
	   std::chrono::duration<double>  benchmarkOverlapModesTime = std::chrono::duration<double>(0);
	   int64_t benchmarkOverlapModesCounter = 0;

	   //An overloaded version of the digHoloOverlapModes(polIdx) routine, which calculates the overlaps for both polarisations.
	   int digHoloOverlapModes()
	   {
		   return digHoloOverlapModes(-1);
	   }

	   int digHoloOverlapModes(int polIDX)
	   {
		   if (config.maxMG > 0)
		   {
			   if (digHoloPixelsCameraPlaneReconstructed16R)
			   {
				   std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
				   digHoloMaxMG = config.maxMG;
				   int totalThreads = digHoloThreadCount;
				   if (totalThreads < 1)
					   totalThreads = 1;
				   int pixelCountX;
				   int pixelCountY;
				   digHoloUpdateBasis(digHoloResolutionIdx, pixelCountX, pixelCountY);

				   //blockSize is the SIMD size (4 packed doubles in this case)
				   const int blockSize = digHoloOverlapModesAllocate(totalThreads);

				   const int polCount = digHoloPolCount;
				   const int modeCount = digHoloModeCount;
				   const int groupCount = digHoloMaxMG;

				   const size_t pixelCount = ((size_t)pixelCountX) * pixelCountY;
				   const size_t batchOffset = polCount * pixelCount;
				   const int batchCount = digHoloBatchCOUNT;

				   unsigned char polCalc[2] = { 1, 1 };

				   int polStart = 0;
				   int polStop = polCount;

				   if (polIDX == 0)
				   {
					   polCalc[0] = 1;
					   polCalc[1] = 0;
					   polStart = 0;
					   polStop = 1;
				   }
				   if (polIDX == 1)
				   {
					   polCalc[0] = 0;
					   polCalc[1] = 1;
					   polStart = 1;
					   polStop = 2;
				   }

				   void* modesX = (void*)digHoloHGX;
				   void* modesY = (void*)digHoloHGY;

				   short* reconstructedFieldR = digHoloPixelsCameraPlaneReconstructed16R;
				   short* reconstructedFieldI = digHoloPixelsCameraPlaneReconstructed16I;

				   if (totalThreads > 1)
				   {
					   // WaitCallback^ digHoloOverlapModes_Callback = gcnew WaitCallback(this, &digHoloObject::overlapWorkerHG);
					   totalThreads = digHoloOverlapModes_InitPool(totalThreads, pixelCountX, pixelCountY, modeCount, blockSize, reconstructedFieldR, reconstructedFieldI, modesX, modesY, polStart, polStop);

					   for (int j = 0; j < totalThreads; j++)
					   {
						   workPackage* workPack = digHoloWorkPacks[j];
						   workPack[0].callback = workFunction::overlapWorkerHG;
						   workPack[0].workCompleteEvent.Reset();
						   workPack[0].workNewEvent.Set();
					   }

					   //Wait for the threads to finish
					   for (int j = 0; j < totalThreads; j++)
					   {
						   workPackage* workPack = digHoloWorkPacks[j];
						   workPack[0].workCompleteEvent.WaitOne();
					   }
				   }
				   else
				   {
					   //   totalThreads, digHoloBatchCOUNT, digHoloPolCount, digHoloModeCount* blockSize * 2
					  // memset(&digHoloOverlapWorkspace[0][0][0][0], 0, sizeof(double) * totalThreads * batchCount * polCount * modeCount * blockSize * 2);
					  // memset(&digHoloOverlapWorkspace[0][0][0], 0, sizeof(double) * batchCount * polCount * modeCount * blockSize * 2);
					   float fieldScales[2];
					   for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
					   {
						   const size_t batchAddr = batchIdx * batchOffset;
						  // double** workspaceBatch = digHoloOverlapWorkspace[0][batchIdx];
						  // double** workspaceBatch = digHoloOverlapWorkspace[batchIdx];

						   for (int polIdx = 0; polIdx < polCount; polIdx++)
						   {
							   fieldScales[polIdx] = (float)1.0f/DIGHOLO_ANALYSIS[IFFT_IDX][DIGHOLO_ANALYSIS_MAXABS][polIdx][batchIdx];
						   }
						   complex64* coefs = digHoloOverlapCoefsHG[batchIdx];
						   float** scaleX = digHoloHGscaleX;
						   float** scaleY = digHoloHGscaleY;

						   overlapFieldSeparable16((__m256i*) & reconstructedFieldR[batchAddr], (__m256i*) & reconstructedFieldI[batchAddr],&fieldScales[0], (__m256i***)modesX, (short***)modesY,scaleX, scaleY, pixelCountX, pixelCountY, coefs, HG_MN, 0, groupCount, groupCount, polCount,modeCount,polStart,polStop);
					   }
				   }

				   digHoloApplyBasisTransform();

				   std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();
				   benchmarkOverlapModesTime += (stopTime - startTime);
				   benchmarkOverlapModesCounter++;
				   return DIGHOLO_ERROR_SUCCESS;
			   }
			   else
			   {
				   return DIGHOLO_ERROR_NULLPOINTER;
			   }
		   }
		   else
		   {
			   digHoloMaxMG = 0;
			   config.maxMG = 0;
			   return DIGHOLO_ERROR_SUCCESS;
		   }
	   }

	   void digHoloApplyBasisTransform()
	   {
		   int basisType = config.basisType;

		   //I don't know how you would manage to specify a basis type that isn't supported.
		   unsigned char basisTypeNotSupported = (basisType < 0 || basisType>DIGHOLO_BASISTYPE_CUSTOM);
		   //If the user has specified they want to use a custom basis, however that custom basis is invalid. Either because the basis transform matrix is a null pointer, or the sizes specified are not >0.
		   unsigned char customBasisInvalid = (basisType == DIGHOLO_BASISTYPE_CUSTOM) && (!digHoloBasisTransformFull || digHoloBasisTransformFullModeCountIn <= 0 || digHoloBasisTransformFullModeCountOut <= 0);

		   //In the event that the user has attempted to load an invalid custom basis, default back to HG
		   if (basisTypeNotSupported || customBasisInvalid)
		   {
			   basisType = DIGHOLO_BASISTYPE_HG;
		   }

		   digHoloBasisType = basisType;

		   //There should never be any issue applying HG or LG bases.
		   //HG is the default and LG transform matrices are automatically calculated when the maxMG is changed.
		   //So HG and LG bases should be ready to go at all times.
		   if (digHoloBasisType == DIGHOLO_BASISTYPE_HG)
		   {
			   digHoloOverlapCoefsPtr = digHoloOverlapCoefsHG;
			   digHoloModeCountOut = digHoloModeCount;
		   }
		   else
		   {
			   if (digHoloBasisType == DIGHOLO_BASISTYPE_LG)
			   {

				   //LG conversion initialisation
				   if (config.maxMG != LGtoHGmaxMG)
				   {
					   LGtoHGmaxMG = digHoloLGInit(config.maxMG, LGtoHG, LG_L, LG_P, HG_M, HG_N);
				   }

				   //allocate memory for the coefficients in the new basis.
				   if (!digHoloOverlapCoefsCustom)
				   {
					   allocate2D(digHoloBatchCOUNT, digHoloModeCount * digHoloPolCount, digHoloOverlapCoefsCustom);
				   }

				   digHoloOverlapCoefsPtr = digHoloOverlapCoefsCustom;
				   digHoloModeCountOut = digHoloModeCount;

				   //Apply the transformation of HG coefficient to LG coefficients
				   HGtoLG(digHoloOverlapCoefsHG, digHoloOverlapCoefsCustom, digHoloBatchCOUNT, digHoloModeCount, digHoloPolCount, false);
			   }
			   else
			   {
				   //Setup the transform matrix if it needs updating.
				   if (!digHoloBasisTransformIsValid || !digHoloBasisTransform || !digHoloBasisTransformModeCountIn)
				   {
					   int dimx = (int)(std::fmin(digHoloModeCount, digHoloBasisTransformFullModeCountIn));
					   int dimy = (int)(std::fmin(digHoloModeCount, digHoloBasisTransformFullModeCountOut));
					   digHoloBasisTransformModeCountIn = dimx;
					   digHoloBasisTransformModeCountOut = dimy;

					   const size_t length = digHoloBasisTransformModeCountIn * digHoloBasisTransformModeCountOut;
					   allocate1D(length, digHoloBasisTransform);
					   memset(&digHoloBasisTransform[0][0], 0, sizeof(complex64) * length);

					   //Copy the transform matrix from it's full-dimension form, to the sub-matrix relevant to the number of HG modes we currently support.
					   //Could be faster, but should only run very rarely. Should use 1D memcpy loop, instead of 2D loop.
					   for (int i = 0; i < dimx; i++)
					   {
						   for (int j = 0; j < dimy; j++)
						   {
							   //  int idx = j * digHoloBasisTransformModeCountIn + i;
							   //  int IDX = j * digHoloBasisTransformFullModeCountIn + i;

							   int idx = i * digHoloBasisTransformModeCountOut + j;
							   int IDX = i * digHoloBasisTransformFullModeCountOut + j;

							   digHoloBasisTransform[idx][0] = digHoloBasisTransformFull[IDX][0];
							   digHoloBasisTransform[idx][1] = digHoloBasisTransformFull[IDX][1];

						   }
					   }
				   }

				   //allocate memory for the coefficients in the new basis, if this hasn't already been done.
				   if (!digHoloOverlapCoefsCustom)
				   {
					   allocate2D(digHoloBatchCOUNT, digHoloBasisTransformModeCountOut * digHoloPolCount, digHoloOverlapCoefsCustom);
				   }

				   digHoloOverlapCoefsPtr = digHoloOverlapCoefsCustom;
				   digHoloModeCountOut = digHoloBasisTransformModeCountOut;

				   digHoloBasisTransformIsValid = true;

				   applyBasisTransform(digHoloOverlapCoefsHG, digHoloOverlapCoefsCustom, digHoloBasisTransform, digHoloBatchCOUNT, digHoloPolCount, digHoloBasisTransformModeCountIn, digHoloBasisTransformModeCountOut, digHoloModeCount, false);
			   }
		   }
	   }

	   //HGtoLG(complex** HGcoefs, complex** LGcoefs, int batchCount, int modeCount, int polCount, int inverseTransform)
	   void applyBasisTransform(complex64** coefsIn, complex64** coefsOut, complex64* transform, int batchCountIn, int polCount, int transformModeCountIn, int transformModeCountOut, int coefsInModeCount, int inverseTransform)
	   {
		   complex64 alpha;
		   alpha[0] = 1;
		   alpha[1] = 0;
		   complex64 beta;
		   beta[0] = 0;
		   beta[1] = 0;

		   int incx = 1;
		   int incy = 1;

		   int batchCount = batchCountIn;

		   int m = transformModeCountIn;
		   int n = transformModeCountOut;
		   complex64* U = transform;

		   for (int polIdx = 0; polIdx < polCount; polIdx++)
		   {
			   size_t idx = polIdx * coefsInModeCount;
			   size_t IDX = polIdx * transformModeCountOut;

			   for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
			   {
				   complex64* coefsHG = &coefsIn[batchIdx][idx];
				   complex64* coefsCustom = &coefsOut[batchIdx][IDX];
#ifdef LAPACKBLAS_ENABLE
#ifdef CBLAS_ENABLE
				   auto tpose = inverseTransform ? CBLAS_TRANSPOSE::CblasConjTrans : CBLAS_TRANSPOSE::CblasNoTrans;
				   cblas_cgemv(CBLAS_LAYOUT::CblasColMajor, tpose, n, m, &alpha, U, n, coefsHG, incx, &beta, coefsCustom, incy);
#else
				   const char trans = inverseTransform ? 'C' : 'N';
				   cgemv(&trans, &n, &m, (BLAS_COMPLEXTYPE*)alpha, (BLAS_COMPLEXTYPE*)U, &n, (BLAS_COMPLEXTYPE*)coefsHG, &incx, (BLAS_COMPLEXTYPE*)beta, (BLAS_COMPLEXTYPE*)coefsCustom, &incy);
#endif
#endif
			   }
		   }
	   }

	   int digHoloOverlapModesAllocate(int totalThreads)
	   {

		   const int blockSize = 4;
		   if (digHoloOverlapModesValid_BatchCount != digHoloBatchCOUNT || digHoloModeCount != digHoloOverlapModesValid_ModeCount ||
			   digHoloOverlapModesValid_PolCount != digHoloPolCount || digHoloOverlapModesValid_ThreadCount != totalThreads)
		   {
			   free2D(digHoloOverlapCoefsCustom);
			   allocate2D(digHoloBatchCOUNT, digHoloModeCount * digHoloPolCount, digHoloOverlapCoefsHG);
			   //Do I actually need to zero this?
			   memset(&digHoloOverlapCoefsHG[0][0][0], 0, sizeof(complex64) * digHoloModeCount * digHoloPolCount * digHoloBatchCOUNT);

			   digHoloOverlapModesValid_BatchCount = digHoloBatchCOUNT;
			   digHoloOverlapModesValid_ModeCount = digHoloModeCount;
			   digHoloOverlapModesValid_PolCount = digHoloPolCount;
			   digHoloOverlapModesValid_ThreadCount = totalThreads;// digHoloThreadCount;

			   //In the power meter routine, this array will be reused as workspace, but it will work with digHoloModeCountOut, rather than digHoloModeCount.
			   //However digHoloModeCountOut should never be higher than digHoloModeCount, there's a fmin call in the BasisTransform function that should ensure this.
			   //Hence this should not need to be reallocated if digHoloModeCountOut!=digHoloModeCount.
			   //allocate4D(totalThreads, digHoloBatchCOUNT, digHoloPolCount, digHoloModeCount * blockSize * 2, digHoloOverlapWorkspace);//x2 for real/imaginary
			   //This used to be over allocated such that every thread had a full copy of the data, but as they all work on separate chunks (either batch or mode) this isn't neccessary
			  // allocate3D(digHoloBatchCOUNT, digHoloPolCount, digHoloModeCount * blockSize * 2, digHoloOverlapWorkspace);//x2 for real/imaginary
			   //allocate4D(totalThreads,digHoloBatchCOUNT, digHoloPolCount, digHoloModeCount * blockSize * 2, digHoloOverlapWorkspace);//x2 for real/imaginary
			   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
			   {
				   fprintf(consoleOut, "Reallocated coefs %i x %i\n\r", digHoloPolCount, digHoloModeCount);
				   fflush(consoleOut);
			   }
		   }
		   return blockSize;
	   }

	   int digHoloOverlapModes_InitPool(int totalThreads0, int width, int height, int modeCount, int blockSize, short* reconstructedFieldR, short* reconstructedFieldI, void* modesX, void* modesY, int polStart, int polStop)
	   {
		   //  const int basisType = config->basisType;
		   int modeGroupCountMax = 0;
		  // unsigned char separableBasis = false;

		   modeGroupCountMax = digHoloMaxMG;
		 //  separableBasis = true;

		   const int polCount = digHoloPolCount;
		   const int pixelCountX = width;
		   const int pixelCountY = height;
		   const int polCount0 = polStop - polStart;
		   const int workThreads = totalThreads0 / polCount0;

		   const size_t pointCount = ((size_t)pixelCountX) * ((size_t)(pixelCountY));
		   // size_t pixelsPerThread = CEILINT(pointCount,totalThreads);
		 //  const float modesPerThreadIdeal = (float)(1.0 * modeCount / workThreads);
		   int modeGroupsPerThread = CEILINT(modeGroupCountMax, workThreads);
		   int batchsPerThread = CEILINT(digHoloBatchCOUNT, workThreads);

		   int modeStart = 0;
		   int modeStop = modeGroupCountMax;

		   int batchStart = 0;
		   int batchStop = digHoloBatchCOUNT;

		  // size_t pixelStart = 0;
		  // size_t pixelStop = pointCount;

		   //zero the working memory for the threads
		   //const size_t memLength = sizeof(double) * totalThreads0 * digHoloBatchCOUNT * digHoloPolCount * modeCount * blockSize * 2;

		   int j = 0;
		  // int groupStart = 0;
		   //int groupStop = 0;

		   for (int polIdx = polStart; polIdx < polStop; polIdx++)
		   {
			   for (int threadIdx = 0; threadIdx < workThreads; threadIdx++)
			   {
				   //If this is a batch, parallelise in the basis of frames
				   if (digHoloBatchCOUNT > 1)
				   {
					   batchStart = threadIdx * batchsPerThread;
					   batchStop = (threadIdx + 1) * batchsPerThread;

					   //Do all the groups
					   modeStart = 0;
					   modeStop = modeGroupCountMax;
				   }
				   else
				   {
					   batchStart = 0;
					   batchStop = digHoloBatchCOUNT;

						modeStart = threadIdx*modeGroupsPerThread;
						modeStop = (threadIdx+1) * modeGroupsPerThread;

				   }

				   //Check we haven't gone over the edge
				   if (modeStop > modeGroupCountMax)
				   {
					   modeStop = modeGroupCountMax;
				   }
				   if (batchStop > digHoloBatchCOUNT)
				   {
					   batchStop = digHoloBatchCOUNT;
				   }

				   workPackage* workPack = digHoloWorkPacks[j];
				   //If there's actually something to process
				   if (modeStart < modeStop && batchStart<batchStop)
				   {
					   workPack[0].threadIdx = j;
					  // workPack[0].totalThreads = totalThreads;

					   workPack[0].ptr1 = (void*)reconstructedFieldR;
					   workPack[0].ptr2 = (void*)reconstructedFieldI;
					   workPack[0].ptr3 = (void*)modesX;
					   workPack[0].ptr4 = (void*)modesY;
					  // workPack[0].ptr5 = (void*)digHoloOverlapWorkspace;
					  // workPack[0].ptr5 = (void*)digHoloOverlapWorkspace[j];
					   workPack[0].ptr6 = (void*)digHoloOverlapCoefsHG;
					   workPack[0].ptr7 = (void*)DIGHOLO_ANALYSIS[IFFT_IDX][DIGHOLO_ANALYSIS_MAXABS];
					   // workPack[0].ptr6 = (void*)digHoloOverlapWorkspaceOut;

						//Used by LG/non-separable overlap. Should really code this flip out
					   float conjB = -1.0;
					   workPack[0].var1 = conjB;

					   workPack[0].idx1 = pointCount * digHoloPolCount;
					   workPack[0].idx2 = pointCount;

					   workPack[0].flag3 = batchStart;
					   workPack[0].flag4 = batchStop;

					   workPack[0].flag5 = polIdx;
					   workPack[0].flag6 = polIdx+1;

					   workPack[0].flag7 = modeStart;
					   workPack[0].flag8 = modeStop;

					   workPack[0].flag10 = polCount;
					   workPack[0].flag11 = modeGroupCountMax;
					   workPack[0].flag12 = pixelCountX;
					   workPack[0].flag13 = pixelCountY;
					   workPack[0].flag14 = modeCount;
					   j++;
				   }
			   }
		   }
		   return j;
	   }

	   void overlapWorkerHG(workPackage& e)
	   {
		   //const int threadIdx = e.threadIdx;
		   short* fieldR = (short*)e.ptr1;
		   short* fieldI = (short*)e.ptr2;
		   short*** modesX = (short***)e.ptr3;
		   short*** modesY = (short***)e.ptr4;
		  // double*** overlapWork = (double***)e.ptr5;
		   complex64** coefs = (complex64**)e.ptr6;
		   float** fieldScales = (float**)e.ptr7;
		  //size_t framePixelCount = e.idx1;

		   int batchStart = e.flag3;
		   int batchStop = e.flag4;

		   int polStart = e.flag5;
		   int polStop = e.flag6;

		   //Start/Stop modeX component
		   int modeStart = e.flag7;
		   int modeStop = e.flag8;

		   // size_t pixelStart = e.idx3;
		   // size_t pixelStop = e.idx4;

		   int modeGroupCount = e.flag11;

		   int pixelCountX = e.flag12;
		   int pixelCountY = e.flag13;
		   int modeCount = e.flag14;
		
		   int polCount = e.flag10;
		  // const size_t blockSize = 4;// sizeof(__m256d);

		   //const int batchCount = (batchStop - batchStart);

		   float** modeScaleX = digHoloHGscaleX;
		   float** modeScaleY = digHoloHGscaleY;

		   //const size_t memLength = sizeof(__m256d) * (modeCount)*polCount * 2;
		   float fScale[2];

		   const size_t fieldStride = pixelCountX * pixelCountY;
		   const size_t batchStride = fieldStride * polCount;

		   for (int batchIdx = batchStart; batchIdx < batchStop; batchIdx++)
		   {
			   const size_t batchOffset = batchIdx * batchStride;
			//   memset(&overlapWork[batchIdx][0][0], 0, memLength);
			//   double** workspace = overlapWork[batchIdx];
			   
			   for (int polIdx = 0; polIdx < polCount; polIdx++)
			   {
				   fScale[polIdx] = 1.0f/fieldScales[polIdx][batchIdx];
			   }
			   
			   overlapFieldSeparable16((__m256i*) &fieldR[batchOffset], (__m256i*) &fieldI[batchOffset],&fScale[0], (__m256i***)modesX, modesY, modeScaleX, modeScaleY, pixelCountX, pixelCountY, coefs[batchIdx], HG_MN, modeStart, modeStop, modeGroupCount, polCount,modeCount, polStart, polStop);

		   }
	   }

	   //Given an LG(l,p) index, and a HG(m,n) index, outputs the coefR/coefI between those modes.
	   template <typename T>
	   void digHoloLGtoHGcoef(int l, int p, int HG_m, int HG_n, T& coefR, T& coefI, T* pu, T* pv)
	   {
		   //Mode group
		   const int N = 2 * p + abs(l);

		   // If the LG mode and HG mode aren't in the same group, the overlap is zero
		   if ((HG_m + HG_n) != N)
		   {
			   coefR = 0;
			   coefI = 0;
		   }
		   else
		   {
			   //Back calculate the n and m indices as per Equation 9.
				//Somewhat confusingly, these are not the same as the m, n indices of the
				 // HG mode being overlapped with(m_, n_).Rather these m, n indices indicate a specific l, p combination.
				 // m, n kept here to keep it the same as Equation 9
			   int n;
			   int m;
			   if (l < 0)
			   {
				   n = p;
				   m = -l + n;
			   }
			   else
			   {
				   m = p;
				   n = l + m;
			   }

			   const int mink = (HG_m < m) ? HG_m : m;
			   // b = b * sqrt((factorial(N - k) * factorial(k)) / (pow(2.0, N) * factorial(n) * factorial(m)));
		//This section evaluates the factorials etc above, in a way which avoids overflows. {b = b * sqrt((factorial(N - k) * factorial(k)) / (pow(2.0, N) * factorial(n) * factorial(m))) / factorial(k)}
			   T b0 = 1.0;
			   T bLog0 = 0;

			   T puv = pu[HG_m];
			   __m256d puv256 = _mm256_set_pd(0, 0, 0, puv);

			  // T pv0 = 1.0;
			   T* puvPtr = (T*)&puv256;

			   for (int termIdx = 1; termIdx <= N; termIdx++)
			   {
				   //This section builds the term {sqrt((factorial(N - k) * factorial(k)) / (pow(2.0, N) * factorial(n) * factorial(m)))};
				   //These should only be 0 or +/- 1.
				   const int termCount = (termIdx <= HG_n) + (termIdx <= (m + n - HG_n)) - (termIdx <= n) - (termIdx <= m);

				   if (termCount)
				   {
					   //Selects to either multiply by 'a' (termCount==1), or divide by 'a' (termCount==-1)
					   b0 = (termCount > 0) ? b0 * (termIdx) : b0 = b0 / (termIdx);
					   //This makes no difference to precision? Could stop overflow, but by that point the precision is lost anyways
					   //bLog0 += floorLog2(b0, b0);
				   }
			   }

			   const int startIdx = HG_m >= (n + 1) ? (HG_m - n) : 1;

			   for (int termIdx = startIdx; termIdx <= mink; termIdx++)
			   {
				   const int idx = HG_m - termIdx;

				   const __m256d pv0256 = _mm256_set_pd(0, 0, 0, pv[termIdx]);
				   const __m256d pu0256 = _mm256_set_pd(0, 0, 0, pu[idx]);
				   puv256 = _mm256_fmadd_pd(pv0256, pu0256, puv256);

			   }

			   b0 = sqrt(b0);
			   bLog0 = bLog0 * 0.5;
			   // /2^N
			   bLog0 -= 0.5 * N;
			   T bLog2 = pow(2, bLog0);

			   puv = (puvPtr[0] + (puvPtr[3] + puvPtr[2] + puvPtr[1])) * b0 * bLog2;

			   if (N % 2 && l > 0)
			   {
				   puv = -puv;
			   }

			   T arg = (T)((piD / 2.0) * (N + HG_m));

			   coefR = (T)(puv * cos(arg));
			   coefI = (T)(puv * sin(arg));
		   }
	   }


	   int digHoloLGInit(int maxMG, complex64**& LGtoHG, int*& L, int*& P, int* M, int* N)
	   {
		  // const size_t totalElementCount = 
			allocate2DSparseBlock(maxMG, LGtoHG);
		   size_t modeCount = 0;
		   for (size_t mgIdx = 1; mgIdx <= maxMG; mgIdx++)
		   {
			   modeCount += mgIdx;
		   }

		   allocate1D(modeCount, L);
		   allocate1D(modeCount, P);

		   //Initialise the lookup table
		   int** LG_UV = 0;
		   allocate2D(maxMG, maxMG, LG_UV);
		   for (int i = 0; i < maxMG; i++)
		   {
			   for (int j = 0; j < maxMG; j++)
			   {
				   LG_UV[i][j] = -1;
			   }
		   }
		   //Setup the LG(l,p) indices to match the HG(m,n) indices.
		   //LG modes are enumerated such that if a HG[idx] has a astigmatic mode converter (cylindrical lens) applied, it'll give you LG[idx]
		   //Also creates a lookup table that allows us to calculate the coefficients in an order such that we can reuse the 'pu' term below.
		   int idx = 0;
		   for (int mgIdx = 1; mgIdx <= maxMG; mgIdx++)
		   {
			   int mgIDX = mgIdx - 1;
			   for (int modeIdx = 1; modeIdx <= mgIdx; modeIdx++)
			   {
				   int m = M[idx];
				   int n = N[idx];
				   int l = m - n;
				   int p = (mgIDX - abs(l)) / 2;
				   L[idx] = l;
				   P[idx] = p;

				   int u, v;

				   if (l < 0)
				   {
					   u = p;
					   v = -l + u;
				   }
				   else
				   {
					   v = p;
					   u = l + v;
				   }
				   LG_UV[u][v] = idx;

				   idx++;
			   }
		   }

		   //Setup the coefficients to convert HGtoLG
		   double* pu = 0;
		   size_t memLength = maxMG + 2;
		   allocate1D(memLength, pu);
		   memset(pu, 0, memLength * sizeof(double));


		   double* pv = 0;
		   allocate1D(memLength, pv);


		   for (int uIdx = 0; uIdx < maxMG; uIdx++)
		   {
			   pu[0] = 1;
			   pu[uIdx] = 1;
			   int dn = uIdx;
			   int halfWay = (int)floor(uIdx / 2.0f) + 1;

			   //These terms are symmetric, e.g. 1 9 36 84 126 126 84 36 9 1
			   //So we'll calculate half, and the copy it to the other half.
			   //This could have some speed benefit, but mostly it's for floating point precision reasons.
			   for (int polyIdx = 1; polyIdx < halfWay; polyIdx++)
			   {
				   pu[polyIdx] = ((pu[polyIdx - 1] * dn) / polyIdx);
				   pu[uIdx - polyIdx + 1 - 1] = pu[polyIdx];

				   dn = dn - 1;
			   }

			   memset(pv, 0, memLength * sizeof(double));
			   for (int vIdx = 0; vIdx < maxMG; vIdx++)
			   {
				   int lgIdx = LG_UV[uIdx][vIdx];

				   int dm = vIdx;
				   int halfWay = (int)floor(vIdx / 2.0f) + 1;
				   double flipIt = (vIdx % 2) ? -1 : 1;
				   pv[0] = 1;
				   pv[vIdx] = flipIt;

				   for (int polyIdx = 1; polyIdx < halfWay; polyIdx++)
				   {
					   pv[polyIdx] = -pv[polyIdx - 1] * dm / polyIdx;
					   pv[vIdx - polyIdx + 1 - 1] = flipIt * pv[polyIdx];
					   dm--;

				   }

				   if (lgIdx >= 0)
				   {
					   int l = L[lgIdx];
					   int p = P[lgIdx];
					   int mg = uIdx + vIdx;// 2 * p + abs(l);
					   complex64* lghgCoef = LGtoHG[mg];
					   int lOffset = ((-l + mg) / 2);

					   for (int mgIdx = 0; mgIdx <= mg; mgIdx++)
					   {
						   int m = mgIdx;
						   int n = mg - m;
						   double coefR = 0;
						   double coefI = 0;
						   digHoloLGtoHGcoef<double>(l, p, m, n, coefR, coefI, pu, pv);

						   idx = n * (mg + 1) + lOffset;
						   lghgCoef[idx][0] = (float)coefR;
						   lghgCoef[idx][1] = (float)coefI;
					   }
				   }
			   }
		   }

		   free1D(pu);
		   free1D(pv);
		   free2D(LG_UV);
		   return maxMG;
	   }

	   unsigned char digHoloGenerateBasisInit(int maxMG, int width, int height, unsigned char squarePixels, int polCount)
	   {
		   unsigned char modeCountChanged = false;
		   unsigned char polCountChanged = false;
		   const int maxGroup = maxMG + 1;
		   unsigned char recalculated = false;

		   //If the number of modes has changed, or this is the first time
		   //Calculate the indices and coefficients
		   if (maxMG != digHoloBasisValid_maxMG || HG_M == 0 || HG_N == 0)
		   {
			   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
			   {
				   fprintf(consoleOut, "Group count changed\n\r");
				   fflush(consoleOut);
			   }
			   int modeCount = 0;
			   allocate2D(maxGroup, maxGroup, HG_MN);
			   
			   for (int i = 1; i < maxGroup; i++)
			   {
				   modeCount += i;
				   for (int j = 0; j < maxGroup; j++)
				   {
					   HG_MN[i - 1][j] = -1;
				   }
			   }

			   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
			   {
				   fprintf(consoleOut, "HG indices.\n\r");
				   fflush(consoleOut);
			   }
			   allocate1D(modeCount, HG_M);
			   allocate1D(modeCount, HG_N);
			   digHoloGenerateHGindices(maxMG, HG_M, HG_N, HG_MN);

			   digHoloModeCount = modeCount;
			   digHoloBasisValid_maxMG = maxMG;
			   modeCountChanged = true;
			   digHoloBasisTransformIsValid = false;
		   }

		   if (polCount != digHoloBasisValid_PolCount || modeCountChanged)
		   {
			   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
			   {
				   fprintf(consoleOut, "Allocating digHoloHGX/Y, scales etc.\n\r");
				   fflush(consoleOut);
			   }
			   int n = 0;
			   //The extra +1 is because we're going to store a backup of the pointers to the second polarisation for switching pointers around when PolLockBasisWaist is enabled/disabled.
			   if (polCount == 1)
			   {
				   n = 0;
			   }
			   else
			   {
				   n = 1;
			   }
			   allocate2D(polCount+n, maxGroup, digHoloHGX);
			   allocate2D(polCount+n, maxGroup, digHoloHGY);

			   allocate2D(polCount, maxGroup, digHoloHGscaleX);
			   allocate2D(polCount, maxGroup, digHoloHGscaleY);
			   allocate1D(polCount, digHoloBasisValid_Waist);
			   memset(digHoloBasisValid_Waist, 0, sizeof(float) * polCount);
			   polCountChanged = true;
		   }

		   if (modeCountChanged || !digHoloBasisValid || polCountChanged)
		   {
			   unsigned char longestAxis = 0;
			   int axisLong = 0;
			   int axisShort = 0;

			   if (width >= height)
			   {
				   axisLong = width;
				   axisShort = height;
				   longestAxis = 0;
			   }
			   else
			   {
				   axisLong = height;
				   axisShort = width;
				   longestAxis = 1;
			   }
			   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
			   {
				   fprintf(consoleOut, "Allocating digHoloHG\n\r");
				   fflush(consoleOut);
			   }
			   //The only axis calculated is the largest axis. The other just points to the correct position in this axis
			   allocate3D(polCount,maxGroup, axisLong * (!squarePixels + 1), digHoloHG);

			   //If the pixels are square. X and Y axis are the same. Hence only a single dimension is calculated and saved (the longest dimension).
			   //If the pixels aren't square. Then separate HG modes must be calculated for both the X and Y axes.
			   //axisOffset sets whether the X and Y modes point to largely overlapping parts of memory, or to their own parts of memory.
			   //Keeping it small/memory reuse might have advantages for caching. The entire mode basis gets so small it might only be a few kB in size.
			   //Speed ups for overlap routine, as the modes can live in cache.
			   int axisOffset = (axisLong - axisShort) / 2;

			   //If the pixels aren't square, then we'll be calculating separate modes for X and Y
			   if (!squarePixels)
			   {
				   axisOffset = axisLong;
			   }
			   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
			   {
				   fprintf(consoleOut, "Referencing pointers\n\r");
				   fflush(consoleOut);
			   }
			   for (int polIdx = 0; polIdx < polCount; polIdx++)
			   {
				   if (longestAxis == 0)
				   {
					   for (int modeIdx = 0; modeIdx < maxGroup; modeIdx++)
					   {
						   digHoloHGX[polIdx][modeIdx] = &digHoloHG[polIdx][modeIdx][0];
						   digHoloHGY[polIdx][modeIdx] = &digHoloHG[polIdx][modeIdx][axisOffset];
					   }
				   }
				   else
				   {
					   for (int modeIdx = 0; modeIdx < maxGroup; modeIdx++)
					   {
						   digHoloHGY[polIdx][modeIdx] = &digHoloHG[polIdx][modeIdx][0];
						   digHoloHGX[polIdx][modeIdx] = &digHoloHG[polIdx][modeIdx][axisOffset];
					   }
				   }
			   }

			   if (polCount > 1)
			   {
				   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
				   {
					   fprintf(consoleOut, "Memcpy digHoloHGX/Y\n\r");
					   fflush(consoleOut);
				   }
				   memcpy(&digHoloHGX[2][0], &digHoloHGX[1][0], sizeof(short**)* maxGroup);
				   memcpy(&digHoloHGY[2][0], &digHoloHGY[1][0], sizeof(short**)* maxGroup);
				  // memcpy(&digHoloHGscaleX[2][0], &digHoloHGscaleX[1][0], sizeof(float)* maxGroup);
				   //memcpy(&digHoloHGscaleY[2][0], &digHoloHGscaleY[1][0], sizeof(float)* maxGroup);
			   }

			   recalculated = true;
			   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
			   {
				   fprintf(consoleOut, "Allocated basis %i x %i x %i \n\r", digHoloModeCount, width, height);
				   fflush(consoleOut);
			   }
		   }

		   return recalculated;
	   }

	   void digHoloUpdateReferenceWave(int polIdx0)
	   {
		   const int fftIdx = digHoloResolutionIdx;

		   const int pixelCountX = digHoloWoiPolWidth[fftIdx];
		   const int pixelCountY = digHoloWoiPolHeight[fftIdx];

		   //This scenario should only occur the first time this routine is run
		   if (!digHoloRefTiltX_Valid)
		   {
			   //Just hard coded as 2, won't bother reallocating if the polCount changes.
			   const size_t polCount = DIGHOLO_POLCOUNTMAX;// digHoloPolCount;
			   const size_t parameterCount = 6;
			   //Just allocate it as 1 array an dereference
			   allocate1D(parameterCount*polCount, digHoloRefTiltX_Valid);

			   digHoloRefTiltY_Valid = &digHoloRefTiltX_Valid[polCount];

			   digHoloRefTiltXoffset_Valid = &digHoloRefTiltX_Valid[2*polCount];
			   digHoloRefTiltYoffset_Valid = &digHoloRefTiltX_Valid[3 * polCount];

			   digHoloRefDefocus_Valid = &digHoloRefTiltX_Valid[4 * polCount];

			   digHoloRefCentreX_Valid = &digHoloRefTiltX_Valid[5 * polCount];
			   digHoloRefCentreY_Valid = &digHoloRefTiltX_Valid[6 * polCount];

			   memset(digHoloRefTiltX_Valid, 0, sizeof(float) * polCount * parameterCount);

			   digHoloRefPixelCountX_Valid = 0;
			   digHoloRefPixelCountY_Valid = 0;
			   digHoloRefPolCount_Valid = 0;

			   digHoloRefWavelengthCount_Valid = 0;

			   digHoloRefValid = false;
		   }

		   if (digHoloRefPixelCountY_Valid != pixelCountY || digHoloRefPixelCountX_Valid != pixelCountX || digHoloPolCount!=digHoloRefPolCount_Valid || digHoloWavelengthCount!=digHoloRefWavelengthCount_Valid)
		   {
			   //We'll allocate the x and y components next to each other for spatial locality. Probably doesn't make much difference.
			   //Also real and imaginary components are stored adjacent, so it goes Xr,Xi,Yr,Yi
			   allocate3D(digHoloWavelengthCount,digHoloPolCount, 2*(pixelCountX+pixelCountY), digHoloRef);
			   digHoloRefValid = false;
		   }
		   
		   if (digHoloWavelengthCount != digHoloRefWavelengthCount_Valid)
		   {
			   digHoloRefValid = false;
			   allocate1D(digHoloWavelengthCount,digHoloRefWavelength_Valid);
			   
		   }
		   else
		   {
			   //Check if the wavelength sweep has changed. Would be better to flag this somewhere, e.g. where it was set.
			   unsigned char valid = true;

			   const int lambdaCount = digHoloWavelengthCount;
			   for (int lambdaIdx = 0; ((lambdaIdx < lambdaCount) && valid); lambdaIdx++)
			   {
				   valid = valid && (digHoloWavelength[lambdaIdx] == digHoloRefWavelength_Valid[lambdaIdx]);
			   }

			   if (!valid)
			   {
				   digHoloRefValid = false;
			   }
		   }

		   int polStart = polIdx0;
		   int polStop = polIdx0 + 1;
		   if (polIdx0 < 0)
		   {
			   polStart = 0;
			   polStop = digHoloPolCount;
		   }

		   //The tilt/defocus parameters from the config object.
		   float** zernCoefs = config.zernCoefs;
		   float tiltX[DIGHOLO_POLCOUNTMAX];
		   float tiltY[DIGHOLO_POLCOUNTMAX];
		   float defocus[DIGHOLO_POLCOUNTMAX];
		   float tiltXoffset[DIGHOLO_POLCOUNTMAX];
		   float tiltYoffset[DIGHOLO_POLCOUNTMAX];
		   float centreX[DIGHOLO_POLCOUNTMAX];
		   float centreY[DIGHOLO_POLCOUNTMAX];

		   unsigned char somethingChanged[DIGHOLO_POLCOUNTMAX];// = { !digHoloRefValid,!digHoloRefValid };
		   memset(&somethingChanged, !digHoloRefValid, sizeof(unsigned char) * DIGHOLO_POLCOUNTMAX);

		   //For each active polarisation
		   for (int polIdx = polStart; polIdx < polStop; polIdx++)
		   {
			   //Go through and enforce and tilt/defocus locks between the two polarisations.
			   //If polLocks are enabled, that means the two polarisation must have the same value.
			   //So if we're currently doing H, V will be locked to this value, and vice-versa.
			   int polIDX = polIdx;
			   //Check if the tilts for the two polarisations are locked together
			   if (config.PolLockTilt)
			   {
				   //If they are, enforce tilt of V polarisation to match H polarisation
				   int polIDX = 0;
				   if (polIdx != polIDX)
				   {
					   zernCoefs[polIdx][TILTX] = zernCoefs[polIDX][TILTX];
					   zernCoefs[polIdx][TILTY] = zernCoefs[polIDX][TILTY];
				   }
			   }

			   //Check if the focus of the two polarisations are locked together
			   if (config.PolLockDefocus)
			   {
				   polIDX = 0;
				   //If they are, lock the V polarisation to the H polarisation (enforce them both to have the same focus)
				   if (polIdx != polIDX)
				   {
					   zernCoefs[polIdx][DEFOCUS] = zernCoefs[polIDX][DEFOCUS];
				   }
			   }

			   //The tilt that must be applied to this polarisation. Converted from degrees (user-facing format) to radians (calculation format)
			   tiltX[polIdx] = (float)(zernCoefs[polIDX][TILTX] * DIGHOLO_UNIT_ANGLE);
			   tiltY[polIdx] = (float)(zernCoefs[polIDX][TILTY] * DIGHOLO_UNIT_ANGLE);

			   //An offset to the tilt which must be applied based on exactly how the FFT window is selected around the area of interest.
			   //Only relevant when the low-res "window only" version of the IFFT is being applied. The centre of the window may not be exactly in the centre of the 2D FFT.
			   //For example it might be centred at a position between the pixels (it usually will be unless you're lucky). But the window itself might also be a bit shifted due to SIMD memory alignment concerns (e.g. multiples of 8 pixels).
			   tiltXoffset[polIdx] = 0;
			   tiltYoffset[polIdx] = 0;
			   if (fftIdx == LOWRES_IDX)
			   {
				   float dkx = (digHoloKXaxis[1] - digHoloKXaxis[0]);
				   float dky = (digHoloKYaxis[1] - digHoloKYaxis[0]);
				   tiltXoffset[polIdx] = (float)((digHoloWoiSX[IFFT_IDX][polIDX] - digHoloWoiPolWidth[FFT_IDX] / 2.0 - 0) * dkx);//Extra -1 was in there? but was compensated later by DX=1 in the CopyWindow routine. That appears to have been an error or no practical consequence because the first colIdx in CopyWindow is almost entirely masked out anyways due to k-filter
				   tiltYoffset[polIdx] = (float)((digHoloWoiSY[IFFT_IDX][polIDX] - digHoloWoiPolHeight[FFT_IDX] / 2.0) * dky);;//
			   }

			   //The defocus term for this polarisation. It's specified as a dioptre.
			   //Tilt is specified as an angle, and focus as a dioptre so that wavelength dependent calibrations can be easily applied.
			   defocus[polIdx] = zernCoefs[polIDX][DEFOCUS];

			   //Beam centre position on the camera. With regards to tilt/focus, this ends up being used for defining the (0,0) position for the tilt.
			   //As the beam is always centred at 0,0 after the IFFT, even if it wasn't in the original camera image, this centre position is still important for removing
			   //a phase term introduced by the tilt applied in the Fourier plane (digHoloCopyWindow), which vs. wavelength would otherwise look like a false delay (wavelength-dependent phase shift).
			   centreX[polIdx] = (float)(config.BeamCentreX[polIdx] * DIGHOLO_UNIT_PIXEL);
			   centreY[polIdx] = (float)(config.BeamCentreY[polIdx] * DIGHOLO_UNIT_PIXEL);

			   //Check for changes if we don't already know we have to update
			   if (!somethingChanged[polIdx])
			   {
				   if (centreX[polIdx] != digHoloRefCentreX_Valid[polIdx] || centreY[polIdx] != digHoloRefCentreY_Valid[polIdx] ||
					   tiltX[polIdx] != digHoloRefTiltX_Valid[polIdx] || tiltY[polIdx] != digHoloRefTiltY_Valid[polIdx] ||
					   tiltXoffset[polIdx] != digHoloRefTiltXoffset_Valid[polIdx] || tiltYoffset[polIdx] != digHoloRefTiltYoffset_Valid[polIdx] ||
					   defocus[polIdx] != digHoloRefDefocus_Valid[polIdx])
				   {
					   somethingChanged[polIdx] = 1;
				   }
			   }

			   digHoloRefCentreX_Valid[polIdx] = centreX[polIdx];
			   digHoloRefCentreY_Valid[polIdx] = centreY[polIdx];
			   digHoloRefTiltX_Valid[polIdx] = tiltX[polIdx];
			   digHoloRefTiltY_Valid[polIdx] = tiltY[polIdx];
			   digHoloRefTiltXoffset_Valid[polIdx] = tiltXoffset[polIdx];
			   digHoloRefTiltYoffset_Valid[polIdx] = tiltYoffset[polIdx];
			   digHoloRefDefocus_Valid[polIdx] = defocus[polIdx];
		   }



		   float* xAxis = digHoloXaxis[fftIdx];
		   float* yAxis = digHoloYaxis[fftIdx];
		   float* lambda = digHoloWavelength;
		   int lambdaCount = digHoloWavelengthCount;
		   
		   digHoloGenerateReferenceWave(polStart, polStop, &somethingChanged[0], &tiltX[0], &tiltY[0], &tiltXoffset[0], &tiltYoffset[0], &defocus[0], &centreX[0], &centreY[0], xAxis, yAxis, pixelCountX, pixelCountY, lambda, lambdaCount);

		   digHoloRefValid = true;

		   digHoloRefPixelCountX_Valid = pixelCountX;
		   digHoloRefPixelCountY_Valid = pixelCountY;
		   digHoloRefPolCount_Valid = digHoloPolCount;
		   digHoloRefWavelengthCount_Valid = digHoloWavelengthCount;
		   memcpy(digHoloRefWavelength_Valid, digHoloWavelength, sizeof(float) * digHoloWavelengthCount);
	   }

	   void digHoloGenerateReferenceWave(int polStart, int polStop, unsigned char* somethingChanged, float* tiltX, float* tiltY, float* tiltOffsetX, float* tiltOffsetY, float* defocus, float* beamCentreX, float* beamCentreY, float* Xf, float* Yf, int pixelCountX, int pixelCountY, float* lambda, int lambdaCount)
	   {
		   for (int polIdx = polStart; polIdx < polStop; polIdx++)
		   {
			   if (somethingChanged[polIdx])
			   {
				   for (int lambdaIdx = 0; lambdaIdx < lambdaCount; lambdaIdx++)
				   {
					   
					   short* digHoloRefXr = &digHoloRef[lambdaIdx][polIdx][0];
					   short* digHoloRefXi = &digHoloRef[lambdaIdx][polIdx][pixelCountX];
					   short* digHoloRefYr = &digHoloRef[lambdaIdx][polIdx][2 * pixelCountX];
					   short* digHoloRefYi = &digHoloRef[lambdaIdx][polIdx][2 * pixelCountX + pixelCountY];

					   const __m256 Float32toInt16Scale = _mm256_set1_ps(16383.0);

					   const size_t blockSize = 8;
					   //const size_t pxCount = pixelCountX / blockSize;

					   const float centreX = beamCentreX[polIdx];
					   const float centreY = beamCentreY[polIdx];
					   const float dioptre = defocus[polIdx];
					   const float thetaX = tiltX[polIdx];
					   const float thetaY = tiltY[polIdx];
					   const float thetaOffsetX = tiltOffsetX[polIdx];
					   const float thetaOffsetY = tiltOffsetY[polIdx];

					   //const __m256 signFlips = _mm256_set_ps(+1, -1, +1, -1, +1, -1, +1, -1);
					   const __m256 dx = _mm256_set1_ps(centreX);
					   const __m256 dy = _mm256_set1_ps(centreY);

					   const float lambda0 = lambda[lambdaIdx];
					   const float k0 = (float)(2.0 * pi / lambda0);

					   const __m256 kx = _mm256_set1_ps(-k0 * (thetaX)+thetaOffsetX);//Could get away with ditching the sinf (small angle), but no different
					   const __m256 ky = _mm256_set1_ps(-k0 * (thetaY)+thetaOffsetY);
					   __m256 kx0;
					   __m256 ky0;

					   if (thetaOffsetX == 0)
					   {
						   kx0 = kx;
					   }
					   else
					   {
						   kx0 = _mm256_set1_ps(k0 * (-(thetaX)));
					   }
					   if (thetaOffsetY == 0)
					   {
						   ky0 = ky;
					   }
					   else
					   {
						   ky0 = _mm256_set1_ps(k0 * (-(thetaY)));
					   }
					   const __m256 focalTerm = _mm256_set1_ps((float)(-pi * dioptre / lambda0));//Make a length of focal length = 1/dioptre

					   //Removes the phase shift that would have been introduced by the Fourier tilt used to centre the beam in the reconstructed camera plane.
					   //Without this, there would be an apparent delay (wavelength-dependent phase shift).




					   for (int axisIdx = 0; axisIdx < 2; axisIdx++)
					   {
						   __m256 pistonTerm;
						   float* axis = 0;
						   size_t pixelCount = 0;

						   short* outR = 0;
						   short* outI = 0;
						   __m256 k;
						   if (axisIdx == 0)
						   {
							   pistonTerm = _mm256_mul_ps(dx, kx0);
							   axis = Xf;
							   pixelCount = pixelCountX;
							   outR = digHoloRefXr;
							   outI = digHoloRefXi;
							   k = kx;
						   }
						   else
						   {
							   pistonTerm = _mm256_mul_ps(dy, ky0);
							   axis = Yf;
							   pixelCount = pixelCountY;
							   outR = digHoloRefYr;
							   outI = digHoloRefYi;
							   k = ky;
						   }

						   for (size_t pixelIdy = 0; pixelIdy < pixelCount; pixelIdy += blockSize)
						   {
							   const __m256 r = _mm256_loadu_ps(&axis[pixelIdy]);
							   const __m256 r2 = _mm256_mul_ps(r, r);
							   const __m256 k_r = _mm256_fmadd_ps(k, r, pistonTerm);

							   const __m256 focalAdjustmentArg = _mm256_mul_ps(focalTerm, r2);
							   const __m256 tiltAdjustmentArg = k_r;
							   const __m256 totalArg = _mm256_add_ps(tiltAdjustmentArg, focalAdjustmentArg);

							   __m256 adjustR;
							   __m256 adjustI = sincos_ps(&adjustR, totalArg);
							   
							   const __m256i aR32 = _mm256_cvtps_epi32(_mm256_mul_ps(Float32toInt16Scale, adjustR));
							   const __m256i aI32 = _mm256_cvtps_epi32(_mm256_mul_ps(Float32toInt16Scale, adjustI));

							   const __m128i adjustR16 = _mm256_extractf128_si256(_mm256_packs_epi32(aR32, _mm256_permute2f128_si256(aR32, aR32, 1)), 0);
							   const __m128i adjustI16 = _mm256_extractf128_si256(_mm256_packs_epi32(aI32, _mm256_permute2f128_si256(aI32, aI32, 1)), 0);

								_mm_storeu_si128((__m128i*) & outR[pixelIdy], adjustR16);
								_mm_storeu_si128((__m128i*) & outI[pixelIdy], adjustI16);

						   }
					   }//axisIdx
				   }
			   }//if something changed
		   }//polIdx
	   }

	   void digHoloUpdateBasis(int basisIdx, int& pixelCountX, int& pixelCountY)
	   {
		   const int maxMG = config.maxMG;

		   if (maxMG > 0)
		   {
			   float waist[2] = { (float)(config.waist[0] * DIGHOLO_UNIT_PIXEL),(float)(config.waist[1] * DIGHOLO_UNIT_PIXEL) };

			   for (int polIDX = 0; polIDX < digHoloPolCount; polIDX++)
			   {
				   if (waist[polIDX] == 0)
				   {
					   waist[polIDX] = 400e-6f;
					   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
					   {
						   fprintf(consoleOut, "Waist cannot be zero. Resetting to default %f\n\r", config.waist[polIDX]);
						   fflush(consoleOut);
					   }
				   }
			   }

			   if (config.PolLockBasisWaist)
			   {
				   waist[1] = waist[0];
				   config.waist[1] = config.waist[0];
			   }

			   pixelCountX = digHoloWoiPolWidth[basisIdx];
			   pixelCountY = digHoloWoiPolHeight[basisIdx];
			   float* xAxis = digHoloXaxis[basisIdx];
			   float* yAxis = digHoloYaxis[basisIdx];
			   digHoloGenerateHGBasis(pixelCountX, pixelCountY, xAxis, yAxis, maxMG, &waist[0],digHoloPolCount);
		   }
	   }


	   //void digHoloGenerateHGBasis(unsigned int precisionIdx, unsigned int polMode, float* waists, float* dx, float* dy, float* dz, int maxMG, int negateZ, float lambda0, int polIDX)
	   void digHoloGenerateHGBasis(int pixelCountX, int pixelCountY, float* xAxis, float* yAxis, int maxMG, float *waist, int polCount)
	   {
		   const float pixelSizeX = (xAxis[1] - xAxis[0]);
		   const float pixelSizeY = (yAxis[1] - yAxis[0]);
		   const unsigned char squarePixels = (pixelSizeX == pixelSizeY);
		   unsigned char polLock = (waist[0] == waist[1]);

		   unsigned char recalculate0 = digHoloGenerateBasisInit(maxMG, pixelCountX, pixelCountY, squarePixels,polCount);

		   int polCOUNT = polCount;

		   if (polCount > 1)
		   {
			   if (polLock)
			   {
				   polCOUNT = 1;
			   }

				   //If the pols are locked, then copy the pointers from the H polarisation to the V polarisation
				   int polSourceIdx = 0;
				   if (polLock)
				   {
				   }
				   else
				   {
					   //If the pols aren't locked, copy the pointers from the backup of the V polarisation.
					   polSourceIdx = 2;
				   }

				   memcpy(&digHoloHGX[1][0], &digHoloHGX[polSourceIdx][0], sizeof(short**) * maxMG);
				   memcpy(&digHoloHGY[1][0], &digHoloHGY[polSourceIdx][0], sizeof(short**) * maxMG);
		   }
		   
		   for (int polIdx = 0; polIdx < polCOUNT; polIdx++)
		   {
			   unsigned char recalculate = false;
			   //If the waist changes, we'll also have to recalculate
			   if (digHoloBasisValid_Waist[polIdx] != waist[polIdx])
			   {
				   recalculate = true;
			   }

			   if (recalculate0 || recalculate)
			   {
				   const unsigned char squarePixels = (pixelSizeX == pixelSizeY);

				   if (squarePixels)
				   {
					   float* AXIS = 0;
					   short*** basisArray = digHoloHG;

					   int pixelCountMax = 0;

					   if (pixelCountX >= pixelCountY)
					   {
						   AXIS = xAxis;
						   pixelCountMax = pixelCountX;

					   }
					   else
					   {
						   AXIS = yAxis;
						   pixelCountMax = pixelCountY;
					   }

					   //This is normally going to be too small to benefit from threading
					   HGbasisXY1D(digHoloHGscaleX[polIdx], waist[polIdx], (float*)AXIS, pixelCountMax, maxMG, (__m128i**)basisArray[polIdx]);
					   memcpy(digHoloHGscaleY[polIdx], digHoloHGscaleX[polIdx], sizeof(float) * maxMG);
				   }
				   else
				   {
					   HGbasisXY1D(digHoloHGscaleX[polIdx], waist[polIdx], (float*)xAxis, pixelCountX, maxMG, (__m128i**)digHoloHGX[polIdx]);
					   HGbasisXY1D(digHoloHGscaleY[polIdx], waist[polIdx], (float*)yAxis, pixelCountY, maxMG, (__m128i**)digHoloHGY[polIdx]);
				   }

				   digHoloBasisValid_maxMG = maxMG;
				   digHoloBasisValid_Waist[polIdx] = waist[polIdx];
				   digHoloBasisValid_PolCount = polCount;
				   digHoloBasisValid = true;
			   }
		   }


		   if (polLock)
		   {
			   if (polCount > 1)
			   {
				   //If the pols are locked, then copy the pointers from the H polarisation to the V polarisation
				   int polSourceIdx = 0;
				   if (polLock)
				   {
				   }
				   else
				   {
					   //If the pols aren't locked, copy the pointers from the backup of the V polarisation.
					   polSourceIdx = 2;
				   }

				   memcpy(&digHoloHGscaleX[1][0], &digHoloHGscaleX[polSourceIdx][0], sizeof(float) * maxMG);
				   memcpy(&digHoloHGscaleY[1][0], &digHoloHGscaleY[polSourceIdx][0], sizeof(float) * maxMG);
			   }
		   }
	   }


	   double factorial(int n)
	   {
		   double b = 1;
		   for (int a = n; a > 1; a--)
		   {
			   b = b * a;
		   }
		   return b;
	   }

	   double factorialDivide(int n, int m)
	   {
		   if (n >= m)
		   {
			   double b = 1;
			   for (int a = n; a > m; a--)
			   {
				   b = b * a;
			   }
			   return b;
		   }
		   else
		   {
			   double b = 1;
			   for (int a = m; a > n; a--)
			   {
				   b = b * a;
			   }
			   return 1.0 / b;
		   }
	   }

	   double factorialDivideSqrt(int n, int m)
	   {
		   if (n >= m)
		   {
			   double b = 1;
			   for (int a = n; a > m; a--)
			   {
				   b = b * sqrt(a);
			   }
			   return b;
		   }
		   else
		   {
			   double b = 1;
			   for (int a = m; a > n; a--)
			   {
				   b = b * sqrt(a);
			   }
			   return 1.0 / b;
		   }
	   }


	   /*
	   void digHoloPowerMeter(float** out, bool polsAreIndependent, bool doIL, bool doMDL, bool doXtalk, bool mulConjTBasis,int lambdaCount,int lambdaStart,int lambdaStop)
	   {
		   int polThreads = 1;
		   int lambdaThreads = 1;
		   digHoloPowerMeterWorkspaceAllocate(polThreads, lambdaThreads, lambdaCount);
		   complex64* workspace = digHoloPowerMeterWorkspace[0][0];
		   return digHoloPowerMeter(out, -1, polsAreIndependent, NULL, doIL, doMDL, doXtalk, mulConjTBasis,workspace,lambdaCount,lambdaStart,lambdaStop);
	   }
	   */
	   //overlapCoefs is a copy of the overlap matrix, needed when you want to launch multiple threads. In that scenario you need a copy
	   void digHoloPowerMeter(float** out, int polIDX, int polsAreIndependent, complex64** overlapCoefs, bool doIL, bool doMDL, bool doXtalk, bool mulConjTBasis, complex64* workspace, int lambdaCount, int lambdaStart, int lambdaStop)
	   {
		   
		   //Special case, where there is no modal basis.
		   if (config.maxMG == 0)
		   {
			  // memset(&out[0][0], 0, DIGHOLO_METRIC_COUNT*lambdaCount);
			   for (int lambdaIdx = lambdaStart; lambdaIdx < lambdaStop && lambdaIdx < digHoloBatchCOUNT && digHoloBatchCOUNT >= lambdaCount; lambdaIdx++)
			   {
				   const size_t batchCount = digHoloBatchCOUNT / lambdaCount;
				   size_t batchOffset = lambdaIdx * batchCount;
				   size_t batchStride = 1;

				   //If the coefficients are organised in wavelength-fast order, we'll have to access and copy them in a strided fashion.
				   if (config.wavelengthOrdering[DIGHOLO_WAVELENGTHORDER_OUTPUT] == DIGHOLO_WAVELENGTHORDER_FAST)
				   {
					   batchOffset = lambdaIdx;
					   batchStride = lambdaCount;
				   }

				   float totalPwr = 0;
				   for (size_t batchIdx = 0; batchIdx < batchCount; batchIdx++)
				   {
					   for (int polIdx = 0; polIdx < digHoloPolCount; polIdx++)
					   {
						   // totalPwr += DIGHOLO_ANALYSIS[FFT_IDX][DIGHOLO_ANALYSIS_TOTALPOWER][polIdx][digHoloBatchCOUNT];
						   totalPwr += DIGHOLO_ANALYSIS[FFT_IDX][DIGHOLO_ANALYSIS_TOTALPOWER][polIdx][batchOffset+lambdaIdx*batchStride];
					   }
				   }

				   if (config.AutoAlignPolIndependence)
				   {
					   totalPwr = totalPwr / digHoloBatchCOUNT;
				   }
				   else
				   {
					   totalPwr = totalPwr / (digHoloPolCount * digHoloBatchCOUNT);
				   }
				   out[DIGHOLO_METRIC_IL][lambdaIdx] = (totalPwr);
			   }
			 //  return;
		   }
		   else
		   {

			   int polIdx = polIDX;
			   //If the user has specified a specific polarisation, then only analyse that polarisation
			   int polCount = 1;
			   if (polIDX < 0)
			   {
				   polIdx = 0;
				   //If both polarisations are to be calculated (polIDX==-1), then we won't be using the overlapCoefs array.
				   overlapCoefs = NULL;
				   //If the user specified a negative polarisation idx, then we're dealing with both polarisations.
				   polCount = digHoloPolCount;
			   }
			   // const size_t lambdaCount = digHoloWavelengthCount;

			   float il = 0;
			   for (int lambdaIdx = lambdaStart; lambdaIdx < lambdaStop && lambdaIdx < digHoloBatchCOUNT && digHoloBatchCOUNT >= lambdaCount; lambdaIdx++)
			   {
				   // int lambdaIdx = digHoloWavelengthCount / 2;
				   const size_t batchCount = digHoloBatchCOUNT / lambdaCount;
				   size_t batchOffset = lambdaIdx * batchCount;
				   size_t batchStride = 1;

				   //If the coefficients are organised in wavelength-fast order, we'll have to access and copy them in a strided fashion.
				   if (config.wavelengthOrdering[DIGHOLO_WAVELENGTHORDER_OUTPUT] == DIGHOLO_WAVELENGTHORDER_FAST)
				   {
					   batchOffset = lambdaIdx;
					   batchStride = lambdaCount;
				   }

				   complex64** sourcePtr = digHoloOverlapCoefsPtr;
				   complex64** sourcePtrOtherPol = digHoloOverlapCoefsPtr;
				   if (overlapCoefs)
				   {
					   sourcePtrOtherPol = overlapCoefs;
				   }

				   bool independentPols = polsAreIndependent || (digHoloPolCount == 1);

				   const size_t matrixStride = batchCount * digHoloPolCount * digHoloModeCountOut;
				   size_t memOffset = ((size_t)(lambdaIdx*0 * digHoloPolCount + polIdx)) * matrixStride;
				   //totalThreads, digHoloBatchCOUNT, digHoloPolCount, digHoloModeCount * blockSize * 2
				   //So each thread has 8 full matrices worth
				  // complex64* workspace = (complex64*)&digHoloOverlapWorkspace[0][0][0][0];
				   complex64* A = &workspace[memOffset];
				   complex64* UU = &workspace[memOffset + 2 * matrixStride];

				   //This is overallocated
				   float* S = (float*)&workspace[memOffset + 4 * matrixStride];// &digHoloSVD_S[memOffset];//Shouldn't this be max/min dim?

				   int totalModeCount = digHoloPolCount * digHoloModeCountOut;


				   float mdl = 0;

				   int svdM = 0;
				   int svdN = 0;
				   int svdCount = 0;

				   if (digHoloBatchCOUNT > 1)
				   {
					   //There is only 1 polarisation, so just copy everything
					   if (digHoloPolCount == 1)
					   {
						   size_t idx = 0;
						   for (size_t batchIdx = 0; batchIdx < batchCount; batchIdx++)
						   {
							   memcpy(&A[idx][0], &sourcePtr[batchIdx * batchStride + batchOffset][0][0], sizeof(complex64) * totalModeCount);
							   idx += totalModeCount;
						   }
						   svdM = totalModeCount;
						   svdN = (int)batchCount;
						   svdCount = 1;
					   }
					   else //There's dual-polarisations
					   {
						   //There's dual-polarisations, but we're interested in one of them.
						   if (polCount == 1)
						   {
							   //If the polarisations are independent, then we can just look at 1 pol, and do a smaller SVD on it.
							   if (independentPols)
							   {

								   size_t idx = 0;
								   const size_t polOffset = polIDX * ((size_t)digHoloModeCountOut);
								   const size_t stride = digHoloModeCountOut;
								   for (size_t batchIdx = 0; batchIdx < batchCount; batchIdx++)
								   {
									   memcpy(&A[idx][0], &sourcePtr[batchIdx * batchStride + batchOffset][polOffset][0], sizeof(complex64) * stride);
									   idx += stride;
								   }
								   svdM = digHoloModeCountOut;
								   svdN = (int)batchCount;
								   svdCount = 1;
							   }
							   else //if the polarisations aren't independent. Then we'll need to construct a full matrix.
							   {//This routine might be running in parallel, whereby each polarisation is attempting to optimise in different threads.
								   //In that scenario, the coefficients for the other polarisations will be taken from a different static copy (sourcePtrOtherPol), so that the different polarisation optimisations
								   //don't interfere with one another.
								   size_t idx = 0;
								   size_t polOffset = polIDX * ((size_t)digHoloModeCountOut);
								   size_t polOffsetOther = !polIDX * digHoloModeCountOut;
								   size_t stride = digHoloModeCountOut * digHoloPolCount;
								   for (size_t batchIdx = 0; batchIdx < batchCount; batchIdx++)
								   {
									   //Copy one polarisation from the source
									   memcpy(&A[idx][0], &sourcePtr[batchIdx * batchStride + batchOffset][polOffset][0], sizeof(complex64) * digHoloModeCountOut);
									   //Copy the other polarisation from potentiall a different source.
									   memcpy(&A[idx + digHoloModeCountOut][0], &sourcePtrOtherPol[batchIdx * batchStride + batchOffset][polOffsetOther][0], sizeof(complex64) * digHoloModeCountOut);
									   idx += stride;
								   }
								   svdM = totalModeCount;
								   svdN = (int)batchCount;
								   svdCount = 1;
							   }
						   }
						   else //There's dual-polarisations and we want to look at both of them
						   {
							   //The polarisation components are completely independent, and can be thought of as independent, mutually incoherent fields.
							   //In this scenario we can do two smaller SVDs on two separate matrices.
							   if (independentPols)
							   {
								   size_t idx = 0;
								   const size_t polOffset = 0;
								   const size_t polOffsetOther = digHoloModeCountOut;
								   const size_t stride = digHoloModeCountOut;
								   const size_t matStride = digHoloModeCountOut * batchCount;
								   for (size_t batchIdx = 0; batchIdx < batchCount; batchIdx++)
								   {
									   //Copy one polarisation from the source
									   memcpy(&A[idx][0], &sourcePtr[batchIdx * batchStride + batchOffset][polOffset][0], sizeof(complex64) * stride);
									   //Copy the other polarisation from potentiall a different source.
									   memcpy(&A[idx + matStride][0], &sourcePtrOtherPol[batchIdx * batchStride + batchOffset][polOffsetOther][0], sizeof(complex64) * stride);
									   idx += stride;
								   }
								   svdM = digHoloModeCountOut;
								   svdN = (int)batchCount;
								   svdCount = 2;

							   }
							   else //The polarisation components are part of the same coherent field, and should be treated together as a state composed of vector of 2x(spatial modes)
							   {
								   //This is the same as digHoloPolCount==1, copy everything as-is.
								   size_t idx = 0;
								   size_t stride = totalModeCount;
								   for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
								   {
									   memcpy(&A[idx][0], &sourcePtr[batchIdx * batchStride + batchOffset][0][0], sizeof(complex64) * stride);
									   idx += stride;
								   }
								   svdM = totalModeCount;
								   svdN = (int)batchCount;
								   svdCount = 1;

							   }
						   }
					   }


					   float totalSignal = 0;
					   float totalNoise = 0;
					   float totalSNR = 0;

					   float totalSignalMG = 0;
					   float totalNoiseMG = 0;

					   float minSignal = FLT_MAX;
					   float maxSignal = -FLT_MAX;
					   float minNoise = FLT_MAX;
					   float maxNoise = -FLT_MAX;
					   float minSNR = FLT_MAX;
					   float maxSNR = -FLT_MAX;
					   int totalSignalCount = 1;
					   if (doXtalk)
					   {
						   for (int svdIdx = 0; svdIdx < svdCount; svdIdx++)
						   {
							   size_t idx = 0;
							   complex64* Aptr = &A[svdIdx * svdN * svdM];
							   int N = svdN;
							   int M = svdM;

							   //Analyse the matrix multiplied by it's conjugate transpose, rather than the raw matrix.
							   //This will give you something like signal and SNR, even if the basis being decomposed in, is not the goal basis.
							   //e.g. if there's an unknown unitary transform between what you measure, and the goal basis, this is useful.
							   //It's effectively a measure similar, but not the same, as il/mdl.
							   if (mulConjTBasis)
							   {
								   UUconj((int)svdM, svdN, Aptr, Aptr, (complex64*)&UU[0][0]);
								   Aptr = UU;
								   if (svdM < svdN)
								   {
									   // M = svdM;
									  //  N = svdN;
									   M = svdM;
									   N = svdM;
								   }
								   else
								   {
									   M = svdN;
									   N = svdN;
								   }
							   }

							   totalSignalCount = M * N * svdCount;
							   for (int i = 0; (i < N && i < digHoloModeCount); i++)//batchCount
							   {
								   float signal = 0;
								   float noise = 0;

								   int MGin = HG_M[i] + HG_N[i] + 1;

								   for (int j = 0; j < M; j++)//modeCount (modeCount x polCount for combined pols)
								   {
									   int MGout = HG_M[j % digHoloModeCountOut] + HG_N[j % digHoloModeCountOut] + 1;
									   float vR = Aptr[idx][0];
									   float vI = Aptr[idx][1];
									   float vPwr = vR * vR + vI * vI;
									   if (i == j || (i == (j - digHoloModeCountOut)))
									   {
										   signal += vPwr;
									   }
									   else
									   {
										   noise += vPwr;
									   }

									   if (MGin == MGout)
									   {
										   totalSignalMG += vPwr / MGin;
									   }
									   else
									   {
										   totalNoiseMG += vPwr / MGin;
									   }

									   idx++;
								   }
								   float SNR = signal / noise;
								   totalSignal += signal;
								   totalNoise += noise;
								   if (signal > maxSignal)
								   {
									   maxSignal = signal;
								   }
								   if (signal < minSignal)
								   {
									   minSignal = signal;
								   }
								   if (noise > maxNoise)
								   {
									   maxNoise = noise;
								   }
								   if (noise < minNoise)
								   {
									   minNoise = noise;
								   }
								   if (SNR < minSNR)
								   {
									   minSNR = SNR;
								   }
								   if (SNR > maxSNR)
								   {
									   maxSNR = SNR;
								   }
							   }
						   }
						   totalSNR = totalSignal / totalNoise;
					   }

					   if (doMDL)
					   {
						   int svdMinDim = svdM;
						   if (svdN < svdM)
						   {
							   svdMinDim = svdN;
						   }

						   int svdDim = 0;
						   S[0] = 0;
						   for (int svdIdx = 0; svdIdx < svdCount; svdIdx++)
						   {
							   svdSingularValues(svdM, svdN, &A[svdIdx * svdM * svdN], &S[svdMinDim * svdIdx]);
							   svdDim += svdMinDim;
						   }
						   il = 0;
						   float maxS = -FLT_MAX;
						   float minS = FLT_MAX;

						   for (int i = 0; i < svdDim; i++)
						   {
							   float s = S[i];

							   float s2 = s * s;

							   il += (s2);
							   if (s < minS)
							   {
								   minS = s;
							   }
							   if (s > maxS)
							   {
								   maxS = s;
							   }
						   }
						   il = il / svdDim;
						   mdl = (minS * minS) / (maxS * maxS);
					   }
					   else
					   {
						   if (doIL)
						   {
							   il = 0;
							   for (int svdIdx = 0; svdIdx < svdCount; svdIdx++)
							   {
								   il += insertionLoss256(svdN, svdM, &A[svdIdx * svdN * svdM]);
							   }
							   // il = (il);
						   }
					   }


					   //These two metrics are basis-independent.
					   out[DIGHOLO_METRIC_IL][lambdaIdx] = il;
					   out[DIGHOLO_METRIC_MDL][lambdaIdx] = mdl;
					   //These metrics will be basis-dependent.
					   out[DIGHOLO_METRIC_DIAG][lambdaIdx] = (totalSignal / totalSignalCount);// signal(diag)
					   out[DIGHOLO_METRIC_SNRAVG][lambdaIdx] = (totalSNR);// SNR(diag)
					   out[DIGHOLO_METRIC_DIAGBEST][lambdaIdx] = (maxSignal);// best pwr(diag)
					   out[DIGHOLO_METRIC_DIAGWORST][lambdaIdx] = (minSignal);// worstpwr(diag)
					   out[DIGHOLO_METRIC_SNRBEST][lambdaIdx] = (maxSNR);// best xtalk
					   out[DIGHOLO_METRIC_SNRWORST][lambdaIdx] = (minSNR);// worst xtalk
					   out[DIGHOLO_METRIC_SNRMG][lambdaIdx] = (totalSignalMG / totalNoiseMG);
				   }
				   else //if batchCount = 1, some metrics don't make sense
				   {
					   int polCount = digHoloPolCount;
					   float totalPwr = 0;
					   float diag = 0;
					   complex64* coefs = digHoloOverlapCoefsPtr[0];
					   size_t idx = 0;
					   for (int polIdx = 0; polIdx < polCount; polIdx++)
					   {
						   for (int modeIdx = 0; modeIdx < digHoloModeCountOut; modeIdx++)
						   {
							   float vR = coefs[idx][0];
							   float vI = coefs[idx][1];
							   float pwr = vR * vR + vI * vI;
							   totalPwr += pwr;
							   if (modeIdx == 0)
							   {
								   diag += pwr;
							   }
							   idx++;
						   }
					   }

					   il = (totalPwr);
					   float diagPwrLog = diag;
					   out[DIGHOLO_METRIC_IL][lambdaIdx] = il;
					   out[DIGHOLO_METRIC_MDL][lambdaIdx] = 0;
					   //These metrics will be basis-dependent.
					   out[DIGHOLO_METRIC_DIAG][lambdaIdx] = diagPwrLog;// signal(diag)
					   out[DIGHOLO_METRIC_SNRAVG][lambdaIdx] = FLT_MAX;// SNR(diag)
					   out[DIGHOLO_METRIC_DIAGBEST][lambdaIdx] = diagPwrLog;// best pwr(diag)
					   out[DIGHOLO_METRIC_DIAGWORST][lambdaIdx] = diagPwrLog;// worstpwr(diag)
					   out[DIGHOLO_METRIC_SNRBEST][lambdaIdx] = FLT_MAX;// best xtalk
					   out[DIGHOLO_METRIC_SNRWORST][lambdaIdx] = FLT_MAX;// worst xtalk
					   out[DIGHOLO_METRIC_SNRMG][lambdaIdx] = FLT_MAX;
				   }
			   }
		   }
	   }


	   void alignmentSave(int polIdx, float** alignmentSettings)
	   {
		   float* settings = alignmentSettings[polIdx];
		   float* zernCoefs = config.zernCoefs[polIdx];

		   settings[AUTOALIGN_TILTX] = zernCoefs[TILTX];
		   settings[AUTOALIGN_TILTY] = zernCoefs[TILTY];
		   settings[AUTOALIGN_DEFOCUS] = zernCoefs[DEFOCUS];

		   settings[AUTOALIGN_WAIST] = config.waist[polIdx];
		   settings[AUTOALIGN_CX] = config.BeamCentreX[polIdx];
		   settings[AUTOALIGN_CY] = config.BeamCentreY[polIdx];
	   }

	   void alignmentLoad(int polIdx, float** alignmentSettings)
	   {
		   float* settings = alignmentSettings[polIdx];
		   float* zernCoefs = config.zernCoefs[polIdx];

		   zernCoefs[TILTX] = settings[AUTOALIGN_TILTX];
		   zernCoefs[TILTY] = settings[AUTOALIGN_TILTY];
		   zernCoefs[DEFOCUS] = settings[AUTOALIGN_DEFOCUS];

		   config.waist[polIdx] = settings[AUTOALIGN_WAIST];
		   config.BeamCentreX[polIdx] = settings[AUTOALIGN_CX];
		   config.BeamCentreY[polIdx] = settings[AUTOALIGN_CY];
	   }

	   //dumb bubble sort in ascending order. 
	   void sortIt(float* x, float* y, int N)
	   {
		   for (int sortIdx = 0; sortIdx < N; sortIdx++)
		   {
			   for (int sortIdy = 1; sortIdy < N; sortIdy++)
			   {
				   if (y[sortIdy] < y[sortIdy - 1])
				   {
					   float swpy = y[sortIdy];
					   y[sortIdy] = y[sortIdy - 1];
					   y[sortIdy - 1] = swpy;

					   float swpx = x[sortIdy];
					   x[sortIdy] = x[sortIdy - 1];
					   x[sortIdy - 1] = swpx;
				   }
			   }
		   }
	   }

	   void digHoloAutoFocus(float* scale, float* bestDefocus)
	   {
		   const float scaleMin = 1.0f;//The minimum search range for the focus
		   float convergenceTol = 0.01f;
		   //const int fftIdx = IFFT_IDX;
		   const int polCount = digHoloPolCount;

		   //const int fftThreads = 1;
		   const int batchCount = config.batchCount;
		  // const int startIdx = 0;
		   //const int stopIdx = batchCount;
		   //const int avgCount = config.avgCount;
		   //const int avgMode = config.avgMode;
		   const size_t strideOut = digHoloWoiPolWidth[digHoloResolutionIdx] * digHoloWoiPolHeight[digHoloResolutionIdx];

		   std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

		   const size_t length = strideOut * polCount * batchCount;


		   //We're going to temporarily swap the order of the input and output of the IFFT routine. So we'll store the original pointers here.
		   complex64* fourierPlaneMasked = digHoloPixelsFourierPlaneMasked;
		   complex64* cameraPlaneReconstructedTilted = digHoloPixelsCameraPlaneReconstructedTilted;

		   complex64* tempArray = 0;
		   allocate1D(length, tempArray);

		   //memcpy(tempArray, digHoloPixelsCameraPlaneReconstructedTilted, length * sizeof(complex64));

		   complex64* backupFourierPlaneMasked = 0;
		   allocate1D(length, backupFourierPlaneMasked);
		   memcpy(backupFourierPlaneMasked, digHoloPixelsFourierPlaneMasked, length * sizeof(complex64));

		   complex64* backupCameraPlaneMasked = 0;
		   allocate1D(length, backupCameraPlaneMasked);
		   memcpy(backupCameraPlaneMasked, digHoloPixelsCameraPlaneReconstructedTilted, length * sizeof(complex64));
		   
		   const int stepCount = 25;
		   const int stepFitCount = 5;
		  // defocuses[0][stepIdx], defocuses[1][stepIdx], Aeffs[0][stepIdx], Aeffs[1][stepIdx], maxima[0][stepIdx], maxima[1][stepIdx]);
		   float** Aeffs = 0;
		   float** defocuses = 0;
		   allocate2D(polCount, stepCount + 3, Aeffs);
		   allocate2D(polCount, stepCount + 3, defocuses);

		   float** Aeffs0 = 0;
		   float** defocuses0 = 0;
		   allocate2D(polCount, stepCount + 3, Aeffs0);
		   allocate2D(polCount, stepCount + 3, defocuses0);

		   float* bestAeff = 0;
		   allocate1D(polCount, bestAeff);
		   int* converged = 0;
		   allocate1D(polCount, converged);

		   for (int polIdx = 0; polIdx < polCount; polIdx++)
		   {
			   bestAeff[polIdx] = FLT_MAX;
			   converged[polIdx] = false;
		   }
		   memset(bestDefocus, 0, polCount * sizeof(float));

		   for (int polIdx = 0; polIdx < polCount; polIdx++)
		   {
			   if (scale[polIdx] < scaleMin)
			   {
				   scale[polIdx] = scaleMin;
			   }
		   }

		   float* scale0 = 0;
		   allocate1D(polCount, scale0);
		   memcpy(scale0, scale, sizeof(float) * polCount);
		   for (int stepIdx = 0; stepIdx < stepCount; stepIdx += 2)
		   {
			   for (int polIdx = 0; polIdx < polCount; polIdx++)
			   {
				   defocuses[polIdx][stepIdx] = scale0[polIdx];
				   defocuses[polIdx][stepIdx + 1] = -scale0[polIdx];
				   scale0[polIdx] += scale[polIdx] * 0.25f;
			   }

		   }

		   float** B = 0;
		   float** B0 = 0;
		   float** maxima = 0;
		   allocate2D(polCount, 3, B);
		   allocate2D(polCount, 3, B0);
		   allocate2D(polCount, stepCount+3, maxima);

		   memset(&maxima[0][0], 0, polCount * stepCount * sizeof(float));
		   memset(&B[0][0], 0, polCount * 3 * sizeof(float));
		   int convergedTotal = 0;
		   int stepCountActual = 0;
		   int nanwarning = false;

		   const float tempSize = digHoloApertureSize;

		   for (int stepIdx = 0; (stepIdx < stepCount) && !convergedTotal; stepIdx++)
		   {
			   stepCountActual++;
			   convergedTotal = 1;

			   for (int polIdx = 0; polIdx < polCount; polIdx++)
			   {
				   config.zernCoefs[polIdx][DEFOCUS] = defocuses[polIdx][stepIdx];
			   }

			   digHoloApplyTilt(-1, tempArray);
			   
			   //By setting this to zero, the IFFTRoutine will skip the digHoloCopyWindow portion.
			   digHoloApertureSize = 0;
			   digHoloPixelsCameraPlaneReconstructedTilted = fourierPlaneMasked;
			   digHoloPixelsFourierPlaneMasked = tempArray;
			 //  digHoloIFFTCalibrationUpdate();
			   digHoloIFFTBatch();

			   float* X = &digHoloKXaxis[0];
			   float* Y = &digHoloKYaxis[0];
			   const float dX = (fabs(X[2] - X[1]));
			   const float dY = (fabs(Y[2] - Y[1]));
			   const float dA = dX * dY;

			   const __m256i permuteMask1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);//move imaginary components to the second lane, real parts to the first lane.
			   const __m256i permuteMask2 = _mm256_set_epi32(6, 4, 2, 0, 7, 5, 3, 1);//move imagingary components to first lane, real parts to second lane.

			   //Calculate the effective area.
			   for (int polIdx = 0; polIdx < polCount; polIdx++)
			   {
				   float* out2 = (float*)DIGHOLO_ANALYSISBatchSum[FFT_IDX + 4][polIdx];
				   memset(out2, 0, sizeof(float) * strideOut);
				   float v2 = 0;
				   float v4 = 0;
				   __m256 pwrTotal = _mm256_set1_ps(0.0);
				   __m256 E4 = _mm256_set1_ps(0.0);
				   for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
				   {
					   const size_t blockSize = 4;
					   const size_t blockSizeDouble = 2 * blockSize;
					   const size_t offset = (batchIdx * polCount + polIdx) * strideOut;

					   for (int pixelIdx = 0; pixelIdx < strideOut; pixelIdx += blockSizeDouble)
					   {
						   __m256 V1 = _mm256_loadu_ps(&digHoloPixelsCameraPlaneReconstructedTilted[offset + pixelIdx][0]);
						   __m256 V2 = _mm256_loadu_ps(&digHoloPixelsCameraPlaneReconstructedTilted[offset + pixelIdx + blockSize][0]);
						   __m256 Vpwr = _mm256_loadu_ps(&out2[pixelIdx]);
						   const __m256 Vr = _mm256_blend_ps(_mm256_permutevar8x32_ps(V1, permuteMask1), _mm256_permutevar8x32_ps(V2, permuteMask2), 0xF0);//0b11110000//Real part
						   const __m256 Vi = _mm256_blend_ps(_mm256_permutevar8x32_ps(V1, permuteMask2), _mm256_permutevar8x32_ps(V2, permuteMask1), 0xF0);//0b11110000//imag part
						   //Get the intensity, abs^2, of the field
						   const __m256 Vsqrd = _mm256_fmadd_ps(Vr, Vr, _mm256_mul_ps(Vi, Vi));
						   pwrTotal = _mm256_add_ps(pwrTotal, Vsqrd);
						   const __m256 E2 = _mm256_add_ps(Vpwr, Vsqrd);
						   _mm256_storeu_ps(&out2[pixelIdx], E2);

						   if (batchIdx == (batchCount - 1))
						   {
							   E4 = _mm256_add_ps(E4, _mm256_mul_ps(E2, E2));
						   }
					   }

					   if (batchIdx == (batchCount - 1))
					   {
						   pwrTotal = _mm256_add_ps(pwrTotal, _mm256_permute_ps(pwrTotal, 0xB1));//0b10110001
						   pwrTotal = _mm256_add_ps(pwrTotal, _mm256_permute_ps(pwrTotal, 0x4E));//0b01001110
						   pwrTotal = _mm256_add_ps(pwrTotal, _mm256_permute2f128_ps(pwrTotal, pwrTotal, 1));

						   E4 = _mm256_add_ps(E4, _mm256_permute_ps(E4, 0xB1));//0b10110001
						   E4 = _mm256_add_ps(E4, _mm256_permute_ps(E4, 0x4E));//0b01001110
						   E4 = _mm256_add_ps(E4, _mm256_permute2f128_ps(E4, E4, 1));
					   }

					   /*  for (int pixelIdx = 0; pixelIdx < strideOut; pixelIdx++)
						 {
							 float vR = digHoloPixelsCameraPlaneReconstructedTilted[offset + pixelIdx][0];
							 float vI = digHoloPixelsCameraPlaneReconstructedTilted[offset + pixelIdx][1];
							 float v = vR * vR + vI * vI;

							  v2 += v;
							 out2[pixelIdx] += v;

						 }
						 */

				   }


				   /* for (int pixelIdx = 0; pixelIdx < strideOut; pixelIdx++)
					{
						float v = out2[pixelIdx];
						v4 += v * v;
					}*/
				   float* v2f = (float*)&pwrTotal;
				   float* v4f = (float*)&E4;

				   v2 = v2f[0];
				   v4 = v4f[0];
				   float Aeff = dA * ((v2 * v2) / v4);

				   Aeffs[polIdx][stepIdx] = Aeff;
				   defocuses[polIdx][stepIdx] = config.zernCoefs[polIdx][DEFOCUS];// defocus;

				   if (bestDefocus)
				   {
					   if (Aeff < bestAeff[polIdx])
					   {
						   bestAeff[polIdx] = Aeff;
						   bestDefocus[polIdx] = defocuses[polIdx][stepIdx];

					   }
				   }

				   //If this is the second step, check which direction to continue the search in
				   if (stepIdx == 1)
				   {
					   float stepScale = 1;
					   float baseValue = 0;
					   //If the second step was better
					   if (Aeffs[polIdx][stepIdx] < Aeffs[polIdx][stepIdx - 1])
					   {
						   stepScale = defocuses[polIdx][stepIdx] - defocuses[polIdx][stepIdx - 1];
						   baseValue = defocuses[polIdx][stepIdx];
					   }
					   else
					   {
						   stepScale = defocuses[polIdx][stepIdx - 1] - defocuses[polIdx][stepIdx];
						   baseValue = defocuses[polIdx][stepIdx - 1];
					   }

					   int idx = 1;
					   //  defocuses[polIdx][stepIdx + 1] = baseValue+ 0.25 * stepScale;
						// defocuses[polIdx][stepIdx + 2] = baseValue - 0.25 * stepScale;
					   for (int i = (stepIdx + 1); i < stepCount; i += 2)
					   {
						   defocuses[polIdx][i] = baseValue + idx * 0.25f * stepScale;
						   defocuses[polIdx][i + 1] = baseValue - idx * 0.25f * stepScale;
						   idx++;
					   }

				   }

				   if (stepIdx >= (stepFitCount - 1))
				   {
					   if (stepIdx == (stepFitCount - 1))
					   {
						   B0[polIdx][0] = 1e7;
						   B0[polIdx][1] = 1e7;
						   B0[polIdx][2] = 0;// meanPwr[i] / (stepIdx + 1);
						   for (int i = 0; i < stepIdx; i++)
						   {
							   B0[polIdx][2] += Aeffs[polIdx][i] / (stepIdx + 1);
						   }
					   }
					   memcpy(&Aeffs0[0][0], &Aeffs[0][0], sizeof(float) * polCount * stepCount);
					   memcpy(&defocuses0[0][0], &defocuses[0][0], sizeof(float) * polCount * stepCount);
					   float* y = &Aeffs0[polIdx][0];
					   float* x = &defocuses0[polIdx][0];
					   sortIt(x, y, stepIdx + 1);
					   const int maxIter = 1000;
					   const float tol = 1e-6f;
					   const int fitCount = stepFitCount - 1;//We'll ignore 1 of the entries. e.g. the first try might not be great, e.g. because we're guessing between +/- focus.
					   maxima[polIdx][stepIdx] = fitToQuadratic(x, y, fitCount, B[polIdx], B0[polIdx], tol, maxIter);
					   memcpy(&B0[polIdx][0], &B[polIdx][0], 3 * sizeof(float));

					   if ((stepIdx + 1) < (stepCount - 1))
					   {
						   if (!isnan(maxima[polIdx][stepIdx]))
						   {
							   defocuses[polIdx][stepIdx + 1] = maxima[polIdx][stepIdx];

						   }
						   else
						   {
							   nanwarning = true;
						   }
					   }
					   if (abs(maxima[polIdx][stepIdx] - bestDefocus[polIdx]) < convergenceTol)
					   {
						   converged[polIdx] = true;
					   }
				   }
				   convergedTotal = convergedTotal & converged[polIdx];
			   }//polIdx

			   //Set to the original pointers
			   digHoloPixelsCameraPlaneReconstructedTilted = cameraPlaneReconstructedTilted;
			   digHoloPixelsFourierPlaneMasked = fourierPlaneMasked;
			   digHoloApertureSize = tempSize;

		   }//defocus

		   std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();
		   std::chrono::duration<double> time_span = (std::chrono::duration_cast<std::chrono::duration<double>>(stopTime - startTime));
		   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
		   {
			   fprintf(consoleOut, "Focus (ms)	%f\n\r", time_span.count() * 1000);

			   if (nanwarning)
			   {
				   fprintf(consoleOut, "Error in Autofocus routine...\n");
				   for (int polIdx = 0; polIdx < polCount; polIdx++)
				   {
					   for (int stepIdx = 0; stepIdx < stepCountActual; stepIdx++)
					   {
						   fprintf(consoleOut, "%i	%6.6f	%6.6e	%6.6e\n", polIdx,defocuses[polIdx][stepIdx],  Aeffs[polIdx][stepIdx], maxima[polIdx][stepIdx]);
					   }
				   }
			   }

			   fflush(consoleOut);
		   }


		   //Restore arrays to the values they were before the routine started.
		   memcpy(digHoloPixelsCameraPlaneReconstructedTilted, backupCameraPlaneMasked, length * sizeof(complex64));
		   memcpy(digHoloPixelsFourierPlaneMasked, backupFourierPlaneMasked, length * sizeof(complex64));
		   free1D(backupCameraPlaneMasked);
		   free1D(backupFourierPlaneMasked);
		   free1D(tempArray);

		   free2D(Aeffs);
		   free2D(defocuses);

		   free2D(Aeffs0);
		   free2D(defocuses0);

		   free2D(maxima);
		   free2D(B0);
		   free2D(B);

		   free1D(bestAeff);
		   free1D(converged);

		   free1D(scale0);
	   }

private: void digHoloAutoAlignSnap(unsigned char doFinalOverlap, int polsAreIndependent, float** outputMetrics)
{
	bool snapTilt = true;
	bool snapFocus = true;
	bool snapWaist = true;
	bool snapCentre = true;

	//Go through and reset the values for whatever parameters are being optimised, to zero.
	//Starts from a clean slate.
	float** zernCoefs = config.zernCoefs;
	//For each polarisation
	for (int polIdx = 0; polIdx < config.polCount; polIdx++)//FFT may not have been run yet, so digHoloPolCount might not yet equal config->polCount
	{
		//If Beam centre is being aligned, reset it to zero
		if (config.AutoAlignCentre && snapCentre)
		{
			config.BeamCentreX[polIdx] = (float)((-(config.polCount - 1) * (config.frameWidth / 4.0) + (polIdx * config.frameWidth / 2.0)) * config.pixelSize);
			config.BeamCentreY[polIdx] = 0;
			//printf("%i	%3.3f	%3.3f	%3.3f\n", polIdx, config.BeamCentreX[polIdx], config.BeamCentreY[polIdx], config.pixelSize);
		}
		//If tilt is being aligned, reset it to  zero
		float* zCoefs = zernCoefs[polIdx];
		if (config.AutoAlignTilt && snapTilt)
		{
			zCoefs[TILTX] = 0;
			zCoefs[TILTY] = 0;
		}
		//If defocus is being aligned, reset it to zero
		if (config.AutoAlignDefocus && snapFocus)
		{
			zCoefs[DEFOCUS] = 0;
		}
	}
	//array which will store estimates of the beam waist in each plane Fourier and reconstructed plane, for both polarisations.
	float waists[2][2];

	//Change the FFT window size to full-size. No matter where the beam is on the camera, we want to be able to find it.
	config.fftWindowSizeX = config.frameWidth / config.polCount;
	config.fftWindowSizeY = config.frameHeight;

	//Do FFT processing of the whole batch
	//Inside this routine will also be digHoloFieldAnalysis, which will give us information about the Fourier plane, excluding the region of radius 2*digHoloApertureSize around the zero-order.
	//Hence that will give us an estimate of where the off-axis term is.
	digHoloFFT();

	//The plane we'll be analysing (Fourier plane)
	int fftIdx = FFT_IDX;

	//The wavenumber for the centre wavelength
	const float k0 = (2 * pi) / (digHoloWavelengthCentre);

	//This routine used to lock onto the tilt using multiple passes. Now it just uses a single pass.
	const int passCount = 1+config.AutoAlignFourierWindowRadius;
	const int passCountMax = 2;
	float tiltsX[passCountMax][2];
	float tiltsY[passCountMax][2];

	int success = digHoloDCTGetPlan();

	if (!success)
	{
		fprintf(consoleOut, "Warning: FFTW Discrete Cosine Transform (DCT) plan failed to initialise. Check you have linked against FFTW library. Intel's MKL library provides an interface which is mostly compatible with FFTW, but does not support real-to-real transforms. When linking, make sure fftw library is listed before MKL library\n\r");
		fflush(consoleOut);
	}

	if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
	{
		fprintf(consoleOut, "Fourier plane\n\r");
		fflush(consoleOut);
	}

	for (int passIdx = 0; passIdx < passCount; passIdx++)
	{

		//For each polarisation that's active
		for (int polIdx = 0; polIdx < digHoloPolCount; polIdx++)
		{
			//get the zernikes for this polarisation
			float* zCoefs = zernCoefs[polIdx];

			//for passIdx==0, digHoloFieldAnalysisRoutine has already been run as part of digHoloFFT()
			//for passIdx>0 we'll have to rerun the analysis for the new window location. However we don't need to redo the FFT, just the analysis part.
			//if (passIdx == 0)
			{
				float* in = (float*)DIGHOLO_ANALYSISBatchSum[FFT_IDX][polIdx];
				float* out = (float*)DIGHOLO_ANALYSISBatchSum[FFT_IDX + 2][polIdx];
				float* out2 = (float*)DIGHOLO_ANALYSISBatchSum[FFT_IDX + 4][polIdx];
				if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
				{
					fprintf(consoleOut, "Execute DCT.\n\r");
					fflush(consoleOut);
				}
				fftwf_execute_r2r(digHoloDCTPlan, in, out);

				//Apply a Gaussian filter, corresponding with the size of the aperture in the Fourier plane.
				const int pixelCountX = digHoloWoiPolWidth[FFT_IDX] / 2 + 1;;
				const int pixelCountY = digHoloWoiPolHeight[FFT_IDX];
				float aperturek = k0 * digHoloApertureSize;
				float wx = (float)((1.0 * pi / aperturek) * sqrt(2.0));
				float wy = (float)((2.0 * pi / aperturek) * sqrt(2.0));

				//Multiplies the DCT from above with a Gaussian filter representing correlation with the Gaussian aperture window in the Fourier plane of the camera.
				//Effectively a type of pattern recognition, looking for the position in Fourier space which captures the most power into the aperture window.
				__m256* out256 = (__m256*)out;
				const __m256 block8 = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
				const __m256 inc = _mm256_set1_ps(8);
				const __m256 wx2 = _mm256_set1_ps((float)(-1.0 / (wx * wx)));
				const __m256 wy2 = _mm256_set1_ps((float)(-1.0 / (wy * wy)));
				const __m256 pxSize = _mm256_set1_ps(digHoloPixelSize);
				const float pxSize2 = digHoloPixelSize * digHoloPixelSize;
				int idx = 0;
				for (int j = 0; j < pixelCountY; j++)
				{
					__m256 y2 = _mm256_set1_ps(j * j * pxSize2);
					__m256 yw2 = _mm256_mul_ps(y2, wy2);
					__m256 xi = block8;
					//Warning : This dimension is Width/2+1, so the final column, won't run as it's not a multiple of 8.
					//However, that's also right at the edge, which we don't use in any case.
					for (int i = 0; i < pixelCountX / 8; i++)
					{
						__m256 x = _mm256_mul_ps(xi, pxSize);
						__m256 x2 = _mm256_mul_ps(x, x);
						__m256 xw2 = _mm256_mul_ps(x2, wx2);
						__m256 exponent = _mm256_add_ps(xw2, yw2);
						//It could be possible to speed this up, by avoiding calculation if exponent is <-88 (and hence becomes zero in float precision).
						__m256 g = exp_ps(exponent);
						out256[idx] = _mm256_mul_ps(out256[idx], g);
						xi = _mm256_add_ps(xi, inc);
						idx++;
					}
				}
				if (config.verbosity >= DIGHOLO_VERBOSITY_COOKED)
				{
					fprintf(consoleOut, "Execute IDCT.\n\r");
					fflush(consoleOut);
				}
				fftwf_execute_r2r(digHoloIDCTPlan, out, out2);
				float* KX = &digHoloKXaxis[0];
				float* KY = &digHoloKYaxis[0];
				float maxPwr = -FLT_MAX;
				float tiltX0 = 0;
				float tiltY0 = 0;
				float ad = aperturek * 2;
				ad = ad * ad;

				//This could definitely be faster. However this routine seems to be dominated by FFT planning delay, rather than any of the other calculations.
				for (int j = 0; j < pixelCountY; j++)
				{
					float ky = KY[j];
					for (int i = 0; i < pixelCountX; i++)
					{
						float kx = KX[i];

						idx = j * pixelCountX + i;
						float pwr = out2[idx];
						float kr = kx * kx + ky * ky;

						if (kr > ad)
						{
							if (pwr > maxPwr)
							{
								maxPwr = pwr;
								tiltX0 = kx;
								tiltY0 = ky;
							}
						}
						else
						{
							out2[idx] = 0;
						}
					}
				}

				tiltX0 = asinf(tiltX0 / k0) / DIGHOLO_UNIT_ANGLE;
				tiltY0 = asinf(tiltY0 / k0) / DIGHOLO_UNIT_ANGLE;

				if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
				{
					fprintf(consoleOut, "Tilt	%i	%f,%f\n\r", polIdx, tiltX0, tiltY0);
					fflush(consoleOut);
				}
				tiltsX[passIdx][polIdx] = tiltX0;
				tiltsY[passIdx][polIdx] = tiltY0;

				if (config.AutoAlignFourierWindowRadius)
				{
					float apertureMax = FourierWindowRadiusRec(tiltX0, tiltY0);
					if (apertureMax < config.apertureSize)
					{
						config.apertureSize = apertureMax;
						digHoloApertureSize = apertureMax * DIGHOLO_UNIT_ANGLE;
						digHoloFFTAnalysis();
						if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
						{
							fprintf(consoleOut, "Fourier Window Radius	%f\n\r", apertureMax); fflush(consoleOut);
						}
					}
				}
			}

			//zernike coefficients for this polarisation
			if (config.AutoAlignTilt && snapTilt)
			{
				zCoefs[TILTX] = tiltsX[passIdx][polIdx];
				zCoefs[TILTY] = tiltsY[passIdx][polIdx];
			}

			//Select a window, centred on the estimated tilt, of radius windowR
			const float windowCX = k0 * ((zCoefs[TILTX]) * DIGHOLO_UNIT_ANGLE);
			const float windowCY = k0 * ((zCoefs[TILTY]) * DIGHOLO_UNIT_ANGLE);
			//The negative radius here indicates only do analysis _inside_ this window. 
			float windowR = -k0 * sinf(digHoloApertureSize);//Small-angle approximation (sin(x) = x)

			//Fourier axis
			float* X = &digHoloKXaxis[0];
			float* Y = &digHoloKYaxis[0];
			//Just process this single polarisation. Could be slightly faster to do both in parallel, but current window is only implemented as a single shared window for both pols.
			const int polStart = polIdx;
			const int polStop = polIdx + 1;
			//Dimensions of the Fourier space (real-to-complex format, (width/2+1)*height)
			const int pixelCountX = digHoloWoiPolWidth[FFT_IDX] / 2 + 1;;
			const int pixelCountY = digHoloWoiPolHeight[FFT_IDX];
			//the total number of batch items
			const int batchCount = digHoloBatchAvgCount * digHoloBatchCOUNT;
			//The number of threads along the y-axis
			int pixelThreads = digHoloThreadCount / digHoloPolCount;
			if (pixelThreads <= 0)
			{
				pixelThreads = 1;
			}
			//The distance between two polarisation components in the Fourier space
			const size_t polStride = ((size_t)(digHoloWoiPolWidth[FFT_IDX] / 2 + 1)) * ((size_t)digHoloWoiPolHeight[FFT_IDX]);
			//The distance from one batch item to the next.
			const size_t batchStride = polStride * digHoloPolCount;
			//The set of fields we want to process.
			complex64* field = (complex64*)&digHoloPixelsFourierPlane[0][0];
			unsigned char wrapYaxis = false;

			//complex64* RefCalibration = 0;
			//int RefCalibrationWavelengthCount = 0;
			//We don't really need to redo the whole batch, we really just need to analyse the sum. Could be faster.
			digHoloFieldAnalysisRoutine(X, Y, field, polStart, polStop, FFT_IDX, pixelCountX, pixelCountY, batchCount, pixelThreads, batchStride, polStride, windowCX, windowCY, windowR, true, wrapYaxis);

			//For the final [batchCount] element, which is the aggregate over the whole batch set.
			for (int batchIdx = batchCount; batchIdx < (batchCount + 1); batchIdx++)
			{
				//Get the centre of mass (x,y) averaged over the whole batch
				//const float comx = digHoloPixelsStats[STAT_COMX][fftIdx][polIdx][batchIdx];
				//const float comy = digHoloPixelsStats[STAT_COMY][fftIdx][polIdx][batchIdx];
				//const float comyWrap = digHoloPixelsStats[STAT_COMYWRAP][fftIdx][polIdx][batchIdx];
				//Get the effective area, averaged over the whole batch
				const float Aeff = DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_AEFF][polIdx][batchIdx];

				//const int* maxIdxPtr = (int*)&digHoloPixelsStats[STAT_MAXABSIDX][fftIdx][polIdx][batchIdx];
				//const float totalPwr = digHoloPixelsStats[STAT_TOTALPOWER][fftIdx][polIdx][batchIdx];
				//const int maxIdx = maxIdxPtr[0] / 2;
				//System::Diagnostics::Debug::WriteLine("IDX " + maxIdx+"	"+polIdx);
				//int widthR2C = (digHoloWoiPolWidth[FFT_IDX] / 2 + 1);
				//int height = (digHoloWoiPolHeight[FFT_IDX]);
				//const int xIdx = maxIdx % widthR2C;
				//const int yIdx = maxIdx / widthR2C;
				//Convert that centre of mass from k-space to degrees.
				//const float tiltX = (float)((comx / k0) * 180.0 / pi);
				//const float tiltY = (float)((comy / k0) * 180.0 / pi);
				//const float tiltYwrap = (float)((comyWrap / k0) * 180.0 / pi);
				//Estimate the beam waist, based on the effective area, and the maximum mode order. It's assuming here that all modes are present.
				const float wEst = AeffToMFD(Aeff, config.maxMG);
				//Convert that estimated beam radius in k-space, to the corresponding beam radius in the plane of the camera.
				const float wCameraPlane = (float)((1.0 / wEst) * pi);
				//Remember this apparent beam waist in the camera plane, we'll use it later to work out how out of focus we might be.
				waists[fftIdx][polIdx] = wCameraPlane;

				//pwrs[passIdx] = totalPwr;
			}
		}
	}


	//Next we'll be processing the camera plane (reconstructed field plane).

	const int maxAttempts = 10;//If you need more than 10 attemps to get a valid result here, you're in trouble. Probably means your Fourier window is wrong.
	bool analysisValid = true;//Are the analysis parameter valid? We'll be checking for things like centre of mass that are NaN. That means somethings wrong, probably with the Fourier window. e.g. if you made the digHoloApertureSize the size of the whole Fourier plane, then digital holography is never going to work, because you'll always be selecting everything.
	int attempts = 0; //current attempts. None so far.

	//We always want to adjust focus at least a little bit. This is so we never get a 'maxFocus' of zero.
	//Sometimes the focus estimate arrived at below will be zero, because a physically unrealistic estimate is provided by FocusEstimate
	float maxFocus = 0.25f;
	float defocusEstimate[2] = { 0,0 };
	//While the analsis is invalid, or we haven't started yet, and if we haven't done too many attempts so as to give up yet.
	while ((!analysisValid || attempts == 0) && attempts < maxAttempts)
	{
		//By default, we assume the analysis will give valid results
		analysisValid = true;
		//Do the IFFT
		digHoloIFFT();
		//We don't really need the tilt to be removed, but the digHoloFieldAnalysis routine is in the digHoloApplyTilt routine.
		//Could also call it externally like we did above for the FFT. Although it's actually rare that this loop runs more than once anyways, and the tilt/focus removal is not a big performance hit compared with digHoloIFFT, 
		//which we will ahve to do each time in this case. Because if we're doing multiple passes here, it means there's something wrong with the digHoloIFFT step (e.g. nothing in the window, or invalid numbers in the window)
		digHoloApplyTilt();

		//We'll be looking at parameters in the camera (reconstructed plane).
		if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
		{
			fprintf(consoleOut, "Camera plane\n\r");
			fflush(consoleOut);
		}
		fftIdx = IFFT_IDX;

		//For each enabled polarisation
		for (int polIdx = 0; polIdx < digHoloPolCount; polIdx++)
		{
			//the zernikes for this polarisation
			float* zCoefs = zernCoefs[polIdx];

			//An offset in x for the centre of mass. To give us centre of mass relative to the absolute coordinates of the camera array, rather than of the FFT sub-window of the camera.
			//i.e. 0,0 is centre of camera, not centre of FFT window on camera.
			float polOffsetX = (float)(((2 * polIdx - 1) * config.pixelSize * digHoloFrameWidth / 4.0) * (digHoloPolCount - 1));

			//For the [batchCount] element, which contains the parameters aggregated over the whole batch
			for (int batchIdx = digHoloBatchCOUNT; batchIdx < (digHoloBatchCOUNT + 1); batchIdx++)
			{
				//Centre of mass in the camera plane
				const float comx = (float)(DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_COMX][polIdx][batchIdx] / DIGHOLO_UNIT_PIXEL + polOffsetX);
				const float comy = (float)(DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_COMY][polIdx][batchIdx] / DIGHOLO_UNIT_PIXEL);
				//Effective area in the camera plane
				const float Aeff = DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_AEFF][polIdx][batchIdx];
				//Total power and maximum absolute value in the camera plane (not used for calculating anything, but printed to screen for the user, mostly for debugging)
				const float totalPwr = DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_TOTALPOWER][polIdx][batchIdx];
				const float maxAbs = DIGHOLO_ANALYSIS[fftIdx][DIGHOLO_ANALYSIS_MAXABS][polIdx][batchIdx];

				//Calculate the estimated beam radius based on the effective area, and the number of mode groups we're dealing with.
				float wEst = (float)(AeffToMFD(Aeff, config.maxMG) * 0.5);
				//Store that waist radius estimate. We'll later be using the different between the apparent waist on the camera, and in the Fourier plane, to estimate the defocus.
				waists[fftIdx][polIdx] = wEst;
				//In fact we'll be using it right now to esimate the defocus. This value could be positive or negative, we have no way to tell.
				//We actually just use this to set the scale of the search step in the next part of the alignment, but here you could also just check the positive and negative scenarios and choose the best.
				defocusEstimate[polIdx] = FocusEstimate(waists[IFFT_IDX][polIdx], waists[FFT_IDX][polIdx]);

				//Print the analysis in the camera plane
				if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
				{
					fprintf(consoleOut, "%i	%i	BeamCentre	(%e	%e)	MFD:%e	Defocus:%f	TotalPwr:%f	MaxV:%f	Aeff:%e\n\r", polIdx, batchIdx, comx, comy, (wEst / DIGHOLO_UNIT_PIXEL), defocusEstimate[polIdx], totalPwr, maxAbs, Aeff / (DIGHOLO_UNIT_PIXEL * DIGHOLO_UNIT_PIXEL));
					fflush(consoleOut);
				}
				//If you're doing things right, you shouldn't need this. But if you've set the aperture huge you might have issues.
				if (isnan(comx) || isnan(comy) || isnan(Aeff))
				{
					//If it's failed, shrink the aperture. We might be capturing the zero-order
					//analsisValid = false means we'll be doing more loops
					analysisValid = false;
					//We'll keep shrinking the aperture size until hopefully we get something we can use.
					config.apertureSize *= 0.9f;
					digHoloApertureSize = (float)(config.apertureSize * DIGHOLO_UNIT_ANGLE);
				}

				//If the analysis is valid, update the values
				if (analysisValid)
				{
					if (config.AutoAlignCentre && snapCentre)
					{
						config.BeamCentreX[polIdx] = comx;
						config.BeamCentreY[polIdx] = comy;
					}

					if (config.AutoAlignBasisWaist && snapWaist)
					{
						config.waist[polIdx] = (float)(wEst / DIGHOLO_UNIT_PIXEL);
					}

					if (config.AutoAlignDefocus && snapFocus)
					{
						// setting the focus to zero, rather than the defocus estimate, we'll use the focus estimate to set the scale of the search in the next step instead of setting it directly.
						// as is current stands we don't know if our focal esimate is positive or negative anyways, and it may not be that accurate unless we've got all the modes, and they all look pretty good anyways.
						zCoefs[DEFOCUS] = 0;
					}

					//Keep track of the largest defocus estimate we've seen thus far
					if (fabsf(defocusEstimate[polIdx]) > maxFocus)
					{
						maxFocus = fabsf(defocusEstimate[polIdx]);
					}
				}
				attempts++;
			}
		}
	}

	if (config.AutoAlignDefocus && snapFocus)
	{
		float bestDefocus[2];
		digHoloAutoFocus(&defocusEstimate[0], &bestDefocus[0]);

		
		for (int polIdx = 0; polIdx < digHoloPolCount; polIdx++)
		{
			//the zernikes for this polarisation
			float* zCoefs = zernCoefs[polIdx];
			zCoefs[DEFOCUS] = bestDefocus[polIdx];

			if (abs(bestDefocus[polIdx]) > maxFocus)
			{
				maxFocus = abs(bestDefocus[polIdx]);
			}
		}
	}

	//If the user has specified to do so, do a final calculation of the overlaps and IL/MDL
	//You'll often get close just from this single pass. You can check the results here if you like.
	if (doFinalOverlap)
	{
		digHoloIFFT();
		digHoloApplyTilt();
		//float out[DIGHOLO_METRIC_COUNT];
		//memset(out, 0, sizeof(float) * DIGHOLO_METRIC_COUNT);

		float** out = outputMetrics;
		int lambdaCount = digHoloWavelengthCount;
		if (!outputMetrics)
		{
			allocate2D(DIGHOLO_METRIC_COUNT, (lambdaCount+1),out);
		}

		digHoloOverlapModes();
		//digHoloPowerMeter(out, polsAreIndependent, true, true, true, config.AutoAlignBasisMulConjTrans, digHoloWavelengthCount,0,digHoloWavelengthCount);
		digHoloPowerMeterThreaded(&out, polsAreIndependent, NULL, true, true, true, config.AutoAlignBasisMulConjTrans, lambdaCount, 1);

		if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
		{
			fprintf(consoleOut, "IL	= %f	MDL = %f\n\r", out[DIGHOLO_METRIC_IL][lambdaCount], out[DIGHOLO_METRIC_MDL][lambdaCount]);
			fflush(consoleOut);
		}
		if (out != outputMetrics)
		{
			free2D(out);
		}

	}

	//Set the scale of the autoalign steps for each parameter, based on the settings we've estimated.
	config.AutoAlignTiltStep = config.apertureSize * 0.1f;//5% of the aperture
	config.AutoAlignCentreStep = config.waist[0] * 0.1f;

	config.AutoAlignDefocusStep = 0.1f*maxFocus;
	config.AutoAlignBasisWaistStep = config.waist[0] * 0.1f;

}

float FourierWindowRadiusMax(float pixelSize, float wavelength)
{
	//If the spatial frequency content of the Signal field has a radius of w_c in Fourier space(green window), then the maximum possible Reference field angle would be given by ~1/3, i.e. [sqrt(2) / (3 + sqrt(2))] = 0.320377 of the maximum resolvable angle set by the frame pixel size(w_max).For example, if the wavelength is 1565e-9 and the pixelSize is 20e-6, w_max would be(1565e-9 / (2 * 20e-6))* (180 / pi) = 2.24 degrees, and window radius(w_c) should be less than 0.3205 * 2.24 = 0.719 degrees.The reference beam tilt angle in x and y would have to be(x, y) = (3w_c, 3w_c) / sqrt(2) = (1.525, 1.525) degrees.If the full resolution of the camera is not required, smaller windows and shallower reference angles can be employed.Smaller windows are also less likely to capture unwanted noise.If the green window is allowed to wrap - around along axis in Fourier - space, a larger fractional window could be used[+root of 0 = 8w ^ 2 + 2 - 2]->[(-2 + sqrt(68)) / 16] = 0.3904.Tilt(x, y) = (w_max - w_c, w_max)
	//const float wc_NoWrap = (float)(sqrt(2.0) / (3.0 + sqrt(2)));
	const float wc = (float)((-2.0 + sqrt(68.0)) / 16.0);
	float wmax = (float)(wavelength / (2.0 * pixelSize));
	return (wc * wmax)/(DIGHOLO_UNIT_ANGLE);
}

float FourierWindowRadiusRec(float tiltX, float tiltY)
{
	return sqrt(tiltX * tiltX + tiltY * tiltY)/3.0f;
}

float FourierWindowRadiusRec(float **zerns, int polCount)
{
	float tiltMin = FLT_MAX;
	for (int polIdx = 0; polIdx < polCount; polIdx++)
	{
		float tiltX = zerns[polIdx][TILTX];
		float tiltY = zerns[polIdx][TILTY];

		float tiltR = sqrt(tiltX * tiltX + tiltY * tiltY);
		if (tiltR < tiltMin)
		{
			tiltMin = tiltR;
		}
	}
	return tiltMin / 3.0f;
}



	   float FocusEstimateInverse(float mfdFourierPlane, float defocus)
	   {
		   defocus = defocus / 2;
		   /*
		   const float a = 0.25;
		   const float b = pi / 2;
		   float mfdRatio = ((1 - a)/((defocus*defocus/(b*b))+1))+a;
		   float mfdCameraPlane = mfdFourierPlane/mfdRatio;
		   return mfdCameraPlane;
		   */
		   const float a = 0.25f;
		   const float b = pi / 2;

		   float mfdRatio = ((1 - a) / (((defocus * defocus * 0.125f) / (b * b)) + 1)) + a;
		   return mfdFourierPlane / mfdRatio;
	   }

	   //Estimates the defocus bases on the apparent beam radii in the camera plane compared with the Fourier plane.
	   //If the mfds are consistent in both planes, that means it's in focus. If they're not, that means it's out of focus
	   //However we can't tell whether the curve is positive or negative from this.
	   float FocusEstimate(float wCameraPlane, float wFourierPlane)
	   {
		   if (wCameraPlane > 0 && wFourierPlane > 0)
		   {
			   float wRatio = wFourierPlane / wCameraPlane;
			   const float fudgeFactor = 1.0;// 0.87f;//Fudge factor added to make the focus estimate more accurate? Shouldn't be necessary.
			   float a = 0.25;
			   float b = pi / 2;
			   float result = fudgeFactor * 2 * (b * sqrtf((1 - a) / (wRatio - a) - 1));
			   if (result > 0)
			   {
				   //return result;
			   }
			   else
			   {
				   //This should be an impossible scenario (Fourier spot smaller than theoretical maximum)
				  // float mfdRatio = mfdCameraPlane / mfdFourierPlane;
				  // float a = 0.25;
				  // float b = pi / 2;
				   //float result = (b * sqrtf((1 - a) / (mfdRatio - a) - 1));
				   // return result;
				   result = 0;
			   }
			   if (isnan(result))
			   {
				   if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
				   {
					   fprintf(consoleOut, "Invalid Focus estimate, defaulting to zero. [%f,%f]\n\r", wCameraPlane, wFourierPlane);
					   fflush(consoleOut);
				   }
				   result = 0;
			   }
			   return result;
		   }
		   else
		   {
			   return 0;
		   }
	   }

	   void digHoloPowerMeterRoutine(workPackage& e)
	   {
		   //workPackage^ e = (workPackage^)o;
		   complex64** overlapCoefsBackup = (complex64**)e.ptr1;
		   float** out = (float**)e.ptr2;
		   // ManualResetEvent resetEvent = e.resetEvent;
		   int polIdx = e.flag1;

		   int doIL = e.flag2;
		   int doMDL = e.flag3;
		   int doXtalk = e.flag4;
		   int polsAreIndependent = e.flag5;
		   int mulConj = e.flag6;
		   int lambdaCount = e.flag7;
		   int lambdaStart = e.flag8;
		   int lambdaStop = e.flag9;
		   complex64* workspace = (complex64*)e.ptr3;

		   digHoloPowerMeter(out, polIdx, polsAreIndependent, overlapCoefsBackup, doIL, doMDL, doXtalk, mulConj,workspace,lambdaCount,lambdaStart, lambdaStop);
		   // e.resetEvent.Set();
	   }

	   int digHoloPowerMeterWorkspaceAllocate(int polThreads, int lambdaThreads, int lambdaCount)
	   {
		   const size_t batchCount = digHoloBatchCOUNT / lambdaCount;
		   const size_t matrixStride = batchCount * digHoloPolCount * digHoloModeCountOut;

		   //(A matrix, UU matrix and S vector)=3. This is over allocated, particularly for S (which is a vector not a matrix and if float32 not complex64). 
		   //You could probably reuse memory as well to allocate less.
		   //Also UU and S aren't even always needed at all, it depends if you're calculating UU* or doing the SVD.
		   const size_t memCount = matrixStride * 3;

		   if (polThreads != digHoloPowerMeterWorkspace_PolCount || lambdaThreads != digHoloPowerMeterWorkspace_WavelengthCount || memCount!=digHoloPowerMeterWorkspace_MemCount)
		   {
			   allocate3D(polThreads, lambdaThreads, memCount, digHoloPowerMeterWorkspace);

			   digHoloPowerMeterWorkspace_PolCount = polThreads;
			   digHoloPowerMeterWorkspace_WavelengthCount = lambdaThreads;
			   digHoloPowerMeterWorkspace_MemCount = memCount;
			   return true;
		   }
		   return false;
	   }

	   void digHoloPowerMeterThreaded(float*** out0, unsigned char polsAreIndependent, complex64** overlapCoefsBackup, bool doIL, bool doMDL, bool doXtalk, bool mulConj, int lambdaCount,int polThreads)
	   {
		   if (polThreads <= 0)
		   {
			   polThreads = 1;
		   }

		   int lambdaThreads = digHoloThreadCount / polThreads;

		   if (lambdaThreads <= 0)
		   {
			   lambdaThreads = 1;
		   }
		   int lambdaEach = (int)(ceil((1.0*lambdaCount) / lambdaThreads));

		   digHoloPowerMeterWorkspaceAllocate(polThreads, lambdaThreads, lambdaCount);

		   int j = 0;
		   for (int polThreadIdx = 0; polThreadIdx < polThreads; polThreadIdx++)
		   {
			   for (int lambdaThreadIdx = 0; lambdaThreadIdx < lambdaThreads; lambdaThreadIdx++)
			   {
				   workPackage* work = digHoloWorkPacks[j];

				   int lambdaStart = lambdaThreadIdx * lambdaEach;
				   int lambdaStop = (lambdaThreadIdx + 1) * lambdaEach;

				   if (lambdaStop > lambdaCount)
				   {
					   lambdaStop = lambdaCount;
				   }

				   work[0].ptr1 = (void*)overlapCoefsBackup;
				   /*
				   if (digHoloOverlapWorkspace)
				   {
					 //  work[0].ptr3 = (complex64*)&digHoloOverlapWorkspace[j][0][0][0];
					   const size_t batchCount = digHoloBatchCOUNT / lambdaCount;
					   const size_t matrixStride = batchCount * digHoloPolCount * digHoloModeCountOut;
					   size_t memOffset = ((size_t)(lambdaStart * digHoloPolCount + polThreadIdx*0)) * matrixStride;
					   complex64* workspace = (complex64*)&digHoloOverlapWorkspace[0][0][0];
					   work[0].ptr3 = &workspace[memOffset];
				   }
				   */
				   work[0].ptr3 = digHoloPowerMeterWorkspace[polThreadIdx][lambdaThreadIdx];
				   if (polThreads == 2)
				   {
					   work[0].flag1 = polThreadIdx;
					   work[0].ptr2 = out0[polThreadIdx];
				   }
				   else
				   {
					   work[0].flag1 = -1;
					   work[0].ptr2 = out0[0];
				   }
				   work[0].flag2 = doIL;
				   work[0].flag3 = doMDL;
				   work[0].flag4 = doXtalk;
				   work[0].flag5 = polsAreIndependent;
				   work[0].flag6 = mulConj;
				   work[0].flag7 = lambdaCount;
				   work[0].flag8 = lambdaStart;
				   work[0].flag9 = lambdaStop;

				   work[0].callback = workFunction::powerMeter;
				   work[0].workCompleteEvent.Reset();
				   work[0].workNewEvent.Set();
				   j++;
			   }
		   }
		   int totalThreads = j;
		   for (int j = 0; j < totalThreads; j++)
		   {
			   workPackage* work = digHoloWorkPacks[j];
			   work[0].workCompleteEvent.WaitOne();
		   }

		   for (int polThreadIdx = 0; polThreadIdx < polThreads; polThreadIdx++)
		   {
			   float** out = out0[polThreadIdx];
			   for (int metricIdx = 0; metricIdx < DIGHOLO_METRIC_COUNT; metricIdx++)
			   {
				   out[metricIdx][lambdaCount] = 0;

				   for (int lambdaIdx = 0; lambdaIdx < lambdaCount; lambdaIdx++)
				   {
					   out[metricIdx][lambdaCount] += out[metricIdx][lambdaIdx];
					   out[metricIdx][lambdaIdx] = 10 * log10(out[metricIdx][lambdaIdx]);
				   }
				   out[metricIdx][lambdaCount] = 10 * log10(out[metricIdx][lambdaCount] / lambdaCount);
			   }
		   }
		  // return out[DIGHOLO_METRIC_IL][lambdaCount];


	   }

public: float* AutoAlignGetMetrics(int metricIdx, int &lambdaCount)
{
	if (digHoloAutoAlignMetrics)
	{
		lambdaCount = digHoloAutoAlignMetricsWavelengthCount;
		return digHoloAutoAlignMetrics[metricIdx];
	}
	else
	{
		lambdaCount = 0;
		return 0;
	}

	
}

	  public: int AutoAlignCalcMetrics()
	  {
		  int lambdaCount = AutoAlignMetricCheck(false);
		  int polsAreIndependent = config.AutoAlignPolIndependence;
		  int MulConjTrans = config.AutoAlignBasisMulConjTrans;
		  float** out = digHoloAutoAlignMetrics;

		  if (out && lambdaCount && digHoloOverlapCoefsPtr)
		  {
			  digHoloPowerMeterThreaded(&out, polsAreIndependent, NULL, true, true, true, MulConjTrans, lambdaCount, 1);
			  return DIGHOLO_ERROR_SUCCESS;
		  }
		  else
		  {
			  return DIGHOLO_ERROR_NULLPOINTER;
		  }
	  }

	  //This does a estimate using the digHoloAutoAlignSnap routine, which should get you pretty close to the answer.
	  //Then the rest of this routine does sweeps of the parameter individually, one at a time (polarisations are handled in parallel)
	  //These individual parameter sweeps are fitted to quadratics using the first few points, the additional samples and re-fits are applied at the positions predicted by the quadratic to be the peak position.
	//mulConj : When measuring properties such as signal and SNR, could the transfer matrix be analysed directly (false), or multiplied by it's conjugate transpose in order to get something ~diagonal (true). 
	  //If there's an unknown unitary transform between your measurement basis, and your goal basis for properties such as xtalk, this should be true.
public: float AutoAlign(int tweakMode, int polsAreIndependent, float terminationTol, float** outputMetrics, int mulConj)
{
	//config.AutoAlignFourierWindowRadius = true;
	if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
	{
		fprintf(consoleOut, "<AUTO ALIGN>\n\r");
		fflush(consoleOut);
	}

	std::chrono::duration<double> fftStartTime = benchmarkFFTTime;
	std::chrono::duration<double> ifftStartTime = benchmarkIFFTTime;
	std::chrono::duration<double> applyTiltStartTime = benchmarkApplyTiltTime;
	std::chrono::duration<double> overlapModesStartTime = benchmarkOverlapModesTime;

	int64_t fftStartCount = benchmarkFFTCounter;
	int64_t ifftStartCount = benchmarkIFFTCounter;
	int64_t applyTiltStartCount = benchmarkApplyTiltCounter;
	int64_t overlapStartCount = benchmarkOverlapModesCounter;

	digHoloPixelBufferConversionFlag = 1;

	std::chrono::steady_clock::time_point startTimeTotal = std::chrono::steady_clock::now();

	int windowSizeX = config.fftWindowSizeX;
	int windowSizeY = config.fftWindowSizeY;

	const int goalIdx0 = config.AutoAlignGoalIdx;

	bool fftPending = true;
	const int maxMG = config.maxMG;

	/*if (outputMetrics)
	{
		for (int i = 0; i < DIGHOLO_METRIC_COUNT; i++)
		{
			outputMetrics[i] = -FLT_MAX;
		}
	}*/
	

	if (config.AutoAlignFourierWindowRadius)
	{
		//If we'll be aligning tilt, then set the window to maximum size
		if (config.AutoAlignTilt)
		{
			const float apertureMax = FourierWindowRadiusMax(config.pixelSize, config.wavelengthCentre);

			config.apertureSize = apertureMax;
		}
		else
		{
			const float apertureMax = FourierWindowRadiusRec(config.zernCoefs, digHoloPolCount);
			config.apertureSize = apertureMax;
		}
	}

	if (tweakMode == DIGHOLO_AUTOALIGNMODE_FULL || tweakMode == DIGHOLO_AUTOALIGNMODE_ESTIMATE || maxMG == 0)
	{
		if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
		{
			fprintf(consoleOut, "<Parameter estimate>\n\r");
			fflush(consoleOut);
		}
		digHoloAutoAlignSnap(!fftPending || (tweakMode == DIGHOLO_AUTOALIGNMODE_ESTIMATE) || maxMG == 0, polsAreIndependent, outputMetrics);
		std::chrono::steady_clock::time_point snapTime = std::chrono::steady_clock::now();
		std::chrono::duration<double> time_span = (std::chrono::duration_cast<std::chrono::duration<double>>(snapTime - startTimeTotal));
		if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
		{
			fprintf(consoleOut, "Parameter estimate time (ms)	%f\n\r", time_span.count() * 1000);
			fprintf(consoleOut, "</Parameter estimate>\n\r");
			fflush(consoleOut);
		}
	}
	else
	{
		config.AutoAlignTiltStep = config.apertureSize * 0.1f;//10% of the aperture
		config.AutoAlignCentreStep = config.waist[0] * 0.1f;

		//Set the defocus step to at least this, or 10% of the current focus, whichever is more.
		config.AutoAlignDefocusStep = 0.4f;
		for (int i = 0; i < digHoloPolCount; i++)
		{
			float minFocus = fabs(config.zernCoefs[i][DEFOCUS]) * 0.1f;
			if (minFocus > config.AutoAlignDefocusStep)
			{
				config.AutoAlignDefocusStep = minFocus;
			}
		}

		config.AutoAlignBasisWaistStep = config.waist[0] * 0.1f;
	}

	if ((tweakMode == DIGHOLO_AUTOALIGNMODE_FULL || tweakMode == DIGHOLO_AUTOALIGNMODE_TWEAK) && maxMG > 0)
	{
		if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
		{
			fprintf(consoleOut, "<Parameter tweak>\n\r");
			fflush(consoleOut);
		}
		const int parameterCount = 6;
		//These values define what consistutes a 'stupid' value to attempt. You should never see these, but if you do, the console will print a warning, that a 'stupid' value has been attempted.
		float LIMS[parameterCount][2];
		//The x-axis can't have negative spatial frequencies along the x-axis (due to the real-to-complex transform not calculating those)
		//and it also can't have any spatial frequencies outside the Fourier space bounds.
		LIMS[AUTOALIGN_TILTX][0] = 0;
		LIMS[AUTOALIGN_TILTX][1] = ((digHoloWavelengthCentre / (2 * digHoloPixelSize)) / DIGHOLO_UNIT_ANGLE);//The maximum angle in Fourier space 

		LIMS[AUTOALIGN_TILTY][0] = -(config.apertureSize + LIMS[AUTOALIGN_TILTX][1]);
		LIMS[AUTOALIGN_TILTY][1] = (config.apertureSize + LIMS[AUTOALIGN_TILTX][1]);

		LIMS[AUTOALIGN_DEFOCUS][0] = -100;
		LIMS[AUTOALIGN_DEFOCUS][1] = 100;

		LIMS[AUTOALIGN_WAIST][0] = (float)(digHoloPixelSize / (3 * DIGHOLO_UNIT_PIXEL));
		LIMS[AUTOALIGN_WAIST][1] = (float)(digHoloPixelSize * fmax(digHoloFrameHeight, digHoloFrameWidth) / DIGHOLO_UNIT_PIXEL);

		LIMS[AUTOALIGN_CX][0] = -(digHoloFrameWidth / 2.0f) * config.pixelSize;// digHoloXaxis[FFT_IDX][0] * 1e6;
		LIMS[AUTOALIGN_CX][1] = (digHoloFrameWidth / 2.0f) * config.pixelSize; //digHoloXaxis[FFT_IDX][digHoloWoiPolWidth[FFT_IDX] - 1] * 1e6;

		LIMS[AUTOALIGN_CY][0] = -(digHoloFrameHeight / 2.0f) * config.pixelSize; ;// digHoloYaxis[FFT_IDX][0] * 1e6;
		LIMS[AUTOALIGN_CY][1] = (digHoloFrameHeight / 2.0f) * config.pixelSize; ;// digHoloYaxis[FFT_IDX][digHoloWoiPolHeight[FFT_IDX] - 1] * 1e6;


		float** bestAlignment = 0;
		allocate2D(2, parameterCount, bestAlignment);

		if (windowSizeX != config.fftWindowSizeX || windowSizeY != config.fftWindowSizeY)
		{
			config.fftWindowSizeX = windowSizeX;
			config.fftWindowSizeY = windowSizeY;
		}
		else
		{
			fftPending = false;
		}

		if (tweakMode)
		{
			fftPending = true;
		}

		const int optimCount = DIGHOLO_METRIC_COUNT;

		int lambdaCount = 0;
		float*** out = 0;
		float** outf = 0;
		float** outBest = 0;
		//allocate2D(2, optimCount, out);
		allocate2D(2, optimCount, outBest);

		int goalIdx = goalIdx0;

		bool doIL = goalIdx == 0;
		bool doMDL = goalIdx == 1;
		bool doXtalk = goalIdx >= 2;

		const int polyTermCount = 3;
		//The number of samples to take before fitting to a quadratic and taking a first estimate at the peak.
		//Must be at least 3.
		const int sampleCount = 3;
		//The total number of steps taken.
		const int stepCount = sampleCount + 2;//Should be an odd number (e.g. 2 below, 2 above and 1 at the centre)

		//int stepIdxes[stepCount] = { 0,  -1, -2 ,1 ,2 };
		float** x = 0;
		float** y = 0;
		//float** yLinear = 0;
		allocate2D(2, stepCount + 2, x);
		allocate2D(2, stepCount + 2, y);
		//allocate2D(2, stepCount + 2, yLinear);
		float** B = 0;
		float** B0 = 0;
		allocate2D(2, polyTermCount, B);
		allocate2D(2, polyTermCount, B0);

		complex64** overlapCoefsBackup = 0;

		bool converged = false;
		float lastErr[] = { FLT_MAX, FLT_MAX };
		float lastErrTol = terminationTol;
		int maxAttempts = 20;
		unsigned char TiltPolLock = config.PolLockTilt;
		unsigned char WaistPolLock = config.PolLockBasisWaist;
		unsigned char DefocusPolLock = config.PolLockDefocus;

		//float** zernCoefs = config.zernCoefs;
		//float bestOut[] = { FLT_MAX,FLT_MAX };
		float bestPwr[] = { -FLT_MAX,-FLT_MAX };
		int attemptCount = 0;
		//const int outputMetricCount = optimCount;
		//float outf[outputMetricCount];
		alignmentSave(0, bestAlignment);
		alignmentSave(1, bestAlignment);
		float** dTxs = 0;
		allocate2D(stepCount + 2, 2, dTxs);
		float** pwrs = 0;
		allocate2D(stepCount + 2, 2, pwrs);


		float stepSize[parameterCount];// = { config->AutoAlignTiltStep, config->AutoAlignTiltStep, config->AutoAlignCentreStep, config->AutoAlignCentreStep, config->AutoAlignDefocusStep, config->AutoAlignBasisWaistStep };
		stepSize[AUTOALIGN_TILTX] = config.AutoAlignTiltStep;
		stepSize[AUTOALIGN_TILTY] = config.AutoAlignTiltStep;
		stepSize[AUTOALIGN_DEFOCUS] = config.AutoAlignDefocusStep;
		stepSize[AUTOALIGN_CX] = config.AutoAlignCentreStep;
		stepSize[AUTOALIGN_CY] = config.AutoAlignCentreStep;
		stepSize[AUTOALIGN_WAIST] = config.AutoAlignBasisWaistStep;

		float stepBounds[parameterCount];

		for (int i = 0; i < parameterCount; i++)
		{
			stepBounds[i] = stepSize[i] * 10;
		}
		unsigned char firstRun = true;

		//Quadratic fit parameters (tolerance and max. iterations)
		const float fitTol = 1e-6f;
		const float fitIterationCount = 101;

		//If we're not using insertion loss as the goal. We'll do 1 pass with IL, then go through again to do the parameter of interest.
		//Less likely to get stuck in weird local minima this way, but it's a bit slower.
		int goalCount = 1;
		if (goalIdx)
		{
			goalCount = 2;
			lastErrTol = 0.1f + terminationTol * 10.0f;
		}

		unsigned char tiltFieldValid = false;

		for (int goalIDX = 0; goalIDX < goalCount; goalIDX++)
		{
			if (!goalIDX)
			{
				goalIdx = 0;
			}
			else
			{
				goalIdx = goalIdx0;
			}
			converged = false;
			firstRun = true;
			bestPwr[0] = -FLT_MAX;
			bestPwr[1] = -FLT_MAX;
			lastErr[0] = FLT_MAX;
			lastErr[1] = FLT_MAX;
			outBest[0][goalIdx] = -FLT_MAX;
			outBest[1][goalIdx] = -FLT_MAX;

			doIL = goalIdx == 0;//If we only want IL, we don't need to do an SVD.
			doMDL = goalIdx == 1;//If we want to do MDL, we'll have to do an SVD.
			doXtalk = goalIdx >= 2;//If we want to look at other parameters, we don't need to do an SVD, but we will need to do some additional calculations.

			attemptCount = 0;

			//If this is a later pass through with a more specific goal (i.e. MDL, XTALK etc), then decrease the step size.
			//We should already be near the solution at this point.
			if (goalIDX)
			{
				stepSize[AUTOALIGN_TILTX] = config.AutoAlignTiltStep / 10.0f;
				stepSize[AUTOALIGN_TILTY] = config.AutoAlignTiltStep / 10.0f;
				stepSize[AUTOALIGN_DEFOCUS] = config.AutoAlignDefocusStep / 10.0f;
				stepSize[AUTOALIGN_CX] = config.AutoAlignCentreStep / 10.0f;
				stepSize[AUTOALIGN_CY] = config.AutoAlignCentreStep / 10.0f;
				stepSize[AUTOALIGN_WAIST] = config.AutoAlignBasisWaistStep / 10.0f;
				lastErrTol = terminationTol;
			}

			//Example, mode tx matrix, email 27th July should give 5.8dB MDL (filebufferCaldNick.mat)
			while (!converged)
			{
				std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
				//This is inside the loop, so that the user can change it on-the-fly
				unsigned char parameterEnabled[parameterCount];// = { config->AutoAlignTilt, config->AutoAlignTilt, config->AutoAlignCentre, config->AutoAlignCentre, config->AutoAlignDefocus, config->AutoAlignWaist };
				parameterEnabled[AUTOALIGN_TILTX] = config.AutoAlignTilt;
				parameterEnabled[AUTOALIGN_TILTY] = config.AutoAlignTilt;
				parameterEnabled[AUTOALIGN_CX] = config.AutoAlignCentre;
				parameterEnabled[AUTOALIGN_CY] = config.AutoAlignCentre;
				parameterEnabled[AUTOALIGN_WAIST] = config.AutoAlignBasisWaist;
				parameterEnabled[AUTOALIGN_DEFOCUS] = config.AutoAlignDefocus;

				//For every parameter (tilt, focus, waist, beam centre)
				for (int parIdx = 0; parIdx < parameterCount; parIdx++)
				{
					//If optimisation of that parameter is enabled...
					if (parameterEnabled[parIdx])
					{
						bool parameterLock = ((parIdx == AUTOALIGN_TILTX && TiltPolLock) || (parIdx == AUTOALIGN_TILTY && TiltPolLock) || (parIdx == AUTOALIGN_DEFOCUS && DefocusPolLock) || (parIdx == AUTOALIGN_WAIST && WaistPolLock));

						//Option to polLock if it's the first attempt, and it's not the beam centre
						//parameterLock = parameterLock || (attemptCount == 0 && (parIdx!=AUTOALIGN_CX && parIdx!=AUTOALIGN_CY));

						float var0[2];

						//Get the current value for that parameter in the H polarisation
						var0[0] = digHoloAutoAlignGet(0, parIdx);

						//If the two polarisations aren't locked together, then also get the V polarisation value
						if (!parameterLock)
						{
							var0[1] = digHoloAutoAlignGet(1, parIdx);
						}
						else //Otherwise, lock the V polarisation value to the H polarisation value
						{
							var0[1] = var0[0];
						}

						float meanPwr[] = { 0,0 };
						float maxPwr[] = { -FLT_MAX,-FLT_MAX };
						//float maxV[] = { 0,0 };
						float maxima[] = { 0,0 };
						unsigned char isValid[] = { true,true };

						//For every step
						for (int stepIdx = 0; stepIdx < (stepCount + 1); stepIdx++)
						{

							//Pointer to the array to store the new values
							float* dTx = dTxs[stepIdx];

							for (int polIdx = 0; polIdx < digHoloPolCount; polIdx++)
							{
								//If this is one of the early steps (<sampleCount), then we'll be sampling +/- either side of the current value at equally spaced points.
								if (stepIdx < sampleCount)
								{
									int sCount = sampleCount;

									dTx[polIdx] = (float)(var0[polIdx] + ((stepIdx - (sCount - 1) / 2.0) * stepSize[parIdx] / ((sCount - 1))));
									//const int stepOrder[] = { 0,2,1 };
									//dTx[polIdx] = (float)(var0[polIdx] + ((stepOrder[stepIdx] - (sCount - 1) / 2.0) * stepSize[parIdx] / ((sCount - 1))));

								}

								//If the two polarisations must have the same value, lock V to H
								if (parameterLock)
								{
									dTx[1] = dTx[0];
								}

								//Performs checks to make sure a 'stupid' value isn't being attempted. e.g. a tilt that's outside the Fourier plane, a waist that's bigger than the camera/smaller than a pixel etc.

								if (dTx[polIdx] > LIMS[parIdx][1])
								{
									dTx[polIdx] = LIMS[parIdx][1];
									if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
									{
										fprintf(consoleOut, "Stupid value attempted %i	%f\n\r", parIdx, dTx[polIdx]);
										fflush(consoleOut);
									}
								}
								if (dTx[polIdx] < LIMS[parIdx][0])
								{
									dTx[polIdx] = LIMS[parIdx][0];
									if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
									{
										fprintf(consoleOut, "Stupid value attempted %i	%f\n\r", parIdx, dTx[polIdx]);
										fflush(consoleOut);
									}
								}

								//Checks that invalid values haven't managed to get in there somehow.
								isValid[polIdx] = !isnan(dTx[polIdx]) && !isinf(dTx[polIdx]);

								//If the H polarisation value is valid, update the config to this value
								if (isValid[polIdx])
								{
									digHoloAutoAlignSet(polIdx, parIdx, dTx[polIdx]);
								}
							}
							//What type of parameter we're changing, will define how much of the digital holography process we'll have to redo. digHoloFFT,IFFT,applyTilt
							//e.g. if the focus is changed, we don't have to redo FFT/IFFT, only the applyTilt method
							unsigned char tiltChanged = (parIdx == AUTOALIGN_TILTX || parIdx == AUTOALIGN_TILTY);
							unsigned char centreChanged = (parIdx == AUTOALIGN_CX || parIdx == AUTOALIGN_CY);
							unsigned char focusChanged = (parIdx == AUTOALIGN_DEFOCUS);
							unsigned char waistChanged = parIdx == AUTOALIGN_WAIST;

							if (config.AutoAlignFourierWindowRadius)
							{
								const float apertureMax = FourierWindowRadiusRec(config.zernCoefs, digHoloPolCount);

								if (config.apertureSize > apertureMax)
								{
									config.apertureSize = apertureMax;
									digHoloApertureSize = apertureMax * DIGHOLO_UNIT_ANGLE;
								}
							}

							//We may have to do an initial FFT, e.g. if this is the first run, or if the FFT window size has changed.
							if (fftPending)
							{
								digHoloFFT();
								tiltFieldValid = false;
							}
							//If the beam centre has changed, we'll have to redo digHoloCopyWindow and the IFFT.
							if (centreChanged || fftPending)
							{
								digHoloIFFT();
								tiltFieldValid = false;
							}

							if (tiltChanged || centreChanged || focusChanged || fftPending)
							{
									digHoloApplyTilt();
									tiltFieldValid = true;
							}

							if (tiltChanged || centreChanged || focusChanged || waistChanged || fftPending)
							{
									digHoloOverlapModes();


								//If this is the first run, make a copy of the overlap coefficients. We need a copy, so that the two polarisations can work in parallel if need be.
								if (firstRun)
								{
									allocate2D(digHoloBatchCOUNT, digHoloModeCountOut * digHoloPolCount, overlapCoefsBackup);
									memcpy(&overlapCoefsBackup[0][0][0], &digHoloOverlapCoefsPtr[0][0][0], sizeof(complex64) * digHoloModeCountOut * digHoloPolCount * digHoloBatchCOUNT);
									firstRun = false;
									
										lambdaCount = digHoloWavelengthCount;
										allocate3D(2, optimCount, (lambdaCount+1), out);
										allocate2D(optimCount, (lambdaCount + 1), outf);
										
								}

								//If the parameters are independent for each polarisation, do two separate evaluations in parallel for each polarisation.
								if ((!parameterLock && digHoloPolCount > 1) || polsAreIndependent)
								{
									int polThreads = 2;
									digHoloPowerMeterThreaded(out, polsAreIndependent, overlapCoefsBackup, doIL, doMDL, doXtalk, mulConj,lambdaCount,polThreads);
								}
								else
								{
									int polThreads = 1;
									digHoloPowerMeterThreaded(out, polsAreIndependent, overlapCoefsBackup, doIL, doMDL, doXtalk, mulConj, lambdaCount, polThreads);
								/*	if (digHoloWavelengthCount == 1)
									{
										digHoloPowerMeter(out[0], false, doIL, doMDL, doXtalk, mulConj, lambdaCount);
									}*/
									memcpy(&out[1][0][0], &out[0][0][0], sizeof(float) * optimCount*(lambdaCount+1));
								}

							}

							//Reset parameters indicating this is a special case/first run. e.g. sometimes we'll have to do an extra digHoloFFT to set things up, but most of the time we can skip that step.
							fftPending = false;
							for (int i = 0; i < digHoloPolCount; i++)
							{
								pwrs[stepIdx][i] = out[i][goalIdx][lambdaCount];
								float pwr = pwrs[stepIdx][i];

								x[i][stepIdx] = dTx[i];
								y[i][stepIdx] = pwr;
								//yLinear[i][stepIdx] = powf(10, pwr / 10);

								if (pwr > maxPwr[i] && isValid[i])
								{
									maxPwr[i] = pwr;
									//maxV[i] = dTx[i];


									if (pwr > bestPwr[i])
									{
										//SaveBestConfig();
										alignmentSave(i, bestAlignment);
										for (int metricIdx = 0; metricIdx < DIGHOLO_METRIC_COUNT; metricIdx++)
										{
											outBest[i][metricIdx] = out[i][metricIdx][lambdaCount];
										}
										//memcpy(&outBest[i][0], &out[i][0], sizeof(float) * optimCount);
										size_t polOffset = ((size_t)digHoloModeCountOut) * i;
										//Create a copy of the overlap coefs so we don't have to re-overlap later if we need to revert to this setting
										//Might be faster just to have two copies for each polarisation so you can copy it as a single chunk?
										for (int batchIdx = 0; batchIdx < digHoloBatchCOUNT; batchIdx++)
										{
											memcpy(&overlapCoefsBackup[batchIdx][polOffset][0], &digHoloOverlapCoefsPtr[batchIdx][polOffset][0], sizeof(complex64) * digHoloModeCountOut);
										}
										bestPwr[i] = pwr;
									}
								}
								meanPwr[i] += pwr;

								if (stepIdx >= (sampleCount - 1))
								{
									if (stepIdx == (sampleCount - 1))
									{
										B0[i][0] = 0;
										B0[i][1] = 0;
										B0[i][2] = meanPwr[i] / (stepIdx + 1);

										//dumb bubble sort in ascending order. This is only needed if we decide to fit to most recent point.
										//and hence we want to make sure the 'most recent' in the array are the best values. i.e. we're fitting a quadratic to the best values.
										for (int sortIdx = 0; sortIdx <= stepIdx; sortIdx++)
										{
											for (int sortIdy = 1; sortIdy <= stepIdx; sortIdy++)
											{
												if (y[i][sortIdy] < y[i][sortIdy - 1])
												{
													float swpy = y[i][sortIdy];
													y[i][sortIdy] = y[i][sortIdy - 1];
													y[i][sortIdy - 1] = swpy;

													float swpx = x[i][sortIdy];
													x[i][sortIdy] = x[i][sortIdy - 1];
													x[i][sortIdy - 1] = swpx;
												}
											}
										}
									}

									//Fit just the 3 most recent points
									//maxima[i] = fitToQuadratic(&x[i][stepIdx - 2], &y[i][stepIdx - 2], (3), B[i], B0[i], fitTol, fitIterationCount);

									//After initial fit to 3 points, keep forgetting the worst points from the first 3, and updating with new points.
									//int dStep0 = stepIdx-(sampleCount-1);
									//maxima[i] = fitToQuadratic(&x[i][stepIdx - 2 - dStep0], &y[i][stepIdx - 2-dStep0], (3+ dStep0), B[i], B0[i], fitTol, fitIterationCount);

									//Fit everything so far
									maxima[i] = fitToQuadratic(x[i], y[i], stepIdx + 1, B[i], B0[i], fitTol, fitIterationCount);

									bool valid = !isnan(maxima[0]) && !isinf(maxima[0]);

									if (valid)
									{
										dTxs[stepIdx + 1][i] = maxima[i];

										//Distance to the best value so far
										float dBest = (maxima[i] - bestAlignment[i][parIdx]);
										//If this new proposed value is too big of a jump, reign it in.
										if (fabsf(dBest) > stepBounds[parIdx])
										{
											if (dBest > stepBounds[parIdx])
											{
												dTxs[stepIdx + 1][i] = bestAlignment[i][parIdx] + stepBounds[parIdx];
											}
											else
											{
												dTxs[stepIdx + 1][i] = bestAlignment[i][parIdx] - stepBounds[parIdx];
											}
										}

									}
									else
									{//If it's an invalid point for some reason, just choose sample point near the best one so far
										dTxs[stepIdx + 1][i] = (float)(bestAlignment[i][parIdx] + ((stepIdx - (stepCount - 1) / 2.0) * stepSize[parIdx] / (stepCount - 1)));
									}
									B0[i][0] = B[i][0];
									B0[i][1] = B[i][1];
									B0[i][2] = B[i][2];

								}
							}//for pol
						}//for step

						for (int polIdx = 0; polIdx < digHoloPolCount; polIdx++)
						{
							alignmentLoad(polIdx, bestAlignment);
						}

						//This could probably be at least partially avoided
						//Final run through, this ensures everything is valid.
						//e.g. we may have avoided doing digHoloFFT when we adjusted tilt, assuming that the change is small, but here we make sure everything is strictly correct before moving on.
						//digHoloFFT(digHoloPixelsCameraPlane);
						digHoloIFFT();
						//

							digHoloApplyTilt();
							digHoloOverlapModes();

						//stepSize[goalIDX] = stepSize[goalIDX] * 2;
					}//if parameter enabled
				}//for each parameter

				//A pretty silly scenario, where you've managed to get through the whole loop, without actually doing anything. e.g. all parameters disabled.
				if (firstRun)
				{
					digHoloFFT();
					digHoloIFFT();

						digHoloApplyTilt();
						digHoloOverlapModes();

					allocate2D(digHoloBatchCOUNT, digHoloModeCountOut * digHoloPolCount, overlapCoefsBackup);
				}

				//digHoloPowerMeter(outf, polsAreIndependent, true, true, true, mulConj,lambdaCount,0,lambdaCount);
				digHoloPowerMeterThreaded(&outf, polsAreIndependent, overlapCoefsBackup, true, true, true, mulConj, lambdaCount, 1);
				memcpy(&overlapCoefsBackup[0][0][0], &digHoloOverlapCoefsPtr[0][0][0], sizeof(complex64) * digHoloModeCountOut * digHoloPolCount * digHoloBatchCOUNT);

				std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();

				std::chrono::duration<double> time_span = (std::chrono::duration_cast<std::chrono::duration<double>>(stopTime - startTime));
				double dt = time_span.count();
				double fps = 1.0 / dt;

				float dErr = 0;

				for (int polIdx = 0; polIdx < digHoloPolCount; polIdx++)
				{

					dErr += fabsf(outBest[polIdx][goalIdx] - lastErr[polIdx]);
					lastErr[polIdx] = outBest[polIdx][goalIdx];
				}
				dErr = dErr / digHoloPolCount;

				if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
				{
					/*
					float* settings = alignmentSettings[polIdx];
					float* zernCoefs = config.zernCoefs[polIdx];

					settings[AUTOALIGN_TILTX] = zernCoefs[TILTX];
					settings[AUTOALIGN_TILTY] = zernCoefs[TILTY];
					settings[AUTOALIGN_DEFOCUS] = zernCoefs[DEFOCUS];

					settings[AUTOALIGN_WAIST] = config.waist[polIdx];
					settings[AUTOALIGN_CX] = config.BeamCentreX[polIdx];
					settings[AUTOALIGN_CY] = config.BeamCentreY[polIdx];
					*/
					fprintf(consoleOut, "	Goal Metric:	%f	IL:%f	MDL:%f	(Hz:%f)	dt: [%f s] dErr	%f \n\r", outf[goalIdx0][lambdaCount], outf[0][lambdaCount], outf[1][lambdaCount], fps, dt, dErr);
					for (int polIDXX = 0; polIDXX < config.polCount; polIDXX++)
					{
						fprintf(consoleOut, "		Pol:%i	Tilt:	%3.3f	%3.3f	Defocus:	%3.3f	Waist:	%3.3f	Centre:	%3.3f	%3.3f\n\r",
							polIDXX,
							bestAlignment[polIDXX][AUTOALIGN_TILTX],
							bestAlignment[polIDXX][AUTOALIGN_TILTY],
							bestAlignment[polIDXX][AUTOALIGN_DEFOCUS],
							bestAlignment[polIDXX][AUTOALIGN_WAIST] * 1e6,
							bestAlignment[polIDXX][AUTOALIGN_CX] * 1e6,
							bestAlignment[polIDXX][AUTOALIGN_CY] * 1e6
						);
					}
					fflush(consoleOut);
				}

				attemptCount++;
				if (dErr <= lastErrTol || attemptCount == maxAttempts)
				{
					converged = true;
				}
			}
			//stepSize[goalIDX] = stepSize[goalIDX] / 2;
		}

		free2D(x);
		free2D(y);
		free2D(B);
		free2D(B0);
		free2D(overlapCoefsBackup);
		free2D(dTxs);
		free3D(out);
		free2D(outBest);

		std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();
		double dt = (std::chrono::duration_cast<std::chrono::duration<double>>(stopTime - startTimeTotal)).count();
		double fps = 1.0 / dt;

		if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
		{
			fprintf(consoleOut, "GOAL:	%f	IL:	%f\n\r", outf[goalIdx0][lambdaCount], outf[DIGHOLO_METRIC_IL][lambdaCount]);
			fprintf(consoleOut, "Completed alignment %f seconds (%f Hz)\n\r", dt, fps);
			
			const float apertureMax = FourierWindowRadiusRec(config.zernCoefs, digHoloPolCount);
			const float apertureABSMax = FourierWindowRadiusMax(config.pixelSize, config.wavelengthCentre);
			if (config.apertureSize>apertureMax)
			{
				fprintf(consoleOut, "WARNING : Fourier window radius (%3.3f) is too large for specified tilt(x,y). Reduce window radius to maximum of %3.3f.\n\r", config.apertureSize,apertureMax);
			}
			if (config.apertureSize > apertureABSMax)
			{
				fprintf(consoleOut, "WARNING : Fourier window radius (%3.3f) is too large for any resolvable tilt(x,y) for pixel size (%3.3e) and wavelength (%3.3e). Reduce window radius to absolute maximum of %3.3f.\n\r", config.apertureSize, config.pixelSize, config.wavelengthCentre,apertureABSMax);
			}
			fflush(consoleOut);
		}
		config.fftWindowSizeX = windowSizeX;
		config.fftWindowSizeY = windowSizeY;
		//ConfigBackupSave();
		if (outputMetrics)
		{
			for (int lambdaIdx = 0; lambdaIdx < (lambdaCount+1); lambdaIdx++)
			{
				for (int metricIdx = 0; metricIdx < DIGHOLO_METRIC_COUNT; metricIdx++)
				{
					outputMetrics[metricIdx][lambdaIdx] = outf[metricIdx][lambdaIdx];
				}
			}
		}

		int64_t fftStopCount = benchmarkFFTCounter;
		int64_t ifftStopCount = benchmarkIFFTCounter;
		int64_t applyTiltStopCount = benchmarkApplyTiltCounter;
		int64_t overlapStopCount = benchmarkOverlapModesCounter;

		std::chrono::duration<double> fftStopTime = benchmarkFFTTime;
		double fftTime = (fftStopTime - fftStartTime).count();
		std::chrono::duration<double> ifftStopTime = benchmarkIFFTTime;
		double ifftTime = (ifftStopTime - ifftStartTime).count();
		std::chrono::duration<double> applyTiltStopTime = benchmarkApplyTiltTime;
		double applyTiltTime = (applyTiltStopTime - applyTiltStartTime).count();
		std::chrono::duration<double> overlapModesStopTime = benchmarkOverlapModesTime;
		double overlapTime = (overlapModesStopTime - overlapModesStartTime).count();

		if (config.verbosity >= DIGHOLO_VERBOSITY_DEBUG)
		{
			fprintf(consoleOut, "Summary routines (time, calls)\n\r");
			fprintf(consoleOut, "FFT        	%4.4f	%zi\n\r", fftTime, fftStopCount - fftStartCount);
			fprintf(consoleOut, "IFFT       	%4.4f	%zi\n\r", ifftTime, ifftStopCount - ifftStartCount);
			fprintf(consoleOut, "ApplyTilt   	%4.4f	%zi\n\r", applyTiltTime, applyTiltStopCount - applyTiltStartCount);
			fprintf(consoleOut, "OverlapModes	%4.4f	%zi\n\r", overlapTime, overlapStopCount - overlapStartCount);

			fprintf(consoleOut, "</Parameter tweak>\n\r");

			fflush(consoleOut);
		}

		digHoloPixelBufferConversionFlag = 0;
		if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
		{
			fprintf(consoleOut, "</AUTO ALIGN>\n\r\n\r");
			fflush(consoleOut);
		}

		return outf[goalIdx0][lambdaCount];
	}
	else //if we just did a snap, then return 0
	{
		config.fftWindowSizeX = windowSizeX;
		config.fftWindowSizeY = windowSizeY;
		digHoloPixelBufferConversionFlag = 0;

		if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
		{
			fprintf(consoleOut, "</AUTO ALIGN>\n\r\n\r");
			fflush(consoleOut);
		}
		
		if (outputMetrics)
		{
			int lambdaCount = digHoloWavelengthCount;
			return outputMetrics[goalIdx0][lambdaCount];
		}
		else
		{
			return -FLT_MAX;
		}
	}
}

private: int AutoAlignMetricCheck(int updateWavelengthCount)
{
	int lambdaCount = 0;
	if (updateWavelengthCount)
	{
		if ((!digHoloWavelengthValid && digHoloWavelengthCountNew > 0))
		{
			lambdaCount = digHoloWavelengthCountNew;
		}
		else
		{
			lambdaCount = 1;
		}
	}
	else
	{
		lambdaCount = digHoloWavelengthCount;
	}

	if (lambdaCount != digHoloAutoAlignMetricsWavelengthCount || digHoloAutoAlignMetricsWavelengthCount == 0)
	{
		digHoloAutoAlignMetricsWavelengthCount = lambdaCount;
		allocate2D(DIGHOLO_METRIC_COUNT, (lambdaCount + 1), digHoloAutoAlignMetrics);
		memset(&digHoloAutoAlignMetrics[0][0], 0, sizeof(float) * (lambdaCount + 1) * DIGHOLO_METRIC_COUNT);
	}
	return lambdaCount;
}

public: float AutoAlign()
{

	//int lambdaCount = 
		AutoAlignMetricCheck(true);

	return AutoAlign(config.AutoAlignMode, config.AutoAlignPolIndependence, config.AutoAlignTol, digHoloAutoAlignMetrics, config.AutoAlignBasisMulConjTrans);
}

	  float digHoloAutoAlignGet(int polIdx, int parIdx)
	  {
		  switch (parIdx)
		  {
			  //Tilt (x)
		  case AUTOALIGN_TILTX:
		  {
			  return config.zernCoefs[polIdx][TILTX];
		  }break;

		  //Tilt (y)
		  case AUTOALIGN_TILTY:
		  {
			  return config.zernCoefs[polIdx][TILTY];
		  }break;

		  //Centre X
		  case AUTOALIGN_CX:
		  {
			  return config.BeamCentreX[polIdx];
		  }break;

		  //Centre Y
		  case AUTOALIGN_CY:
		  {
			  return config.BeamCentreY[polIdx];
		  }break;
		  //Defocus
		  case AUTOALIGN_DEFOCUS:
		  {
			  return config.zernCoefs[polIdx][DEFOCUS];
		  }break;
		  //Waist
		  case AUTOALIGN_WAIST:
		  {
			  return config.waist[polIdx];
		  }break;
		  }
		  //This should never happen
		  return -FLT_MAX;
	  }

	  void digHoloAutoAlignSet(int polIdx, int parIdx, float value)
	  {
		  switch (parIdx)
		  {
			  //Tilt (x)
		  case AUTOALIGN_TILTX:
		  {
			  config.zernCoefs[polIdx][TILTX] = value;
		  }break;

		  //Tilt (y)
		  case AUTOALIGN_TILTY:
		  {
			  config.zernCoefs[polIdx][TILTY] = value;
		  }break;

		  //Centre X
		  case AUTOALIGN_CX:
		  {
			  config.BeamCentreX[polIdx] = value;
		  }break;

		  //Centre Y
		  case AUTOALIGN_CY:
		  {
			  config.BeamCentreY[polIdx] = value;
		  }break;
		  //Defocus
		  case AUTOALIGN_DEFOCUS:
		  {
			  config.zernCoefs[polIdx][DEFOCUS] = value;
		  }break;
		  //Waist
		  case AUTOALIGN_WAIST:
		  {
			  config.waist[polIdx] = value;
		  }break;
		  }
	  }

	  int ConfigBackupSave()
	  {
		  //if configBackup hasn't been initialised
		  if (configBackup.isNull)
		  {
			  configBackup.ConfigInit();
		  }

		  //if config hasn't been initialised
		  if (config.isNull)
		  {
			  return DIGHOLO_ERROR_NULLPOINTER;
		  }
		  else
		  {
			  configBackup.Copy(configBackup, config);
			  return DIGHOLO_ERROR_SUCCESS;
		  }
	  }

	  int ConfigBackupLoad()
	  {
		  if (config.isNull)
		  {
			  config.ConfigInit();
		  }

		  if (configBackup.isNull)
		  {
			  return DIGHOLO_ERROR_NULLPOINTER;
		  }
		  else
		  {
			  configBackup.Copy(config, configBackup);
			  return DIGHOLO_ERROR_SUCCESS;
		  }
	  }

	  int ThreadDiagnostic(float goalRuntime)
	  {
		  if (goalRuntime < 0)
		  {
			  return 0;
		  }
		  if (digHoloIsValidPixelBuffer())
		  {
			  if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
			  {
				  fprintf(consoleOut, "Estimating optimal thread count...\n\r");
			  }
			  const int threadCount = THREADCOUNT_MAX + 1;
			  const int timingCount = 7;
			  const int totalRuntimeIdx = 6;
			  float** timings = 0;
			  allocate2D(threadCount, timingCount, timings);
			  memset(&timings[0][0], 0, sizeof(float) * (threadCount)*timingCount);
			  int* bestThreadCount = 0;
			  allocate1D(timingCount, bestThreadCount);
			  char** functionNames = 0;
			  const int stringMax = 64;
			  char printoutEnabled[timingCount];
			  allocate2D(timingCount, stringMax, functionNames);
			  memset(&functionNames[0][0], 0, sizeof(char) * timingCount * stringMax);
			  strcpy(functionNames[0], "FFT      "); printoutEnabled[0] = 1;
			  strcpy(functionNames[1], "IFFT     "); printoutEnabled[1] = 1;
			  strcpy(functionNames[2], "applyTilt"); printoutEnabled[2] = 1;
			  strcpy(functionNames[3], "Basis    "); printoutEnabled[3] = 0;//Basis isn't multithreaded
			  strcpy(functionNames[4], "Overlap  "); printoutEnabled[4] = 1;
			  strcpy(functionNames[5], "Total    "); printoutEnabled[5] = 0;
			  strcpy(functionNames[6], "Total    "); printoutEnabled[6] = 1;

			  const int threadCount0 = digHoloThreadCount;

			  float perBenchmarkRuntime = goalRuntime / threadCount;
			  for (int threadIdx = 1; threadIdx < (threadCount); threadIdx++)
			  {
				  SetThreadCount(threadIdx);
				  benchmarkRoutine(timings[threadIdx], perBenchmarkRuntime, true);
			  }
			  if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
			  {
				  fprintf(consoleOut, "CPU count (logical) %i\n\r", CPU_COUNT);
				  fprintf(consoleOut, "\n\rRoutine	Speed(Hz)	threads (optimal)\n\r");
			  }
			  for (int timingIdx = 0; timingIdx < timingCount; timingIdx++)
			  {
				  float bestRate = 0;
				  bestThreadCount[timingIdx] = 0;
				  for (int threadIdx = 1; threadIdx < threadCount; threadIdx++)
				  {
					  if (timings[threadIdx][timingIdx] > bestRate)
					  {
						  bestRate = timings[threadIdx][timingIdx];
						  bestThreadCount[timingIdx] = threadIdx;
					  }
				  }
				  if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC && printoutEnabled[timingIdx])
				  {
					  fprintf(consoleOut, "%s	%3.3f	%i\n\r", functionNames[timingIdx], bestRate, bestThreadCount[timingIdx]);
				  }
			  }

			  if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
			  {
				  fprintf(consoleOut, "\n\rDefault : Threads = CPU\n\r");
				  for (int timingIdx = 0; timingIdx < timingCount; timingIdx++)
				  {
					  if (printoutEnabled[timingIdx])
					  {
						  fprintf(consoleOut, "%s	%3.3f	%i\n\r", functionNames[timingIdx], timings[CPU_COUNT][timingIdx], CPU_COUNT);
					  }
				  }
				  fprintf(consoleOut, "\n\rThreads	Speed(Hz)\n\r");
				  for (int threadIdx = 1; threadIdx < threadCount; threadIdx++)
				  {
					  fprintf(consoleOut, "%2.2i	%3.3f	\n\r", threadIdx, timings[threadIdx][totalRuntimeIdx]);
				  }
			  }
			  int bestOverallThreadCount = bestThreadCount[totalRuntimeIdx];
			  SetThreadCount(threadCount0);
			  digHoloThreadCount = threadCount0;

			  if (config.verbosity >= DIGHOLO_VERBOSITY_BASIC)
			  {
				  fprintf(consoleOut, "\nOptimal thread count %i	(%3.3f percent speedup over current thread count [%i])\n\r", bestOverallThreadCount, 100.0 * (timings[bestOverallThreadCount][totalRuntimeIdx] / timings[digHoloThreadCount][totalRuntimeIdx] - 1.0), digHoloThreadCount);
			  }
			  free1D(bestThreadCount);
			  free2D(timings);
			  return bestOverallThreadCount;
		  }
		  else
		  {
			  return 0;
		  }
	  }



	  float benchmarkRoutine(float* info, float goalDuration, int printOut)
	  {
		  if (printOut)
		  {
			  fprintf(consoleOut, "<BENCHMARK>\n\r");
			  fflush(consoleOut);
		  }
		  if (digHoloIsValidPixelBuffer() && goalDuration > 0)
		  {

			  //Warm-up pass (e.g. so that planning isn't included)
			  std::chrono::steady_clock::time_point nowTime = std::chrono::steady_clock::now();
			  digHoloFFT();
			  digHoloIFFT();
			  digHoloApplyTilt();
			  digHoloOverlapModes();
			  std::chrono::steady_clock::time_point nowTime2 = std::chrono::steady_clock::now();
			  //Take a guess at how long 1 iteration takes.
			  std::chrono::duration<double> loopEstimate = std::chrono::duration<double>(nowTime2 - nowTime);
			  double dtEstimate = (std::chrono::duration_cast<std::chrono::duration<double>>(loopEstimate)).count();
			  //How long do we want the benchmark to run approximately, in seconds.
			  //unsigned int trialCount = (unsigned int)ceil(100000.0 / (digHoloBatchCOUNT * digHoloModeCount));
			  unsigned int trialCount = (unsigned int)ceil(goalDuration / dtEstimate);
			  if (!(trialCount > 1))
			  {
				  trialCount = 1;
			  }
			  nowTime = std::chrono::steady_clock::now();
			  std::chrono::duration<double> zeroTime = std::chrono::duration<double>(nowTime - nowTime);
			  std::chrono::duration<double> benchTimer = zeroTime;
			  //float* currentFrame = 0;
			  int64_t trialIdx = 0;

			  benchmarkFFTTime = zeroTime;
			  benchmarkIFFTTime = zeroTime;
			  benchmarkApplyTiltTime = zeroTime;
			  benchmarkOverlapModesTime = zeroTime;

			  benchmarkFFTCounter = 0;
			  benchmarkIFFTCounter = 0;
			  benchmarkApplyTiltCounter = 0;
			  benchmarkOverlapModesCounter = 0;

			  //We'll create a new input buffer that's at least 1GB in size. To make sure that the frames aren't being stored in CPU cache, which could make the benchmark unrealistically fast.
			  const double GBsize = 1024 * 1024 * 1024;
			  const size_t bufferSize = (size_t)digHoloBatchCOUNT * (size_t)digHoloFrameHeight * (size_t)digHoloFrameWidth * sizeof(float);

			  int bufferCount = (int)ceil((1.0 * GBsize) / bufferSize);
			  float* tempBufferPtr = digHoloPixelsCameraPlane;

			  float** buffer = 0;

			  if (bufferCount > 1)
			  {
				  allocate2D(bufferCount, bufferSize, buffer);

				  for (int bufferIdx = 0; bufferIdx < bufferCount; bufferIdx++)
				  {
					  memcpy(&buffer[bufferIdx][0], digHoloPixelsCameraPlane, bufferSize);
				  }
			  }
			  
			  while (trialIdx < trialCount)
			  {
				  if (bufferCount > 1)
				  {
					  digHoloPixelsCameraPlane = buffer[trialIdx % bufferCount]; //digHoloPixelsCameraPlane;
				  }
				  std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
				  digHoloFFT();
				  digHoloIFFT();
				  digHoloApplyTilt();
				  digHoloOverlapModes();
				  std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();
				  benchTimer += (stopTime - startTime);
				  trialIdx++;

			  }
			  digHoloPixelsCameraPlane = tempBufferPtr;
			  float tempWaist = config.waist[0];
			  std::chrono::steady_clock::time_point startTime2 = std::chrono::steady_clock::now();
			  int trialCount2 = trialCount * 100;
			  float waistStep = config.waist[0] / 100000;
			  for (int i = 0; i < trialCount2; i++)
			  {
				  config.waist[0] += waistStep;
				  int x, y;
				  digHoloUpdateBasis(digHoloResolutionIdx, x, y);
			  }

			  std::chrono::steady_clock::time_point stopTime2 = std::chrono::steady_clock::now();
			  config.waist[0] = tempWaist;
			  int x, y;
			  digHoloUpdateBasis(digHoloResolutionIdx, x, y);

			  double dt = (std::chrono::duration_cast<std::chrono::duration<double>>(benchmarkFFTTime)).count();
			  double hzFFT = benchmarkFFTCounter / dt;
			  dt = (std::chrono::duration_cast<std::chrono::duration<double>>(benchmarkIFFTTime)).count();
			  double hzIFFT = benchmarkIFFTCounter / dt;
			  dt = (std::chrono::duration_cast<std::chrono::duration<double>>(benchmarkApplyTiltTime)).count();
			  double hzApplyTilt = benchmarkApplyTiltCounter / dt;
			  dt = (std::chrono::duration_cast<std::chrono::duration<double>>(benchmarkOverlapModesTime)).count();
			  double hzOverlap = benchmarkOverlapModesCounter / dt;
			  dt = (std::chrono::duration_cast<std::chrono::duration<double>>(benchTimer)).count();
			  double hzTotalMeas = trialCount / dt;
			  dt = (std::chrono::duration_cast<std::chrono::duration<double>>(stopTime2 - startTime2)).count();
			  double hzBasis = trialCount2 / dt;
			  double hzTotalMeas2 = (1.0 / (1.0 / hzFFT + 1.0 / hzIFFT + 1.0 / hzApplyTilt + 1.0 / hzOverlap));

			  dt = (std::chrono::duration_cast<std::chrono::duration<double>>(benchmarkFFTAnalysisTime)).count();
			  double hzFFTAnalysis = benchmarkFFTAnalysisCounter / dt;

			  dt = (std::chrono::duration_cast<std::chrono::duration<double>>(benchmarkIFFTAnalysisTime)).count();
			  double hzIFFTAnalysis = benchmarkIFFTAnalysisCounter / dt;

			  int batchCount = digHoloBatchCOUNT;
			  if (printOut)
			  {
				  fprintf(consoleOut, "Thread count : %i\n\r", digHoloThreadCount);
				  fprintf(consoleOut, "		Batches per second (frames per second)\n\r");
				  fprintf(consoleOut, "	FFT       (Hz)	%3.3f	%3.3f\n\r", hzFFT, hzFFT * batchCount);
				  fprintf(consoleOut, "	-Analysis (Hz)	%3.3f	%3.3f\n\r", hzFFTAnalysis, hzFFTAnalysis * batchCount);
				  fprintf(consoleOut, "	IFFT      (Hz)	%3.3f	%3.3f\n\r", hzIFFT, hzIFFT * batchCount);
				  fprintf(consoleOut, "	ApplyTilt (Hz)	%3.3f	%3.3f\n\r", hzApplyTilt, hzApplyTilt * batchCount);
				  fprintf(consoleOut, "	-Analysis (Hz)	%3.3f	%3.3f\n\r", hzIFFTAnalysis, hzIFFTAnalysis * batchCount);
				  fprintf(consoleOut, "	Basis     (Hz)	%3.3f\n\r", hzBasis);
				  fprintf(consoleOut, "	Overlap   (Hz)	%3.3f	%3.3f\n\r", hzOverlap, hzOverlap * batchCount);

				 // fprintf(consoleOut, "	Total     (Hz)	%3.3f	%3.3f\n", hzTotalMeas2, hzTotalMeas2 * batchCount);
				  fprintf(consoleOut, "	Total     (Hz)	%3.3f	%3.3f\n\r", hzTotalMeas, hzTotalMeas * batchCount);
				  fflush(consoleOut);
			  }
			  if (info)
			  {
				  info[0] = (float)hzFFT;
				  info[1] = (float)hzIFFT;
				  info[2] = (float)hzApplyTilt;
				  info[3] = (float)hzBasis;
				  info[4] = (float)hzOverlap;
				  info[5] = (float)hzTotalMeas2;
				  info[6] = (float)hzTotalMeas;
			  }
			  //One final pass to make sure that valid coefficients etc. are loaded after the benchmark runs.
			  digHoloFFT();
			  digHoloIFFT();
			  digHoloApplyTilt();
			  digHoloOverlapModes();
			  if (bufferCount > 1)
			  {
				  free2D(buffer);
			  }
			  if (printOut)
			  {
				  fprintf(consoleOut, "</BENCHMARK>\n\r\n\r");
				  fflush(consoleOut);
			  }
			  return (float)hzTotalMeas;
		  }
		  else
		  {
		  if (printOut)
		  {
			  fprintf(consoleOut, "</BENCHMARK>\n\r\n\r");
			  fflush(consoleOut);
		  }
			  return 0;
		  }
	  }
private:
};

//Returns 0 : Valid 1 : Never existed 2 : Destroyed
int digHoloHandleCheck(int handleIdx)
{
	if (handleIdx < 0 || handleIdx >= digHoloObjectsIdx.size())
	{
		return 1;
	}
	else
	{
		const int idx = digHoloObjectsIdx[handleIdx];
		if (idx)
		{
			return 0;
		}
		else
		{
			return 2;
		}
	}
}

//Creates a default digHoloObject and adds it to the array
EXT_C int digHoloCreate()
{
	digHoloObject* newObj = new digHoloObject();
	digHoloObjects.push_back(newObj);

	const size_t size = digHoloObjects.size();
	const int idx = (int)(size - 1);
	digHoloObjectsCount++;

	newObj->digHoloInit();

	digHoloObjectsIdx.push_back(1);
	return idx;
}

//This would have some memory leaks in it. The large arrays within the object are freed, but not the objects themselves.
int digHoloDestroy(int i)
{
	const int size = (int)digHoloObjects.size();
	if (i < size)
	{
		digHoloObject* oldObj = digHoloObjects.at(i);
		//This is not really the proper way to destroy this.
		oldObj->Destroy();
		digHoloObjectsIdx[i] = 0;
		digHoloObjectsCount--;
		return false;
	}
	else
	{
		return true;
	}
}

EXT_C int digHoloFFTWWisdomForget()
{
	fftwf_forget_wisdom();
	return DIGHOLO_ERROR_SUCCESS;
}

EXT_C int digHoloFFTWWisdomFilename(const char* filename)
{
	memset(FFTW_WISDOM_FILENAME, 0, sizeof(char) * FFTW_WISDOM_FILENAME_MAXLENGTH);
	strcpy(&FFTW_WISDOM_FILENAME[0], filename);
	fftwWisdomCustomFilename = 1;
	return DIGHOLO_ERROR_SUCCESS;
}

//Get-Set the number of threads used for digHolo processing
EXT_C int digHoloConfigGetThreadCount(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetThreadCount();
	}
}
EXT_C int digHoloConfigSetThreadCount(int handleIdx, int threadCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->SetThreadCount(threadCount);
	}
}

//Get-Set aperture size
EXT_C float digHoloConfigGetFourierWindowRadius(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].apertureSize;
	}
}
EXT_C int digHoloConfigSetFourierWindowRadius(int handleIdx, float windowRadius)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		if (windowRadius > 0)
		{
			config[0].apertureSize = windowRadius;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDDIMENSION;
		}
	}
}

EXT_C int digHoloConfigSetAutoAlignBeamCentre(int handleIdx, int AutoAlignCentre)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (AutoAlignCentre) { AutoAlignCentre = 1; }
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].AutoAlignCentre = AutoAlignCentre;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

//Get-Set AutoAlignCentre
EXT_C int digHoloConfigGetAutoAlignBeamCentre(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].AutoAlignCentre;
	}
}


EXT_C int digHoloConfigSetAutoAlignDefocus(int handleIdx, int AutoAlignDefocus)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (AutoAlignDefocus) { AutoAlignDefocus = 1; }

		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].AutoAlignDefocus = AutoAlignDefocus;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

//Get-Set AutoAlignDefocus
EXT_C int digHoloConfigGetAutoAlignDefocus(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].AutoAlignDefocus;
	}
}

//Get-Set AutoAlignTilt
EXT_C int digHoloConfigGetAutoAlignTilt(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].AutoAlignTilt;
	}
}

EXT_C int digHoloConfigSetAutoAlignTilt(int handleIdx, int AutoAlignTilt)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (AutoAlignTilt) AutoAlignTilt = 1;

		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].AutoAlignTilt = AutoAlignTilt;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

//Get-Set AutoAlignTilt
EXT_C int digHoloConfigGetAutoAlignMode(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].AutoAlignMode;
	}
}
EXT_C int digHoloConfigSetAutoAlignMode(int handleIdx, int AutoAlignMode)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (AutoAlignMode >= 0 && AutoAlignMode < DIGHOLO_AUTOALIGNMODE_COUNT)
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			config[0].AutoAlignMode = AutoAlignMode;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}
//Get-Set AutoAlignWaist
EXT_C int digHoloConfigGetAutoAlignBasisWaist(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].AutoAlignBasisWaist;
	}
}
EXT_C int digHoloConfigSetAutoAlignBasisWaist(int handleIdx, int AutoAlignWaist)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (AutoAlignWaist) AutoAlignWaist = 1;

		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].AutoAlignBasisWaist = AutoAlignWaist;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

//Get-Set AutoAlignFourierWindowRadius
EXT_C int digHoloConfigGetAutoAlignFourierWindowRadius(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].AutoAlignFourierWindowRadius;
	}
}

EXT_C int digHoloConfigSetAutoAlignFourierWindowRadius(int handleIdx, int AutoAlignFourierWindowRadius)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (AutoAlignFourierWindowRadius) AutoAlignFourierWindowRadius = 1;

		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].AutoAlignFourierWindowRadius = AutoAlignFourierWindowRadius;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

EXT_C int digHoloConfigSetBeamCentre(int handleIdx, int axisIdx, int polIdx, float value)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (isfinite(value))
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;

			const int polCount = config[0].polCount;
			if (polIdx < polCount && polIdx >= 0)
			{
				if (axisIdx == 0)
				{
					config[0].BeamCentreX[polIdx] = value;
				}
				else
				{
					if (axisIdx == 1)
					{
						config[0].BeamCentreY[polIdx] = value;
					}
					else
					{
						return DIGHOLO_ERROR_INVALIDAXIS;
					}
				}
			}
			else
			{
				return DIGHOLO_ERROR_INVALIDPOLARISATION;
			}
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}

EXT_C float digHoloConfigGetBeamCentre(int handleIdx, int axisIdx, int polIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0.0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		float value = 0.0;

		const int polCount = config[0].polCount;
		if (polIdx < polCount && polIdx >= 0)
		{
			if (axisIdx == 0)
			{
				value = config[0].BeamCentreX[polIdx];
			}
			else
			{
				if (axisIdx == 1)
				{
					value = config[0].BeamCentreY[polIdx];
				}
			}
		}
		return value;
	}
}

//Get BeamCentreX
EXT_C float* digHoloConfigGetBeamCentreX(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].BeamCentreX;
	}
}

//Get BeamCentreY
EXT_C float* digHoloConfigGetBeamCentreY(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].BeamCentreY;
	}
}

//Get-Set pixelSize
EXT_C float digHoloConfigGetFramePixelSize(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].pixelSize;
	}
}
EXT_C int digHoloConfigSetFramePixelSize(int handleIdx, float pixelSize)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (pixelSize > 0 && isfinite(pixelSize))
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			config[0].pixelSize = pixelSize;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDDIMENSION;
		}
	}
}

EXT_C int digHoloConfigSetFillFactorCorrectionEnabled(int handleIdx, int enabled)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			config[0].fillFactorCorrection = !(enabled==0);
			return DIGHOLO_ERROR_SUCCESS;
	}
}

EXT_C int digHoloConfigGetFillFactorCorrectionEnabled(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].fillFactorCorrection;
	}
}

//Get-Set polCount
EXT_C int digHoloConfigGetPolCount(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].polCount;
	}
}

EXT_C int digHoloConfigSetPolCount(int handleIdx, int polCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (polCount > 0 && polCount <= DIGHOLO_POLCOUNTMAX)
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			config[0].polCount = polCount;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDPOLARISATION;
		}
	}
}

//Get-Set PolLockTilt
EXT_C int digHoloConfigGetPolLockTilt(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].PolLockTilt;
	}
}
EXT_C int digHoloConfigSetPolLockTilt(int handleIdx, int polLockTilt)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (polLockTilt) { polLockTilt = 1; }

		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].PolLockTilt = polLockTilt;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

//Get-Set PolLockDefocus
EXT_C int digHoloConfigGetPolLockDefocus(int handleIdx)
{
	digHoloObject* obj = digHoloObjects[handleIdx];
	digHoloConfig* config = &obj[0].config;
	return config[0].PolLockDefocus;
}
EXT_C int digHoloConfigSetPolLockDefocus(int handleIdx, int polLockDefocus)
{
	digHoloObject* obj = digHoloObjects[handleIdx];
	digHoloConfig* config = &obj[0].config;
	config[0].PolLockDefocus = polLockDefocus;
	return config[0].PolLockDefocus;
}

//Get-Set PolLockBasisWaist
EXT_C int digHoloConfigGetPolLockBasisWaist(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].PolLockBasisWaist;
	}
}

EXT_C int digHoloConfigSetPolLockBasisWaist(int handleIdx, int polLockWaist)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (polLockWaist) { polLockWaist = 1; }

		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].PolLockBasisWaist = polLockWaist;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

//Get waist
EXT_C float digHoloConfigGetBasisWaist(int handleIdx, int polIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;

		if (polIdx < DIGHOLO_POLCOUNTMAX && polIdx >= 0)
		{
			return config[0].waist[polIdx];
		}
		else
		{
			return 0;
		}
	}
}
EXT_C int digHoloConfigSetBasisWaist(int handleIdx, int polIdx, float value)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (value > 0 && isfinite(value))
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			int polCount = config->polCount;

			if (polIdx < DIGHOLO_POLCOUNTMAX && polIdx >= 0)
			{
				for (int polIdx = 0; polIdx < polCount; polIdx++)
				{
					config[0].waist[polIdx] = value;
				}
				return DIGHOLO_ERROR_SUCCESS;
			}
			else
			{
				return DIGHOLO_ERROR_INVALIDPOLARISATION;
			}
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDDIMENSION;
		}
	}
}
//Get zernCoefs
EXT_C float digHoloConfigGetTilt(int handleIdx, int axisIdx, int polIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;

		if (polIdx < DIGHOLO_POLCOUNTMAX && polIdx >= 0)
		{
			if (axisIdx == 0)
			{
				return config[0].zernCoefs[polIdx][TILTX];
			}
			else
			{
				if (axisIdx == 1)
				{
					return config[0].zernCoefs[polIdx][TILTY];
				}
				else
				{
					return 0;
				}
			}
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDPOLARISATION;
		}
	}
}

EXT_C int digHoloConfigSetTilt(int handleIdx, int axisIdx, int polIdx, float value)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		//Could check for non-physical values as well, but could be annoying.
		if (isfinite(value))
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;

			if (polIdx < DIGHOLO_POLCOUNTMAX && polIdx >= 0)
			{
				if (axisIdx == 0)
				{
					config[0].zernCoefs[polIdx][TILTX] = value;
				}
				else
				{
					if (axisIdx == 1)
					{
						config[0].zernCoefs[polIdx][TILTY] = value;
					}
					else
					{
						return DIGHOLO_ERROR_INVALIDAXIS;
					}
				}
				return DIGHOLO_ERROR_SUCCESS;
			}
			else
			{
				return DIGHOLO_ERROR_INVALIDPOLARISATION;
			}
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}

EXT_C float digHoloConfigGetDefocus(int handleIdx, int polIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0.0f;
	}
	else
	{
		if (polIdx < DIGHOLO_POLCOUNTMAX && polIdx >= 0)
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			return config[0].zernCoefs[polIdx][DEFOCUS];
		}
		else
		{
			return 0.0f;
		}
	}
}

EXT_C int digHoloConfigSetDefocus(int handleIdx, int polIdx, float value)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (polIdx < DIGHOLO_POLCOUNTMAX && polIdx >= 0)
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			config[0].zernCoefs[polIdx][DEFOCUS] = value;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDPOLARISATION;
		}
	}
}

//Get zernCoefs
EXT_C float** digHoloConfigGetZernCoefs(int handleIdx)
{
	digHoloObject* obj = digHoloObjects[handleIdx];
	digHoloConfig* config = &obj[0].config;
	return config[0].zernCoefs;
}

//Get zernCount
EXT_C int digHoloConfigGetZernCount(int handleIdx)
{
	digHoloObject* obj = digHoloObjects[handleIdx];
	digHoloConfig* config = &obj[0].config;
	return config[0].zernCount;
}

//Get-set fftWindowSizeX
EXT_C int digHoloConfigGetfftWindowSizeX(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].fftWindowSizeX;
	}
}

//Values will be set to multiple of 16
EXT_C int digHoloConfigSetfftWindowSizeX(int handleIdx, int fftWindowSizeX)
{
	if (digHoloHandleCheck(handleIdx) || fftWindowSizeX < 0)
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		fftWindowSizeX = DIGHOLO_PIXEL_QUANTA * ((fftWindowSizeX) / DIGHOLO_PIXEL_QUANTA);
		config[0].fftWindowSizeX = fftWindowSizeX;
		return config[0].fftWindowSizeX;
	}
}

EXT_C int digHoloConfigGetfftWindowSizeY(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].fftWindowSizeY;
	}
}
//Values will be set to multiple of 16 (DIGHOLO_PIXEL_QUANTA)
EXT_C int digHoloConfigSetfftWindowSizeY(int handleIdx, int fftWindowSizeY)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		fftWindowSizeY = DIGHOLO_PIXEL_QUANTA * ((fftWindowSizeY) / DIGHOLO_PIXEL_QUANTA);
		config[0].fftWindowSizeY = fftWindowSizeY;
		return config[0].fftWindowSizeY;
	}
}

EXT_C int digHoloConfigSetfftWindowSize(int handleIdx, int fftWindowSizeX, int fftWindowSizeY)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (fftWindowSizeX < 0 || fftWindowSizeY < 0)
		{
			return DIGHOLO_ERROR_INVALIDDIMENSION;
		}
		else
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			fftWindowSizeX = DIGHOLO_PIXEL_QUANTA * ((fftWindowSizeX) / DIGHOLO_PIXEL_QUANTA);
			fftWindowSizeY = DIGHOLO_PIXEL_QUANTA * ((fftWindowSizeY) / DIGHOLO_PIXEL_QUANTA);
			config[0].fftWindowSizeX = fftWindowSizeX;
			config[0].fftWindowSizeY = fftWindowSizeY;
			return DIGHOLO_ERROR_SUCCESS;
		}
	}
}

EXT_C int digHoloConfigGetfftWindowSize(int handleIdx, int *fftWindowSizeX, int *fftWindowSizeY)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			fftWindowSizeX[0] = config[0].fftWindowSizeX;
			fftWindowSizeY[0] = config[0].fftWindowSizeY;
			return DIGHOLO_ERROR_SUCCESS;
	}
}

EXT_C int digHoloConfigGetFFTWPlanMode(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].FFTW_PLANMODE;
	}
}
EXT_C int digHoloConfigSetFFTWPlanMode(int handleIdx, int FFTW_PLANMODE)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (FFTW_PLANMODE >= 0 && FFTW_PLANMODE < FFTW_PLANMODECOUNT)
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			config[0].FFTW_PLANMODE = FFTW_PLANMODE;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}

//Get-set the dimensions of the camera frame
EXT_C int digHoloConfigSetFrameDimensions(int handleIdx, int width, int height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->SetFrameDimensions(width, height);
	}

}
EXT_C int digHoloConfigSetFrameWidth(int handleIdx, int width)
{
	return digHoloObjects[handleIdx]->SetFrameWidth(width);

}
EXT_C int digHoloConfigSetFrameHeight(int handleIdx, int height)
{
	return digHoloObjects[handleIdx]->SetFrameHeight(height);

}
EXT_C int digHoloConfigGetFrameDimensions(int handleIdx, int* width, int* height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		digHoloObjects[handleIdx]->GetFrameDimensions(width[0], height[0]);
		return DIGHOLO_ERROR_SUCCESS;
	}
}
EXT_C int digHoloConfigGetFrameWidth(int handleIdx)
{
	return digHoloObjects[handleIdx]->GetFrameWidth();
}
EXT_C int digHoloConfigGetFrameHeight(int handleIdx)
{
	return digHoloObjects[handleIdx]->GetFrameHeight();
}

//Get-Set PolLockBasisWaist
EXT_C int digHoloConfigGetIFFTResolutionMode(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].resolutionIdx;
	}
}
EXT_C int digHoloConfigSetIFFTResolutionMode(int handleIdx, int resolutionIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (resolutionIdx == FFT_IDX || resolutionIdx == IFFT_IDX)
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			config[0].resolutionIdx = resolutionIdx;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}

EXT_C int digHoloConfigSetAutoAlignGoalIdx(int handleIdx, int goalIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (goalIdx >= 0 && goalIdx < DIGHOLO_METRIC_COUNT)
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			config[0].AutoAlignGoalIdx = goalIdx;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}

EXT_C int digHoloConfigGetAutoAlignGoalIdx(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].AutoAlignGoalIdx;
	}

}

EXT_C int digHoloConfigSetAutoAlignTol(int handleIdx, float tol)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		//Ignore sign
		if (tol < 0)
		{
			tol = abs(tol);
		}
		else
		{
			//Don't feed the trolls
			if (!(tol >= 0) || tol == FLT_MAX)
			{
				tol = 0;
			}
		}
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].AutoAlignTol = tol;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

EXT_C float digHoloConfigGetAutoAlignTol(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].AutoAlignTol;
	}
}

EXT_C int digHoloConfigSetAutoAlignPolIndependence(int handleIdx, int polIndependence)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (polIndependence) { polIndependence = 1; }
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].AutoAlignPolIndependence = polIndependence;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

EXT_C int digHoloConfigGetAutoAlignPolIndependence(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].AutoAlignPolIndependence;
	}
}

EXT_C int digHoloConfigSetAutoAlignBasisMulConjTrans(int handleIdx, int mulConjTranspose)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (mulConjTranspose) { mulConjTranspose = 1; }
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].AutoAlignBasisMulConjTrans = mulConjTranspose;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

EXT_C int digHoloConfigGetAutoAlignBasisMulConjTrans(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].AutoAlignBasisMulConjTrans;
	}
}

EXT_C float digHoloConfigGetWavelengthCentre(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].wavelengthCentre;
	}
}

EXT_C int digHoloConfigSetWavelengthCentre(int handleIdx, float lambda0)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (lambda0 > 0 && isfinite(lambda0))
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			config[0].wavelengthCentre = lambda0;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}

EXT_C int digHoloConfigSetWavelengths(int handleIdx, float *lambdas, int lambdaCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (lambdaCount>0)
		{
			return digHoloObjects[handleIdx]->SetWavelengthArbitrary(lambdas, lambdaCount);
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}

EXT_C float* digHoloConfigGetWavelengths(int handleIdx, int *lambdaCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{

			return digHoloObjects[handleIdx]->GetWavelengths(&lambdaCount[0]);

	}
}

EXT_C int digHoloConfigSetWavelengthsLinearFrequency(int handleIdx, float lambdaStart, float lambdaStop, int lambdaCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if (lambdaCount)
		{
			return digHoloObjects[handleIdx]->SetWavelengthFrequencyLinear(lambdaStart, lambdaStop, lambdaCount);
		}
		else
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}
EXT_C int digHoloConfigSetWavelengthOrdering(int handleIdx, int inoutIdx, int ordering)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		if ((inoutIdx == DIGHOLO_WAVELENGTHORDER_INPUT || inoutIdx == DIGHOLO_WAVELENGTHORDER_OUTPUT) && (ordering==DIGHOLO_WAVELENGTHORDER_FAST || ordering==DIGHOLO_WAVELENGTHORDER_SLOW))
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			config[0].wavelengthOrdering[inoutIdx] = ordering;
			return DIGHOLO_ERROR_SUCCESS;
		}
		{
			return DIGHOLO_ERROR_INVALIDARGUMENT;
		}
	}
}

EXT_C int digHoloConfigGetWavelengthOrdering(int handleIdx, int inoutIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		if (inoutIdx == DIGHOLO_WAVELENGTHORDER_INPUT || inoutIdx == DIGHOLO_WAVELENGTHORDER_OUTPUT)
		{
			digHoloObject* obj = digHoloObjects[handleIdx];
			digHoloConfig* config = &obj[0].config;
			return config[0].wavelengthOrdering[inoutIdx];
		}
		{
			return 0;
		}
	}
}

EXT_C int digHoloConfigGetBasisGroupCount(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].maxMG;
	}
}

EXT_C int digHoloConfigSetBasisGroupCount(int handleIdx, int maxMG)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		if (maxMG < 0)
		{
			maxMG = abs(maxMG);
		}
		config[0].maxMG = maxMG;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

EXT_C int digHoloConfigSetBasisType(int handleIdx, int typeIdx)
{
	digHoloObject* obj = digHoloObjects[handleIdx];
	digHoloConfig* config = &obj[0].config;
	config[0].basisType = typeIdx;
	return config[0].basisType;
}

EXT_C int digHoloConfigSetBasisTypeHG(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].basisType = DIGHOLO_BASISTYPE_HG;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

EXT_C int digHoloConfigSetBasisTypeLG(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		config[0].basisType = DIGHOLO_BASISTYPE_LG;
		return DIGHOLO_ERROR_SUCCESS;
	}
}

EXT_C int digHoloConfigSetBasisTypeCustom(int handleIdx, int modeCountIn, int modeCountOut, complex64* transform)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		//If not a null pointer, and all specified dimensions are non-zero.
		if (transform)
		{
			if (modeCountIn > 0 && modeCountOut > 0)
			{
				digHoloObject* obj = digHoloObjects[handleIdx];
				digHoloConfig* config = &obj[0].config;
				config[0].basisType = DIGHOLO_BASISTYPE_CUSTOM;
				return digHoloObjects[handleIdx]->SetBasisTypeCustom(modeCountIn, modeCountOut, transform);
			}
			else
			{
				return DIGHOLO_ERROR_INVALIDDIMENSION;
			}
		}
		else
		{
			return DIGHOLO_ERROR_NULLPOINTER;
		}
	}

}

EXT_C int digHoloConfigGetBasisType(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].basisType;
	}
}

EXT_C int digHoloConfigGetBatchCount(int handleIdx)
{
	digHoloObject* obj = digHoloObjects[handleIdx];
	digHoloConfig* config = &obj[0].config;
	return config[0].batchCount;
}

EXT_C int digHoloConfigSetBatchCount(int handleIdx, int batchCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;

		if (batchCount < 1)
		{
			batchCount = 1;
		}

		config[0].batchCount = batchCount;

		return DIGHOLO_ERROR_SUCCESS;
	}
}
EXT_C int digHoloConfigSetBatchAvgMode(int handleIdx, int avgMode)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		//Out of range, default to zero
		if (avgMode < 0 || avgMode >= DIGHOLO_AVGMODE_COUNT)
		{
			avgMode = 0;
		}
		config[0].avgMode = avgMode;
		return DIGHOLO_ERROR_SUCCESS;
	}

}
EXT_C int digHoloConfigGetBatchAvgMode(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;

		return config[0].avgMode;
	}
}
EXT_C int digHoloConfigGetBatchAvgCount(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;
		return config[0].avgCount;
	}
}

EXT_C int digHoloConfigSetBatchAvgCount(int handleIdx, int avgCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		digHoloObject* obj = digHoloObjects[handleIdx];
		digHoloConfig* config = &obj[0].config;

		if (avgCount < 1)
		{
			avgCount = 1;
		}

		config[0].avgCount = avgCount;

		return DIGHOLO_ERROR_SUCCESS;
	}
}

EXT_C int digHoloConfigSetBatchCalibration(int handleIdx, complex64* cal, int polCount, int batchCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		//digHoloObject* obj = digHoloObjects[handleIdx];

		return digHoloObjects[handleIdx]->SetBatchCalibration(cal, polCount, batchCount);
	}
}

EXT_C complex64* digHoloConfigGetBatchCalibration(int handleIdx, int *polCount, int *batchCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		//digHoloObject* obj = digHoloObjects[handleIdx];

		return digHoloObjects[handleIdx]->GetBatchCalibration(&polCount[0], &batchCount[0]);
	}
}



EXT_C int digHoloConfigSetBatchCalibrationFromFile(int handleIdx, const char* fname, int polCount, int batchCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->SetBatchCalibrationLoadFromFile(fname, polCount, batchCount);
	}
}



EXT_C int digHoloConfigSetBatchCalibrationEnabled(int handleIdx, int enabled)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		//digHoloObject* obj = digHoloObjects[handleIdx];

		return digHoloObjects[handleIdx]->SetBatchCalibrationEnabled(enabled);
	}
}

EXT_C int digHoloConfigGetBatchCalibrationEnabled(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		//digHoloObject* obj = digHoloObjects[handleIdx];

		return digHoloObjects[handleIdx]->GetBatchCalibrationEnabled();
	}
}

EXT_C float digHoloAutoAlign(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0.0f;
	}
	else
	{
		return digHoloObjects[handleIdx]->AutoAlign();
	}
}

EXT_C float* digHoloAutoAlignGetMetrics(int handleIdx, int metricIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		int lambdaCount = 0;
		return digHoloObjects[handleIdx]->AutoAlignGetMetrics(metricIdx,lambdaCount);
	}
}

EXT_C float digHoloAutoAlignGetMetric(int handleIdx, int metricIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0.0f;
	}
	else
	{
		if (metricIdx < DIGHOLO_METRIC_COUNT && metricIdx >= 0)
		{
			int lambdaCount = 0;
			float* metrics = digHoloObjects[handleIdx]->AutoAlignGetMetrics(metricIdx,lambdaCount);
			//int lambdaCount = digHoloAutoAlignMetricsWavelengthCount;
			if (lambdaCount >= 0)
			{
				return metrics[lambdaCount];
			}
			else
			{
				return -FLT_MAX;
			}
		}
		else
		{
			return 0.0f;
		}
	}
}

EXT_C int digHoloAutoAlignCalcMetrics(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->AutoAlignCalcMetrics();
	}
}

/*
EXT_C void digHoloViewportResumeNewFrame(int handleIdx)
{
	digHoloObjects[handleIdx]->ViewportResumeNewFrame();
}
EXT_C void digHoloViewportResume(int handleIdx)
{
	digHoloObjects[handleIdx]->Resume();
}
EXT_C void digHoloViewportPause(int handleIdx)
{
	digHoloObjects[handleIdx]->ViewportPause();
}
EXT_C int digHoloInitialisingEventWait(int handleIdx)
{
	return !digHoloObjects[handleIdx]->InitialisingEvent.WaitOne();
}
*/
EXT_C int digHoloSetBatch(int handleIdx, int batchCount, float* buffer)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->SetBatch(batchCount, buffer, 1, 0);
	}
}

EXT_C int digHoloSetBatchUint16(int handleIdx, int batchCount, unsigned short* buffer, int transposed)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		const int avgCount = 1;
		const int avgMode = 0;
		return digHoloObjects[handleIdx]->SetBatchUint16(batchCount, buffer, avgCount, avgMode, transposed);
	}
}

EXT_C int digHoloSetBatchAvg(int handleIdx, int batchCount, float* buffer, int avgCount, int avgMode)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->SetBatch(batchCount, buffer, avgCount, avgMode);
	}
}

EXT_C int digHoloSetBatchAvgUint16(int handleIdx, int batchCount, unsigned short* buffer, int avgCount, int avgMode, int transposed)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->SetBatchUint16(batchCount, buffer, avgCount, avgMode, transposed);
	}
}

EXT_C complex64* digHoloProcessBatch(int handleIdx, int* batchCount, int* modeCount, int* polCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		batchCount[0] = 0;
		modeCount[0] = 0;
		polCount[0] = 0;
		return 0;
	}
	else
	{
		complex64** returnPtr = digHoloObjects[handleIdx]->ProcessBatch(batchCount[0], modeCount[0], polCount[0]);
		if (returnPtr)
		{
			return (complex64*)&returnPtr[0][0][0];
		}
		else
		{
			return 0;
		}
	}
}

EXT_C int digHoloProcessFFT(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->ProcessFFT();
	}
}

EXT_C int digHoloProcessIFFT(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->ProcessIFFT();
	}
}

EXT_C int digHoloProcessRemoveTilt(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->ProcessApplyTilt();
	}
}

EXT_C int digHoloProcessBasisExtractCoefs(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->ProcessOverlapModes();
	}
}

EXT_C complex64* digHoloProcessBatchFrequencySweepLinear(int handleIdx, int* batchCount, int* modeCount, int* polCount, float lambdaStart, float lambdaStop, int lambdaCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		batchCount[0] = 0;
		modeCount[0] = 0;
		polCount[0] = 0;
		return 0;
	}
	else
	{
		complex64** returnPtr = digHoloObjects[handleIdx]->ProcessBatchFrequencySweepLinear(batchCount[0], modeCount[0], polCount[0], lambdaStart, lambdaStop, lambdaCount);
		if (returnPtr)
		{
			return (complex64*)&returnPtr[0][0][0];
		}
		else
		{
			return 0;
		}
	}
}

EXT_C complex64* digHoloProcessBatchWavelengthSweepArbitrary(int handleIdx, int* batchCount, int* modeCount, int* polCount, float* wavelengths, int lambdaCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		batchCount[0] = 0;
		modeCount[0] = 0;
		polCount[0] = 0;
		return 0;
	}
	else
	{
		complex64** returnPtr = digHoloObjects[handleIdx]->ProcessBatchFrequencySweepArbitrary(batchCount[0], modeCount[0], polCount[0], wavelengths, lambdaCount);
		if (returnPtr)
		{
			return (complex64*)&returnPtr[0][0][0];
		}
		else
		{
			return 0;
		}
	}
}

EXT_C int digHoloBatchGetSummary(int handleIdx, int planeIdx, int* parameterCount, int* batchCount, int* polCount, float** parameters, int* pixelCountX, int* pixelCountY, float** xAxis, float** yAxis, float** intensity)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetBatchSummary(planeIdx, parameterCount[0], batchCount[0], polCount[0], pixelCountX[0], pixelCountY[0], parameters[0], intensity[0], xAxis[0], yAxis[0]);
	}
}

EXT_C complex64* digHoloBasisGetCoefs(int handleIdx, int* batchCount, int* modeCount, int* polCount)
{
	if (digHoloHandleCheck(handleIdx))
	{
		batchCount[0] = 0;
		modeCount[0] = 0;
		polCount[0] = 0;
		return 0;
	}
	else
	{
		complex64** returnPtr = digHoloObjects[handleIdx]->GetCoefs(batchCount[0], modeCount[0], polCount[0]);
		if (returnPtr)
		{
			return (complex64*)&returnPtr[0][0][0];
		}
		else
		{
			batchCount[0] = 0;
			modeCount[0] = 0;
			polCount[0] = 0;
			return 0;
		}
	}
}

EXT_C int digHoloGetEnabled(int handleIdx)
{
	return digHoloObjects[handleIdx]->Enabled;
}

EXT_C int digHoloSetEnabled(int handleIdx, int enabled)
{
	digHoloObjects[handleIdx]->Enabled = enabled;
	return digHoloObjects[handleIdx]->Enabled;
}
EXT_C int digHoloGetFrameThreadIsRunning(int handleIdx)
{
	return digHoloObjects[handleIdx]->FrameThreadIsRunning;
}

EXT_C int digHoloSetFrameThreadIsRunning(int handleIdx, int frameIsRunning)
{
	digHoloObjects[handleIdx]->FrameThreadIsRunning = frameIsRunning;
	return digHoloObjects[handleIdx]->FrameThreadIsRunning;
}

EXT_C void digHoloDebugRoutine(int handleIdx)
{
	digHoloObjects[handleIdx]->debugRoutine();
}

EXT_C float digHoloBenchmark(int handleIdx, float goalDuration, float* info)
{
	digHoloObject* obj = digHoloObjects[handleIdx];
	digHoloConfig* config = &obj[0].config;
	int printOut = config->verbosity;
	return digHoloObjects[handleIdx]->benchmarkRoutine(info, goalDuration, printOut);
}

EXT_C int digHoloBenchmarkEstimateThreadCountOptimal(int handleIdx, float goalDuration)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->ThreadDiagnostic(goalDuration);
	}
}

EXT_C int digHoloViewportSetDisplayMode(int handleIdx, int displayMode)
{
	digHoloObjects[handleIdx]->DisplayMode = displayMode;
	return digHoloObjects[handleIdx]->DisplayMode;
}
/*
EXT_C complex64* digHoloGetField(int handleIdx, float* pixelBuff, int* pixelCountX, int* pixelCountY)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetField(pixelBuff, pixelCountX[0], pixelCountY[0]);
	}
}
*/

EXT_C complex64* digHoloGetFields(int handleIdx, int* batchCount, int* polCount, float** x, float** y, int* width, int* height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		batchCount[0] = 0;
		polCount[0] = 0;
		x[0] = 0;
		y[0] = 0;
		width[0] = 0;
		height[0] = 0;
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetFields(batchCount[0], polCount[0], x[0], y[0], width[0], height[0]);
	}
}

EXT_C complex64* digHoloConfigGetRefCalibrationFields(int handleIdx, int* lambdaCount, int* polCount, float** x, float** y, int* width, int* height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		lambdaCount[0] = 0;
		polCount[0] = 0;
		x[0] = 0;
		y[0] = 0;
		width[0] = 0;
		height[0] = 0;
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetRefCalibration(lambdaCount[0], polCount[0], x[0], y[0], width[0], height[0]);
	}
}
EXT_C complex64* digHoloBasisGetFields(int handleIdx, int* modeCount, int* polCount, float** x, float** y, int* width, int* height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		modeCount[0] = 0;
		polCount[0] = 0;
		x[0] = 0;
		y[0] = 0;
		width[0] = 0;
		height[0] = 0;
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetFieldsBasis(modeCount[0], polCount[0], x[0], y[0], width[0], height[0]);
	}
}

EXT_C int digHoloGetFields16(int handleIdx, int* batchCount, int* polCount, short** fieldR, short** fieldI, float** fieldScale, float** x, float** y, int* width, int* height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		batchCount[0] = 0;
		polCount[0] = 0;
		fieldR[0] = 0;
		fieldI[0] = 0;
		fieldScale[0] = 0;
		x[0] = 0;
		y[0] = 0;
		width[0] = 0;
		height[0] = 0;
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetFields16(batchCount[0], polCount[0], fieldR[0], fieldI[0], fieldScale[0], x[0], y[0], width[0], height[0]);
	}
}

EXT_C int digHoloConfigBackupLoad(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->ConfigBackupLoad();
	}
}

EXT_C int digHoloConfigBackupSave(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->ConfigBackupSave();
	}
}

EXT_C int digHoloSetFrameBuffer(int handleIdx, float* buff)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->SetFrameBuffer(buff);
	}
}

EXT_C int digHoloSetFrameBufferUint16(int handleIdx, unsigned short* buff, int transpose)
{
	return digHoloObjects[handleIdx]->SetFrameBufferUint16(buff, transpose);
}

EXT_C int digHoloSetFrameBufferFromFile(int handleIdx, const char* fname)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->SetFrameBufferFromFile(fname);
	}
}

EXT_C float* digHoloGetFrameBuffer(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetFrameBuffer();
	}
}

EXT_C unsigned short* digHoloGetFrameBufferUint16(int handleIdx, int* transpose)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetFrameBufferUint16(transpose[0]);
	}
}

EXT_C int digHoloConfigSetRefCalibrationFromFile(int handleIdx, const char* fname, int lambdaCount, int width,int height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->RefCalibrationLoadFromFile(fname,width,height,lambdaCount);
	}
}

EXT_C int digHoloConfigSetRefCalibrationIntensity(int handleIdx, unsigned short* cal, int wavelengthCount, int width, int height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->RefCalibrationSet16(cal, width, height, wavelengthCount);
	}
}

EXT_C int digHoloConfigSetRefCalibrationField(int handleIdx, complex64* cal, int wavelengthCount, int width, int height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->RefCalibrationSet32(cal, width, height, wavelengthCount);
	}
}

EXT_C int digHoloConfigGetRefCalibrationEnabled(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->RefCalibrationGetEnabled();
	}
}

EXT_C int digHoloConfigSetRefCalibrationEnabled(int handleIdx, int enabled)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->RefCalibrationSetEnabled(enabled);
	}
}

EXT_C complex64* digHoloGetFourierPlaneFull(int handleIdx, int* batchCount, int* polCount, int* width, int* height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetFourierPlaneFull(batchCount[0], polCount[0], width[0], height[0]);
	}
}

EXT_C complex64* digHoloGetFourierPlaneWindow(int handleIdx, int* batchCount, int* polCount, int* width, int* height)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetFourierPlaneWindow(batchCount[0], polCount[0], width[0], height[0]);
	}
}

EXT_C unsigned char* digHoloGetViewport(int handleIdx, int displayMode, int waitForNewFrame, int* width, int* height, char** windowString)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetViewport(displayMode, waitForNewFrame, width[0], height[0], windowString[0]);
	}
}

EXT_C int digHoloGetViewportToFile(int handleIdx, int displayMode, int forceProcessing, int* width, int* height, char** windowString, const char* filename)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetViewportToFile(displayMode, forceProcessing, width[0],height[0], windowString[0], filename);
	}
}

EXT_C complex64* digHoloHGtoLG(int handleIdx, complex64** HGcoefs, complex64** LGcoefs, int batchCount, int modeCount, int polCount)
{
	return digHoloObjects[handleIdx]->HGtoLG(HGcoefs, LGcoefs, batchCount, modeCount, polCount, false);
}

EXT_C int digHoloConfigSetVerbosity(int handleIdx, int verboseMode)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return DIGHOLO_ERROR_INVALIDHANDLE;
	}
	else
	{
		return digHoloObjects[handleIdx]->SetVerbosity(verboseMode);
	}

}
EXT_C int digHoloConfigGetVerbosity(int handleIdx)
{
	if (digHoloHandleCheck(handleIdx))
	{
		return 0;
	}
	else
	{
		return digHoloObjects[handleIdx]->GetVerbosity();
	}
}

EXT_C int digHoloConsoleRestore()
{
	//If the console is current stdout. Leave it.
	if (consoleOut == stdout)
	{
		return DIGHOLO_ERROR_SUCCESS;
	}
	else
	{
		//If the console isn't stdout, but it is something else (presumably an open file)
		if (consoleOut)
		{
			//Create a temporary copy of the pointer
			FILE* fileTemp = consoleOut;
			//Redirect the console to stdout
			consoleOut = stdout;
			fprintf(fileTemp, "File closed. Redirected back to stdout.\n");
			fflush(fileTemp);
			//close the file
			fclose(fileTemp);
			return DIGHOLO_ERROR_SUCCESS;
		}
		else //not sure how you'd get here, but if the console is null, then set it to stdout
		{
			consoleOut = stdout;
			return DIGHOLO_ERROR_SUCCESS;
		}
	}
}

EXT_C int digHoloConsoleRedirectToFile(char* filename)
{
	if (filename)
	{
		//FILE* fileOut = fopen(filename, "w");

		FILE* fileOut = fopen(filename, "w");//FILE* fileOut = 0;;errno_t err = fopen(&fileOut, filename, "w");
		if (fileOut)
		{
			consoleOut = fileOut;
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			consoleOut = stdout;
			return DIGHOLO_ERROR_FILENOTCREATED;
		}
	}
	else
	{
		digHoloConsoleRestore();
		return DIGHOLO_ERROR_NULLPOINTER;
	}
}

EXT_C int digHoloFrameSimulatorDestroy(float* pixelBuffer)
{
	if (pixelBuffer)
	{
		free1D(pixelBuffer);
		if (!pixelBuffer)
		{
			return DIGHOLO_ERROR_SUCCESS;
		}
		else
		{
			return DIGHOLO_ERROR_MEMORYALLOCATION;
		}
	}
	else
	{
		return DIGHOLO_ERROR_NULLPOINTER;
	}
}

void digHoloFrameSimulatorRoutine(void *work)
{
	workPackage* e= (workPackage*)work;
	//workPackage&e = work0;
	const int threadIdx = e[0].threadIdx;

	const size_t frameStart = e[0].start;
	const size_t frameStop = e[0].stop;

	const size_t frameCount = e[0].flag1;
	const size_t polCount = e[0].flag2;

	const size_t frameWidth = e[0].flag3;
	const size_t frameHeight = e[0].flag4;

	const int modeCount = e[0].flag5;

	const int wavelengthOrdering = e[0].flag6;

	const int wavelengthCount = e[0].flag7;
	const int fillFactorCorrection = e[0].flag8;
	const int cameraPixelLevelCount = e[0].flag9;

	const size_t height = frameWidth / polCount;
	const size_t width = frameHeight;

	const size_t pixelCount = width * height;

	const int heightFFT = (int)frameHeight;
	const int widthFFT = (int)frameWidth;

	const int R2C = 1;
	const int widthOutFFT = R2C ? (widthFFT / 2 + 1) : widthFFT;

	complex64* beamCoefs = (complex64*)e[0].ptr1;
	complex64*** refX = (complex64***)e[0].ptr2;
	complex64*** refY = (complex64***)e[0].ptr3;
	complex64** tempMemory = (complex64**)e[0].ptr4;
	float* pixelBufferOut = (float*)e[0].ptr5;
	float*** HGscale = (float***)e[0].ptr6;
	float** HGscaleX = HGscale[0];
	float** HGscaleY = HGscale[1];
	short*** HGx = (short***)e[0].ptr7;
	short*** HGy = (short***)e[0].ptr8;
	int* M = (int*)e[0].ptr9;
	int* N = (int*)e[0].ptr10;

	fftwf_plan* fftPlans = (fftwf_plan*)e[0].ptr11;
	fftwf_plan forwardPlanR2C = fftPlans[0];
	fftwf_plan backwardPlanC2R = fftPlans[1];

	float* sincX = (float*)e[0].ptr12;
	float* sincY = (float*)e[0].ptr13;
	short* pixelBuffer16 = (short*)e[0].ptr14;
	
	float normFactorFFT = e[0].var1;

	float maxV = 0;
	complex64* fieldTemp = tempMemory[threadIdx];

	for (size_t frameIdx = frameStart; frameIdx < frameStop; frameIdx++)
	{
		complex64* modeCoefs = &beamCoefs[frameIdx * polCount * modeCount];

		const size_t frameOffset = frameIdx * pixelCount * polCount;

		int transpose = 1;

		int lambdaIdx = 0;
		int subBatchIdx = 0;
		int wavelengthOrderingOut = wavelengthOrdering;

		WavelengthOrderingCalc((int)frameIdx, wavelengthCount, (int)frameCount, wavelengthOrdering, wavelengthOrderingOut, lambdaIdx, subBatchIdx);
		complex64** rfX = refX[lambdaIdx];
		complex64** rfY = refY[lambdaIdx];

		
		
		float* intensityOut = &pixelBufferOut[frameOffset];
		float maxV0 = generateModeSeparable(HGscaleX, HGscaleY, fieldTemp, HGx, HGy, (int)width, (int)height, modeCoefs, M, N, modeCount, (int)polCount, transpose, rfX, rfY, intensityOut);

		if (maxV0 > maxV)
		{
			maxV = maxV0;
		}

		if (fillFactorCorrection)
		{
			const size_t blockSize = 8;
			const size_t widthOutBlock = blockSize * (widthOutFFT / blockSize);

			float* in = &pixelBufferOut[frameIdx * widthFFT * heightFFT];

			fftwf_execute_dft_r2c(forwardPlanR2C, in, fieldTemp);

			for (int pixelIdy = 0; pixelIdy < heightFFT; pixelIdy++)
			{
				float sincVY = sincY[pixelIdy] * normFactorFFT;
				const __m256 sincVY256 = _mm256_set1_ps(sincVY);

				for (int pixelIdx = 0; pixelIdx < widthOutBlock; pixelIdx += blockSize)
				{
					__m256 sincVX256 = _mm256_loadu_ps(&sincX[pixelIdx]);

					__m256 env = _mm256_mul_ps(sincVY256, sincVX256);

					__m256 envA = _mm256_permutevar8x32_ps(env, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					__m256 envB = _mm256_permutevar8x32_ps(env, _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4));

					const size_t idx = pixelIdy * widthOutFFT + pixelIdx;

					__m256 outA = _mm256_loadu_ps(&fieldTemp[idx][0]);
					__m256 outB = _mm256_loadu_ps(&fieldTemp[idx + 4][0]);

					_mm256_storeu_ps(&fieldTemp[idx][0], _mm256_mul_ps(outA, envA));
					_mm256_storeu_ps(&fieldTemp[idx + 4][0], _mm256_mul_ps(outB, envB));
				}

				//For the remaining pixels that aren't a multiple of blockSize
				for (size_t pixelIdx = widthOutBlock; pixelIdx < widthOutFFT; pixelIdx++)
				{
					float sincVX = sincX[pixelIdx];
					float env = (sincVX * sincVY);

					const size_t idx = pixelIdy * widthOutFFT + pixelIdx;
					fieldTemp[idx][0] *= env;
					fieldTemp[idx][1] *= env;
				}
			}

			fftwf_execute_dft_c2r(backwardPlanC2R, fieldTemp, in);
		}
	}

	fieldTemp[0][0] = maxV;
	e[0].workCompleteEvent.Set();

	///WAIT
	e[0].workNewEvent.WaitOne();
	maxV = fieldTemp[0][0];
	


	const float maxVinv = (cameraPixelLevelCount - 1) / maxV;
	const __m256 maxVinv256 = _mm256_set1_ps(maxVinv);

	const size_t totalPixelCount = (size_t)width * (size_t)height * (size_t)polCount;

	const size_t blockSize = 16;
	for (size_t frameIdx = frameStart; frameIdx < frameStop; frameIdx++)
	{
		size_t frameOffset = frameIdx * totalPixelCount;
		for (size_t pixelIdx = 0; pixelIdx < totalPixelCount; pixelIdx += blockSize)
		{
			const size_t idx = frameOffset + pixelIdx;
			__m256 pxOutA = _mm256_loadu_ps(&pixelBufferOut[idx]);
			__m256 pxOutB = _mm256_loadu_ps(&pixelBufferOut[idx + 8]);

			pxOutA = _mm256_mul_ps(pxOutA, maxVinv256);
			pxOutA = _mm256_round_ps(pxOutA, _MM_ROUND_NEAREST);

			pxOutB = _mm256_mul_ps(pxOutB, maxVinv256);
			pxOutB = _mm256_round_ps(pxOutB, _MM_ROUND_NEAREST);

			__m256i pxOuti16 = cvtps_epu16(pxOutA, pxOutB);
			_mm256_storeu_ps(&pixelBufferOut[idx], pxOutA);
			_mm256_storeu_ps(&pixelBufferOut[idx + 8], pxOutB);
			_mm256_storeu_si256((__m256i*) & pixelBuffer16[idx], pxOuti16);
		}
	}
	e[0].workCompleteEvent.Set();
}

EXT_C float* digHoloFrameSimulatorCreateSimple(int frameCount, int frameWidth, int frameHeight, float pixelSize, int polCount, float wavelength, int printToConsole)
{

	int frameWidth0 = DIGHOLO_PIXEL_QUANTA  * ((frameWidth) / DIGHOLO_PIXEL_QUANTA);
	int frameHeight0 = DIGHOLO_PIXEL_QUANTA * ((frameHeight) / DIGHOLO_PIXEL_QUANTA);

	if (frameWidth != frameWidth0 || frameHeight != frameHeight0)
	{
		if (printToConsole)
		{
			fprintf(consoleOut, "Frame dimensions must be a multiple of %i. Aborting.\n\r", DIGHOLO_PIXEL_QUANTA);
			fflush(consoleOut);
		}
		return 0;
	}

	if (frameCount < 1 || frameWidth < 1 || frameHeight < 1 || pixelSize <= 0 || polCount < 1 || wavelength <= 0)
	{
		if (printToConsole)
		{
			fprintf(consoleOut, "Invalid parameter. Aborting.\n\r");
			fflush(consoleOut);
		}
		return 0;
	}

	float* refTiltX = 0;
	float* refTiltY = 0;
	float* refDefocus = 0;
	float* refWaist = 0;
	float* refBeamCentreX = 0;
	float* refBeamCentreY = 0;
	complex64* refAmplitude = 0;
	int beamGroupCount = 0;
	float* beamWaist = 0;
	complex64* beamCoefs = 0;
	float* beamCentreX = 0;
	float* beamCentreY = 0;
	int cameraPixelLevelCount = 16384;
	int fillFactorCorrection = 1;
	float* wavelengthPtr = &wavelength;
	int wavelengthCount = 1;
	int wavelengthOrdering = 0;
	unsigned short* pixelBuffer16 = 0;
	const char* fname = 0;

	float* pixelBuffer32 = digHoloFrameSimulatorCreate(frameCount, &frameWidth, &frameHeight, &pixelSize, &polCount,
		&refTiltX, &refTiltY, &refDefocus, &refWaist, &refBeamCentreX, &refBeamCentreY, &refAmplitude,
		&beamGroupCount, &beamWaist, &beamCoefs, &beamCentreX, &beamCentreY,
		&cameraPixelLevelCount, fillFactorCorrection, &wavelengthPtr, &wavelengthCount, wavelengthOrdering,
		printToConsole, &pixelBuffer16, fname);

	return pixelBuffer32;
}

float* frameSimulatorCreate(int frameCount, int& frameWidth, int& frameHeight, float& pixelSize, int& polCount,
	float*& refTiltX, float*& refTiltY, float*& refDefocus, float*& refWaist, float*& refBeamCentreX, float*& refBeamCentreY, complex64*& refAmplitude,
	int& beamGroupCount, float*& beamWaist, complex64*& beamCoefs, float*& beamCentreX, float*& beamCentreY,
	int& cameraPixelLevelCount, int fillFactorCorrection, float*& wavelengths, int& wavelengthCount, int wavelengthOrdering, int printOut, unsigned short*& pixelBuffer16, const char* fname)
{
	std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

	if (frameCount > 0 && frameWidth > 0 && frameHeight > 0 && polCount > 0)
	{
		if (printOut)
		{
			fprintf(consoleOut, "<FRAME SIMULATOR>\n\r");
			fflush(consoleOut);
		}
		//If there's not at least 2 levels (1 bit) then it's not really valid data.
		//Assume the user doesn't know what's going on and just set it to 16-bit.
		if (cameraPixelLevelCount < 2)
		{
			cameraPixelLevelCount = 65536;

			if (printOut)
			{
				fprintf(consoleOut, "Invalid cameraPixelLevelCount. Default: %i\n\r", cameraPixelLevelCount);
				fflush(consoleOut);
			}
		}
		if (cameraPixelLevelCount > 65536)
		{
			cameraPixelLevelCount = 65536;
			if (printOut)
			{
				fprintf(consoleOut, "Invalid cameraPixelLevelCount. Default: %i\n\r", cameraPixelLevelCount);
				fflush(consoleOut);
			}
		}

		frameWidth = DIGHOLO_PIXEL_QUANTA * ((frameWidth) / DIGHOLO_PIXEL_QUANTA);
		frameHeight = DIGHOLO_PIXEL_QUANTA * ((frameHeight) / DIGHOLO_PIXEL_QUANTA);

		if (pixelSize <= 0)
		{
			pixelSize = 20e-6f;
			if (printOut)
			{
				fprintf(consoleOut, "Invalid pixelSize. Default: %e\n\r", pixelSize);
				fflush(consoleOut);
			}
		}

		if (polCount > DIGHOLO_POLCOUNTMAX)
		{
			polCount = DIGHOLO_POLCOUNTMAX;
			if (printOut)
			{
				fprintf(consoleOut, "Invalid polCount. Default: %i\n\r", polCount);
				fflush(consoleOut);
			}
		}

		int modeCount = 0;

		//If no beamGroupCount is specified, then just generate a basis with enough orthogonal modes to satisfy the frameCount
		if (beamGroupCount <= 0)
		{
			int groupIdx = 1;

			while (modeCount < frameCount)
			{
				modeCount += groupIdx;
				groupIdx++;
			}
			if (modeCount == frameCount)
			{
				beamGroupCount = groupIdx - 1;
			}
			else
			{
				beamGroupCount = groupIdx;
			}
			if (printOut)
			{
				fprintf(consoleOut, "Invalid beamGroupCount.Default to match frameCount %i\n\r", beamGroupCount);
				fflush(consoleOut);
			}
		}
		else
		{
			int groupCount = 1;
			while (groupCount <= beamGroupCount)
			{
				modeCount += groupCount;
				groupCount++;
			}
		}

		if (wavelengthCount <= 0 || !wavelengths)
		{
			wavelengthCount = 1;
			if (printOut)
			{
				fprintf(consoleOut, "Invalid wavelengthCount. Default: %i\n\r", wavelengthCount);
				fflush(consoleOut);
			}
		}

		int height = frameWidth / polCount;
		int width = frameHeight;


		float* centreOffsetX = 0;
		allocate1D(polCount, centreOffsetX);

		for (int polIdx = 0; polIdx < polCount; polIdx++)
		{
			centreOffsetX[polIdx] = (float)((pixelSize * (frameWidth / polCount) / 2.0) * (polCount * polIdx - 1 * (polCount - 1)));
		}


		const size_t pixelCount = (size_t)width * (size_t)height;
		const size_t bufferLength = (size_t)frameCount * pixelCount * (size_t)polCount;
		const size_t memLength = (bufferLength * 3) / 2 + 2;

		//Extra memory elements
		size_t parameterMemLength = 0;
		parameterMemLength += polCount * 6; //for the ref tilt (tiltx, tilty, defocus, beamCentreX, beamCentreY, waist)
		parameterMemLength += polCount * 2;// for the refAmplitude
		parameterMemLength += polCount * 3 * 2; //for beamWaist,centre
		parameterMemLength += modeCount * polCount * frameCount * 2;//for beamCoefs
		parameterMemLength += wavelengthCount; //for wavelength

		//Initially fields will store the field of the interference between the Signal and the Reference.
		//Then the same memory will be overwritten with the intensity of the interference |S+R|^2, both in float32 form and uint16 form
		float* pixelBufferOut = 0;
		allocate1D(memLength + parameterMemLength, pixelBufferOut);

		//We'll write the uint16 version of the output buffer to the same memory space
		pixelBuffer16 = (unsigned short*)&pixelBufferOut[bufferLength];

		//These parameters will be aliased into the fields array. This is done so that all we have to do is free(fields) and all memory is freed. Hence all an outside user has to remember is the pointer to fields. Freeing 'fields' frees everything.
		float* refTiltX0 = (float*)&pixelBuffer16[bufferLength];
		float* refTiltY0 = &refTiltX0[polCount];
		float* refDefocus0 = &refTiltY0[polCount];
		float* refBeamCentreX0 = &refDefocus0[polCount];
		float* refBeamCentreY0 = &refBeamCentreX0[polCount];
		float* refWaist0 = &refBeamCentreY0[polCount];
		complex64* refAmplitude0 = (complex64*)&refWaist0[polCount];

		float* beamWaist0 = (float*)&refAmplitude0[polCount];
		float* beamCentreX0 = &beamWaist0[polCount];
		float* beamCentreY0 = &beamCentreX0[polCount];
		complex64* beamCoefs0 = (complex64*)&beamCentreY0[polCount];
		float* wavelengths0 = (float*)&beamCoefs0[polCount * modeCount * frameCount];

		const int threadCount = std::thread::hardware_concurrency();

		complex64** tempMemory = 0;
		allocate2D(threadCount, pixelCount * polCount, tempMemory);

		if (!wavelengths)
		{
			wavelengths = wavelengths0;
			wavelengthCount = 1;
			wavelengths0[0] = (float)DIGHOLO_WAVELENGTH_DEFAULT;

			if (printOut)
			{
				fprintf(consoleOut, "Wavelength(s) not specified. Default: %e\n\r", wavelengths0[0]);
				fflush(consoleOut);
			}
		}

		//The average wavenumber
		float k0 = 0;
		for (int lambdaIdx = 0; lambdaIdx < wavelengthCount; lambdaIdx++)
		{
			k0 += (2 * pi) / wavelengths[lambdaIdx];
		}

		k0 = k0 / wavelengthCount;

		float lambda0 = 2 * pi / k0;
		float wmax = lambda0 / (2 * pixelSize);

		//The maximum supported tilt for the maximum possible window size (wmax)
		float tiltMax = (float)(wmax * (1 - DIGHOLO_WMAXRATIO) / DIGHOLO_UNIT_ANGLE);

		//If no tiltX was specified
		if (!refTiltX)
		{
			refTiltX = refTiltX0;

			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				refTiltX[polIdx] = tiltMax;
			}

			if (printOut)
			{
				fprintf(consoleOut, "tilt(x) not specified. Default: %f\n\r", tiltMax);
				fflush(consoleOut);
			}
		}
		else //refTiltX was specified
		{
			memcpy(refTiltX0, refTiltX, polCount * sizeof(float));
		}

		//If no tiltY was specified
		if (!refTiltY)
		{
			refTiltY = refTiltY0;
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				refTiltY[polIdx] = -tiltMax;//Make it negative so it's easier to tell difference between x and y
			}
			if (printOut)
			{
				fprintf(consoleOut, "tilt(y) not specified. Default: %f\n\r", tiltMax);
				fflush(consoleOut);
			}
		}
		else //refTiltY was specified
		{
			memcpy(refTiltY0, refTiltY, polCount * sizeof(float));
		}

		//If no defocus was specified
		if (!refDefocus)
		{
			refDefocus = refDefocus0;
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				refDefocus[polIdx] = 0;
			}
			if (printOut)
			{
				fprintf(consoleOut, "defocus not specified. Default: %f\n\r", 0.0);
				fflush(consoleOut);
			}
		}
		else //refDefocus was specified
		{
			memcpy(refDefocus0, refDefocus, polCount * sizeof(float));
		}

		//if refBeamCentreX not specified
		if (!refBeamCentreX)
		{
			refBeamCentreX = refBeamCentreX0;
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				refBeamCentreX[polIdx] = centreOffsetX[polIdx]; //(pixelSize*(frameWidth/polCount)/2.0)*(polCount*polIdx-1*(polCount-1));

				if (printOut)
				{
					fprintf(consoleOut, "refBeamCentreX not specified. Default: %e\n\r", refBeamCentreX[polIdx]);
					fflush(consoleOut);
				}
			}

		}
		else
		{
			memcpy(refBeamCentreX0, refBeamCentreX, polCount * sizeof(float));
		}

		if (!refBeamCentreY)
		{
			refBeamCentreY = refBeamCentreY0;
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				refBeamCentreY[polIdx] = 0;

			}
			if (printOut)
			{
				fprintf(consoleOut, "refBeamCentreY not specified. Default: %e\n\r", 0.0);
				fflush(consoleOut);
			}
		}
		else
		{
			memcpy(refBeamCentreY0, refBeamCentreY, polCount * sizeof(float));
		}

		//If reference waist not specified
		if (!refWaist)
		{
			refWaist = refWaist0;

			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				refWaist[polIdx] = 0;
			}
			if (printOut)
			{
				fprintf(consoleOut, "refWaist not specified. Default: %e\n\r", 0.0);
				fflush(consoleOut);
			}
		}
		else
		{
			memcpy(refWaist0, refWaist, polCount * sizeof(float));
		}

		if (!refAmplitude)
		{
			refAmplitude = refAmplitude0;
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				refAmplitude[polIdx][0] = 1;
				refAmplitude[polIdx][1] = 0;

				if (printOut)
				{
					fprintf(consoleOut, "refAmplitude not specified. Default: %f\n\r", 1.0);
					fflush(consoleOut);
				}
			}
		}
		else
		{
			memcpy(refAmplitude0, refAmplitude, polCount * sizeof(complex64));
		}

		if (!beamWaist)
		{
			beamWaist = beamWaist0;
			//Estimate a waist parameter value for this number of mode groups that will fit inside the window
			int minDim = width < height ? width : height;
			float r = (float)(0.25 * minDim * pixelSize);
			float Aeff = pi * r * r;
			float w = AeffToMFD(Aeff, beamGroupCount) * 0.5f;
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				beamWaist[polIdx] = w;
			}

			if (printOut)
			{
				fprintf(consoleOut, "beamWaist not specified. Default: %e\n\r", w);
				fflush(consoleOut);
			}
		}
		else
		{
			memcpy(beamWaist0, beamWaist, polCount * sizeof(float));
		}

		if (!beamCentreX)
		{
			beamCentreX = beamCentreX0;
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				beamCentreX[polIdx] = centreOffsetX[polIdx];
				if (printOut)
				{
					fprintf(consoleOut, "beamCentreX not specified. Default: %e\n\r", beamCentreX[polIdx]);
					fflush(consoleOut);
				}
			}
		}
		else
		{
			memcpy(beamCentreX0, beamCentreX, polCount * sizeof(float));
		}

		if (!beamCentreY)
		{
			beamCentreY = beamCentreY0;
			for (int polIdx = 0; polIdx < polCount; polIdx++)
			{
				beamCentreY[polIdx] = 0.0;
			}
			if (printOut)
			{
				fprintf(consoleOut, "beamCentreY not specified. Default: %e\n\r", 0.0);
				fflush(consoleOut);
			}
		}
		else
		{
			memcpy(beamCentreY0, beamCentreY, polCount * sizeof(float));
		}

		if (!beamCoefs)
		{
			beamCoefs = beamCoefs0;
			for (int frameIdx = 0; frameIdx < frameCount; frameIdx++)
			{
				for (int polIdx = 0; polIdx < polCount; polIdx++)
				{
					for (int modeIdx = 0; modeIdx < modeCount; modeIdx++)
					{
						size_t idx = frameIdx * modeCount * polCount + polIdx * modeCount + modeIdx;
						beamCoefs[idx][0] = frameIdx == modeIdx;
						beamCoefs[idx][1] = 0;
					}
				}
			}

			if (printOut)
			{
				fprintf(consoleOut, "beamCoefs not specified. Default: frameIdx == modeIdx\n\r");
				fflush(consoleOut);
			}
		}
		else
		{
			memcpy(beamCoefs0, beamCoefs, polCount * modeCount * frameCount * sizeof(complex64));
		}

		int maxMG = beamGroupCount;
		int maxGroup = beamGroupCount + 1;;

		int** MN = 0;

		allocate2D(maxGroup, maxGroup, MN);

		modeCount = 0;
		for (int i = 1; i < maxGroup; i++)
		{
			modeCount += i;
			for (int j = 0; j < maxGroup; j++)
			{
				MN[i - 1][j] = -1;
			}
		}

		int* M = 0;
		int* N = 0;
		allocate1D(modeCount, M);
		allocate1D(modeCount, N);

		digHoloGenerateHGindices(maxMG, M, N, MN);

		float*** HGscale = 0;
		//float** HGscaleY = 0;
		allocate3D(2, polCount, maxMG, HGscale);
		//allocate2D(polCount, maxMG, HGscaleY);

		short*** HGx = 0;
		short*** HGy = 0;
		allocate3D(polCount, maxMG, width, HGx);
		allocate3D(polCount, maxMG, height, HGy);

		float** axisX = 0;
		float** axisY = 0;
		allocate2D(polCount, width, axisX);
		allocate2D(polCount, height, axisY);

		for (int polIdx = 0; polIdx < polCount; polIdx++)
		{
			float xShift = +centreOffsetX[polIdx] - beamCentreX[polIdx];
			float yShift = -beamCentreY[polIdx];

			for (int pixelIdx = 0; pixelIdx < width; pixelIdx++)
			{
				axisX[polIdx][pixelIdx] = pixelSize * (pixelIdx - width / 2) + xShift;// +(polIdx * 2 - 1) * width / 2);
			}

			for (int pixelIdx = 0; pixelIdx < height; pixelIdx++)
			{
				axisY[polIdx][pixelIdx] = pixelSize * (pixelIdx - height / 2) + yShift;
			}
		}

		for (int polIdx = 0; polIdx < polCount; polIdx++)
		{
			HGbasisXY1D(HGscale[0][polIdx], beamWaist[polIdx], axisX[polIdx], width, maxMG, (__m128i**)HGx[polIdx]);
			HGbasisXY1D(HGscale[1][polIdx], beamWaist[polIdx], axisY[polIdx], height, maxMG, (__m128i**)HGy[polIdx]);
		}

		//Make the reference wave
		complex64*** refX = 0;
		allocate3D(wavelengthCount, polCount, width, refX);
		complex64*** refY = 0;
		allocate3D(wavelengthCount, polCount, height, refY);

		//Remake the x/y axis for a potentially different reference beam centre
		//You could check if this needs redoing or not, but it's not a big calculation.
		for (int polIdx = 0; polIdx < polCount; polIdx++)
		{
			float xShift = +centreOffsetX[polIdx] - refBeamCentreX[polIdx];
			float yShift = -refBeamCentreY[polIdx];

			for (int pixelIdx = 0; pixelIdx < width; pixelIdx++)
			{
				axisX[polIdx][pixelIdx] = pixelSize * (pixelIdx - width / 2) + xShift;// +(polIdx * 2 - 1) * width / 2);
			}

			for (int pixelIdx = 0; pixelIdx < height; pixelIdx++)
			{
				axisY[polIdx][pixelIdx] = pixelSize * (pixelIdx - height / 2) + yShift;
			}
		}

		for (int polIdx = 0; polIdx < polCount; polIdx++)
		{
			float* xaxis = axisX[polIdx];
			float* yaxis = axisY[polIdx];
			float kx = k0 * sinf(refTiltX[polIdx] * DIGHOLO_UNIT_ANGLE);
			float ky = k0 * sinf(refTiltY[polIdx] * DIGHOLO_UNIT_ANGLE);
			float kfocus = k0 * (0.5f * refDefocus[polIdx]);

			float w = refWaist[polIdx];
			float w2 = w * w;
			for (int lambdaIdx = 0; lambdaIdx < wavelengthCount; lambdaIdx++)
			{
				for (int axisIdx = 0; axisIdx < 2; axisIdx++)
				{
					float* axis = 0;
					int stopIdx = 0;
					complex64*** ref = 0;
					float A0 = 0;
					float kr = 0;
					if (axisIdx == 0)
					{
						axis = xaxis;
						stopIdx = width;
						ref = refX;
						A0 = sqrtf(1.0f / width);
						kr = kx;
					}
					else
					{
						axis = yaxis;
						stopIdx = height;
						ref = refY;
						A0 = sqrtf(1.0f / height);
						kr = ky;
					}

					for (int pixelIdx = 0; pixelIdx < stopIdx; pixelIdx++)
					{
						float r = axis[pixelIdx];
						float r2 = r * r;

						float arg = kr * r + kfocus * r2;
						float cosV = cosf(arg);
						float sinV = sinf(arg);
						float A = A0;

						if (w2 > 0)
						{
							A = A0 * expf(-r2 / w2);
						}

						ref[lambdaIdx][polIdx][pixelIdx][0] = A * cosV;
						ref[lambdaIdx][polIdx][pixelIdx][1] = A * sinV;
					}
				}//pixel
			}//polIdx
		}//lambdaIdx

		float* sincX = 0;
		float* sincY = 0;
		float* axisKX = 0;
		float* axisKY = 0;

		const int heightFFT = frameHeight;
		const int widthFFT = frameWidth;

		const int R2C = 1;
		const int widthOutFFT = R2C ? (widthFFT / 2 + 1) : widthFFT;
		float normFactorFFT = 1.0f / (widthFFT * heightFFT);

		//fftwf_plan forwardPlanR2C = 0;
		//fftwf_plan backwardPlanC2R = 0;
		const size_t fftPlanCount = 2;
		fftwf_plan* fftPlans = 0;
		allocate1D(fftPlanCount, fftPlans);
		fftPlans[0] = 0;
		fftPlans[1] = 0;

		if (fillFactorCorrection)
		{
			const double piD = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229;
			allocate1D(widthFFT, axisKX);
			allocate1D(heightFFT, axisKY);

			//Dimensions of a single pixel in Fourier space along kx/ky axes.
			const float pixelPeriod = (float)(2.0 * pixelSize);
			const float kFactorX = (float)(2 * (2.0 * piD / pixelPeriod) / widthFFT);
			const float kFactorY = (float)(2 * (2.0 * piD / pixelPeriod) / heightFFT);
			//Recalculate Fourier-space kx/ky axes. The 'true' flag at the end indicates this axis should be calculated in FFTshifted form, so that the axes match the output of the FFT
			//The full Fourier-space axis is calculated (not just the R2C part)
			GenerateCoordinatesXY(widthFFT, heightFFT, kFactorX, kFactorY, axisKX, axisKY, true);

			//Concise alias of FFT, pol and frame dimensions
			const int heightFull = heightFFT;// / polCount; If you divide by pol or not doesn't make a difference here, it's the width that's important due to the way it's set out in memory
			const int widthFull = widthFFT;

			//FFTW properties
			const int rank = 2;//2D Fourier transform
			const int n[2] = { heightFFT,widthFFT };//dimensions of the Fourier transform
			//const int nC2R[2] = { heightFFT,widthOutFFT };

			//Use this temporary memory for planning
			float* in = pixelBufferOut;
			complex64* out = tempMemory[0];

			//For picking off sub-window, embedded in a larger matrix (selecting a smaller window inside the camera frame).
			const int inembed[2] = { heightFull,widthFull };//Dimension of the raw camera frame
			const int onembed[2] = { heightFFT ,widthOutFFT };//Dimension of the output Fourier-space (Real-to-complex transform, means dimensions are (width/2+1)*height, because the Hermitian symmetric half isn't calculated.
			//Indicates that the input/output arrays are contigious in memory. e.g. as opposed to some matrix where the columns aren't stored in memory next to each other, or we're not making a matrix be selecting every Nth column of a larger matrix etc.
			//Just a normal input/output matrix as 1 big chunk of memory.
			const int istride = 1;
			const int ostride = 1;

			//Parameters for planning multiple FFTs using a single fftwf_execute call...
			//The distance between input arrays (1 camera frame worth)
			const int idist = (widthFull * heightFull);
			//Distance between output arrays (R2C length x 2 for polarisation)
			const int odist = (heightFFT * (widthOutFFT));

			//Use whatever FFTW plan mode the user has specified, e.g. FFTW_ESTIMATE, FFTW_PATIENT etc.
			const int planMode = FFTW_ESTIMATE;

			int howmany = 1;// fftIdx + 1;
			int threadIdx = 0;

			fftwf_plan_with_nthreads(threadIdx + 1);

			fftPlans[0] = fftwf_plan_many_dft_r2c(rank, &n[0], howmany, in, &inembed[0], istride, idist, out, &onembed[0], ostride, odist, planMode | FFTW_PRESERVE_INPUT);
			fftPlans[1] = fftwf_plan_many_dft_c2r(rank, &n[0], howmany, out, &onembed[0], ostride, odist, in, &inembed[0], istride, idist, planMode);//C2R doesn't like FFTW_PRESERVE_INPUT

			allocate1D(widthOutFFT, sincX);
			allocate1D(heightFFT, sincY);

			for (int pixelIdx = 0; pixelIdx < widthOutFFT; pixelIdx++)
			{
				if (axisKX[pixelIdx])
				{
					float kxPixelSize = axisKX[pixelIdx] * 0.5f * pixelSize;
					sincX[pixelIdx] = sinf(kxPixelSize) / kxPixelSize;
				}
				else
				{
					sincX[pixelIdx] = 1;
				}
			}

			for (int pixelIdx = 0; pixelIdx < heightFFT; pixelIdx++)
			{
				if (axisKY[pixelIdx])
				{
					float kyPixelSize = axisKY[pixelIdx] * 0.5f * pixelSize;
					sincY[pixelIdx] = sinf(kyPixelSize) / kyPixelSize;
				}
				else
				{
					sincY[pixelIdx] = 1;
				}
			}
		}



		std::vector<workPackage*> workPacks(threadCount);
		std::vector<std::thread> threadPool(threadCount);

		int framesPerThread = (int)ceil((1.0f * frameCount) / threadCount);

		for (int i = 0; i < threadCount; i++)
		{
			workPacks[i] = new workPackage();
			int startIdx = i * framesPerThread;
			int stopIdx = (i + 1) * framesPerThread;

			if (stopIdx > frameCount)
			{
				stopIdx = frameCount;
			}

			workPacks[i][0].start = startIdx;
			workPacks[i][0].stop = stopIdx;

			workPacks[i][0].threadIdx = i;// const int threadIdx = e.threadIdx;

			workPacks[i][0].flag1 = frameCount;// const size_t frameCount = e.flag1;
			workPacks[i][0].flag2 = polCount;// const size_t polCount = e.flag2;

			workPacks[i][0].flag3 = frameWidth;// const size_t frameWidth = e.flag3;
			workPacks[i][0].flag4 = frameHeight;// const size_t frameHeight = e.flag4;

			workPacks[i][0].flag5 = modeCount;// const int modeCount = e.flag5;

			workPacks[i][0].flag6 = wavelengthOrdering;// const int wavelengthOrdering = e.flag6;

			workPacks[i][0].flag7 = wavelengthCount;// const int wavelengthCount = e.flag7;
			workPacks[i][0].flag8 = fillFactorCorrection;// const int fillFactorCorrection = e.flag8;
			workPacks[i][0].flag9 = cameraPixelLevelCount;// const int cameraPixelLevelCount = e.flag9;

			workPacks[i][0].ptr1 = beamCoefs;// complex64* beamCoefs = (complex64*)e.ptr1;
			workPacks[i][0].ptr2 = refX;// complex64*** refX = (complex64***)e.ptr2;
			workPacks[i][0].ptr3 = refY;// complex64*** refY = (complex64***)e.ptr3;
			workPacks[i][0].ptr4 = tempMemory;// complex64** tempMemory = (complex64**)e.ptr4;
			workPacks[i][0].ptr5 = pixelBufferOut;// float* pixelBufferOut = (float*)e.ptr5;
			workPacks[i][0].ptr6 = HGscale;// float*** HGscale = (float***)e.ptr6;
			//float** HGscaleX = HGscale[0];
			//float** HGscaleY = HGscale[1];
			workPacks[i][0].ptr7 = HGx;// short*** HGx = (short***)e.ptr7;
			workPacks[i][0].ptr8 = HGy;// short*** HGy = (short***)e.ptr8;
			workPacks[i][0].ptr9 = M;// int* M = (int*)e.ptr9;
			workPacks[i][0].ptr10 = N;// int* N = (int*)e.ptr10;

			workPacks[i][0].ptr11 = fftPlans;// fftwf_plan* fftPlans = (fftwf_plan*)e.ptr11;
			//fftwf_plan forwardPlanR2C = fftPlans[0];
			//fftwf_plan backwardPlanC2R = fftPlans[1];

			workPacks[i][0].ptr12 = sincX;// float* sincX = (float*)e.ptr12;
			workPacks[i][0].ptr13 = sincY;// float* sincY = (float*)e.ptr13;
			workPacks[i][0].ptr14 = pixelBuffer16;// short* pixelBuffer16 = (short*)e.ptr14;

			workPacks[i][0].var1 = normFactorFFT;//float normFactorFFT = e.var1;
			workPacks[i][0].workCompleteEvent.Reset();
			workPacks[i][0].workNewEvent.Reset();


			//std::thread thread(digHoloFrameSimulatorRoutine, workPacks[i]);
			threadPool[i] = std::thread(digHoloFrameSimulatorRoutine, (void*)&workPacks[i][0]);
		}

		//Wait for completion
		float maxV = 0;
		for (int i = 0; i < threadCount; i++)
		{
			workPacks[i][0].workCompleteEvent.WaitOne();
			if (tempMemory[i][0][0] > maxV)
			{
				maxV = tempMemory[i][0][0];
			}
		}

		for (int i = 0; i < threadCount; i++)
		{
			tempMemory[i][0][0] = maxV;
			workPacks[i][0].workCompleteEvent.Reset();
			workPacks[i][0].workNewEvent.Set();
		}

		for (int i = 0; i < threadCount; i++)
		{
			workPacks[i][0].workCompleteEvent.WaitOne();
			threadPool[i].join();
			threadPool[i].~thread();
			delete workPacks[i];
		}

		/*for (int i = 0; i < threadCount; i++)
		{
			workPacks.push_back(new workPackage());

			workPackage* work = workPacks[i];
			work[0].workNewEvent.Reset();
			work[0].workCompleteEvent.Reset();
			work[0].active = true;
			work[0].callback = workFunction::NA;
			threadPool.push_back(std::thread(&digHoloFrameSimulatorRoutine, this, i));
		}
		*/

		std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();

		std::chrono::duration<double> time_spanTotal = (std::chrono::duration_cast<std::chrono::duration<double>>(stopTime - startTime));

		if (printOut)
		{
			fprintf(consoleOut, "Frames generated in %f seconds.\n\r", time_spanTotal.count());
			fflush(consoleOut);
		}

		if (fname)
		{
			FILE* coefsFile = fopen(fname, "wb");
			if (coefsFile)
			{
				fwrite(pixelBuffer16, sizeof(unsigned short), frameCount * width * height * polCount, coefsFile);
				//fwrite(field32, sizeof(float), frameCount* width* height* polCount, coefsFile);
				fclose(coefsFile);
			}
		}

		free3D(refX);
		free3D(refY);

		free2D(MN);
		free1D(M);
		free1D(N);
		free3D(HGscale);
		free3D(HGx);
		free3D(HGy);
		free2D(axisX);
		free2D(axisY);

		free2D(tempMemory);
		free1D(sincX);
		free1D(sincY);
		free1D(axisKX);
		free1D(axisKY);

		for (int fftIdx = 0; fftIdx < fftPlanCount; fftIdx++)
			if (fftPlans[fftIdx])
			{
				fftwf_destroy_plan(fftPlans[fftIdx]);
			}


		if (printOut)
		{
			fprintf(consoleOut, "</FRAME SIMULATOR> \n\r");
			fflush(consoleOut);
		}

		return (float*)pixelBufferOut;
	}

	return 0;
}

EXT_C  float* digHoloFrameSimulatorCreate(int frameCount, int* frameWidth, int* frameHeight, float *pixelSize, int *polCount,
	float* *refTiltX, float* *refTiltY, float* *refDefocus, float* *refWaist, float* *refBeamCentreX, float* *refBeamCentreY, complex64* *refAmplitude,
	int *beamGroupCount, float* *beamWaist, complex64* *beamCoefs, float* *beamCentreX, float* *beamCentreY,
	int *cameraPixelLevelCount, int fillFactorCorrection, float * *wavelengths, int *wavelengthCount, int wavelengthOrdering, int printOut, unsigned short* *pixelBuffer16, const char* fname)
{
	return frameSimulatorCreate(frameCount, frameWidth[0], frameHeight[0], pixelSize[0], polCount[0],
		refTiltX[0],  refTiltY[0],refDefocus[0], refWaist[0], refBeamCentreX[0], refBeamCentreY[0], refAmplitude[0],
		beamGroupCount[0], beamWaist[0], beamCoefs[0], beamCentreX[0], beamCentreY[0],
		cameraPixelLevelCount[0], fillFactorCorrection, wavelengths[0], wavelengthCount[0], wavelengthOrdering,printOut, pixelBuffer16[0],fname);
}

EXT_C int digHoloRunBatchFromConfigFile(char* filename)
{
	if (filename)
	{
		std::ifstream configFile(filename);
		std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

		if (configFile.is_open())
		{
			int handleIdx = 0;
			if (digHoloObjectsCount == 0)
			{
				handleIdx = digHoloCreate();
			}
			else
			{

			}

			std::string line;
			unsigned short* frameBuffer16 = 0;
			float* frameBuffer = 0;
			//size_t frameBufferLength = 0;
		
			int errorCode = 0;

			std::string consoleFilename = "";
			std::string OutputFilenameSummary = "";
			std::string OutputFilenameCoefs = "";
			std::string OutputFilenameFields = "";
			std::string OutputFilenameBasis = "";
			std::string OutputFilenameXaxis = "";
			std::string OutputFilenameYaxis = "";
			std::string OutputFilenameCalibration = "";
			std::string OutputFilenameViewport = "";

			std::string BasisFilename = "";			
			int BasisCustomModeCountIn = 0;
			int BasisCustomModeCountOut = 0;

			std::string CalibrationFilename = "";
			int CalibrationExists = 0;

			std::string BatchCalibrationFilename = "";
			int BatchCalibrationExists = 0;
			int BatchCalibrationPolCount = 0;
			int BatchCalibrationBatchCount = 0;

			int doAutoAlign = 0;
			int doThreadDiagnostic = 0;
			float doBenchmark = 0;
			int doDebug = 0;
			//int doViewport = 0;
			int* doViewport = 0;
			int doViewportCount = 0;
			int doTestFrames = 0;

			int wavelengthCount = 0;
			float wavelengthStart = 0;
			float wavelengthStop = 0;

			int tiltSpecified = 0;
			int centreSpecified = 0;
			int waistSpecified = 0;
			int defocusSpecified = 0;


			while (std::getline(configFile, line) && !errorCode)
			{
				if (line.length() > 0)
				{
					if (line[line.length() - 1] == '\r')
					{
						line[line.length() - 1] = 0;
					}
					std::istringstream iline;
					iline.str(line);

					std::string parameterName;
					std::getline(iline, parameterName, '	');
					const char* pName = parameterName.c_str();

					std::vector<std::string> args;
					std::string argValue;

					while (std::getline(iline, argValue, '	'))
					{
						args.push_back(argValue);
					}
					const size_t argCount = args.size();

					if (!strcmp("FrameBufferFilename", pName))
					{


						if (argCount > 0)
						{
							std::string value = args[0];
							int errorCode = digHoloSetFrameBufferFromFile(handleIdx, value.c_str());
							if (errorCode != DIGHOLO_ERROR_SUCCESS)
							{
								std::cout << "Fatal error. Aborting.\n\r";
								return errorCode;
							}
						}
						else
						{
							std::cout << "No file supplied. Generating test frames.\n\r";
							doTestFrames = true;
						}
						//	goto NEXTLINE;
					}

					if (argCount > 0)
					{
						try
						{

							if (!strcmp("FrameWidth", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetFrameWidth(handleIdx, value);
								goto NEXTLINE;
							}
							if (!strcmp("FrameHeight", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetFrameHeight(handleIdx, value);
								goto NEXTLINE;
							}
							if (!strcmp("BatchCount", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetBatchCount(handleIdx, value);
								goto NEXTLINE;
							}
							if (!strcmp("PolCount", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetPolCount(handleIdx, value);
								goto NEXTLINE;
							}
							if (!strcmp("ConsoleFilename", pName))
							{
								std::string value = args[0];
								consoleFilename = value;
								goto NEXTLINE;
							}
							if (!strcmp("Verbosity", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetVerbosity(handleIdx, value);
								goto NEXTLINE;
							}
							if (!strcmp("OutputFilenameSummary", pName))
							{
								std::string value = args[0];
								OutputFilenameSummary = value;
								goto NEXTLINE;
							}
							if (!strcmp("OutputFilenameCoefs", pName))
							{
								std::string value = args[0];
								OutputFilenameCoefs = value;
								goto NEXTLINE;
							}
							if (!strcmp("OutputFilenameFields", pName))
							{
								std::string value = args[0];
								OutputFilenameFields = value;
								goto NEXTLINE;
							}
							if (!strcmp("OutputFilenameBasis", pName))
							{
								std::string value = args[0];
								OutputFilenameBasis = value;
								goto NEXTLINE;
							}
							if (!strcmp("OutputFilenameXaxis", pName))
							{
								std::string value = args[0];
								OutputFilenameXaxis = value;
								goto NEXTLINE;
							}
							if (!strcmp("OutputFilenameYaxis", pName))
							{
								std::string value = args[0];
								OutputFilenameYaxis = value;
								goto NEXTLINE;
							}
							if (!strcmp("OutputFilenameRefCalibration", pName))
							{
								std::string value = args[0];
								OutputFilenameCalibration = value;
								goto NEXTLINE;
							}
							if (!strcmp("OutputFilenameViewport", pName))
							{
								std::string value = args[0];
								OutputFilenameViewport = value;
								goto NEXTLINE;
							}
							if (!strcmp("ThreadCount", pName))
							{
								int value = stoi(args[0]);
								if (value <= 0)
								{
									doThreadDiagnostic = 1;
								}
								digHoloConfigSetThreadCount(handleIdx, value);
								goto NEXTLINE;
							}
							if (!strcmp("Benchmark", pName))
							{
								float value = stof(args[0]);
								doBenchmark = value;
								goto NEXTLINE;
							}
							if (!strcmp("Debug", pName))
							{
								int value = stoi(args[0]);
								doDebug = value;
								goto NEXTLINE;
							}
							if (!strcmp("FourierWindowRadius", pName))
							{
								float value = stof(args[0]);
								digHoloConfigSetFourierWindowRadius(handleIdx, value);
								goto NEXTLINE;
							}
							if (!strcmp("BeamCentreX", pName))
							{
								for (int argIdx = 0; argIdx < argCount; argIdx++)
								{
									float value = stof(args[argIdx]);
									centreSpecified = centreSpecified | (value != 0);
									digHoloConfigSetBeamCentre(handleIdx, 0, argIdx, value);
								}
								goto NEXTLINE;
							}
							if (!strcmp("BeamCentreY", pName))
							{
								for (int argIdx = 0; argIdx < argCount; argIdx++)
								{
									float value = stof(args[argIdx]);
									centreSpecified = centreSpecified | (value != 0);
									digHoloConfigSetBeamCentre(handleIdx, 1, argIdx, value);
								}
								goto NEXTLINE;
							}
							if (!strcmp("TiltX", pName))
							{
								for (int argIdx = 0; argIdx < argCount; argIdx++)
								{
									float value = stof(args[argIdx]);
									tiltSpecified = tiltSpecified | (value != 0);
									digHoloConfigSetTilt(handleIdx, 0, argIdx, value);
								}
								goto NEXTLINE;
							}
							if (!strcmp("TiltY", pName))
							{
								for (int argIdx = 0; argIdx < argCount; argIdx++)
								{
									float value = stof(args[argIdx]);
									tiltSpecified = tiltSpecified | (value != 0);
									digHoloConfigSetTilt(handleIdx, 1, argIdx, value);
								}
								goto NEXTLINE;
							}

							if (!strcmp("Defocus", pName))
							{
								for (int argIdx = 0; argIdx < argCount; argIdx++)
								{
									float value = stof(args[argIdx]);
									defocusSpecified = defocusSpecified || value != 0;
									digHoloConfigSetDefocus(handleIdx, argIdx, value);
								}
								goto NEXTLINE;
							}
							if (!strcmp("PixelSize", pName) | !strcmp("FramePixelSize", pName))
							{
								float value = stof(args[0]);
								digHoloConfigSetFramePixelSize(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("PolLockTilt", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetPolLockTilt(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("PolLockDefocus", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetPolLockDefocus(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("PolLockBasisWaist", pName) || !strcmp("PolLockWaist", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetPolLockBasisWaist(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("fftWindowSizeX", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetfftWindowSizeX(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("fftWindowSizeY", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetfftWindowSizeY(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("FFTWPlanMode", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetFFTWPlanMode(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("FFTWWisdomFilename", pName))
							{
								std::string value = args[0];
								digHoloFFTWWisdomFilename(value.c_str());

								goto NEXTLINE;
							}
							if (!strcmp("IFFTResolutionMode", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetIFFTResolutionMode(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("WavelengthCentre", pName))
							{
								float value = stof(args[0]);
								digHoloConfigSetWavelengthCentre(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("WavelengthCount", pName))
							{
								wavelengthCount = stoi(args[0]);
								goto NEXTLINE;
							}
							if (!strcmp("WavelengthStart", pName))
							{
								wavelengthStart = stof(args[0]);
								goto NEXTLINE;
							}
							if (!strcmp("WavelengthStop", pName))
							{
								wavelengthStop = stof(args[0]);
								goto NEXTLINE;
							}
							if (!strcmp("WavelengthOrdering", pName))
							{
								for (int argIdx = 0; argIdx < argCount; argIdx++)
								{
									int value = stoi(args[argIdx]);
									digHoloConfigSetWavelengthOrdering(handleIdx, argIdx, value);
								}
								goto NEXTLINE;
							}


							if (!strcmp("MaxMG", pName) || !strcmp("BasisGroupCount", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetBasisGroupCount(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("BasisType", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetBasisType(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("Waist", pName) || !strcmp("BasisWaist", pName))
							{
								for (int argIdx = 0; argIdx < argCount; argIdx++)
								{
									float value = stof(args[argIdx]);
									waistSpecified = waistSpecified | (value != 0);
									digHoloConfigSetBasisWaist(handleIdx, argIdx, value);
								}
								//If less waist number were specified than polarisations
								int polCount = digHoloConfigGetPolCount(handleIdx);
								for (int polIdx = (int)argCount; polIdx < polCount; polIdx++)
								{
									float value = stof(args[argCount - 1]);
									digHoloConfigSetBasisWaist(handleIdx, polIdx, value);
								}

								goto NEXTLINE;
							}
							if (!strcmp("BatchAvgCount", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetBatchAvgCount(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("BatchAvgMode", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetBatchAvgMode(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("AutoAlignCentre", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetAutoAlignBeamCentre(handleIdx, value);
								doAutoAlign = doAutoAlign | value;
								goto NEXTLINE;
							}
							if (!strcmp("AutoAlignDefocus", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetAutoAlignDefocus(handleIdx, value);
								doAutoAlign = doAutoAlign | value;
								goto NEXTLINE;
							}
							if (!strcmp("AutoAlignTilt", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetAutoAlignTilt(handleIdx, value);
								doAutoAlign = doAutoAlign | value;
								goto NEXTLINE;
							}
							if (!strcmp("AutoAlignWaist", pName) || !strcmp("AutoAlignBasisWaist", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetAutoAlignBasisWaist(handleIdx, value);
								doAutoAlign = doAutoAlign | value;
								goto NEXTLINE;
							}
							if (!strcmp("AutoAlignFourierWindowRadius", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetAutoAlignFourierWindowRadius(handleIdx, value);
								doAutoAlign = doAutoAlign | value;
								goto NEXTLINE;
							}

							if (!strcmp("AutoAlignTol", pName))
							{
								float value = stof(args[0]);
								digHoloConfigSetAutoAlignTol(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("AutoAlignPolIndependence", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetAutoAlignPolIndependence(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("AutoAlignBasisMulConjTrans", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetAutoAlignBasisMulConjTrans(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("AutoAlignMode", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetAutoAlignMode(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("AutoAlignGoalIdx", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetAutoAlignGoalIdx(handleIdx, value);

								goto NEXTLINE;
							}
							if (!strcmp("BasisCustomFilename", pName))
							{
								std::string value = args[0];
								BasisFilename = value;

								goto NEXTLINE;
							}
							if (!strcmp("BasisCustomModeCountIn", pName))
							{
								int value = stoi(args[0]);
								BasisCustomModeCountIn = value;

								goto NEXTLINE;
							}
							if (!strcmp("BasisCustomModeCountOut", pName))
							{
								int value = stoi(args[0]);
								BasisCustomModeCountOut = value;

								goto NEXTLINE;
							}
							if (!strcmp("RefCalibrationFilename", pName))
							{
								std::string value = args[0];
								CalibrationFilename = value;
								CalibrationExists = true;
								goto NEXTLINE;
							}

							if (!strcmp("BatchCalibrationFilename", pName))
							{
								std::string value = args[0];
								BatchCalibrationFilename = value;
								BatchCalibrationExists = true;
								goto NEXTLINE;
							}

							if (!strcmp("BatchCalibrationPolCount", pName))
							{
								int value = stoi(args[0]);
								BatchCalibrationPolCount = value;
								goto NEXTLINE;
							}
							if (!strcmp("BatchCalibrationBatchCount", pName))
							{
								int value = stoi(args[0]);
								BatchCalibrationBatchCount = value;
								goto NEXTLINE;
							}
							if (!strcmp("FillFactorCorrection", pName))
							{
								int value = stoi(args[0]);
								digHoloConfigSetFillFactorCorrectionEnabled(handleIdx, value);
								goto NEXTLINE;
							}
							if (!strcmp("Viewport", pName))
							{
								allocate1D(argCount, doViewport);
								for (int argIdx = 0; argIdx < argCount; argIdx++)
								{
									int value = stoi(args[argIdx]);
									doViewport[argIdx] = value;
								}
								doViewportCount = (int)argCount;

								goto NEXTLINE;
							}
						}
						catch (const std::exception& ex)
						{
							std::cout << ex.what() << "\n";
						}
					}

				NEXTLINE: {}
				}
			}//while new lines
			configFile.close();
			if (!errorCode)
			{
				int batchCount = digHoloConfigGetBatchCount(handleIdx);
				size_t avgCount = digHoloConfigGetBatchAvgCount(handleIdx);
				size_t frameWidth = digHoloConfigGetFrameWidth(handleIdx);
				size_t frameHeight = digHoloConfigGetFrameHeight(handleIdx);

				//size_t frameBufferExpectedLength = batchCount * avgCount * frameWidth * frameHeight;

				if (consoleFilename.length())
				{
					const char* cFilename = consoleFilename.c_str();
					int success = digHoloConsoleRedirectToFile((char*)&cFilename[0]);
					if (!success)
					{
						fprintf(consoleOut, "Specified console output filename could not be created. %s\n", cFilename); fflush(consoleOut);
					}
				}
				/*
				if (frameBufferExpectedLength > frameBufferLength)
				{
					fprintf(consoleOut, "Error: too few pixels in supplied frame buffer for the specified frame dimensions and batch count\n");
					fprintf(consoleOut, "	Expected pixel count : %zu\n", frameBufferExpectedLength);
					fprintf(consoleOut, "	Actual pixel count   : %zu\n", frameBufferLength); fflush(consoleOut);
					//return 0;
				}
				*/

				if (wavelengthCount > 0)
				{
					digHoloConfigSetWavelengthsLinearFrequency(handleIdx, wavelengthStart, wavelengthStop,wavelengthCount);
				}

				if (CalibrationExists)
				{
					const char* fname = CalibrationFilename.c_str();
					//int success = 
					digHoloConfigSetRefCalibrationFromFile(handleIdx, fname, (int)wavelengthCount, (int)frameWidth, (int)frameHeight);
				}

				if (BatchCalibrationExists)
				{
					const char* fname = BatchCalibrationFilename.c_str();
					//int success = 
					digHoloConfigSetBatchCalibrationFromFile(handleIdx, fname, BatchCalibrationPolCount, BatchCalibrationBatchCount);
				}

				const int basisType = digHoloConfigGetBasisType(handleIdx);
				complex64* basisTransformMatrix = 0;
				if (basisType == DIGHOLO_BASISTYPE_CUSTOM)
				{
					FILE* basisTransformMatrixFile = fopen(BasisFilename.c_str(), "r");
					if (basisTransformMatrixFile && BasisCustomModeCountIn && BasisCustomModeCountOut)
					{
						fseek(basisTransformMatrixFile, 0, SEEK_END);
						size_t sizeBytes = ftell(basisTransformMatrixFile);
						rewind(basisTransformMatrixFile);
						size_t basisTransformMatrixLength = sizeBytes / sizeof(complex64);
						allocate1D(basisTransformMatrixLength, basisTransformMatrix);
						size_t result = fread(basisTransformMatrix, sizeof(complex64), basisTransformMatrixLength, basisTransformMatrixFile);
						//This should never happen because of the fseek/tell above.
						if (result != sizeBytes)
						{
							fprintf(consoleOut, "Error : Read did not match expected number of bytes\n"); fflush(consoleOut);
						}
						fclose(basisTransformMatrixFile);
						int modeCountIn = BasisCustomModeCountIn;
						int modeCountOut = BasisCustomModeCountOut;
						digHoloConfigSetBasisTypeCustom(handleIdx,  modeCountIn, modeCountOut, basisTransformMatrix);
					}
					else
					{
						fprintf(consoleOut, "Error : BasisFilename could not be found. %s\n", BasisFilename.c_str()); fflush(consoleOut);
						//return 0;
					}
				}

				if (doTestFrames)
				{						
					int frameCount  = batchCount;
					int frameWidth  = digHoloConfigGetFrameWidth(handleIdx);
					int frameHeight = digHoloConfigGetFrameHeight(handleIdx);
					float pixelSize = digHoloConfigGetFramePixelSize(handleIdx);
					int polCount    = digHoloConfigGetPolCount(handleIdx);

					float* refTiltX = 0;
					float* refTiltY = 0;
					float* refDefocus = 0;
					float* refWaist = 0;
					float* refBeamCentreX = 0;
					float* refBeamCentreY = 0;

					float* beamWaist = 0;
					float* beamCentreX = 0;
					float* beamCentreY = 0;

					if (tiltSpecified)
					{
						allocate1D(polCount,refTiltX);
						allocate1D(polCount, refTiltY);

						for (int axisIdx = 0; axisIdx < 2; axisIdx++)
						{
							float* refTilt = 0;

							if (axisIdx == 0)
							{
								refTilt = refTiltX;

							}
							else
							{
								refTilt = refTiltY;

							}

							for (int polIdx = 0; polIdx < polCount; polIdx++)
							{
								float tilt = digHoloConfigGetTilt(handleIdx, axisIdx, polIdx);
								refTilt[polIdx] = tilt;

							}
						}
					}

					if (centreSpecified)
					{
						allocate1D(polCount, refBeamCentreX);
						allocate1D(polCount, refBeamCentreY);
						allocate1D(polCount, beamCentreX);
						allocate1D(polCount, beamCentreY);

						for (int axisIdx = 0; axisIdx < 2; axisIdx++)
						{
							float* refCentre = 0;
							float* beamCentre = 0;
							if (axisIdx == 0)
							{
								refCentre = refBeamCentreX;
								beamCentre = beamCentreX;
							}
							else
							{
								refCentre = refBeamCentreY;
								beamCentre = beamCentreY;
							}

							for (int polIdx = 0; polIdx < polCount; polIdx++)
							{

								float centre = digHoloConfigGetBeamCentre(handleIdx, axisIdx, polIdx);

								refCentre[polIdx] = centre;
								beamCentre[polIdx] = centre;
							}
						}
					}

					if (defocusSpecified)
					{
						allocate1D(polCount, refDefocus);

						for (int polIdx = 0; polIdx < polCount; polIdx++)
						{
							float defocus = digHoloConfigGetDefocus(handleIdx, polIdx);
							refDefocus[polIdx] = defocus;
							defocusSpecified = defocusSpecified || defocus != 0;

						}
					}

					if (waistSpecified)
					{
						allocate1D(polCount, beamWaist);
						for (int polIdx = 0; polIdx < polCount; polIdx++)
						{
							float waist = digHoloConfigGetBasisWaist(handleIdx, polIdx);
							beamWaist[polIdx] = waist;
						}
					}

					complex64* refAmplitude = 0;

					int beamGroupCount = digHoloConfigGetBasisGroupCount(handleIdx);


					complex64* beamCoefs = 0;

					int cameraPixelLevelCount = 16384;
					int fillFactorCorrection = digHoloConfigGetFillFactorCorrectionEnabled(handleIdx);
					int wavelengthCount = 1;
					float* wavelength = digHoloConfigGetWavelengths(handleIdx, &wavelengthCount);
					int wavelengthOrdering = digHoloConfigGetWavelengthOrdering(handleIdx,DIGHOLO_WAVELENGTHORDER_INPUT);

					int printOut = 1;
					unsigned short* pixelBuffer16 = 0;
					const char* fname = "testFrame.bin";

					frameBuffer = digHoloFrameSimulatorCreate(frameCount, &frameWidth, &frameHeight, &pixelSize, &polCount,
						&refTiltX, &refTiltY, &refDefocus, &refWaist, &refBeamCentreX, &refBeamCentreY, &refAmplitude,
						&beamGroupCount, &beamWaist, &beamCoefs, &beamCentreX, &beamCentreY,
						&cameraPixelLevelCount, fillFactorCorrection, &wavelength, &wavelengthCount, wavelengthOrdering,
						printOut, &pixelBuffer16, fname);

					if (tiltSpecified)
					{
						free1D(refTiltX);
						free1D(refTiltY);
					}

					if (defocusSpecified)
					{
						free1D(refDefocus);
					}

					if (waistSpecified)
					{
						free1D(beamWaist);
					}

					if (centreSpecified)
					{
						free1D(refBeamCentreX);
						free1D(refBeamCentreY);
						free1D(beamCentreX);
						free1D(beamCentreY);
					}
				}


				int avgMode = digHoloConfigGetBatchAvgMode(handleIdx);
				digHoloSetBatchAvg(handleIdx, (int)batchCount, frameBuffer, (int)avgCount, (int)avgMode);

				int modeCount = 0;
				int polCount = 0;

				float il = -1000;
				float mdl = -1000;
				float snravg = -1000;
				double runtimeTotal = 0;
				double runtimeAutoAlign = 0;
				double runtimeProcessBatch = 0;

				if (!doAutoAlign)
				{
					fprintf(consoleOut, "No AutoAlign parameters set(e.g. tilt, focus, waist, beam centre). Hence no AutoAlign will be performed. Using user specified digHolo parameters.\n"); fflush(consoleOut);
				}
				else
				{
					std::chrono::steady_clock::time_point autoAlignStartTime = std::chrono::steady_clock::now();
					digHoloAutoAlign(handleIdx);
					std::chrono::steady_clock::time_point autoAlignStopTime = std::chrono::steady_clock::now();

					std::chrono::duration<double> time_span = (std::chrono::duration_cast<std::chrono::duration<double>>(autoAlignStopTime - autoAlignStartTime));
					runtimeAutoAlign = time_span.count();

					il = digHoloAutoAlignGetMetric(handleIdx, DIGHOLO_METRIC_IL);
					mdl = digHoloAutoAlignGetMetric(handleIdx, DIGHOLO_METRIC_MDL);
					snravg = digHoloAutoAlignGetMetric(handleIdx, DIGHOLO_METRIC_SNRAVG);
					
				}

				if (doThreadDiagnostic)
				{
					int threadCount = digHoloBenchmarkEstimateThreadCountOptimal(handleIdx, 15.0);
					digHoloConfigSetThreadCount(handleIdx, threadCount);
				}

				if (doBenchmark > 0)
				{

					digHoloBenchmark(handleIdx, doBenchmark,0);

				}

				if (doDebug > 0)
				{
					digHoloDebugRoutine(handleIdx);
				}

				int maxMG = digHoloConfigGetBasisGroupCount(handleIdx);




				//ProcessBatch
				if (OutputFilenameCoefs.length())
				{
					std::chrono::steady_clock::time_point processBatchStartTime = std::chrono::steady_clock::now();
					complex64* coefs = digHoloProcessBatch(handleIdx, &batchCount, &modeCount, &polCount);
					//complex64* coefs = digHoloProcessBatchFrequencySweepLinear(handleIdx, &batchCount, &modeCount, &polCount, wavelengthStart, wavelengthStop, wavelengthCount);
					std::chrono::steady_clock::time_point processBatchStopTime = std::chrono::steady_clock::now();

					std::chrono::duration<double> time_span = (std::chrono::duration_cast<std::chrono::duration<double>>(processBatchStopTime - processBatchStartTime));
					runtimeProcessBatch = time_span.count();

					if (coefs && batchCount && modeCount && polCount)
					{
						FILE* coefsFile = fopen(OutputFilenameCoefs.c_str(), "wb");
						if (coefsFile)
						{
							fwrite(coefs, sizeof(complex64), batchCount * modeCount * polCount, coefsFile);
							fclose(coefsFile);
						}
						else
						{
							fprintf(consoleOut, "Error : Could not open OutputFilenameCoefs %s\n", OutputFilenameCoefs.c_str()); fflush(consoleOut);
						}
					}
				}

				int fieldWidth = 0;
				int fieldHeight = 0;
				float* fieldXaxis = 0;
				float* fieldYaxis = 0;
				if (OutputFilenameBasis.length() && maxMG)
				{
					complex64* basisFields = digHoloBasisGetFields(handleIdx, &modeCount, &polCount, &fieldXaxis, &fieldYaxis, &fieldWidth, &fieldHeight);
					if (basisFields && modeCount && polCount && fieldWidth && fieldHeight)
					{
						FILE* basisFile = fopen(OutputFilenameBasis.c_str(), "wb");
						if (basisFile)
						{
							fwrite(basisFields, sizeof(complex64), modeCount * polCount * fieldWidth * fieldHeight, basisFile);
							fclose(basisFile);
						}
						else
						{
							fprintf(consoleOut, "Error : Could not open OutputFilenameBasis %s\n", OutputFilenameBasis.c_str()); fflush(consoleOut);
						}
					}
				}

				if (OutputFilenameFields.length())
				{
					complex64* fields = digHoloGetFields(handleIdx, &batchCount, &polCount, &fieldXaxis, &fieldYaxis, &fieldWidth, &fieldHeight);

					if (fields && batchCount && polCount && fieldWidth && fieldHeight)
					{
						FILE* fieldsFile = fopen(OutputFilenameFields.c_str(), "wb");
						if (fieldsFile)
						{
							size_t elementCount = (size_t)batchCount;
							elementCount *= polCount * fieldWidth * fieldHeight;
							fwrite(fields, sizeof(complex64), elementCount, fieldsFile);
							fclose(fieldsFile);
						}
						else
						{
							fprintf(consoleOut, "Error : Could not open OutputFilenameFields %s\n", OutputFilenameFields.c_str()); fflush(consoleOut);
						}
					}
				}

				
				if (CalibrationExists)
				{
					FILE* fieldsFile = fopen(OutputFilenameCalibration.c_str(), "wb");
					if (fieldsFile)
					{
						complex64* fields = digHoloConfigGetRefCalibrationFields(handleIdx, &batchCount, &polCount, &fieldXaxis, &fieldYaxis, &fieldWidth, &fieldHeight);
						size_t elementCount = (size_t)batchCount;
						elementCount *= polCount * fieldWidth * fieldHeight;
						fwrite(fields, sizeof(complex64), elementCount, fieldsFile);
						fclose(fieldsFile);
						
					}

				}
				
				if (doViewport)
				{
					if (OutputFilenameViewport.length())
					{
						int fileExPos = (int)OutputFilenameViewport.length();

						if (doViewportCount >= 1)
						{
							int idx = (int)OutputFilenameViewport.rfind('.');
							if (fileExPos >= 0)
							{
								fileExPos = idx;
							}
						}

						for (int viewIdx = 0; viewIdx < doViewportCount; viewIdx++)
						{
							int displayMode = doViewport[viewIdx];

							if (displayMode && displayMode< DIGHOLO_VIEWPORT_COUNT)
							{
								std::string filename0 = OutputFilenameViewport;
								char tempBuff[1024];
								sprintf(&tempBuff[0],"%2.2i", displayMode);
								filename0.insert(fileExPos, tempBuff);
								int forceProcessing = 0;
								int width = 0;
								int height = 0;
								char* windowString = 0;
								
								digHoloGetViewportToFile(handleIdx, displayMode, forceProcessing, &width, &height, &windowString, filename0.c_str());
							}
						}
					}
					free1D(doViewport);
				}

				//if (OutputFilenameSummary.length())
				{

					FILE* summaryFile = 0;
					if (OutputFilenameSummary.length())
					{
						summaryFile = fopen(OutputFilenameSummary.c_str(), "w");
					}
					else
					{
						summaryFile = stdout;
					}

					if (summaryFile)
					{
						fprintf(summaryFile, "<SUMMARY>\n\r");
						for (int axisIdx = 0; axisIdx < 2; axisIdx++)
						{
							if (axisIdx == 0)
							{
								fprintf(summaryFile, "BeamCentreX	");
							}
							else
							{
								fprintf(summaryFile, "BeamCentreY	");
							}
							for (int polIdx = 0; polIdx < polCount; polIdx++)
							{
								float beamCentreX = digHoloConfigGetBeamCentre(handleIdx, axisIdx, polIdx);
								if (polIdx != (polCount - 1))
								{
									fprintf(summaryFile, "%e	", beamCentreX);
								}
								else
								{
									fprintf(summaryFile, "%e\n\r", beamCentreX);
								}
							}
						}
						for (int axisIdx = 0; axisIdx < 2; axisIdx++)
						{
							if (axisIdx == 0)
							{
								fprintf(summaryFile, "TiltX	");
							}
							else
							{
								fprintf(summaryFile, "TiltY	");
							}
							for (int polIdx = 0; polIdx < polCount; polIdx++)
							{
								float tilt = digHoloConfigGetTilt(handleIdx, axisIdx, polIdx);
								if (polIdx != (polCount - 1))
								{
									fprintf(summaryFile, "%f	", tilt);
								}
								else
								{
									fprintf(summaryFile, "%f\n\r", tilt);
								}
							}
						}

						fprintf(summaryFile, "Defocus	");
						for (int polIdx = 0; polIdx < polCount; polIdx++)
						{
							float defocus = digHoloConfigGetDefocus(handleIdx, polIdx);
							if (polIdx != (polCount - 1))
							{
								fprintf(summaryFile, "%f	", defocus);
							}
							else
							{
								fprintf(summaryFile, "%f\n\r", defocus);
							}
						}
						fprintf(summaryFile, "Waist	");
						for (int polIdx = 0; polIdx < polCount; polIdx++)
						{
							float waist = digHoloConfigGetBasisWaist(handleIdx, polIdx);
							if (polIdx != (polCount - 1))
							{
								fprintf(summaryFile, "%e	", waist);
							}
							else
							{
								fprintf(summaryFile, "%e\n\r", waist);
							}
						}
						float fourierWindow = digHoloConfigGetFourierWindowRadius(handleIdx);
						fprintf(summaryFile, "FourierWindow	%3.3f\n\r", fourierWindow);
						if (fieldWidth && fieldHeight)
						{
							fprintf(summaryFile, "FieldWidth	%i\n\r", fieldWidth);
							fprintf(summaryFile, "FieldHeight	%i\n\r", fieldHeight);
							if (OutputFilenameXaxis.length())
							{
								FILE* xaxisFile = fopen(OutputFilenameXaxis.c_str(), "w");
								if (xaxisFile)
								{
									fwrite(fieldXaxis, sizeof(float), fieldWidth, xaxisFile);
									fclose(xaxisFile);
								}
							}
							if (OutputFilenameYaxis.length())
							{
								FILE* yaxisFile = fopen(OutputFilenameYaxis.c_str(), "w");
								if (yaxisFile)
								{
									fwrite(fieldYaxis, sizeof(float), fieldHeight, yaxisFile);
									fclose(yaxisFile);
								}
							}
						}
						if (!doAutoAlign)
						{
							digHoloAutoAlignCalcMetrics(handleIdx);

							il = digHoloAutoAlignGetMetric(handleIdx, DIGHOLO_METRIC_IL);
							mdl = digHoloAutoAlignGetMetric(handleIdx, DIGHOLO_METRIC_MDL);
							snravg = digHoloAutoAlignGetMetric(handleIdx, DIGHOLO_METRIC_SNRAVG);
						}
					//	if (doAutoAlign)
						{
							fprintf(summaryFile, "IL	%f\n\r", il);
							fprintf(summaryFile, "MDL	%f\n\r", mdl);
							fprintf(summaryFile, "SNRAVG	%f\n\r", snravg);
						}

						std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();

						std::chrono::duration<double> time_spanTotal = (std::chrono::duration_cast<std::chrono::duration<double>>(stopTime - startTime));
						runtimeTotal = time_spanTotal.count();
						//if (doAutoAlign)
						{
							fprintf(summaryFile, "Runtime (AutoAlign,sec. Hz)	%f	%f\n\r", runtimeAutoAlign, 1.0 / runtimeAutoAlign);
							fprintf(summaryFile, "Runtime (ProcessBatch,sec. Hz)	%f	%f\n\r", runtimeProcessBatch, 1.0 / runtimeProcessBatch);
							fprintf(summaryFile, "Runtime (Total,sec. Hz)	%f	%f\n\r", runtimeTotal, 1.0 / runtimeTotal);
						}
						fprintf(summaryFile, "</SUMMARY>\n\r\n\r"); fflush(summaryFile);
					}
					else
					{
						fprintf(consoleOut, "Error : Could not open OutputFilenameSummary %s\n\r", OutputFilenameSummary.c_str());
						fprintf(summaryFile, "</SUMMARY>\n\r\n\r"); fflush(summaryFile);
					}
					if (OutputFilenameSummary.length())
					{
						fclose(summaryFile);
					}
				}

				if (doTestFrames)
				{
					digHoloFrameSimulatorDestroy(frameBuffer);


				}
				else
				{
					free1D(frameBuffer16);
					free1D(frameBuffer);
				}
				digHoloDestroy(handleIdx);



				return DIGHOLO_ERROR_SUCCESS;
			}
			else
			{
				free1D(frameBuffer16);
				free1D(frameBuffer);
				//digHoloDestroy(handleIdx);
				return DIGHOLO_ERROR_ERROR;
			}
		}
		else
		{
			fprintf(consoleOut, "File could not be opened. %s\n", filename); fflush(consoleOut);
			return DIGHOLO_ERROR_FILENOTFOUND;
		}
	}
	else
	{
		return DIGHOLO_ERROR_NULLPOINTER;
	}
}

int main(int argc, char* argv[]) 
{

	for (int argIdx = 1; argIdx < argc; argIdx++)
	{
		std::cout << (argIdx - 1) << "	" << argv[argIdx] << "\n";
		digHoloRunBatchFromConfigFile(argv[argIdx]);
	}

}