#pragma once

#include <cutil.h>
#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"

#include "common_resources.cu"

// Same plugIn dependencies
#include "include\[PName]_kernel.cuh"
#include "[PName]_resources.cu"

// Other plugIn dependencies
[PDependenciesPaths]

// ////////////////////////////////////////////////////////
// Kernels
__global__ void [PName]_kernelExample()
{
	
}

// ////////////////////////////////////////////////////////

extern "C"
{
	// ////////////////////////////////////////////////////////////////////////
	// Bundary functions
	void [PName]::[PName]_beforeKernelCall()
	{
		bind_field_texture(h[PName]Fields.position, oldPosTex);
	}

	void [PName]::[PName]_afterKernelCall()
	{
		unbind_field_texture(oldPosTex);
	}

	void [PName]::[PName]_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d[PName]Fields, &h[PName]Fields, sizeof([PName]Fields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d[PName]Params, &h[PName]Params, sizeof([PName]Params)) );
	}

	// ////////////////////////////////////////////////////////////////////////
	// Generic kernel calls

	genericKernelFuncPointer 
		[PName]_kernelExampleRef() { return &[PName]_kernelExample; }

	// ////////////////////////////////////////////////////////////////////////
	// Custom kernel calls

}



