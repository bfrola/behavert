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
// Auto-generated code START
[PDependenciesPaths]
// Auto-generated code END

// ////////////////////////////////////////////////////////
// Kernels

__global__ void [PName]Example_kernel()
{
	// Get individual index
	int individualIndex = BehaveRT::getIndividualIndex();

	// Example of feature chaced access
	// cacheName is the texture name
	//   pick it up from [PName]_resources.cu
	//
	// float4 element = 
	//	getInputFeatureCachedElement(cacheName, individualIndex);
	
	// OR, Example of feature access without cache
	// float4* feature = 
	//	getInputFeature(dBody3DFields.position);
	// float4 featureElement = feature[ individualIndex ];

	// Device funcs from [PName]_resources.cu
	[PName]::deviceFuncExample( individualIndex );

	// Example of feature storing
	//
	// BehaveRT::setOutputFeatureElement<float4>(
	//	dBody3DFields.position, individualIndex, featureElement );

} // [PName]Example_kernel

// -------------------------------------------------
// -------------------------------------------------

// ////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////

extern "C"
{
	// ////////////////////////////////////////////////////////////////////////
	// Bundary functions
	void [PName]::[PName]_beforeKernelCall()
	{
		// Other plugIns data binding
		// Auto-generated code START
		[PFeaturesDependencyBefoceCall]
		// Auto-generated code END

		// bind_field_texture( h[PName]Fields.featureName, cacheName );
		//
		// Where:
		// cacheName is the texture name
		//   pick it up from [PName]_resources.cu
		// featureName is the name of the fiature on whiche enable caching
		//   pick it up from [PName]_kernel.cuh

		// Auto-generated code START
		[PFeaturesTexturesBinding]
		// Auto-generated code END
	}

	void [PName]::[PName]_afterKernelCall()
	{
		// Other plugIns data unbinding
		// Auto-generated code START
		[PFeaturesDependencyAfterCall]
		// Auto-generated code END

		// unbind_field_texture( cacheName );
		//
		// Where:
		// cacheName is the texture name
		//   pick it up from [PName]_resources.cu

		// Auto-generated code START
		[PFeaturesTexturesUnbinding]
		// Auto-generated code END
	}

	void [PName]::[PName]_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d[PName]Fields, &h[PName]Fields, sizeof([PName]Fields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(d[PName]Params, &h[PName]Params, sizeof([PName]Params)) );
	}

	// ////////////////////////////////////////////////////////////////////////
	// Generic kernel calls

	BehaveRT_exportKernel( [PName]Example_kernel );

	// ////////////////////////////////////////////////////////////////////////
	// Custom kernel calls

}



