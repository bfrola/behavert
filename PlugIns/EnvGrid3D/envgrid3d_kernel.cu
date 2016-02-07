// ----------------------------------------------------------------------------
// This source file is part of BehaveRT 
// http://isis.dia.unisa.it/projects/behavert/
//
// Copyright (c) 2008-2010 ISISLab - University of Salerno
// Original author: Bernardino Frola <frola@dia.unisa.it>
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// ----------------
// Change log
//
// 01-09 bf: Created
//
// ----------------

#pragma once

#include <cutil.h>
#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

#include "common_resources.cu"

// Same plugIn dependencies
#include "include\envgrid3d_kernel.cuh"
#include "envgrid3d_resources.cu"

// Other plugIn dependencies
#include "..\Body\include\body3d_kernel.cuh"
#include "..\Body\body3d_resources.cu"

__global__ void genCalcHashD()
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float4 p = tex1Dfetch(oldPosTex, index);

    // get address in grid
    int3 gridPos = calcGridPos(p, dEnvGrid3DParams.worldOrigin, dEnvGrid3DParams.cellSize);
    uint gridHash = calcGridHash(gridPos, dEnvGrid3DParams.gridSize);
	
	declare_output(agentHash, uint2, dEnvGrid3DFields.hash);
    // store grid hash and particle index
    agentHash[index] = make_uint2(gridHash, index);
}

// calculate the inverse hash used for host debugging
__global__ void
calcInverseHashD(uint2* agentHash, uint* inverseHash)
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint sortedIndex = FETCH(agentHash, index).y;

	inverseHash[sortedIndex] = index;
}

// --------------------------------------------------------------------------
// Rearrange agent data into sorted order
__global__ void
reorderDataD(
		uint2*  agentHash,  // particle id sorted by hash
		float4* oldArray,
		float4* sortedArray)
{
	const int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	// Read the cached agent hash
	uint sortedIndex = FETCH(agentHash, index).y;

	// Change the position
	sortedArray[index] = FETCH(oldArray, sortedIndex);
}

template <class Type>
__global__ void genericReorderDataD()
{
	const int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
	uint sortedIndex = tex1Dfetch(agentHashTex, index).y;

	declare_input(oldFeature, Type, dEnvGrid3DFields.featureToReorder);
	declare_output(newFeature, Type, dEnvGrid3DFields.featureToReorder);
	//newFeature[index] = oldFeature[sortedIndex];
	oldFeature[sortedIndex] = newFeature[index];
}

__global__ void reorderDataFloat4()
{
	const int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
	uint sortedIndex = tex1Dfetch(agentHashTex, index).y;

	declare_input(oldFeature, float4, dEnvGrid3DFields.featureToReorder);
	declare_output(newFeature, float4, dEnvGrid3DFields.featureToReorder);
	newFeature[index] = oldFeature[sortedIndex];
	//oldFeature[index] = oldFeature[sortedIndex]
		//make_float4(0, 100, 0, 1);
	//newFeature[index];
}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------


extern "C"
{
	// ////////////////////////////////////////////////////////////////////////
	// Bundary functions

	void EnvGrid3D::EnvGrid3D_beforeKernelCall()
	{
		Body3D::Body3D_beforeKernelCall();
		bind_field_texture(hEnvGrid3DFields.hash, agentHashTex);
	}

	void EnvGrid3D::EnvGrid3D_afterKernelCall()
	{
		Body3D::Body3D_afterKernelCall();
		unbind_field_texture(agentHashTex);
	}

	void EnvGrid3D::EnvGrid3D_beforeKernelCallSimple()
	{
		bind_field_texture(hEnvGrid3DFields.hash, agentHashTex);
	}

	void EnvGrid3D::EnvGrid3D_afterKernelCallSimple()
	{
		unbind_field_texture(agentHashTex);
	}

	void EnvGrid3D::EnvGrid3D_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dEnvGrid3DFields, &hEnvGrid3DFields, sizeof(EnvGrid3DFields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dEnvGrid3DParams, &hEnvGrid3DParams, sizeof(EnvGrid3DParams)) );
	}

	// ////////////////////////////////////////////////////////////////////////
	// Generic kernel call

	BehaveRT::genericKernelFuncPointer 
		genericCalcHashDRef() { return &genCalcHashD; }

	BehaveRT::genericKernelFuncPointer 
		reorderDataFloat4Ref() { return &reorderDataFloat4; }

#define	define_genericReoredRef(type)						\
	BehaveRT::genericKernelFuncPointer						\
	genericReorderDataRef_##type() {						\
	return &genericReorderDataD<##type##>; }

	define_genericReoredRef(float4);
	define_genericReoredRef(uint);
	define_genericReoredRef(float);

	// ////////////////////////////////////////////////////////////////////////
	// Custom kernel call
	void EnvGrid3D::
	reorderArray( 
			uint* agentHash, 
			int agentHashElementSize,
			uint vboReadArray,
			uint vboWriteArray,
			float4* oldArray,
			float4* sortedArray,
			int oldArrayElementSize,
			int numBodies,
			int blockSize)
	{
		int numThreads, numBlocks;
		
		computeGridSize(
			numBodies, blockSize, 
			numBlocks, numThreads);	

		if (vboReadArray >= 0)
		{
			CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&oldArray, vboReadArray));
			CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&sortedArray, vboWriteArray));
		}

		CUDA_SAFE_CALL(cudaBindTexture(0, agentHashTex, agentHash, 
			numBodies * agentHashElementSize));

		//printf("%d %d %d\n", numBodies, agentHashElementSize, numBodies * agentHashElementSize);
		CUDA_SAFE_CALL(cudaBindTexture(0, oldArrayTex, oldArray, 
			numBodies * oldArrayElementSize));

		//printf("%d %d %d\n", numBodies, oldArrayElementSize, numBodies * oldArrayElementSize);

		reorderDataD<<< numBlocks, numThreads >>>(
			(uint2 *)  agentHash,
			(float4 *) oldArray,
			(float4 *) sortedArray);
			
		if (vboReadArray >= 0)
		{
			CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&oldArray, vboReadArray));
			CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&sortedArray, vboWriteArray));
		}

		CUDA_SAFE_CALL(cudaUnbindTexture(agentHashTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(oldArrayTex));

	}
}
