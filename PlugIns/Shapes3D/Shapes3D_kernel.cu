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

#include "common_resources.cu"

// Same plugIn dependencies
#include "include\Shapes3D_kernel.cuh"
#include "Shapes3D_resources.cu"

// Other plugIn dependencies
#include "..\EnvGrid3D\include\envgrid3d_kernel.cuh"
#include "..\EnvGrid3D\envgrid3d_resources.cu"

#include "..\Proximity3D\include\Proximity3D_kernel.cuh"
#include "..\Proximity3D\Proximity3D_resources.cu"

#include "..\Body\include\body3d_kernel.cuh"
#include "..\Body\body3d_resources.cu"

#include "..\OpenSteerWrapper\include\OpenSteerWrapper_kernel.cuh"
#include "..\OpenSteerWrapper\OpenSteerWrapper_resources.cu"


// ----------------------------------------------------

__global__ void moveTowardsTargetD()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint2 sortedData = tex1Dfetch(agentHashTex, index);
	int shiftedIndex = sortedData.y + dShapes3DParams.indexShift;
	// Primitive modulus
	if (shiftedIndex > dBody3DParams.numBodies)
		shiftedIndex -= dBody3DParams.numBodies;

	// Get desired target and current pos
	float4 finalTarget = tex1Dfetch(finalTargetTex, shiftedIndex);
	float3 pos = make_float3(FETCH(oldPos, index));

	// Steering is an attractive vector
	float3 steering = 
		make_float3(finalTarget) + dShapes3DParams.targetBase - pos;
	
	// Store the steering force
	OpenSteerWrapper::blendIntoSteeringForce(index, normalize(steering) * 12 * finalTarget.w); 

	//declare_output(newPos, float4, dBody3DFields.position);
	//newPos[index] = make_float4(make_float3(finalTarget) + dShapes3DParams.targetBase, 1.0f);
		 
}


// ----------------------------------------------------

extern "C"
{

	// ////////////////////////////////////////////////////////////////////////
	// Export kernels
	BehaveRT::genericKernelFuncPointer 
		moveTowardsTargetDRef() { return &moveTowardsTargetD; }

	
	
	void Shapes3D::Shapes3D_beforeKernelCall()
	{
		OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall();
		bind_field_texture(hShapes3DFields.finalTarget, finalTargetTex);
	}

	void Shapes3D::Shapes3D_afterKernelCall()
	{
		OpenSteerWrapper::OpenSteerWrapper_afterKernelCall();
		unbind_field_texture(finalTargetTex);
	}

	void Shapes3D::Shapes3D_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dShapes3DFields, &hShapes3DFields, sizeof(Shapes3DFields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dShapes3DParams, &hShapes3DParams, sizeof(Shapes3DParams)) );
	}
	
}

