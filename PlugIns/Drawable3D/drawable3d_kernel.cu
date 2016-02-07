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
#include "include\drawable3d_kernel.cuh"
#include "drawable3d_resources.cu"

// Other plugIn dependencies
#include "..\EnvGrid3D\include\envgrid3d_kernel.cuh"
#include "..\EnvGrid3D\envgrid3d_resources.cu"

#include "..\Proximity3D\include\Proximity3D_kernel.cuh"
#include "..\Proximity3D\Proximity3D_resources.cu"

#include "..\Body\include\body3d_kernel.cuh"
#include "..\Body\body3d_resources.cu"

#include "..\OpenSteerWrapper\include\OpenSteerWrapper_kernel.cuh"
#include "..\OpenSteerWrapper\OpenSteerWrapper_resources.cu"

__global__ void extractColorFromNeighborhoodD() 
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	const int neighIndexBase = 
		__mul24(index, dProximity3DParams.numNeighWordsPerAgent);

	uint4 neighWord = tex1Dfetch(oldNeighListTex, neighIndexBase);

	//BehaveRT::uint neighNum = neighWord.x;

	uint2 sortedData = tex1Dfetch(agentHashTex, index);

	declare_output(color, float4, dDrawable3DFields.color);

	float colorFactor = ((float)neighWord.x) / dProximity3DParams.maxNeighbors;

	// Write the result in an ordered fashion
	float4 colorFinal =
		make_float4(
			dDrawable3DParams.colorBase.x + colorFactor , 
			dDrawable3DParams.colorBase.y + 0.3 + 1 - colorFactor, 
			dDrawable3DParams.colorBase.z + colorFactor * (1 - colorFactor), 
			1);

	if (dDrawable3DParams.useCUDAGeometry)
	{
		color[sortedData.y * dDrawable3DParams.numVertexes] = colorFinal - make_float4(0.5, 0.5, 0.5, 0);
		color[sortedData.y * dDrawable3DParams.numVertexes + 1] = colorFinal;
		color[sortedData.y * dDrawable3DParams.numVertexes + 2] = colorFinal;
	}
	else
		color[sortedData.y] = colorFinal;
}

__global__ void smoothColorD() 
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	float4 targetColor = tex1Dfetch(colorTex, index);
	float4 oldSmoothedColor = tex1Dfetch(smoothedColorTex, index);

	uint2 sortedData = tex1Dfetch(agentHashTex, index);

	if (calcDist4(targetColor, oldSmoothedColor) < 0.1)
		return;

	declare_output(smoothedColor, float4, dDrawable3DFields.smoothedColor);
	smoothedColor[index] = 
		oldSmoothedColor + (targetColor - oldSmoothedColor) / 150;

	declare_output(newTargetColor, float4, dDrawable3DFields.color);
	newTargetColor[index] = targetColor;
}

__global__ void createGeometryD()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	uint2 sortedData = tex1Dfetch(agentHashTex, index);

	float4 pos = tex1Dfetch(oldPosTex, index);
	float4 forward = tex1Dfetch(oldForwardTex, index);

	declare_output(geometry, float4, dDrawable3DFields.geometry);
	
	geometry[sortedData.y * 3] = pos;

	geometry[sortedData.y * 3 + 1] = 
		make_float4(
			make_float3(pos) - 
			perpendicularComponent(make_float3(forward) * dBody3DParams.commonRadius * 2 + 
			make_float3(forward) * 
			dBody3DParams.commonRadius,
				make_float3(0, 1, 0)), 
			pos.w);

	
	geometry[sortedData.y * 3 + 2] = 
		make_float4(
			make_float3(pos) + 
			make_float3(forward) * 
			dBody3DParams.commonRadius * 4, pos.w);
	
}


// ----------------------------------------------------
// ----------------------------------------------------

__device__ float distanceFromLine (const float3 point,
                                   const float3 lineOrigin,
                                   const float3 lineUnitTangent)
{
	const float3 offset = point - lineOrigin;
	const float3 perp = perpendicularComponent (offset, lineUnitTangent);
    return length(perp);
}

// ----------------------------------------------------
// ----------------------------------------------------

__global__ void computeMouseDistance_kernel()
{
	int individualIndex = BehaveRT::getIndividualIndex();
	
	float3 position = 
		make_float3(
			getInputFeatureCachedElement(oldPosTex, individualIndex));

	float distance = distanceFromLine(
		position, 
		dDrawable3DParams.cameraPosition,
		dDrawable3DParams.mouseDirection);

	//uint sortedIndex = FETCH(agentHash, individualIndex).y;

	// Store the distance into the output array
	BehaveRT::setOutputFeatureElement<float>(
		dDrawable3DFields.mouseDistance, 
		individualIndex, distance);
}

// ----------------------------------------------------
// ----------------------------------------------------

__global__ void dummyKernel()
{
	// Do nothing
}

// ----------------------------------------------------

extern "C"
{
	// ////////////////////////////////////////////////////////////////////////
	// Global vars
	

	// ////////////////////////////////////////////////////////////////////////
	// Export kernels
	BehaveRT::genericKernelFuncPointer 
		extractColorFromNeighborhoodDRef() { return &extractColorFromNeighborhoodD; }

	BehaveRT::genericKernelFuncPointer 
		smoothColorDRef() { return &smoothColorD; }

	BehaveRT::genericKernelFuncPointer 
		createGeometryDRef() { return &createGeometryD; }

	BehaveRT::genericKernelFuncPointer 
		dummyKernelRef() { return &dummyKernel; }

	BehaveRT_exportKernel(computeMouseDistance_kernel);
	
	void Drawable3D::Drawable3D_beforeKernelCall()
	{
		OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall();

		bind_field_texture(hDrawable3DFields.color, colorTex);
		bind_field_texture(hDrawable3DFields.smoothedColor, smoothedColorTex);
	}

	void Drawable3D::Drawable3D_afterKernelCall()
	{
		OpenSteerWrapper::OpenSteerWrapper_afterKernelCall();

		unbind_field_texture(colorTex);
		unbind_field_texture(smoothedColorTex);
	}

	void Drawable3D::Drawable3D_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dDrawable3DFields, &hDrawable3DFields, sizeof(Drawable3DFields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dDrawable3DParams, &hDrawable3DParams, sizeof(Drawable3DParams)) );
	}

	
}

