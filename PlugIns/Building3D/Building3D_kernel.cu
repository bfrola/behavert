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
//
// ----------------
// Change log
//
//    03-09 bf: Created
// 31-03-10 bf: Floating behavior modified
// 08-09-10 bf: Added soruce-destination management
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
#include "include\Building3D_kernel.cuh"
#include "Building3D_resources.cu"

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

#define ACCEL_VAR1 10
#define ACCEL_VAR2 15

// --------------------------------------------------------
// --------------------------------------------------------

__global__ void computeFloatingBehavior_kernel()
{
	// -----------------------------------
	// Read data

	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float4 pos = FETCH(oldPos, index);

	// Sampling position
	pos.y -= 2.0f;

	// Get input/output
	float4 oldSmoothedAccel = getInputFeatureCachedElement(oldSmoothedAccelTex, index);
	float4 oldForward = getInputFeatureCachedElement(oldForwardTex, index);
	
	float4* newSmoothedAccel = 
		BehaveRT::getOutputFeature<float4>(dOpenSteerWrapperFields.smoothAccel);
	float4* newForward = 
		BehaveRT::getOutputFeature<float4>(dOpenSteerWrapperFields.forward);
	
	// -----------------------------------
	// Cell information retrieval

	// Future forward position block info (ground)
	float4 futurePosition = 
		pos + oldForward * dEnvGrid3DParams.cellSize.x;
	// Look for future horizontal position
	futurePosition.y = pos.y;

	// Current position block info (ground)
	//pos.y -= dEnvGrid3DParams.cellSize.y;
	int3 gridPos = calcGridPos(pos, dEnvGrid3DParams.worldOrigin, dEnvGrid3DParams.cellSize);
	uint gridHash = calcGridHash(gridPos, dEnvGrid3DParams.gridSize);
	uint blockInfo = getInputFeatureCachedElement(blockInfoTex, gridHash);

	int3 futureGridPos = calcGridPos(futurePosition, dEnvGrid3DParams.worldOrigin, dEnvGrid3DParams.cellSize);
	uint futureGridHash = calcGridHash(futureGridPos, dEnvGrid3DParams.gridSize);
	uint futureBlockInfo = getInputFeatureCachedElement(blockInfoTex, futureGridHash);

	// -----------------------------------
	// Behavior adoption

	// Future cell is filled
	if (futureBlockInfo == 1)
	{
		// Acceleration upward factor
		oldSmoothedAccel.y = ACCEL_VAR1;
		oldSmoothedAccel.y += (oldSmoothedAccel.y * 0.5);

		// Slow speed during climbing
		if (oldForward.w > dOpenSteerWrapperParams.commonMaxSpeed * 0.5)
			oldForward.w *= 0.7;

		// Slow y reduction
		if (oldForward.y < 0.0f)
			// Mod 31-03-10
			oldForward.y -= oldForward.y * 0.001;

		newSmoothedAccel[index] = oldSmoothedAccel;
		newForward[index] = oldForward;

		return;
	} // (futureBlockInfo == 1)


	// Current cell is filled
	if (blockInfo == 1)
	{
		// Stronger upward factor
		oldSmoothedAccel.y = ACCEL_VAR2;
		newSmoothedAccel[index] = oldSmoothedAccel;
		newForward[index] = oldForward;

		return;
	} // (blockInfo == 1)
	
	// Downward factor
	oldSmoothedAccel.y = -ACCEL_VAR2;

	newSmoothedAccel[index] = oldSmoothedAccel;
	newForward[index] = oldForward;
} // computeFloatingBehavior_kernel

// --------------------------------------------------------
// --------------------------------------------------------

__global__ void followTerrain_kernel()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float4 oldSmoothedAccel = getInputFeatureCachedElement(oldSmoothedAccelTex, index);
	float3 oldForward = make_float3(getInputFeatureCachedElement(oldForwardTex, index));

	float3 steering = make_float3(.0f, .0f, .0f);

	float4 pos = FETCH(oldPos, index);
	int3 gridPos = calcGridPos(pos, dEnvGrid3DParams.worldOrigin, dEnvGrid3DParams.cellSize);
	uint gridHash = calcGridHash(gridPos, dEnvGrid3DParams.gridSize);

	float3 flow = make_float3(getInputFeatureCachedElement(flowFieldTex, gridHash));

	if (flow.x != .0f || flow.y != .0f || flow.z != .0f)
	{
		steering = flow * 2;
		OpenSteerWrapper::blendIntoSteeringForce(index, steering);
		return;
	}

	OpenSteerWrapper::blendIntoSteeringForce(index, steering);

} // followTerrain_kernel

// --------------------------------------------------------
// --------------------------------------------------------

__global__ void throwIndividuals()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float4 oldPos = FETCH(oldPos, index);
	float4 oldSmoothedAccel = FETCH(oldSmoothedAccel, index);
	float4 oldForward = FETCH(oldForward, index);

	if (index >= dBuilding3DParams.individualThrowingIndex && 
			index < dBuilding3DParams.individualThrowingIndex + 
				dBuilding3DParams.individualThrowingBlockSize)
	{
		
		oldPos = make_float4(
			dBuilding3DParams.
				individualThrowingPosition + make_float3(
			(index - dBuilding3DParams.individualThrowingIndex), 0, 0), oldPos.w);

		oldForward = make_float4(
			dBuilding3DParams.individualThrowingDirection, 
			oldForward.w);
	}

	declare_output(newPos, float4, dBody3DFields.position);
	newPos[index] = oldPos;
	declare_output(newSmoothedAccel, float4, dOpenSteerWrapperFields.smoothAccel);
	newSmoothedAccel[index] = oldSmoothedAccel;
	declare_output(newForward, float4, dOpenSteerWrapperFields.forward);
	newForward[index] = oldForward;
} // throwIndividuals	


// --------------------------------------------------------
// --------------------------------------------------------

__global__ void manageSourceDestination_kernel()
{
	int individualIndex = BehaveRT::getIndividualIndex();

	// Get current position
	float4 oldPos = 
		getInputFeatureCachedElement(oldPosTex, individualIndex);

	float3 sourcePos = make_float3(1500, 60, 1500);
	float3 destPost = make_float3(800, 20, 500);

	float3 pos2Dest = dBuilding3DParams.individualsDestPos - make_float3(oldPos);

	// Add destination check
	if (length(pos2Dest) < dBuilding3DParams.individualsDestSize)
	{
		oldPos = make_float4(dBuilding3DParams.individualsSourcePos, oldPos.w);
	}

	// Store position and steering force
	BehaveRT::setOutputFeatureElement<float4>(
		dBody3DFields.position, individualIndex, oldPos);
	
	OpenSteerWrapper::blendIntoSteeringForce(individualIndex, 
		normalize(pos2Dest) * dBuilding3DParams.individualsDestAttractionForce);
	
} // manageSourceDestination_kernel



// ----------------------------------------------------

extern "C"
{

	// ////////////////////////////////////////////////////////////////////////
	// Export kernels

	BehaveRT_exportKernel(computeFloatingBehavior_kernel);
	BehaveRT_exportKernel(followTerrain_kernel);
	BehaveRT_exportKernel(throwIndividuals);
	BehaveRT_exportKernel(manageSourceDestination_kernel);
	
	void Building3D::Building3D_beforeKernelCall()
	{
		OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall();
		bind_field_texture(hBuilding3DFields.stateInfo, stateInfoTex);
		bind_field_texture(hBuilding3DFields.blockInfo, blockInfoTex);
		bind_field_texture(hBuilding3DFields.flowField, flowFieldTex);
	}

	void Building3D::Building3D_afterKernelCall()
	{
		OpenSteerWrapper::OpenSteerWrapper_afterKernelCall();
		unbind_field_texture(stateInfoTex);
		unbind_field_texture(blockInfoTex);
		unbind_field_texture(flowFieldTex);
	}

	void Building3D::Building3D_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dBuilding3DFields, &hBuilding3DFields, sizeof(Building3DFields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dBuilding3DParams, &hBuilding3DParams, sizeof(Building3DParams)) );
	}
	
}

