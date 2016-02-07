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
// 01-09 bf: Created
// 09-10 bf: Generalized macros and kernels (moveAway...) 
//
// ----------------

#pragma once

#include <cutil.h>
#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"

#include "common_resources.cu"

#include "include\OpenSteerWrapper_kernel.cuh"
#include "OpenSteerWrapper_resources.cu"

// Other plugIn dependencies
#include "..\EnvGrid3D\include\envgrid3d_kernel.cuh"
#include "..\EnvGrid3D\envgrid3d_resources.cu"

#include "..\Proximity3D\include\Proximity3D_kernel.cuh"
#include "..\Proximity3D\Proximity3D_resources.cu"

#include "..\Body\include\body3d_kernel.cuh"
#include "..\Body\body3d_resources.cu"


// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

__global__ void genericCohesionD()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	float3 myPos, myForward, steering;
	uint neighNum;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	myPos = make_float3(tex1Dfetch(oldPosTex, index));			
	myForward =  make_float3(tex1Dfetch(oldForwardTex, index));
	Proximity3D::getNeighborsList(index, neighNum, neighList);
	
	steering = OpenSteerWrapper::calcCohesion(
			myPos, myForward, 
			neighNum, neighList) *
		dOpenSteerWrapperParams.cohesionParams.x;

	OpenSteerWrapper::blendIntoSteeringForce(index, steering);

} //genericCohesionD


// ------------------------------------------------------------------------------

__global__ void genericSeparationD()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	float3 myPos, myForward, steering;
	uint neighNum;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	myPos = make_float3(tex1Dfetch(oldPosTex, index));			
	myForward =  make_float3(tex1Dfetch(oldForwardTex, index));
	Proximity3D::getNeighborsList(index, neighNum, neighList);
	
	steering = OpenSteerWrapper::calcSeparation(
			myPos, myForward, 
			neighNum, neighList) *
		dOpenSteerWrapperParams.separationParams.x;

	OpenSteerWrapper::blendIntoSteeringForce(index, steering);

}//genericSeparationD


// ------------------------------------------------------------------------------

__global__ void steerToAvoidNeighbors()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	float3 myPos, myForward, steering;
	uint neighNum;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	myPos = make_float3(tex1Dfetch(oldPosTex, index));			
	myForward =  make_float3(tex1Dfetch(oldForwardTex, index));
	Proximity3D::getNeighborsList(index, neighNum, neighList);
	
	steering = OpenSteerWrapper::calcSeparation(
			myPos, myForward, 
			neighNum, neighList) *
		dOpenSteerWrapperParams.separationParams.x;

	// ------------------
	// NeighNum balanced steering

	const int neighNumThreshold = 16;
	const float steerForceMultipler = 0.5f;

	if (neighNum > neighNumThreshold)
		steering *= (neighNum - neighNumThreshold) * steerForceMultipler;

	// ------------------

	float mySpeed = tex1Dfetch(oldForwardTex, index).w;
	declare_output(newForward, float4, dOpenSteerWrapperFields.forward);
	if (dot(steering, myForward) < -0.8)
	{
		mySpeed *= 0.9;
	}
	else if (dot(steering, myForward) > 0.5)
	{
		mySpeed *= 1.02;
	}

	newForward[index] = make_float4(myForward, mySpeed);

	OpenSteerWrapper::blendIntoSteeringForce(index, steering);


}//steerToAvoidNeighbors

// ------------------------------------------------------------------------------

__global__ void genericAlignmentD()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	//uint2 sortedData;
	float3 myPos, myForward, steering;
	uint neighNum;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	myPos = make_float3(tex1Dfetch(oldPosTex, index));		
	myForward =  make_float3(tex1Dfetch(oldForwardTex, index));
	Proximity3D::getNeighborsList(index, neighNum, neighList);

	steering = OpenSteerWrapper::calcAlignment(
			myPos, myForward, 
			neighNum, neighList) *
		dOpenSteerWrapperParams.alignmentParams.x;

	OpenSteerWrapper::blendIntoSteeringForce(index, steering);

}//genericAlignmentD


// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

__global__ void moveAwayFromTarget_kernel()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	float3 pos = make_float3(FETCH(oldPos, index));
	float4 oldSmoothedAccel = tex1Dfetch(oldSmoothedAccelTex, index);
	declare_output(newAccel, float4, dOpenSteerWrapperFields.smoothAccel);

	float3 steering = pos - dOpenSteerWrapperParams.avoidBase;
	float distance = length(steering);

	// dist < radius						==> steering + acceleration
	// dist > radius && dist < radius * 2	==> steering
	// dist > radius * 2					==> none

	if (distance < dOpenSteerWrapperParams.avoidBaseParams.z)
	{
		float3 newAccelValue = (steering + make_float3(oldSmoothedAccel)) * 
			dOpenSteerWrapperParams.avoidBaseParams.y;
		newAccel[index] = make_float4(newAccelValue, oldSmoothedAccel.w);
	} 
	else if (distance > dOpenSteerWrapperParams.avoidBaseParams.z * 2)
		steering = make_float3(0, 0, 0);
	
	newAccel[index] = oldSmoothedAccel;	
	OpenSteerWrapper::blendIntoSteeringForce(index, 
		dOpenSteerWrapperParams.avoidBaseParams.x * steering / distance); 
}


// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

template<bool useSfericBoundings>
__global__ void genericSeekingWorldCenterD()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	float3 myPos = make_float3(tex1Dfetch(oldPosTex, index));

	if (useSfericBoundings)	// Compile-time condition check
	{
		if (length(myPos - dEnvGrid3DParams.worldCenter) < dEnvGrid3DParams.worldRadius.x)
		{
			// Inalterated steerForce
			OpenSteerWrapper::blendIntoSteeringForce(index, make_float3(0, 0, 0));
			return;
		}
	}
	else					// Compile-time condition check
	{
		if ((myPos.x > dEnvGrid3DParams.worldCenter.x - dEnvGrid3DParams.worldRadius.x) && 
			(myPos.x < dEnvGrid3DParams.worldCenter.x + dEnvGrid3DParams.worldRadius.x) &&
			(myPos.y > dEnvGrid3DParams.worldCenter.y - dEnvGrid3DParams.worldRadius.y) && 
			(myPos.y < dEnvGrid3DParams.worldCenter.y + dEnvGrid3DParams.worldRadius.y) &&
			(myPos.z > dEnvGrid3DParams.worldCenter.z - dEnvGrid3DParams.worldRadius.z) && 
			(myPos.z < dEnvGrid3DParams.worldCenter.z + dEnvGrid3DParams.worldRadius.z))
		{
			// Inalterated steerForce
			OpenSteerWrapper::blendIntoSteeringForce(index, make_float3(0, 0, 0));
			return;
		}
	}
	

	float mySpeed = tex1Dfetch(oldForwardTex, index).w;
	float3 myForward = make_float3(tex1Dfetch(oldForwardTex, index));

	float3 target = dEnvGrid3DParams.worldCenter;

	float3 steering = OpenSteerWrapper::xxxsteerForSeek(
		myPos, mySpeed, myForward, target, dOpenSteerWrapperParams.commonMaxSpeed);

	OpenSteerWrapper::blendIntoSteeringForce(index, steering);
} // genericSeekingWorldCenterD

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

__global__ void genericApplySteeringForceD()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
	float3 myPos = make_float3(tex1Dfetch(oldPosTex, index));
	float3 myForward = make_float3(tex1Dfetch(oldForwardTex, index));
	
	// Read from the w element of forward array
	float mySpeed =tex1Dfetch(oldForwardTex, index).w;

	float3 mySmoothedAccel = make_float3(tex1Dfetch(oldSmoothedAccelTex, index));
	float3 myForce =make_float3(tex1Dfetch(oldForceTex, index));

	uint sortedIndex = tex1Dfetch(agentHashTex, index).y;
	
	// Outputs
	declare_output(newPos, float4, dBody3DFields.position);	
	declare_output(newSmoothedAccel, float4, dOpenSteerWrapperFields.smoothAccel);
	declare_output(newForward, float4, dOpenSteerWrapperFields.forward);

	// The agent does not move
	if (mySpeed < 0)
	{
		//declare_output(oldPosWritePosition, float4, dBody3DFields.position);	
		// Restore old value
		newPos[sortedIndex] = FETCH(oldPos, index);
		newSmoothedAccel[sortedIndex] = FETCH(oldSmoothedAccel, index);
		newForward[sortedIndex] = FETCH(oldForce, index);
		return;
	}

	if (dBody3DParams.use2DProjection)
	{
		myForce.y = 0;
	}

	// --

	OpenSteerWrapper::applySteeringForceSingle(myForce, 
		dOpenSteerWrapperDynParams.elapsedTime, 
		dOpenSteerWrapperParams.commonMaxForce, 
		dOpenSteerWrapperParams.commonMaxSpeed, 
		dOpenSteerWrapperParams.commonMass, 
		mySpeed, myForward, myPos,
		mySmoothedAccel);
	
	if (dBody3DParams.use2DProjection)
	{
		myPos.y = 0.5;
	} 

	
	// -----------
	// DEBUG
	//float forceDotForward = dot(myForce, myForward);
	//if (sortedIndex == 0)
	//	cuPrintf("OSW: %f\n", forceDotForward);
	// -----------

	newPos[sortedIndex] = 
		make_float4(myPos, 1);

	newSmoothedAccel[sortedIndex] = 
		make_float4(mySmoothedAccel, FETCH(oldSmoothedAccel, index).w);

	newForward[sortedIndex] = 
		make_float4(myForward, mySpeed);

}

extern "C"
{
	// ////////////////////////////////////////////////////////////////////////
	// Global vars

	// ////////////////////////////////////////////////////////////////////////
	// Export kernels
	BehaveRT::genericKernelFuncPointer 
		genericSeparationDRef() { return &genericSeparationD; }

	BehaveRT::genericKernelFuncPointer 
		genericCohesionDRef() { return &genericCohesionD; }

	BehaveRT::genericKernelFuncPointer 
		genericAlignmentDRef() { return &genericAlignmentD; }

	BehaveRT::genericKernelFuncPointer 
		steerToAvoidNeighborsRef() { return &steerToAvoidNeighbors; }

	// TODO duplicate with <true> value of template
	BehaveRT::genericKernelFuncPointer 
		genericSeekingWorldCenterDRef() { return &genericSeekingWorldCenterD<false>; }

	BehaveRT::genericKernelFuncPointer 
		genericApplySteeringForceDRef() { return &genericApplySteeringForceD; }

	BehaveRT_exportKernel(moveAwayFromTarget_kernel);

	
	void OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall()
	{
		Proximity3D::Proximity3D_beforeKernelCall();

		bind_field_texture(hOpenSteerWrapperFields.forward, oldForwardTex);
		bind_field_texture(hOpenSteerWrapperFields.smoothAccel, oldSmoothedAccelTex);
		bind_field_texture(hOpenSteerWrapperFields.steerForce, oldForceTex);

	}

	void OpenSteerWrapper::OpenSteerWrapper_afterKernelCall()
	{
		Proximity3D::Proximity3D_afterKernelCall();
		
		unbind_field_texture(oldForwardTex);
		unbind_field_texture(oldSmoothedAccelTex);
		unbind_field_texture(oldForceTex);	

	}

	void OpenSteerWrapper::OpenSteerWrapper_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dOpenSteerWrapperFields, &hOpenSteerWrapperFields, sizeof(OpenSteerWrapperFields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dOpenSteerWrapperParams, &hOpenSteerWrapperParams, sizeof(OpenSteerWrapperParams)) );
	}

	void OpenSteerWrapper::OpenSteerWrapper_copyDynParams()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dOpenSteerWrapperDynParams, &hOpenSteerWrapperDynParams, sizeof(OpenSteerWrapperDynParams)) );
	}
}