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
// 03-09 bf: Created
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
#include "include\Schooling_kernel.cuh"
#include "Schooling_resources.cu"

// Other plugIn dependencies
#include "..\Body\include\Body3D_kernel.cuh"
#include "..\Body\Body3D_resources.cu"

#include "..\EnvGrid3D\include\EnvGrid3D_kernel.cuh"
#include "..\EnvGrid3D\EnvGrid3D_resources.cu"

#include "..\Proximity3D\include\Proximity3D_kernel.cuh"
#include "..\Proximity3D\Proximity3D_resources.cu"


#include "..\OpenSteerWrapper\include\OpenSteerWrapper_kernel.cuh"
#include "..\OpenSteerWrapper\OpenSteerWrapper_resources.cu"

// ////////////////////////////////////////////////////////
// Kernels

__global__ void Schooling_repulsion ()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	float3 myPos, myForward;
	uint neighNum;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	myPos = make_float3(tex1Dfetch(oldPosTex, index));			
	Proximity3D::getNeighborsList(index, neighNum, neighList);

	float3 newForwardElement = make_float3(.0f, .0f, .0f);
	
	// Neighs list does not contain the idividual that is executing the behavior
	for (int i = 0; i < neighNum; i ++)
	{
		uint otherIndex = neighList[i];
		float3 otherPos = make_float3((float4)FETCH(oldPos, otherIndex));

		float dist = calcDist3(myPos, otherPos);

		if (dist < dSchoolingParams.r_r)
		{
			float3 offset = otherPos - myPos;
			newForwardElement += offset / length(offset);
		}
	}

	if (length(newForwardElement) > 0)
		newForwardElement = -normalize(newForwardElement);

	declare_output(newForce, float4, dOpenSteerWrapperFields.steerForce);
	newForce[index] = make_float4(newForwardElement);
}

// -----------------------------------------------------------------------
// -----------------------------------------------------------------------

__global__ void Schooling_attractionOrientation ()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	float3 myPos, myForward, steering;
	uint neighNum;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	myPos = make_float3(tex1Dfetch(oldPosTex, index));			
	myForward =  make_float3(tex1Dfetch(oldForwardTex, index));
	Proximity3D::getNeighborsList(index, neighNum, neighList);

	// Gets the steer force
	float3 newForwardElement = 
		make_float3(tex1Dfetch(oldForceTex, index));

	// Normalize forward
	if (length(newForwardElement) > 0)
	{
		declare_output(newForce, float4, dOpenSteerWrapperFields.steerForce);
		newForce[index] = make_float4(newForwardElement);
		return;
	}

	float3 orientation = make_float3(.0f, .0f, .0f);
	float3 attraction = make_float3(.0f, .0f, .0f);

	orientation += myForward / length(myForward);

	for (int i = 0; i < neighNum; i ++)
	{
		uint otherIndex = neighList[i];
		float3 otherPos = make_float3((float4)FETCH(oldPos, otherIndex));
		float3 otherForward = make_float3(tex1Dfetch(oldForwardTex, otherIndex));

		checkForAttractionOrientation(
			myPos, otherPos, myForward, otherForward, 
			attraction, orientation);
	}

	newForwardElement = 
		attraction * dSchoolingParams.w_a + 
		orientation * dSchoolingParams.w_o;

	if (length(newForwardElement) > 0)
		newForwardElement = normalize(newForwardElement);

	declare_output(newForce, float4, dOpenSteerWrapperFields.steerForce);
	newForce[index] = make_float4(newForwardElement);
}

// -----------------------------------------------------------------------
// -----------------------------------------------------------------------

__global__ void Schooling_move ()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	float3 myPos = make_float3(tex1Dfetch(oldPosTex, index));	
	
	// Gets the steer force
	float3 newForwardElement = 
		make_float3(tex1Dfetch(oldForceTex, index));

	if (!((myPos.x > dEnvGrid3DParams.worldCenter.x - dEnvGrid3DParams.worldRadius.x) && 
		(myPos.x < dEnvGrid3DParams.worldCenter.x + dEnvGrid3DParams.worldRadius.x) &&
		(myPos.y > dEnvGrid3DParams.worldCenter.y - dEnvGrid3DParams.worldRadius.y) && 
		(myPos.y < dEnvGrid3DParams.worldCenter.y + dEnvGrid3DParams.worldRadius.y) &&
		(myPos.z > dEnvGrid3DParams.worldCenter.z - dEnvGrid3DParams.worldRadius.z) && 
		(myPos.z < dEnvGrid3DParams.worldCenter.z + dEnvGrid3DParams.worldRadius.z)))

	{
		newForwardElement -= myPos / length(myPos);
		newForwardElement = normalize(newForwardElement);
	}
	
	// Add ramdom factor
	if (dSchoolingParams.randomDeviationWeight > 0.0)
	{
		float3 randomDirection = 
			make_float3(tex1Dfetch(randomValueTex, index)) * dSchoolingParams.randomDeviationWeight * 2 - 
			make_float3(1.0, 1.0, 1.0) * dSchoolingParams.randomDeviationWeight;

		newForwardElement = normalize(newForwardElement + randomDirection);
	}

	float3 myForward =  
		make_float3(tex1Dfetch(oldForwardTex, index));

	
	// Applies it to forward direction
	myForward = vecLimitDeviationAngleUtility (
		true, newForwardElement, dSchoolingParams.theta, myForward);

	// Calculates new position
	myPos +=  dSchoolingParams.s * myForward * 0.2;

	/*if (myPos.x < dEnvGrid3DParams.worldCenter.x  - dEnvGrid3DParams.worldRadius.x)
		myPos.x = dEnvGrid3DParams.worldCenter.x  + dEnvGrid3DParams.worldRadius.x;
	else if (myPos.x > dEnvGrid3DParams.worldCenter.x  + dEnvGrid3DParams.worldRadius.x)
		myPos.x = dEnvGrid3DParams.worldCenter.x - dEnvGrid3DParams.worldRadius.x;

	if (myPos.y < dEnvGrid3DParams.worldCenter.y  - dEnvGrid3DParams.worldRadius.y)
		myPos.y = dEnvGrid3DParams.worldCenter.y  + dEnvGrid3DParams.worldRadius.y;
	else if (myPos.y > dEnvGrid3DParams.worldCenter.y  + dEnvGrid3DParams.worldRadius.y)
		myPos.y = dEnvGrid3DParams.worldCenter.y - dEnvGrid3DParams.worldRadius.y;

	if (myPos.z < dEnvGrid3DParams.worldCenter.z  - dEnvGrid3DParams.worldRadius.z)
		myPos.z = dEnvGrid3DParams.worldCenter.z  + dEnvGrid3DParams.worldRadius.z;
	else if (myPos.z > dEnvGrid3DParams.worldCenter.z  + dEnvGrid3DParams.worldRadius.z)
		myPos.z = dEnvGrid3DParams.worldCenter.z - dEnvGrid3DParams.worldRadius.z;

	*/


	if (dBody3DParams.use2DProjection)
	{
		myPos.y = 0.5f;
	} 

	// Store new position and forward
	uint sortedIndex = tex1Dfetch(agentHashTex, index).y;
	declare_output(newPos, float4, dBody3DFields.position);
	newPos[sortedIndex] = 
		make_float4(myPos, 1);

	declare_output(newForward, float4, dOpenSteerWrapperFields.forward);
	newForward[sortedIndex] = 
		make_float4(myForward, dSchoolingParams.s);
}

// -----------------------------------------------------------------------
// -----------------------------------------------------------------------

__global__ void Schooling_animateSchool ()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	float3 myPos, myForward;
	uint neighNum;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	myPos = make_float3(tex1Dfetch(oldPosTex, index));			
	myForward =  make_float3(tex1Dfetch(oldForwardTex, index));
	Proximity3D::getNeighborsList(index, neighNum, neighList);

	uint foundNeighs = 0;
	float3 newForwardElement = make_float3(.0f, .0f, .0f);
	
	// Neighs list does not contain the idividual that is executing the behavior
	for (int i = 0; i < neighNum; i ++)
	{
		uint otherIndex = neighList[i];
		float3 otherPos = make_float3((float4)FETCH(oldPos, otherIndex));

		float dist = calcDist3(myPos, otherPos);

		if (dist < dSchoolingParams.r_r)
		{
			float3 offset = otherPos - myPos;
			newForwardElement += offset / length(offset);
			foundNeighs ++;
		}
	}

	newForwardElement = -newForwardElement;

	if (foundNeighs == 0)
	{
		float3 orientation = make_float3(.0f, .0f, .0f);
		float3 attraction = make_float3(.0f, .0f, .0f);

		orientation += myForward / length(myForward);

		for (int i = 0; i < neighNum; i ++)
		{
			uint otherIndex = neighList[i];
			float3 otherPos = make_float3((float4)FETCH(oldPos, otherIndex));
			float3 otherForward = make_float3(tex1Dfetch(oldForwardTex, otherIndex));

			float dist = calcDist3(myPos, otherPos);

			if (dist < dSchoolingParams.r_p)
			{
				float3 offset =  otherPos - myPos;
				attraction += offset / length(offset);

				orientation += otherForward / length(otherForward);
				foundNeighs ++;
			}
		}

		newForwardElement = 
			attraction * dSchoolingParams.w_a + 
			orientation * dSchoolingParams.w_o;
	}

	//newForwardElement += myForward / length(myForward);

	if (dBody3DParams.use2DProjection)
	{
		newForwardElement.y = 0.0f;
	} 	

	// Handle boundings 
	if (!((myPos.x > dEnvGrid3DParams.worldCenter.x - dEnvGrid3DParams.worldRadius.x) && 
		(myPos.x < dEnvGrid3DParams.worldCenter.x + dEnvGrid3DParams.worldRadius.x) &&
		(myPos.y > dEnvGrid3DParams.worldCenter.y - dEnvGrid3DParams.worldRadius.y) && 
		(myPos.y < dEnvGrid3DParams.worldCenter.y + dEnvGrid3DParams.worldRadius.y) &&
		(myPos.z > dEnvGrid3DParams.worldCenter.z - dEnvGrid3DParams.worldRadius.z) && 
		(myPos.z < dEnvGrid3DParams.worldCenter.z + dEnvGrid3DParams.worldRadius.z)))

	{
		newForwardElement -= myPos / length(myPos);
	}

	// Normalize forward
	if (length(newForwardElement) > 0)
		newForwardElement = normalize(newForwardElement);
		
	myForward = vecLimitDeviationAngleUtility (
		true, newForwardElement, dSchoolingParams.theta, myForward);

	// Calc new position
	myPos +=  dSchoolingParams.s * myForward * 0.1;

	if (dBody3DParams.use2DProjection)
	{
		myPos.y = 0.5f;
	} 

	// Store new position and forward
	uint sortedIndex = tex1Dfetch(agentHashTex, index).y;
	declare_output(newPos, float4, dBody3DFields.position);
	newPos[sortedIndex] = 
		make_float4(myPos, 1);

	declare_output(newForward, float4, dOpenSteerWrapperFields.forward);
	newForward[sortedIndex] = 
		make_float4(myForward, dSchoolingParams.s);
}

// ////////////////////////////////////////////////////////

extern "C"
{
	// ////////////////////////////////////////////////////////////////////////
	// Bundary functions
	void Schooling::Schooling_beforeKernelCall()
	{
		OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall();
		bind_field_texture(hSchoolingFields.randomValue, randomValueTex);
	}

	void Schooling::Schooling_afterKernelCall()
	{
		OpenSteerWrapper::OpenSteerWrapper_afterKernelCall();
		unbind_field_texture(randomValueTex);
	}

	void Schooling::Schooling_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dSchoolingFields, &hSchoolingFields, sizeof(SchoolingFields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dSchoolingParams, &hSchoolingParams, sizeof(SchoolingParams)) );
	}

	// ////////////////////////////////////////////////////////////////////////
	// Generic kernel calls

	BehaveRT::genericKernelFuncPointer 
		Schooling_animateSchoolRef() { return &Schooling_animateSchool; }

	BehaveRT::genericKernelFuncPointer 
		Schooling_attractionOrientationRef() { return &Schooling_attractionOrientation; }

	BehaveRT::genericKernelFuncPointer 
		Schooling_repulsionRef() { return &Schooling_repulsion; }

	BehaveRT::genericKernelFuncPointer 
		Schooling_moveRef() { return &Schooling_move; }
	
	// ////////////////////////////////////////////////////////////////////////
	// Custom kernel calls

}



