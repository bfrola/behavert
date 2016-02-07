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
//    11-2008: Created
//    03-2009: Updated, Schooling
//    05-2009: Updated, Buildings
// 01-06-2010: Updated, Clustering
//
// ----------------
#pragma once

#include "DeviceData.cuh"
#include "include\OpenSteerWrapper_kernel.cuh"

// Other plugIn dependencies
#include "..\EnvGrid3D\include\envgrid3d_kernel.cuh"
#include "..\EnvGrid3D\envgrid3d_resources.cu"

#include "..\Proximity3D\include\Proximity3D_kernel.cuh"
#include "..\Proximity3D\Proximity3D_resources.cu"

#include "..\Body\include\body3d_kernel.cuh"
#include "..\Body\body3d_resources.cu"


share_device_struct(OpenSteerWrapperParams);
share_device_struct(OpenSteerWrapperDynParams);
share_device_struct(OpenSteerWrapperFields);

texture<float4, 1, cudaReadModeElementType> oldForwardTex;
texture<float4, 1, cudaReadModeElementType> oldSmoothedAccelTex;
texture<float4, 1, cudaReadModeElementType> oldForceTex;

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------


namespace OpenSteerWrapper
{

	__device__ bool inBoidNeighborhood(
		float3 myPos, 
		float3 otherPos, 
		float minDist, 
		float maxDist, 
		float cosMaxAngle, 
		float3 forward)
	{
		float3 offset;
		offset.x = otherPos.x - myPos.x;
		offset.y = otherPos.y - myPos.y;
		offset.z = otherPos.z - myPos.z;

		float dist = length(offset);

		if ( dist < minDist )
			return true;

		if ( dist > maxDist )
			return false;

		// Check angular data
		float cosAngle = dot(forward, normalize(offset));
		return cosAngle > cosMaxAngle;
		
	}

// ------------------------------------------------------------------------------

	/*__device__ void blendIntoSteeringForce2(
		// In params (arrays)
		const float4* oldForce,
		float4* newForce,
		const uint index,
		const uint sortedIndex,
		float3 steering,
		const bool useCumulativeForce)
	{
		if (useCumulativeForce)
		{
			steering += make_float3(oldForce[index]);

		}
		newForce[index] = make_float4(steering);
	}*/

	template <bool blendIntoSteringForce>
	__device__ void storeSteeringForce(int index, float3 steering)
	{
		declare_output(newForce, float4, dOpenSteerWrapperFields.steerForce);

		if (blendIntoSteringForce)
		{
			steering += make_float3(tex1Dfetch(oldForceTex, index));
		}

		newForce[index] = make_float4(steering);
	}

	__device__ void blendIntoSteeringForce(int index, float3 steering)
	{
		storeSteeringForce<true>(index, steering);
	}

// ------------------------------------------------------------------------------

	/** 
		Read common behavior parameters
	*/
	__device__ void loadCommonBehaviorParams(
		// In params (arrays)
		int index,
		float4* oldPos,
		uint2* particleHash,
		float4* oldForward,
		// out params
		float3& myPos,
		uint2& sortedData,
		float3& myForward,
		float3& steering,
		uint& neighNum,
		uint* neighList)
	{
		sortedData = particleHash[index];
		
		myPos = make_float3((float4)FETCH(oldPos, index));	
		
		myForward = FETCH_FLOAT3(oldForward, index);

		// This var will used on implementing item sight
		steering = make_float3(0.0f, 0.0f, 0.0f);

		Proximity3D::getNeighborsList(index, neighNum, neighList);

		// -- 
	}
	
// ------------------------------------------------------------------------------

	__device__ float3 calcSeparation(
		float3 myPos,
		float3 myForward,
		uint neighNum,
		uint* neighList)
	{
		int otherIndex;
		float3 otherPos;
		float3 steering = make_float3(0.0, 0.0, 0.0);

		uint neighbors = 0;

		for (int i = 0; i < neighNum; i ++)
		{
			//otherIndex = neighborhood.neighborsIndex[i];
			otherIndex = neighList[i];
	
			if (otherIndex < 0)
				break;

			otherPos = make_float3((float4)FETCH(oldPos, otherIndex));

			if (inBoidNeighborhood(myPos, otherPos, 
				dOpenSteerWrapperParams.commonSensingMinRadius, 
				dOpenSteerWrapperParams.separationParams.y, 
				dOpenSteerWrapperParams.separationParams.z, 
				myForward))
			{	
				// add in steering contribution
				// (opposite of the offset direction, divided once by distance
				// to normalize, divided another time to get 1/d falloff)
				float3 offset = otherPos - myPos;
				float distanceSquared = dot(offset, offset);
				steering += (offset / -distanceSquared);
				
				// count neighbors
				neighbors ++;
			}
		}

		// divide by neighbors, then normalize to pure direction
		if (neighbors > 0)
			steering = normalize((steering / (float)neighbors));

		return steering;
	}


	// ------------------------------------------------------------------------------

	__device__ float3 calcCohesion(
		float3 myPos,
		float3 myForward,
		uint neighNum,
		uint* neighList)
	{
		int otherIndex;
		float3 otherPos;
		float3 steering = make_float3(0.0, 0.0, 0.0);

		uint neighbors = 0;

		for (int i = 0; i < neighNum; i ++)
		{
			otherIndex = neighList[i];
			otherPos = make_float3(tex1Dfetch(oldPosTex, otherIndex));
			
			if (OpenSteerWrapper::inBoidNeighborhood(
				myPos, otherPos, 
				dOpenSteerWrapperParams.commonSensingMinRadius, 
				dOpenSteerWrapperParams.cohesionParams.y, 
				dOpenSteerWrapperParams.cohesionParams.z, 
				myForward))
			{
				steering = steering + otherPos;
				neighbors ++;
			}
		}

		if (neighbors > 0)	
			steering = normalize((steering / (float)neighbors) - myPos);

		return steering;
	}

	// ------------------------------------------------------------------------------

	__device__ float3 calcAlignment(
		float3 myPos,
		float3 myForward,
		uint neighNum,
		uint* neighList )
	{

		uint neighbors = 0;
		int otherIndex;
		float3 otherPos;
		float3 steering = make_float3(0.0, 0.0, 0.0);
		
		for (int i = 0; i < neighNum; i ++)
		{
			otherIndex = neighList[i];
			otherPos = make_float3(tex1Dfetch(oldPosTex, otherIndex));

			if (OpenSteerWrapper::inBoidNeighborhood(
				myPos, otherPos, 
				dOpenSteerWrapperParams.commonSensingMinRadius, 
				dOpenSteerWrapperParams.alignmentParams.y, 
				dOpenSteerWrapperParams.alignmentParams.z, 
				myForward))
			{	
				float3 otherForward = make_float3(tex1Dfetch(oldForwardTex, otherIndex));
				// accumulate sum of neighbor's heading
				steering += otherForward;
				neighbors++;
			}
		}

		// divide by neighbors, subtract off current heading to get error-
		// correcting direction, then normalize to pure direction
		if (neighbors > 0)	
			steering = normalize((steering / (float)neighbors) - myForward);
		
		return steering;
	}

	// ------------------------------------------------------------------------------

	__device__ float3 xxxsteerForSeek (
				const float3 myPos,
				const float mySpeed,
				const float3 myForward, 
				const float3 target,
				const float maxSpeed)
	{
		float3 offset = target - myPos;
		float3 desiredVelocity = truncateLength(offset, maxSpeed); //xxxnew
		float3 myVel = myForward * mySpeed;
		//return perpendicularComponent(desiredVelocity - myVel, myForward); 
		return desiredVelocity - myVel; // UPD 01-06-2010
	} // xxxsteerForSeek

	// ------------------------------------------------------------------------------

	// ADD 01-06-2010
	__device__ float3 xxxsteerForFlee (
				const float3 myPos,
				const float mySpeed,
				const float3 myForward, 
				const float3 target,
				const float maxSpeed)
	{
		float3 offset = myPos - target;
		float3 desiredVelocity = truncateLength(offset, maxSpeed); //xxxnew
		float3 myVel = myForward * mySpeed;
		return desiredVelocity - myVel; 
	}

	// ------------------------------------------------------------------------------

	__device__ void
	applySteeringForceSingle (float3 force, float elapsedTime, 
							float maxForce, float maxSpeed, 
							float mass, float& speed, 
							float3& forward, float3& position, 
							float3 &_smoothedAccel )
	{
		const float3 adjustedForce = adjustRawSteeringForce (force, speed, forward, maxSpeed);

		// enforce limit on magnitude of steering force
		const float3 clippedForce = truncateLength (adjustedForce, maxForce);

		// compute Accel and velocity
		const float3 newAccel = (clippedForce / mass);
	    
		// damp out abrupt changes and oscillations in steering Accel
		// (rate is proportional to time step, then clipped into useful range)
		if (elapsedTime > 0)
		{
			// Clamp substitute clip
			float smoothRate = clamp (9 * elapsedTime, 0.15f, 0.4f);
			blendIntoAccumulator3 (smoothRate,
								  newAccel,
								  _smoothedAccel);
		}

		// Euler integrate (per frame) Accel into velocity
		float3 newVelocity = forward * speed + 
			_smoothedAccel * elapsedTime;
	    
		// enforce speed limit
		newVelocity = truncateLength (newVelocity, maxSpeed);

		// extract speed
		speed = length(newVelocity);

		// Euler integrate (per frame) velocity into position
		position += newVelocity * elapsedTime;

		if (dBody3DParams.use2DProjection)
		{
			newVelocity.y = 0;
		}

		forward = normalize(newVelocity);

		/*

		// regenerate local space (by default: align vehicle's forward axis with
		// new velocity, but this behavior may be overridden by derived classes.)
		//regenerateLocalSpace (newVelocity, elapsedTime);

		*/
	} // applySteeringForceSingle
}