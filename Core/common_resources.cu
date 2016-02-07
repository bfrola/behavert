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
// 12-08 bf: Created
//
// ----------------

#pragma once

#include "DeviceData.cuh"

__constant__ BehaveRT::SimParams params;
__constant__ BehaveRT::DeviceDataRepository deviceDataRepository;

#include "cutil_math.h"

// //////////////////////////////////////////////////////////////////////////////
// Host funcs
// //////////////////////////////////////////////////////////////////////////////

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b){
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
inline void computeGridSize(int n, int blockSize, int &numBlocks, int &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

// //////////////////////////////////////////////////////////////////////////////
// Device funcs
// //////////////////////////////////////////////////////////////////////////////

// ------------------------------------------------------------------------------

__host__ __device__ float calcDist4(float4 posA, float4 posB)
{
	float3 relPos;
    relPos.x = posB.x - posA.x;
    relPos.y = posB.y - posA.y;
    relPos.z = posB.z - posA.z;

    return length(relPos);
}

// ------------------------------------------------------------------------------

__host__ __device__ float calcSquaredDist4(float4 posA, float4 posB)
{
	float3 relPos;
    relPos.x = posB.x - posA.x;
    relPos.y = posB.y - posA.y;
    relPos.z = posB.z - posA.z;

    return dot(relPos, relPos);
}

// ------------------------------------------------------------------------------

__host__ __device__ float calcDist3(float3 posA, float3 posB)
{
	float3 relPos;
    relPos.x = posB.x - posA.x;
    relPos.y = posB.y - posA.y;
    relPos.z = posB.z - posA.z;

    return length(relPos);
}

// ------------------------------------------------------------------------------

__host__ __device__ float calcSquaredDist3(float3 posA, float3 posB)
{
	float3 relPos;
    relPos.x = posB.x - posA.x;
    relPos.y = posB.y - posA.y;
    relPos.z = posB.z - posA.z;

    return dot(relPos, relPos);
}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

// return component of vector parallel to a unit basis vector
// (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))

__host__ __device__  float3 parallelComponent (float3 vector, float3 unitBasis)
{
    const float projection = dot(vector, unitBasis);
    return unitBasis * projection;
}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

// return component of vector perpendicular to a unit basis vector
// (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))

__host__ __device__  float3 perpendicularComponent (float3 vector, float3 unitBasis)
{
    return vector - parallelComponent (vector, unitBasis);
}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

// Generic interpolation
__host__ __device__ float interpolate (float alpha, float x0, float x1)
{
    return x0 + ((x1 - x0) * alpha);
}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

__host__ __device__ float3 interpolate3 (float alpha, float3 x0, float3 x1)
{
    return x0 + ((x1 - x0) * alpha);
}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

__host__ __device__ void blendIntoAccumulator (	float smoothRate,
										float newValue,
										float& smoothedAccumulator)
{
	// Function Clamp is equivalent to function clip
    smoothedAccumulator = interpolate (clamp (smoothRate, 0.0f, 1.0f),
                                       smoothedAccumulator,
                                       newValue);
}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

__host__ __device__ void blendIntoAccumulator3 (	float smoothRate,
										float3 newValue,
										float3& smoothedAccumulator)
{
	// Function Clamp is equivalent to function clip
    smoothedAccumulator = interpolate3 (clamp (smoothRate, 0.0f, 1.0f),
                                       smoothedAccumulator,
                                       newValue);
}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

__host__ __device__ 
float3 vecLimitDeviationAngleUtility (bool insideOrOutside,
                                      float3 source,
                                      float cosineOfConeAngle,
									  float3 basis)
{
    // immediately return zero length input vectors
    const float sourceLength = length(source);
    if (sourceLength == 0) return source;

    // measure the angular diviation of "source" from "basis"
	const float3 direction = source / sourceLength;
    const float cosineOfSourceAngle = dot(direction, basis);

    // Simply return "source" if it already meets the angle criteria.
    // (note: we hope this top "if" gets compiled out since the flag
    // is a constant when the function is inlined into its caller)
    if (insideOrOutside)
    {
	// source vector is already inside the cone, just return it
	if (cosineOfSourceAngle >= cosineOfConeAngle) return source;
    }
    else
    {
	// source vector is already outside the cone, just return it
	if (cosineOfSourceAngle <= cosineOfConeAngle) return source;
    }

    // find the portion of "source" that is perpendicular to "basis"
    const float3 perp = perpendicularComponent(source, basis);

    // normalize that perpendicular
    const float3 unitPerp = normalize(perp);

    // construct a new vector whose length equals the source vector,
    // and lies on the intersection of a plane (formed the source and
    // basis vectors) and a cone (whose axis is "basis" and whose
    // angle corresponds to cosineOfConeAngle)
    const float perpDist = sqrtf (1 - (cosineOfConeAngle * cosineOfConeAngle));
    const float3 c0 = basis * cosineOfConeAngle;
    const float3 c1 = unitPerp * perpDist;
    return (c0 + c1) * sourceLength;
} // vecLimitDeviationAngleUtility

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

__host__ __device__ 
float3 adjustRawSteeringForce (float3 force, float speed, float3 forward, float maxSpeed)
{
    float maxAdjustedSpeed = 0.2f * maxSpeed;

    if ((speed > maxAdjustedSpeed) || (force.x == 0 && force.y == 0 && force.z == 0))
    {
        return force;
    }
    else
    {
        const float range = speed / maxAdjustedSpeed;
        // const float cosine = interpolate (pow (range, 6), 1.0f, -1.0f);
        // const float cosine = interpolate (pow (range, 10), 1.0f, -1.0f);
        // const float cosine = interpolate (pow (range, 20), 1.0f, -1.0f);
        // const float cosine = interpolate (pow (range, 100), 1.0f, -1.0f);
        // const float cosine = interpolate (pow (range, 50), 1.0f, -1.0f);
        const float cosine = interpolate (powf (range, 20), 1.0f, -1.0f);
        return vecLimitDeviationAngleUtility (true, force, cosine, forward);
    }
}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

__host__ __device__ float3 truncateLength (float3 vector, float maxLength)
{
 
	float vecLength = length(vector);
	
	if (vecLength <= maxLength)
		return vector;

	//clamp(vecLength, 0, maxLength);
	
	return vector * (maxLength / vecLength);
}


// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

namespace BehaveRT
{
	__device__ int getIndividualIndex()
	{
		return __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	}

	// ------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------

	template<class DataType>   
	__device__ DataType* getInputFeature(int dataInfoIndex)
	{
		return (DataType*)
			deviceDataRepository[dataInfoIndex].inputDataPointer;
	}

	// ------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------

	template<class DataType>   
	__device__ DataType* getOutputFeature(int dataInfoIndex)
	{
		return (DataType*)
			deviceDataRepository[dataInfoIndex].outputDataPointer;
	}

	template<class DataType>   
	__device__ DataType* setOutputFeatureElement(int dataInfoIndex, 
		int elementIndex, DataType elementValue)
	{
		DataType* outputFeature = 
			BehaveRT::getOutputFeature<DataType>(dataInfoIndex);

		outputFeature[elementIndex] = elementValue;
	}

	

};



// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

extern "C"
{
	BehaveRT::DeviceDataRepository m_DeviceDataRepository;
}

