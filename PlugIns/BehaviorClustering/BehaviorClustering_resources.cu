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
// 03-10 bf: Created
//
// ----------------

#pragma once

#include "DeviceData.cuh"
#include "common_resources.cu"
#include "include\BehaviorClustering_kernel.cuh"

// Other plugIn dependencies


// Structures sharing
share_device_struct(BehaviorClusteringParams);
share_device_struct(BehaviorClusteringFields);

// Textures
texture<float, 1, cudaReadModeElementType> featuresVectorTex;

texture<float, 1, cudaReadModeElementType> neighSimilaritiesTex;
texture<float, 1, cudaReadModeElementType> neighIndexesTex;

texture<float4, 1, cudaReadModeElementType> similarityStatsTex;


// ----------------------------------------------------------------------
// Utility device functions
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------


__device__ float getLinearInterpolation(
	float x,
	float xa, float ya,
	float xb, float yb)
{
	return 
		ya * ((x - xb) / (xa - xb)) +
		yb * ((x - xa) / (xb - xa));
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

__device__ float interpolateSimilarity(float val, float min, float avg, float max)
{
	
	// Linear interpolation
	if (val > avg)
	{
		// Clamp for precision related errors
		return clamp(getLinearInterpolation(val, avg, 0.5f, max, 1.0f), 
			0.0f, 1.0f);
	}

	return clamp(getLinearInterpolation(val, min, 0.0f, avg, 0.5f),
		0.0f, 1.0f);
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

__device__ float4 updateSimilarityStats(float similarity, float4 oldStats)
{
	float minVal = oldStats.x;
	float avgVal = oldStats.y;
	float maxVal = oldStats.z;

	// --------------------------------------
	float valueToBlendInAvg =
		similarity *
			dBehaviorClusteringParams.avgSimilarityController +
		((maxVal + minVal) / 2.0f) * 
			(1 - dBehaviorClusteringParams.avgSimilarityController);


	avgVal +=
		(valueToBlendInAvg - oldStats.y) * 
			dBehaviorClusteringParams.similarityStatsParams.x;

	avgVal = clamp(avgVal, 0.0f, 1.0f);

	// --------------------------------------

	if (similarity < minVal)
	{
		minVal += (similarity - minVal) * 
			dBehaviorClusteringParams.similarityStatsParams.y;

		// Bind to avgVal
		if (minVal > avgVal)
			minVal = clamp(minVal, 0.0f, similarity);
	}
	else if (minVal < avgVal)
		minVal += dBehaviorClusteringParams.similarityStatsParams.z; // "Memory"

	// --------------------------------------

	if (similarity > maxVal)
	{
		maxVal += (similarity - maxVal) * 
			dBehaviorClusteringParams.similarityStatsParams.y;

		// Bind to avgVal
		if (maxVal < avgVal)
			maxVal =  clamp(maxVal, similarity, 1.0f);
	}
	else if (maxVal > avgVal)
		maxVal -= dBehaviorClusteringParams.similarityStatsParams.z; // "Memory"
	
	// --------------------------------------

	oldStats.x = minVal;
	oldStats.y = avgVal;
	oldStats.z = maxVal;

	return oldStats;
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

// Device functions

// Compute the distance between two features
// vector of length dBehaviorClusteringParams.featuresCount
__device__ float computeSimilarity(
	float* feturesVectorRef1, 
	float* feturesVectorRef2)
{
	float norma1 = 0.0;
	float norma2 = 0.0;
	float dot_scalar = 0.0;
	
	for(int k = 0; k < dBehaviorClusteringParams.featuresCount; k ++)
	{
		float featureValue1 = feturesVectorRef1[k];
		float featureValue2 = feturesVectorRef2[k];

		// Calculate the dot product
		dot_scalar += (featureValue1 * featureValue2); 
		norma1 += (featureValue1 * featureValue1);
		norma2 += (featureValue2 * featureValue2);
	}
	norma1 = sqrt(norma1);
	norma2 = sqrt(norma2);

	float similarity = dot_scalar /(norma1*norma2); // Similarity [-1..1]

	return (similarity + 1.0f) / 2.0f; // Get a value [0..1]
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

__device__ float computeSimilarityFromTex(
	int individualIndex1, int individualIndex2)
{
	float norma1 = 0.0;
	float norma2 = 0.0;
	float dot_scalar = 0.0;

	individualIndex1 *= dBehaviorClusteringParams.featuresCount;
	individualIndex2 *= dBehaviorClusteringParams.featuresCount;
	
	for(int k = 0; k < dBehaviorClusteringParams.featuresCount; k ++)
	{
		float featureValue1 = FETCH(featuresVector, individualIndex1 + k);
		float featureValue2 = FETCH(featuresVector, individualIndex2 + k);

		// Calculate the dot product
		dot_scalar += (featureValue1 * featureValue2); 
		norma1 += (featureValue1 * featureValue1);
		norma2 += (featureValue2 * featureValue2);
	}
	norma1 = sqrt(norma1);
	norma2 = sqrt(norma2);

	float similarity = dot_scalar /(norma1*norma2); // Similarity [-1..1]

	return (similarity + 1.0f) / 2.0f; // Get a value [0..1]
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

__device__ float computeSimilarityFromShrdTex(
	float* feturesVectorShrdRef1, 
	int individualIndex2)
{
	float norma1 = 0.0;
	float norma2 = 0.0;
	float dot_scalar = 0.0;

	individualIndex2 *= dBehaviorClusteringParams.featuresCount;

	for(int k = 0; k < dBehaviorClusteringParams.featuresCount; k ++)
	{
		float featureValue1 = feturesVectorShrdRef1[k];
		float featureValue2 = FETCH(featuresVector, individualIndex2 + k);

		// Calculate the dot product
		dot_scalar += (featureValue1 * featureValue2); 
		norma1 += (featureValue1 * featureValue1);
		norma2 += (featureValue2 * featureValue2);
	}
	norma1 = sqrt(norma1);
	norma2 = sqrt(norma2);

	float similarity = dot_scalar /(norma1*norma2); // Similarity [-1..1]

	return (similarity + 1.0f) / 2.0f; // Get a value [0..1]
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

__device__ void getNeighborsSimilarities(int index, float* neighSimilarities)
{
	declare_input(neighSimilaritiesList, float, dBehaviorClusteringFields.neighSimilarities);
	const int neighIndexBase = __mul24(index, dProximity3DParams.maxNeighbors + 1);
	neighSimilarities = neighSimilaritiesList + neighIndexBase;
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

// Declare and init neigh lists and similarities
#define BEHAVIORCLUSTERING_PREPARENEIGHLISTS					\
	uint neighNum;												\
	uint neighList[Proximity3d_MAX_NEIGHBORS];					\
	Proximity3D::getNeighborsList(index, neighNum, neighList);	\
	declare_input(neighSimilaritiesList, float, dBehaviorClusteringFields.neighSimilarities);	\
	const int neighIndexBase = __mul24(index, dProximity3DParams.maxNeighbors + 1);				\
	float* neighSimilarities = neighSimilaritiesList + neighIndexBase;							

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

__global__ void BehaviorClustering_steerForClusterCohesion()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint sortedIndex = FETCH(agentHash, index).y;

	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	// Vars
	float3 myPos = make_float3(tex1Dfetch(oldPosTex, index));
	float3 myForward = make_float3(tex1Dfetch(oldForwardTex, index));
	float mySpeed = tex1Dfetch(oldForwardTex, index).w;
	float3 steering = make_float3(0, 0, 0);

	// Macro
	BEHAVIORCLUSTERING_PREPARENEIGHLISTS

	/*uint neighNum;												
	uint neighList[Proximity3d_MAX_NEIGHBORS];					
	//float neighSimilarities[Proximity3d_MAX_NEIGHBORS];			
	Proximity3D::getNeighborsList(index, neighNum, neighList);	
	//getNeighborsSimilarities(index, neighSimilarities);			
	declare_input(neighSimilaritiesList, float, dBehaviorClusteringFields.neighSimilarities);
	const int neighIndexBase = __mul24(index, dProximity3DParams.maxNeighbors + 1);
	float* neighSimilarities = neighSimilaritiesList + neighIndexBase;
	*/

	

	// Neighs list does not contain the idividual that is executing the behavior
	for (int i = 0; i < neighNum; i ++)
	{

		// Get indexes and similarities of neighbors
		uint otherIndex = neighList[i];
		float similarity = neighSimilarities[i];
		//float similarity = 1.0f;

		// --------------------------------
		// DEBUG

		//uint otherSortedIndex = FETCH(agentHash, otherIndex).y;
		//if (sortedIndex == dBehaviorClusteringParams.debugIndex)
		//	cuPrintf("C %d-%d) %f\n", sortedIndex, otherSortedIndex, similarity);

		// --------------------------------

		float3 otherPos = make_float3((float4)FETCH(oldPos, otherIndex));

		// Calculate perpendicular forces
		float3 seekForce =
			OpenSteerWrapper::xxxsteerForSeek(
				myPos, mySpeed, myForward, otherPos, 
				dOpenSteerWrapperParams.commonMaxSpeed);

		float3 fleeForce = 
			OpenSteerWrapper::xxxsteerForFlee(
				myPos, mySpeed, myForward, otherPos, 
				dOpenSteerWrapperParams.commonMaxSpeed);

		// Combine forces using the similary value
		steering += 
			seekForce  * similarity + 
			fleeForce * (1.0f - similarity);
	}

	// Normalize and add a force weight
	if (neighNum > 0)
	{
		steering = normalize(steering / (float)neighNum) * 
			dBehaviorClusteringParams.clusteringForceParams.x;
	}

	OpenSteerWrapper::blendIntoSteeringForce(index, steering);
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

__global__ void BehaviorClustering_steerForClusterAlignment()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint sortedIndex = FETCH(agentHash, index).y;

	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	// Var init
	float3 myForward = make_float3(tex1Dfetch(oldForwardTex, index));
	float3 steering = make_float3(0, 0, 0);

	// Macro
	BEHAVIORCLUSTERING_PREPARENEIGHLISTS

	// Neighs list does not contain the idividual that is executing the behavior
	for (int i = 0; i < neighNum; i ++)
	{
		// Get indexes and similarities of neighbors
		uint otherIndex = neighList[i];
		float similarity = neighSimilarities[i];

		float3 otherForward = make_float3(tex1Dfetch(oldForwardTex, otherIndex));

		// Calc the similarity alignment
		steering += otherForward * similarity;
	}

	// Normalize and add a force weight
	if (neighNum > 0)
	{
		steering = normalize((steering / (float)neighNum) - myForward) *
			dBehaviorClusteringParams.alignmentClusteringForceParams.x;
	}

	OpenSteerWrapper::blendIntoSteeringForce(index, steering);
}
