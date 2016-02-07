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

#include <cutil.h>
#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"

#include "common_resources.cu"

// Same plugIn dependencies
#include "include\BehaviorClustering_kernel.cuh"
#include "BehaviorClustering_resources.cu"

// Other plugIn dependencies
#include "..\EnvGrid3D\include\envgrid3d_kernel.cuh"
#include "..\EnvGrid3D\envgrid3d_resources.cu"

#include "..\Proximity3D\include\Proximity3D_kernel.cuh"
#include "..\Proximity3D\Proximity3D_resources.cu"

#include "..\Body\include\body3d_kernel.cuh"
#include "..\Body\body3d_resources.cu"

#include "..\OpenSteerWrapper\include\OpenSteerWrapper_kernel.cuh"
#include "..\OpenSteerWrapper\OpenSteerWrapper_resources.cu"




// ////////////////////////////////////////////////////////
// Kernels
template <bool c_UseAdaptiveSimilarity>
__global__ void BehaviorClustering_steerForClustering()
{
	
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint sortedIndex = FETCH(agentHash, index).y;

	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	uint neighNum;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	float3 myPos = make_float3(tex1Dfetch(oldPosTex, index));
	float3 myForward = make_float3(tex1Dfetch(oldForwardTex, index));
	float mySpeed = tex1Dfetch(oldForwardTex, index).w;

	declare_input(featuresVector, float, dBehaviorClusteringFields.featuresVector);
	declare_input(similarityStats, float4, dBehaviorClusteringFields.similarityStats);

	float4 mySimilarityStats = similarityStats[sortedIndex];

	// Get a features vector pointer
	float* feturesVectorRef = &featuresVector[sortedIndex * 
		dBehaviorClusteringParams.featuresCount];

	// Get neighbors
	Proximity3D::getNeighborsList(index, neighNum, neighList);

	float3 steering = make_float3(0, 0, 0);
	float3 alignmentSteering = make_float3(0, 0, 0);

	// --------------------------------
	// DEBUG

	//if (sortedIndex == dBehaviorClusteringParams.debugIndex)
	//	cuPrintf("\n%d) %f %f %f\n", sortedIndex, mySimilarityStats.x, 
	//		mySimilarityStats.y, mySimilarityStats.z);

	// --------------------------------

	declare_output(newNeighSimilarities, float, dBehaviorClusteringFields.neighSimilarities);
	const int neighIndexBase = __mul24(index, dProximity3DParams.maxNeighbors + 1);
	float* neighSimilarities = newNeighSimilarities + neighIndexBase;

	// Neighs list does not contain the idividual that is executing the behavior
	for (int i = 0; i < neighNum; i ++)
	{
		// Get neighbor data
		uint otherIndex = neighList[i];

		float3 otherPos = make_float3((float4)FETCH(oldPos, otherIndex));
		float3 otherForward = make_float3(tex1Dfetch(oldForwardTex, otherIndex));

		uint otherSortedIndex = FETCH(agentHash, otherIndex).y;
		float* otherFeturesVectorRef = &featuresVector[otherSortedIndex * 
			dBehaviorClusteringParams.featuresCount];

		// ----------------------------
		// Compute similarity
		
		float similarity = computeSimilarity(
			feturesVectorRef, 
			otherFeturesVectorRef);

		// --------------------------------
		// DEBUG
		//if (false && sortedIndex == dBehaviorClusteringParams.debugIndex)
		//{
		//	cuPrintf("OI: %d Sim: %f\n", otherSortedIndex, similarity);
		//	//if (similarity < 0)
		//	{
		//		cuPrintf("\tF1[%f, %f]\n", 
		//			feturesVectorRef[0],
		//			feturesVectorRef[1]);
		//		cuPrintf("\tF2[%f, %f]\n", 
		//			otherFeturesVectorRef[0],
		//			otherFeturesVectorRef[1]);
		//	}
		//}
		// --------------------------------
		
		mySimilarityStats = 
			updateSimilarityStats(similarity, mySimilarityStats);

		if (c_UseAdaptiveSimilarity)
		{
			
			similarity = interpolateSimilarity(
				similarity, 
				mySimilarityStats.x,	// Min 
				mySimilarityStats.y,	// Avg
				mySimilarityStats.z);	// Max
		}

		// -----------------------------------------

		// --------------------------------
		// DEBUG

		//if (sortedIndex == dBehaviorClusteringParams.debugIndex)
		//	cuPrintf("A %d-%d) %f\n", sortedIndex, otherSortedIndex, similarity);

		// --------------------------------

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

		// Calc the similarity alignment
		alignmentSteering += 
			otherForward * similarity;

		neighSimilarities[i] = similarity;

	} // Neighs

	// Normalize and add a force weight
	if (neighNum > 0)
	{
		steering = normalize(steering / (float)neighNum) * 
			dBehaviorClusteringParams.clusteringForceParams.x;

		alignmentSteering = normalize((alignmentSteering / (float)neighNum) - myForward) *
			dBehaviorClusteringParams.alignmentClusteringForceParams.x;

		steering += alignmentSteering;
	}

	

	OpenSteerWrapper::blendIntoSteeringForce(index, steering);
	declare_output(newSimilarityStats, float4, dBehaviorClusteringFields.similarityStats);
	newSimilarityStats[sortedIndex] = mySimilarityStats;

}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

template <bool c_UseAdaptiveSimilarity>
__global__ void BehaviorClustering_computeNeighSimilarities()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint sortedIndex = FETCH(agentHash, index).y;

	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	declare_input(featuresVector, float, dBehaviorClusteringFields.featuresVector);
	declare_input(similarityStats, float4, dBehaviorClusteringFields.similarityStats);

	uint neighNum;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	float4 mySimilarityStats = similarityStats[sortedIndex];

	// Get a features vector pointer
	float* feturesVectorRef = &featuresVector[sortedIndex * 
		dBehaviorClusteringParams.featuresCount];

	// Get neighbors
	Proximity3D::getNeighborsList(index, neighNum, neighList);

	// --------------------------------
	// DEBUG

	//if (sortedIndex == dBehaviorClusteringParams.debugIndex)
	//	cuPrintf("\n%d] %f %f %f\n", sortedIndex, mySimilarityStats.x, 
	//		mySimilarityStats.y, mySimilarityStats.z);

	// --------------------------------

	declare_output(newNeighSimilarities, float, dBehaviorClusteringFields.neighSimilarities);
	const int neighIndexBase = __mul24(index, dProximity3DParams.maxNeighbors + 1);
	float* neighSimilarities = newNeighSimilarities + neighIndexBase;
	
	// Neighs list does not contain the idividual that is executing the behavior
	for (int i = 0; i < neighNum; i ++)
	{
		// Get neighbor data
		uint otherIndex = neighList[i];

		float3 otherPos = make_float3((float4)FETCH(oldPos, otherIndex));
		float3 otherForward = make_float3(tex1Dfetch(oldForwardTex, otherIndex));

		uint otherSortedIndex = FETCH(agentHash, otherIndex).y;
		float* otherFeturesVectorRef = &featuresVector[otherSortedIndex * 
			dBehaviorClusteringParams.featuresCount];

		// ----------------------------
		// Compute similarity
		
		float similarity = computeSimilarity(
			feturesVectorRef, 
			otherFeturesVectorRef);

		mySimilarityStats = 
			updateSimilarityStats(similarity, mySimilarityStats);

		if (c_UseAdaptiveSimilarity)
		{
			
			similarity = interpolateSimilarity(
				similarity, 
				mySimilarityStats.x,	// Min 
				mySimilarityStats.y,	// Avg
				mySimilarityStats.z);	// Max
		}

		neighSimilarities[i] = similarity;

		//if (sortedIndex == dBehaviorClusteringParams.debugIndex)
		//	cuPrintf("B %d-%d] %f\n", sortedIndex, 
		//		otherSortedIndex, neighSimilarities[i]);

	} // Neighs


	// --------------------
	// Write outputs

	declare_output(newSimilarityStats, float4, dBehaviorClusteringFields.similarityStats);
	newSimilarityStats[sortedIndex] = mySimilarityStats;
}

// ----------------------------------------------------------------
// ----------------------------------------------------------------

__global__ void BehaviorClustering_computeNeighNaiveSimilarities()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint sortedIndex = FETCH(agentHash, index).y;

	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	declare_input(featuresVector, float, dBehaviorClusteringFields.featuresVector);
	//declare_input(similarityStats, float4, dBehaviorClusteringFields.similarityStats);

	uint neighNum;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	// Get neighbors
	Proximity3D::getNeighborsList(index, neighNum, neighList);

	// --------------------------------
	// DEBUG

	//if (sortedIndex == dBehaviorClusteringParams.debugIndex)
	//	cuPrintf("\n%d] %f %f %f\n", sortedIndex, mySimilarityStats.x, 
	//		mySimilarityStats.y, mySimilarityStats.z);

	// --------------------------------

	declare_output(newNeighSimilarities, float, dBehaviorClusteringFields.neighSimilarities);
	const int neighIndexBase = __mul24(index, dProximity3DParams.maxNeighbors + 1);
	float* neighSimilarities = newNeighSimilarities + neighIndexBase;
	
	// Neighs list does not contain the idividual that is executing the behavior
	for (int i = 0; i < neighNum; i ++)
	{
		// Get neighbor data
		uint otherIndex = neighList[i];
		uint otherSortedIndex = FETCH(agentHash, otherIndex).y;

		// ----------------------------
		// Compute similarity
		
		float similarity = computeSimilarityFromTex(
			sortedIndex, otherSortedIndex);

		neighSimilarities[i] = similarity; // Store naive similarity

	} // Neighs

}

// ----------------------------------------------------------------
// ----------------------------------------------------------------

// With shared memory
__global__ void BehaviorClustering_computeNeighNaiveSimilarities_shrd()
{
	const int maxNeighsAdded1 = dProximity3DParams.maxNeighbors + 1;
	const int threadOffset = //(threadIdx.x >> log2(neighsBlockSize));
		threadIdx.x / maxNeighsAdded1;

	// blockIdx.x + threadIdx.x DIV blockDim.x
	int index = blockIdx.x + threadOffset;

	uint sortedIndex = FETCH(agentHash, index).y;
	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	const int neighIndexBase = __mul24(index, maxNeighsAdded1);

	// Current index
	// blockIdx.x MOD neighsBlockSize
	int i = threadIdx.x & (maxNeighsAdded1 - 1);

	declare_input(neighListArray, uint, dProximity3DFields.neighList);	
	__shared__ uint neighNum;

	if (i == 0)
	{
		neighNum = neighListArray[neighIndexBase];
	}

	// Sync and share neighNum
	__syncthreads();

	if (i > neighNum)
		return;

	const int individualsPerBlock = 1;//blockDim.x / maxNeighsAdded1;

	const int feturesVectorRefSize = BehaviorClustering_MAX_FEATUREDIM;
	const int feturesVectorRefOffset = feturesVectorRefSize * threadOffset;
	__shared__ float feturesVectorRef[feturesVectorRefSize * individualsPerBlock];
	
	// Parallel read
	declare_input(featuresVector, float, dBehaviorClusteringFields.featuresVector);	
	if (i < dBehaviorClusteringParams.featuresCount)
	{
		feturesVectorRef[i + feturesVectorRefOffset] = 
			featuresVector[sortedIndex * dBehaviorClusteringParams.featuresCount + i];
	}

	// Sync and share the features vector
	__syncthreads();

	// Get neighbor data
	uint otherIndex = neighListArray[neighIndexBase + i + 1];
	uint otherSortedIndex = FETCH(agentHash, otherIndex).y;

	// --------------------------------
	// DEBUG

	//if (index == dBehaviorClusteringParams.debugIndex)
	//	cuPrintf("\n%d] %d < %d - %d \n", i, neighNum, sortedIndex, otherSortedIndex);

	// --------------------------------

	// ----------------------------
	// Compute similarity

	// Current features fector read from shared memory
	// Neighbor features vector read from texture
	float similarity = computeSimilarityFromShrdTex(
		feturesVectorRef + feturesVectorRefOffset, 
		otherSortedIndex);

	// Store result
	declare_output(newNeighSimilarities, float, dBehaviorClusteringFields.neighSimilarities);
	float* neighSimilarities = newNeighSimilarities + neighIndexBase;
	neighSimilarities[i] = similarity; // Store naive similarity
}

// ----------------------------------------------------------------
// ----------------------------------------------------------------

__global__ void BehaviorClustering_computeAdaptiveSimilarities()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint sortedIndex = FETCH(agentHash, index).y;

	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	float4 mySimilarityStats = FETCH(similarityStats, sortedIndex);

	declare_input(oldNeighSimilaritiesArray, float, dBehaviorClusteringFields.neighSimilarities);
	declare_output(newNeighSimilaritiesArray, float, dBehaviorClusteringFields.neighSimilarities);
	const int neighIndexBase = __mul24(index, dProximity3DParams.maxNeighbors + 1);
	float* oldNeighSimilarities = oldNeighSimilaritiesArray + neighIndexBase;
	float* newNeighSimilarities = newNeighSimilaritiesArray + neighIndexBase;

	// Get neighbors
	const int neighIndexBaseTex = __mul24(index, dProximity3DParams.numNeighWordsPerAgent);
	uint4 neighWord = FETCH(oldNeighList, neighIndexBaseTex);
	uint neighNum = neighWord.x;
	
	// --------------------------------
	// DEBUG

	//if (sortedIndex == dBehaviorClusteringParams.debugIndex)
	//	cuPrintf("\n%d) %f %f %f\n", sortedIndex, mySimilarityStats.x, 
	//		mySimilarityStats.y, mySimilarityStats.z);

	// --------------------------------

	// Neighs list does not contain the idividual that is executing the behavior
	for (int i = 0; i < neighNum; i ++)
	{
		float similarity = oldNeighSimilarities[i];

		// --------------------------------
		// DEBUG

		//if (sortedIndex == dBehaviorClusteringParams.debugIndex)
		//	cuPrintf("\n%d) %f\n", sortedIndex, similarity);

		// --------------------------------

		// Update similarity
		mySimilarityStats = 
			updateSimilarityStats(similarity, mySimilarityStats);

		similarity = interpolateSimilarity(
			similarity, 
			mySimilarityStats.x,	// Min 
			mySimilarityStats.y,	// Avg
			mySimilarityStats.z);	// Max

		newNeighSimilarities[i] = similarity;
	} // Neighs

	declare_output(newSimilarityStats, float4, dBehaviorClusteringFields.similarityStats);
	newSimilarityStats[sortedIndex] = mySimilarityStats;
}

// ----------------------------------------------------------------
// ----------------------------------------------------------------

__global__ void BehaviorClustering_clusterHitRate()
{
	declare_input(oldClusterMembership, float3, dBehaviorClusteringFields.clusterMembership);
	declare_output(newClusterMembership, float3, dBehaviorClusteringFields.clusterMembership);

	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint sortedIndex = FETCH(agentHash, index).y;

	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	float myClusterMembership = oldClusterMembership[sortedIndex].x;

	
	uint neighNum;												
	uint neighList[Proximity3d_MAX_NEIGHBORS];					
	Proximity3D::getNeighborsList(index, neighNum, neighList);	

	float3 myPos = make_float3(tex1Dfetch(oldPosTex, index));

	int closeNeighbors = 0;
	float realClusterMembership = 0.0f;

	float clusterHitRate = 0.0f;
	for (int i = 0; i < neighNum; i ++)
	{
		uint otherIndex = neighList[i];
		uint otherSortedIndex = FETCH(agentHash, otherIndex).y;
		float otherClusterMembership = oldClusterMembership[otherSortedIndex].x;

		float3 otherPos = make_float3((float4)FETCH(oldPos, otherIndex));

		float3 offset;
		offset.x = otherPos.x - myPos.x;
		offset.y = otherPos.y - myPos.y;
		offset.z = otherPos.z - myPos.z;

		float distance = length(offset);

		if (distance > dBehaviorClusteringParams.sameClusterMaxRadius)
			continue;

		closeNeighbors ++;

		realClusterMembership += otherClusterMembership;

		if (myClusterMembership == otherClusterMembership)
		{
			clusterHitRate += 1.0f;
		}

		// --------------------------------
		// DEBUG
		//if (sortedIndex == dBehaviorClusteringParams.debugIndex)
		//	cuPrintf("\n%f %f\n", myClusterMembership, otherClusterMembership);
		// --------------------------------
	}

	if (closeNeighbors > 0)
	{
		clusterHitRate /= (float) closeNeighbors;
		realClusterMembership /= closeNeighbors;
	}
	else
		realClusterMembership = -1.0;

	// --------------------------------
	// DEBUG
	//if (sortedIndex == dBehaviorClusteringParams.debugIndex)
	//	cuPrintf("\n%d) %f [%d/%d]\n", sortedIndex, clusterHitRate, closeNeighbors, neighNum);
	// --------------------------------

	newClusterMembership[sortedIndex] = 
		make_float3(myClusterMembership, clusterHitRate, realClusterMembership);
}

// ----------------------------------------------------------------
// ----------------------------------------------------------------

// Set the clusterMembership as the higher membership among all neighbors
// Step of the local propagation algorithm
__global__ void BehaviorClustering_computeClusterMembership_kernel()
{
	int individualIndex = BehaveRT::getIndividualIndex();

	// Setup input
	float3* clusterMembershipFeature = 
		BehaveRT::getInputFeature<float3>( dBehaviorClusteringFields.clusterMembership );

	uint sortedIndex = BehaveRT::EnvGrid3D::getIndividualSortedIndex(individualIndex);
	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	// Ininitalize input and cluster membership
	float myClusterMembership = clusterMembershipFeature[sortedIndex].x;
	float minClusterMembership = myClusterMembership;

	// Get neighbors list
	uint neighNum;												
	uint neighList[Proximity3d_MAX_NEIGHBORS];					
	Proximity3D::getNeighborsList(individualIndex, neighNum, neighList);	

	// Iterate neighbors list
	for (int i = 0; i < neighNum; i ++)
	{
		// Get the neighbor's cluster memberhsip
		uint otherIndividualIndex = neighList[i];
		uint otherSortedIndex = //FETCH(agentHash, otherIndividualIndex).y; // TODO: generalize
			BehaveRT::EnvGrid3D::getIndividualSortedIndex(otherIndividualIndex);
		float otherClusterMembership = clusterMembershipFeature[otherSortedIndex].x;

		if (minClusterMembership > otherClusterMembership)
			minClusterMembership = otherClusterMembership;
	}

	// Store output
	BehaveRT::setOutputFeatureElement<float3>(
		dBehaviorClusteringFields.clusterMembership,
		sortedIndex,
		make_float3(
			minClusterMembership, // Update myClusterMembership
			clusterMembershipFeature[sortedIndex].y, 
			clusterMembershipFeature[sortedIndex].z));
} //BehaviorClustering_computeClusterMembership_kernel

// ----------------------------------------------------------------
// ----------------------------------------------------------------

// Set the individual unique index as cluster membership
// First step of the local propagation algorithm
__global__ void BehaviorClustering_resetClusterMembership_kernel()
{
	int individualIndex = BehaveRT::getIndividualIndex();

	float3* clusterMembershipFeature = 
		BehaveRT::getInputFeature<float3>( dBehaviorClusteringFields.clusterMembership );

	uint sortedIndex = BehaveRT::EnvGrid3D::getIndividualSortedIndex(individualIndex);
	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	// We use "sortedIndex" as individual identificator
	BehaveRT::setOutputFeatureElement<float3>(
		dBehaviorClusteringFields.clusterMembership,
		sortedIndex, // --> myClusterMembership
		make_float3(
			individualIndex, // Set the individual unique index as cluster membership
			clusterMembershipFeature[sortedIndex].y, 
			clusterMembershipFeature[sortedIndex].z));

} // BehaviorClustering_resetClusterMembership_kernel


// ----------------------------------------------------------------
// ----------------------------------------------------------------

// Connection: <individual, leader>
__global__ void BehaviorClustering_CMConnection_kernel()
{
	int individualIndex = BehaveRT::getIndividualIndex();

	float3* clusterMembershipFeature = 
		BehaveRT::getInputFeature<float3>( dBehaviorClusteringFields.clusterMembership );

	uint sortedIndex = BehaveRT::EnvGrid3D::getIndividualSortedIndex(individualIndex);
	if (sortedIndex > dBehaviorClusteringParams.elementsNumber)
		return;

	// myClusterMembership is the "leader"'s sortedIndex
	float myClusterMembership = clusterMembershipFeature[sortedIndex].x;
	
	// Individual's position
	float4 myPos = tex1Dfetch(oldPosTex, individualIndex);
	// Cluster Membership leader's position
	float4 CMPos = tex1Dfetch(oldPosTex, myClusterMembership);
	
	BehaveRT::setOutputFeatureElement<float4>(
		dBehaviorClusteringFields.CMConnector,
		sortedIndex * 2,
		myPos);

	BehaveRT::setOutputFeatureElement<float4>(
		dBehaviorClusteringFields.CMConnector,
		sortedIndex * 2 + 1,
		CMPos);


} // BehaviorClustering_CMConnection_kernel


// ----------------------------------------------------------------
// ----------------------------------------------------------------

// Cause collision between two clusters
// Change the direction of individuals
__global__ void BehaviorClustering_MergeClusters_kernel()
{
	int individualIndex = BehaveRT::getIndividualIndex();
	uint sortedIndex = BehaveRT::EnvGrid3D::getIndividualSortedIndex(individualIndex);
	
	// Setup input
	float3* clusterMembershipFeature = 
		BehaveRT::getInputFeature<float3>( dBehaviorClusteringFields.clusterMembership );

	// Compute leader 
	int myLeader = clusterMembershipFeature[sortedIndex].x; //SLOW conversion

	int mergeLeader;
	if (myLeader == dBehaviorClusteringParams.mergeClusterLeaders[0])
	{
		mergeLeader = dBehaviorClusteringParams.mergeClusterLeaders[1];
	} 
	else if (myLeader == dBehaviorClusteringParams.mergeClusterLeaders[1])
	{
		mergeLeader = dBehaviorClusteringParams.mergeClusterLeaders[0];
	} 
	else
	{
		// This individual is not involved
		// Store the old value
		BehaveRT::setOutputFeatureElement<float4>(
			dOpenSteerWrapperFields.forward,
			individualIndex,
			tex1Dfetch(oldForwardTex, individualIndex));
	
		return;
	}
	
	// Get positions
	// 
	float3 myLeaderPos = make_float3(tex1Dfetch(oldPosTex, myLeader));
	float3 mergeLeaderPos = make_float3(tex1Dfetch(oldPosTex, mergeLeader));

	float3 newDirection = normalize(mergeLeaderPos - myLeaderPos);

	BehaveRT::setOutputFeatureElement<float4>(
		dOpenSteerWrapperFields.forward,
		individualIndex,
		make_float4(newDirection, 
			dOpenSteerWrapperParams.commonMaxSpeed));

}


// ----------------------------------------------------------------
// ----------------------------------------------------------------

// Move a cluster outside the simulation area
__global__ void BehaviorClustering_FreezeCluster_kernel()
{
	int individualIndex = BehaveRT::getIndividualIndex();
	uint sortedIndex = BehaveRT::EnvGrid3D::getIndividualSortedIndex(individualIndex);


	
}
	



// ////////////////////////////////////////////////////////

extern "C"
{
	// ////////////////////////////////////////////////////////////////////////
	// Bundary functions
	void BehaviorClustering::BehaviorClustering_beforeKernelCall()
	{
		OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall();
		bind_field_texture(hBehaviorClusteringFields.featuresVector, featuresVectorTex);
		bind_field_texture(hBehaviorClusteringFields.similarityStats, similarityStatsTex);
		bind_field_texture(hProximity3DFields.neighList, neighIndexesTex);
		
	}

	void BehaviorClustering::BehaviorClustering_afterKernelCall()
	{
		OpenSteerWrapper::OpenSteerWrapper_afterKernelCall();
		unbind_field_texture(featuresVectorTex);
		unbind_field_texture(similarityStatsTex);
		unbind_field_texture(neighIndexesTex);
	}

	void BehaviorClustering::BehaviorClustering_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dBehaviorClusteringFields, &hBehaviorClusteringFields, sizeof(BehaviorClusteringFields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dBehaviorClusteringParams, &hBehaviorClusteringParams, sizeof(BehaviorClusteringParams)) );
	}

	// ////////////////////////////////////////////////////////////////////////
	// Generic kernel calls

	BehaveRT_exportKernelWithName(BehaviorClustering_steerForClustering<true>, 
		BehaviorClustering_steerForClustering);

	BehaveRT_exportKernelWithName(BehaviorClustering_computeNeighSimilarities<true>, 
		BehaviorClustering_computeNeighSimilarities);
	
	BehaveRT_exportKernel(BehaviorClustering_computeNeighNaiveSimilarities);
	BehaveRT_exportKernel(BehaviorClustering_computeAdaptiveSimilarities);

	BehaveRT_exportKernel(BehaviorClustering_steerForClusterCohesion);
	BehaveRT_exportKernel(BehaviorClustering_steerForClusterAlignment);

	BehaveRT_exportKernel(BehaviorClustering_computeNeighNaiveSimilarities_shrd);

	BehaveRT_exportKernel(BehaviorClustering_clusterHitRate);

	BehaveRT_exportKernel(BehaviorClustering_computeClusterMembership_kernel);
	BehaveRT_exportKernel(BehaviorClustering_resetClusterMembership_kernel);
	BehaveRT_exportKernel(BehaviorClustering_CMConnection_kernel);

	BehaveRT_exportKernel(
		BehaviorClustering_MergeClusters_kernel);
	BehaveRT_exportKernel(
		BehaviorClustering_FreezeCluster_kernel);

	// ////////////////////////////////////////////////////////////////////////
	// Custom kernel calls

}



