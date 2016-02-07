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

#define BehaviorClustering_MAX_FEATUREDIM	32

/// Body PlugIn parameters list
struct BehaviorClusteringParams
{
	float3 clusteringForceParams;
	float3 alignmentClusteringForceParams;
	int featuresCount;

	float3 similarityStatsParams;

	int elementsNumber;

	float avgSimilarityController; // Works on interpolateSimilarity

	int debugIndex;

	float sameClusterMaxRadius;
	int iterationLP; // Local propagation's #iterations

	int mergeClusterLeaders[2];
	int freezeClusterLeader;
};

share_struct(BehaviorClusteringParams);

// -----------------------------------------------------------

/// Body PlugIn features declaration
struct BehaviorClusteringFields
{
	int featuresVector;
	int similarityStats;
	int clusterMembership;
	int neighSimilarities;

	// ---------------
	// MOD 24-10-10
	int CMConnector;
	// ---------------
};


share_struct(BehaviorClusteringFields);

// -----------------------------------------------------------

// Kernel declarations
extern "C"
{
	namespace BehaviorClustering
	{
		// Kernel declarations
		BehaveRT_declareKernel(BehaviorClustering_steerForClustering);

		BehaveRT_declareKernel(BehaviorClustering_computeNeighNaiveSimilarities);
		BehaveRT_declareKernel(BehaviorClustering_computeAdaptiveSimilarities);

		BehaveRT_declareKernel(BehaviorClustering_computeNeighSimilarities);
		BehaveRT_declareKernel(BehaviorClustering_steerForClusterCohesion);
		BehaveRT_declareKernel(BehaviorClustering_steerForClusterAlignment);	

		BehaveRT_declareKernel(BehaviorClustering_computeNeighNaiveSimilarities_shrd);

		BehaveRT_declareKernel(BehaviorClustering_clusterHitRate);

		BehaveRT_declareKernel(BehaviorClustering_computeClusterMembership_kernel);
		BehaveRT_declareKernel(BehaviorClustering_resetClusterMembership_kernel);
		BehaveRT_declareKernel(BehaviorClustering_CMConnection_kernel);

		BehaveRT_declareKernel(BehaviorClustering_MergeClusters_kernel);
		BehaveRT_declareKernel(BehaviorClustering_FreezeCluster_kernel);

		// Boundary functions
		void BehaviorClustering_beforeKernelCall();
		void BehaviorClustering_afterKernelCall();
		void BehaviorClustering_copyFieldsToDevice();
	}	
}

