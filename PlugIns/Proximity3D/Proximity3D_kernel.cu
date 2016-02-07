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
#include "include\proximity3d_kernel.cuh"
#include "proximity3d_resources.cu"

// Other plugIn dependencies
#include "..\EnvGrid3D\include\envgrid3d_kernel.cuh"
#include "..\EnvGrid3D\envgrid3d_resources.cu"

#include "..\Body\include\body3d_kernel.cuh"
#include "..\Body\body3d_resources.cu"

__global__ void genericFindCellStartD()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
	declare_input(agentHash, uint2, dEnvGrid3DFields.hash);

	uint2 sortedData = agentHash[index];

	// Load hash data into shared memory so that we can look 
	// at neighboring agent's hash value without loading
	// two hash values per thread
	__shared__ uint sharedHash[257];
	sharedHash[threadIdx.x+1] = sortedData.x;
	if (index > 0 && threadIdx.x == 0)
	{
		// first thread in block must load neighbor particle hash
		volatile uint2 prevData = agentHash[index-1];
		sharedHash[0] = prevData.x;
	}
	
	__syncthreads();

	declare_input(cellStart, uint, dProximity3DFields.cellStart);
	//uint* cellStart = (uint*) deviceDataRepository[P3DFields.cellStart].inputDataPointer;

	if (index == 0 || sortedData.x != sharedHash[threadIdx.x])
	{
		cellStart[sortedData.x] = index;
	}

}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

__global__ void genericCalcNeighgborhoodD()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	const float4 pos = FETCH(oldPos, index);
		
    // get address in grid
    const int3 gridPos = calcGridPos(pos, dEnvGrid3DParams.worldOrigin, dEnvGrid3DParams.cellSize);

	uint neighNum = 0;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	// Take the neighbors distances
	float distList[Proximity3d_MAX_NEIGHBORS];
	float maxDistValue = 0;
	const int searchDepth = 1;
	
    // examine only neighbouring cells	
	// Unroll the assembler instruction exactly to 3 repetitions 
    for(int x=-searchDepth; x<=searchDepth; x++) {
        for(int y=-searchDepth; y<=searchDepth; y++) {
            for(int z=-searchDepth; z<=searchDepth; z++) {
				Proximity3D::checkForNeighborsInCell(gridPos + make_int3(x, y, z), index, pos, 
					neighNum, neighList, distList, maxDistValue);
            }
        }
    }

	declare_output(oldNeighList, uint, dProximity3DFields.neighList);	
	const int neighIndexBase = __mul24(index, dProximity3DParams.maxNeighbors + 1);
	oldNeighList[neighIndexBase] = neighNum; 	
	// Copy the neighList to memory
	for (int i = 0; i < neighNum; i ++)
	{
		oldNeighList[neighIndexBase + i + 1] = neighList[i];
	}
}

// ------------------------------------------------------------------

__global__ void computeExploreFieldD()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	const float4 pos = FETCH(oldPos, index);
		
    // get address in grid
    const int3 gridPos = calcGridPos(pos, dEnvGrid3DParams.worldOrigin, dEnvGrid3DParams.cellSize);
	
	uint exploreFieldIndex = 0;
	declare_output(exploreField, uint, dProximity3DFields.exploreField);

	uint baseIndex = __mul24(index, dProximity3DParams.exploreFieldSize);
	for(int x=-1; x<=1; x++) {
		for(int y=-1; y<=1; y++) {
			for(int z=-1; z<=1; z++) {
				// Exit if the cell is out of the boundings [0, griddim]
				int3 currentGridPos = gridPos + make_int3(x, y, z);
				
				if ((currentGridPos.x < 0) || (currentGridPos.x > dEnvGrid3DParams.gridSize.x-1) ||
						(currentGridPos.y < 0) || (currentGridPos.y > dEnvGrid3DParams.gridSize.y-1) ||
						(currentGridPos.z < 0) || (currentGridPos.z > dEnvGrid3DParams.gridSize.z-1)) {
					exploreFieldIndex ++;
					continue;
				}
				uint gridHash = calcGridHash(currentGridPos, dEnvGrid3DParams.gridSize);
				exploreField[baseIndex + exploreFieldIndex] = gridHash;
				exploreFieldIndex ++;
			}
		}
	}

}

__global__ void splittedFindNeighborhoodD()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	//declare_cached_var(float4, pos, _EnvGrid3DFields.position, index);
	//const float4 pos = oldPos[index];
	const float4 pos = FETCH(oldPos, index);
		
    // get address in grid
    const int3 gridPos = calcGridPos(pos, dEnvGrid3DParams.worldOrigin, dEnvGrid3DParams.cellSize);

	//const uint sortedIndex = 0;
	uint neighNum = 0;

	uint neighList[Proximity3d_MAX_NEIGHBORS];

	// Take the neighbors distances
	float distList[Proximity3d_MAX_NEIGHBORS];
	float maxDistValue = 0;

	//declare_input(exploreField, uint, dProximity3DFields.exploreField);

	uint baseIndex = __mul24(index, dProximity3DParams.exploreFieldSize);
	for(int i = 0; i < dProximity3DParams.exploreFieldSize; i++) {
		Proximity3D::checkForNeighborsInCell2(tex1Dfetch(exploreFieldTex, baseIndex + i), 
			index, pos, neighNum, neighList, distList, maxDistValue);
	}

	declare_output(oldNeighList, uint, dProximity3DFields.neighList);
	
	const int neighIndexBase = __mul24(index, dProximity3DParams.maxNeighbors + 1);
	oldNeighList[neighIndexBase] = neighNum; 
	
	// Copy the neighList to memory
	for (int i = 0; i < neighNum; i ++)
	{
		oldNeighList[neighIndexBase + i + 1] = neighList[i];
	}
}


/*__global__ void findNeighborhoodD()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	const float4 pos = FETCH(oldPos, index);
		
	const uint sortedIndex = 0;
	uint neighNum = 0;
	uint neighList[Proximity3d_MAX_NEIGHBORS];

	// Take the neighbors distances
	float distList[Proximity3d_MAX_NEIGHBORS];
	float maxDistValue = 0;

	uint num = FETCH(lastStepIndex, index * dEnvGrid3DParams.maxBodiesPerCell);

	for (int j=0; j<num; j++)
	{
		uint curr = FETCH(lastStepIndex, index * dEnvGrid3DParams.maxBodiesPerCell + j);
		uint bucketStart = FETCH(cellStart, curr);
		uint bucketEnd = FETCH(cellStart, curr);
			
		// iterate over particles in this cell
		for(uint i=0; i< bucketEnd; i++) {
			uint index2 = bucketStart + i;
			//declare_cached_var(uint2, cellData, _EnvGrid3DFields.hash, index2);
			//uint2 cellData = agentHash[index2];
			//uint2 cellData = FETCH(agentHash, index2);
			//if (cellData.x != curr) break;   // no longer in same bucket

			if (index2 != index) 
			{
				//declare_cached_var(float4, pos2, _EnvGrid3DFields.position, index2);
				//float4 pos2 = oldPos[index2];
				float4 pos2 = FETCH(oldPos, index2);
				// collide two spheres
				Proximity3D::checkForNeighbor(pos, pos2,
					index, index2, 
					neighNum, neighList,
					distList, maxDistValue);
			}
	        
		}//for
	}
}*/

/*
__global__ void useOldNeighborhoodD()
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint sortedIndex = tex1Dfetch(agentHashTex, index).y;
	int oldIndex = tex1Dfetch(lastStepIndexTex, sortedIndex);

	//uint neighNum;
	//uint neighList[Proximity3d_MAX_NEIGHBORS];
	//Proximity3D::getNeighborsList(oldIndex, neighNum, neighList);


	declare_input(oldNeighList, uint, dProximity3DFields.neighList);
	declare_output(newNeighList, uint, dProximity3DFields.neighList);

	int neighIndexBase = __mul24(index, dProximity3DParams.maxNeighbors + 1);
	const int oldNeighIndexBase = __mul24(oldIndex, dProximity3DParams.maxNeighbors + 1);

	//int oldNeighIndexBase = __mul24(oldIndex, dProximity3DParams.numNeighWordsPerAgent);
	
	// Copy the neighList to memory
	
	uint neighNum = oldNeighList[oldNeighIndexBase];
	newNeighList[neighIndexBase] = neighNum;
	for (int i = 0; i < neighNum; i ++)
	{
		newNeighList[neighIndexBase + i + 1] = oldNeighList[oldNeighIndexBase + i + 1]; 
	}
}
*/

// ------------------------------------------------------------------

__global__ void backupHashD()
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	uint sortedIndex = tex1Dfetch(agentHashTex, index).y;

	// Sign up the sorted data
	declare_output(lastStepIndex, uint, dProximity3DFields.lastStepIndex);
	lastStepIndex[index] =  sortedIndex;
}


__global__ void reloadOldNeighborhoodD()
{
	
}


extern "C"
{
	// ////////////////////////////////////////////////////////////////////////
	// Global vars

	// ////////////////////////////////////////////////////////////////////////
	// Export kernels
	BehaveRT::genericKernelFuncPointer 
		genericFindCellStardDRef() { return &genericFindCellStartD; }
	
	BehaveRT::genericKernelFuncPointer 
		calcNeighgborhoodDRef() { return &genericCalcNeighgborhoodD; }

	BehaveRT::genericKernelFuncPointer 
		backupHashDRef() { return &backupHashD; }

	BehaveRT::genericKernelFuncPointer 
		computeExploreFieldDRef() { return &computeExploreFieldD; }

	BehaveRT::genericKernelFuncPointer 
		splittedFindNeighborhoodDRef() { return &splittedFindNeighborhoodD; }


	//BehaveRT::genericKernelFuncPointer 
	//	useOldNeighborhoodDRef() { return &useOldNeighborhoodD; }

	void Proximity3D::Proximity3D_beforeKernelCall()
	{
		//init_boundary_func();

		//printf("prox>> ");
		EnvGrid3D::EnvGrid3D_beforeKernelCall();
		
		bind_field_texture(hProximity3DFields.cellStart, cellStartTex);
		bind_field_texture(hProximity3DFields.neighList, oldNeighListTex);
		bind_field_texture(hProximity3DFields.lastStepIndex, lastStepIndexTex);
		bind_field_texture(hProximity3DFields.exploreField, exploreFieldTex);
	}

	void Proximity3D::Proximity3D_afterKernelCall()
	{
		//init_boundary_func();

		EnvGrid3D::EnvGrid3D_afterKernelCall();
		
		unbind_field_texture(cellStartTex);
		unbind_field_texture(oldNeighListTex);
		unbind_field_texture(lastStepIndexTex);
		unbind_field_texture(exploreFieldTex);
	}

	void Proximity3D::Proximity3D_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dProximity3DFields, &hProximity3DFields, sizeof(Proximity3DFields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dProximity3DParams, &hProximity3DParams, sizeof(Proximity3DParams)) );
	}

	
}

