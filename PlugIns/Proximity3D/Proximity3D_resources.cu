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

#include "DeviceData.cuh"
#include "common_resources.cu"

#include "include\proximity3d_kernel.cuh"

// Other plugIn dependencies
#include "..\EnvGrid3D\include\envgrid3d_kernel.cuh"
#include "..\EnvGrid3D\envgrid3d_resources.cu"

#include "..\Body\include\body3d_kernel.cuh"
#include "..\Body\body3d_resources.cu"

share_device_struct(Proximity3DParams);
share_device_struct(Proximity3DFields);

texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint4, 1, cudaReadModeElementType> oldNeighListTex;
texture<uint, 1, cudaReadModeElementType> lastStepIndexTex;
texture<uint, 1, cudaReadModeElementType> exploreFieldTex;

// ------------------------------------------------------------------------------
namespace Proximity3D
{
	// collide two spheres using DEM method
	__device__ void checkForNeighbor(float4 posA, float4 posB,
									 uint index, uint index2, // *** MOD added params
									 uint& neighNum, 
									 uint* neighList,
									 float* distList,
									 float& maxDistValue)  // *** MOD added param
	{
		const float dist = calcDist4(posA, posB);

		if (dist > dProximity3DParams.commonSearchRadius)
			return;

		if (neighNum < dProximity3DParams.maxNeighbors)
		{
			neighList[neighNum] = index2;

			if (dProximity3DParams.useKnn)
			{
				// Save the distance
				distList[neighNum] = dist;
					
				// Update the maxdist value
				maxDistValue = max(maxDistValue, dist);
			}

			neighNum ++;
		}
		else if (dProximity3DParams.useKnn && dist < maxDistValue)
		{
			maxDistValue = distList[0];
			uint maxDistIndex = 0;

			// Find the farthest neighbor, and recompute maxDistValue 
			for(uint i = 1; i < neighNum; i ++)
			{
				if (maxDistValue < distList[i])
				{
					maxDistValue = distList[i];
					maxDistIndex = i;
				}
			} //for

			neighList[maxDistIndex] = index2;
			distList[maxDistIndex] = dist;

		}//if 
	}

	// ------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------

	__device__ float4 getCellCenter(int3 gridPos)
	{
		float3 gridRealPos;
		gridRealPos.x = (gridPos.x + 0.5) * 
			dEnvGrid3DParams.cellSize.x + dEnvGrid3DParams.worldOrigin.x;
		gridRealPos.y = (gridPos.y + 0.5) * 
			dEnvGrid3DParams.cellSize.y + dEnvGrid3DParams.worldOrigin.y;
		gridRealPos.z = (gridPos.z + 0.5) * 
			dEnvGrid3DParams.cellSize.z + dEnvGrid3DParams.worldOrigin.z;

		return make_float4(gridRealPos, 1);
	}

	// version using sorted grid
	__device__
	void checkForNeighborsInCell(
		int3   gridPos,
		uint    index,
		float4  pos,
		uint& neighNum, 
		uint* neighList,
		float* distList,
		float& maxDistValue)
	{
		// Exit if the cell is out of the boundings [0, griddim]
		if ((gridPos.x < 0) || (gridPos.x > dEnvGrid3DParams.gridSize.x-1) ||
			(gridPos.y < 0) || (gridPos.y > dEnvGrid3DParams.gridSize.y-1) ||
			(gridPos.z < 0) || (gridPos.z > dEnvGrid3DParams.gridSize.z-1)) {
			return;
		}
		
		const uint gridHash = calcGridHash(gridPos, dEnvGrid3DParams.gridSize);
		
		// get start of bucket for this cell

		//declare_cached_var(uint, bucketStart, _Proximity3DFields.cellStart, gridHash);
		//const uint bucketStart = cellStart[gridHash];
		const uint bucketStart = FETCH(cellStart, gridHash);
		if (bucketStart == 0xffffffff)
			return;   // cell empty
		
		// iterate over particles in this cell
		for(uint i=0; i< dEnvGrid3DParams.maxBodiesPerCell; i++) {
			uint index2 = bucketStart + i;
			//declare_cached_var(uint2, cellData, _EnvGrid3DFields.hash, index2);
			//uint2 cellData = agentHash[index2];
			uint2 cellData = FETCH(agentHash, index2);
			if (cellData.x != gridHash) break;   // no longer in same bucket

			if (index2 != index) 
			{
				//declare_cached_var(float4, pos2, _EnvGrid3DFields.position, index2);
				//float4 pos2 = oldPos[index2];
				float4 pos2 = FETCH(oldPos, index2);
				// collide two spheres
				checkForNeighbor(pos, pos2,
					index, index2, 
					neighNum, neighList,
					distList, maxDistValue);
			}
	        
		}//for
	}


	__device__ void getNeighborsList(int index, uint& neighNum, uint* neighList)
	{
		const int neighIndexBase = 
			__mul24(index, dProximity3DParams.numNeighWordsPerAgent);

		for (int i = 0; i < dProximity3DParams.numNeighWordsPerAgent; i ++)
		{
			uint4 neighWord = FETCH(oldNeighList, 
				neighIndexBase + i);
			
			int shiftedI = __mul24(i, 4) - 1;

			// Only the first word contains the value of neighNum 
			if (shiftedI >= 0)
				neighList[shiftedI] = neighWord.x;
			else
				neighNum = neighWord.x;

			neighList[shiftedI + 1 ] = neighWord.y;
			neighList[shiftedI + 2 ] = neighWord.z;
			neighList[shiftedI + 3 ] = neighWord.w;
				
		}
	}

	__device__ void checkForNeighborsInCell2(
		uint   gridHash,
		uint    index,
		float4  pos,
		uint& neighNum, 
		uint* neighList,
		float* distList,
		float& maxDistValue)
	{
		// get start of bucket for this cell

		//declare_cached_var(uint, bucketStart, _Proximity3DFields.cellStart, gridHash);
		//const uint bucketStart = cellStart[gridHash];
		const uint bucketStart = FETCH(cellStart, gridHash);
		if (bucketStart == 0xffffffff)
			return;   // cell empty
		
		// iterate over particles in this cell
		for(uint i=0; i< dEnvGrid3DParams.maxBodiesPerCell; i++) {
			uint index2 = bucketStart + i;
			//declare_cached_var(uint2, cellData, _EnvGrid3DFields.hash, index2);
			//uint2 cellData = agentHash[index2];
			uint2 cellData = FETCH(agentHash, index2);
			if (cellData.x != gridHash) break;   // no longer in same bucket

			if (index2 != index) 
			{
				//declare_cached_var(float4, pos2, _EnvGrid3DFields.position, index2);
				//float4 pos2 = oldPos[index2];
				float4 pos2 = FETCH(oldPos, index2);
				// collide two spheres
				checkForNeighbor(pos, pos2,
					index, index2, 
					neighNum, neighList,
					distList, maxDistValue);
			}
	        
		}//for
	}
}