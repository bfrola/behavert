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
#include "include\envgrid3d_kernel.cuh"

share_device_struct(EnvGrid3DParams);
share_device_struct(EnvGrid3DFields);

texture<uint2, 1, cudaReadModeElementType> agentHashTex;

// A generic float4 array texture
texture<float4, 1, cudaReadModeElementType> oldArrayTex;

texture<float4, 1, cudaReadModeElementType> float4Tex;
texture<float2, 1, cudaReadModeElementType> float2Tex;
texture<float, 1, cudaReadModeElementType> floatTex;

texture<uint4, 1, cudaReadModeElementType> uint4Tex;
texture<uint2, 1, cudaReadModeElementType> uint2Tex;
texture<BehaveRT::uint, 1, cudaReadModeElementType> uintTex;

texture<int4, 1, cudaReadModeElementType> int4Tex;
texture<int2, 1, cudaReadModeElementType> int2Tex;
texture<int, 1, cudaReadModeElementType> intTex;


// calculate position in uniform grid
__device__ int3 calcGridPos(float4 p, float3 worldOrigin, float3 cellSize)
{
    int3 gridPos;
    gridPos.x = floor((p.x - worldOrigin.x) / cellSize.x);
    gridPos.y = floor((p.y - worldOrigin.y) / cellSize.y);
    gridPos.z = floor((p.z - worldOrigin.z) / cellSize.z);
    return gridPos;
}


// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

// calculate address in grid from position (clamping to edges)
__device__  BehaveRT::uint calcGridHash (int3 gridPos, uint3 gridSize)
{
    gridPos.x = max(0, min(gridPos.x, gridSize.x-1));
    gridPos.y = max(0, min(gridPos.y, gridSize.y-1));
    gridPos.z = max(0, min(gridPos.z, gridSize.z-1));
    return 
		__mul24(__mul24(gridPos.z, gridSize.y), gridSize.x) + 
		__mul24(gridPos.y, gridSize.x) + 
		gridPos.x;
}

__device__ float4 getCellPosition(int3 gridPos)
{
	float4 position;
	
	position.x = dEnvGrid3DParams.worldOrigin.x + (gridPos.x + 0.5) * dEnvGrid3DParams.cellSize.x;
	position.y = dEnvGrid3DParams.worldOrigin.y + (gridPos.y + 0.5) * dEnvGrid3DParams.cellSize.y;
	position.z = dEnvGrid3DParams.worldOrigin.z + (gridPos.z + 0.5) * dEnvGrid3DParams.cellSize.z;
	position.w = .0f;

    return position;
}

namespace BehaveRT
{
	namespace EnvGrid3D
	{
		__device__ uint getIndividualSortedIndex(uint individualIndex)
		{
			return FETCH(agentHash, individualIndex).y; 
		}
	} // namespace EnvGrid3D
} // namespace BehaveRT