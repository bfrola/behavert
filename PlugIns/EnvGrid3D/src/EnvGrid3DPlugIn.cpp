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

#include "BehaveRT.h"
#include "EnvGrid3DPlugIn.h"
#include <math.h>


/// Calculates the grid cell coordinates (gridPos) of a given point in the space
int3 EnvGrid3D::calcGridPosH(float4 p, float3 worldOrigin, float3 cellSize)
{
	int3 gridPos;
    gridPos.x = floor((p.x - worldOrigin.x) / cellSize.x);
    gridPos.y = floor((p.y - worldOrigin.y) / cellSize.y);
    gridPos.z = floor((p.z - worldOrigin.z) / cellSize.z);
    return gridPos;
}

/// Calculates the hash of a given cell, which is indexed by its grid coordinates (gridPos)
uint EnvGrid3D::calcGridHashH (int3 gridPos, uint3 gridSize)
{
	if (gridPos.x < 0 || gridPos.x > gridSize.x ||
		gridPos.y < 0 || gridPos.y > gridSize.y ||
		gridPos.z < 0 || gridPos.z > gridSize.z)
		return 0;
	//gridPos.x = std::max(0, std::min(gridPos.x, gridSize.x-1));
    //gridPos.y = std::max(0, std::min(gridPos.y, gridSize.y-1));
    //gridPos.z = std::max(0, std::min(gridPos.z, gridSize.z-1));
    return 
		gridPos.z * gridSize.y * gridSize.x + 
		gridPos.y * gridSize.x + 
		gridPos.x;
}