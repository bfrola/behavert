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

/// EnvGrid3D includes parameters regarding environment and grid subdivition of the space
struct EnvGrid3DParams
{
	/// Side size of the grid
	uint3 gridSize;				

	/// Total number of cell. If the grid is 3D then numCells =  gridSize ^ 3
	BehaveRT::uint numCells;

	/// Position of the grid start vertex 
    float3 worldOrigin;

	/// A scalar radius of the sphere which represent the world
	float3 worldRadius;

	/// \brief Dimension of each side of one cell.
	/// Important parameter for performance-tuning
    float3 cellSize;

	/// \brief Upper-bound to the number of bodies which can enter into a cell at the same moment.
	/// Important parameter for performance-tuning
	BehaveRT::uint maxBodiesPerCell;

	float3 worldCenter;

	bool disableSorting;

	bool lockWorldProportions;
};

/// EnvGrid3D features list
struct EnvGrid3DFields
{
	/// Hash value associated to each body, based on cell membership
	int hash;

	/// Used to map temporaneally the feature to reorder
	int featureToReorder;
};

share_struct(EnvGrid3DParams);
share_struct(EnvGrid3DFields);

// Kernel declarations
extern "C"
{
	namespace EnvGrid3D
	{
		// ////////////////////////////////////////////////////////////////////////
		// Bundary functions
		void EnvGrid3D_beforeKernelCall();
		void EnvGrid3D_afterKernelCall();
		void EnvGrid3D_copyFieldsToDevice();

		void EnvGrid3D_beforeKernelCallSimple();
		void EnvGrid3D_afterKernelCallSimple();

		// ////////////////////////////////////////////////////////////////////////
		// Generic kernel calls
		BehaveRT::genericKernelFuncPointer genericCalcHashDRef();

		/// Reorder a single feature which type is float4
		BehaveRT::genericKernelFuncPointer genericReorderDataRef_float4();

		/// Reorder a single feature which type is uint
		BehaveRT::genericKernelFuncPointer genericReorderDataRef_uint();

		/// Reorder a single feature which type is float
		BehaveRT::genericKernelFuncPointer genericReorderDataRef_float();

		BehaveRT::genericKernelFuncPointer reorderDataFloat4Ref();


		// ////////////////////////////////////////////////////////////////////////
		// Custom kernel calls

		/// Deprecated
		void reorderArray( 
			BehaveRT::uint* agentHash, 
			int agentHashElementSize,
			BehaveRT::uint vboReadArray,
			BehaveRT::uint vboWriteArray,
			float4* oldArray,
			float4* sortedArray,
			int oldArrayElementSize,
			int numBodies,
			int blockSize);
	}

}

