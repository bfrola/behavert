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

#define Proximity3d_MAX_NEIGHBORS	128

/// Proximity3D parameters 
struct Proximity3DParams
{

	/// This param refers to how many neighbors cells each body have to look at during the neighborhood search
	uint3 searchDepth;

	/// Flag used to abilitate the KNN algorithm
	bool useKnn;

	/// Max size of the neighbors lists
	BehaveRT::uint maxNeighbors;

	/// Each neighbors list is divided into a certain number of words which size is 4.
	/// numNeighWordsPerAgent = UPPER( maxNeighbors + 1 / 4 )
	BehaveRT::uint numNeighWordsPerAgent;

	/// Each body search neighbors bodies which are at most distant a certain value from it
	float commonSearchRadius;

	/// Deprecated
	bool useDiscreteApprox;

	/// Deprecated
	bool useDiscreteApproxThisFrame;

	/// Deprecated
	int discreteApproxStep;

	/// Deprecated
	bool useSplittedNeigCalc;

	/// Deprecated
	int exploreFieldSize;
};

share_struct(Proximity3DParams);


/// Proximity3D features declaration
struct Proximity3DFields
{
	/// Index of the firt body included into each cell 
	int cellStart;

	/// Neighbors lists
	int neighList;

	/// Deprecated
	int lastStepIndex;

	/// Deprecated
	int exploreField;
};

share_struct(Proximity3DFields);


// Kernel declarations
extern "C"
{
	// Declare kernels references
	BehaveRT_declareKernel(genericFindCellStardD);
	BehaveRT_declareKernel(calcNeighgborhoodD);
	//BehaveRT_declareKernel(backupHashD);
	//BehaveRT_declareKernel(useOldNeighborhoodD);

	BehaveRT_declareKernel(computeExploreFieldD);
	BehaveRT_declareKernel(splittedFindNeighborhoodD);
	
	namespace Proximity3D
	{
		void Proximity3D_beforeKernelCall();
		void Proximity3D_afterKernelCall();
		void Proximity3D_copyFieldsToDevice();
	}

	
}

