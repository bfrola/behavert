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

// Params declaration
struct Drawable3DParams
{
	bool neighborhoodColoring;
	float3 colorBase;
	bool disableDrawing;
	BehaveRT::uint renderingType;
	int numVertexes;
	bool useCUDAGeometry;
	
	// The value of these parameter has to be sent to the
	// GPU at each simulation/draw step
	float3 mouseDirection;
	float3 cameraPosition;
};


share_struct(Drawable3DParams);

// Field declaration
struct Drawable3DFields
{
	int color;
	int smoothedColor;
	int geometry;
	int highLightPositions;

	int mouseDistance;
};

share_struct(Drawable3DFields);


// Kernel declarations
extern "C"
{
	// Declare kernels references
	BehaveRT_declareKernel(extractColorFromNeighborhoodD);
	BehaveRT_declareKernel(smoothColorD);
	BehaveRT_declareKernel(createGeometryD);

	BehaveRT_declareKernel(dummyKernel);

	BehaveRT_declareKernel(computeMouseDistance_kernel);
		
	namespace Drawable3D
	{
		void Drawable3D_beforeKernelCall();
		void Drawable3D_afterKernelCall();
		void Drawable3D_copyFieldsToDevice();
	}
}

