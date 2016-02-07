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
struct OpenSteerWrapperParams
{
	float3 commonTarget;

	float commonMaxSpeed;
	float commonMaxForce;
	float commonMass;
	float commonSensingMinRadius;

	float3 separationParams; // Weight, radius, maxCosAngle
	float3 cohesionParams;
	float3 alignmentParams; 

	float3 avoidBase;
	float3 avoidBaseParams; // Weight, radius, UNSET

	bool useForwardVBO; // True if forward is renderable (for openGL geometry instancing)
};

struct OpenSteerWrapperDynParams
{
	float elapsedTime;
};

// Field declaration
struct OpenSteerWrapperFields
{
	int smoothAccel;
	int steerForce;
	int forward;
};

share_struct(OpenSteerWrapperParams);
share_struct(OpenSteerWrapperDynParams);
share_struct(OpenSteerWrapperFields);

// Kernel declarations
extern "C"
{
	// Declare kernels references
	BehaveRT_declareKernel(genericSeparationD);
	BehaveRT_declareKernel(genericCohesionD);
	BehaveRT_declareKernel(genericAlignmentD);
	BehaveRT_declareKernel(steerToAvoidNeighbors);
	BehaveRT_declareKernel(genericSeekingWorldCenterD);
	BehaveRT_declareKernel(genericApplySteeringForceD);

	BehaveRT_declareKernel(moveAwayFromTarget_kernel);
	
	namespace OpenSteerWrapper
	{
		void OpenSteerWrapper_beforeKernelCall();
		void OpenSteerWrapper_afterKernelCall();
		void OpenSteerWrapper_copyFieldsToDevice();
		void OpenSteerWrapper_copyDynParams();
	}

	
}