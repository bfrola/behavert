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
#include "include\body3d_kernel.cuh"
#include "body3d_resources.cu"

// Other plugIn dependencies


extern "C"
{
	// ////////////////////////////////////////////////////////////////////////
	// Bundary functions
	void Body3D::Body3D_beforeKernelCall()
	{
		/*printf("Pointer: %d\n", 
			m_DeviceDataRepository[hBody3DFields.position].inputDataPointer);
		printf("Size: %d\n", 
			m_DeviceDataRepository[hBody3DFields.position].dataSize);*/

		bind_field_texture(hBody3DFields.position, oldPosTex);
	
		/*
		// #define bind_field_texture(dataInfoIndex, textureReference)			
		size_t offset;			
		CUDA_SAFE_CALL(cudaBindTexture(&offset, oldPosTex,				
			m_DeviceDataRepository[hBody3DFields.position].inputDataPointer,		
			m_DeviceDataRepository[hBody3DFields.position].dataSize));
		*/
	}

	void Body3D::Body3D_afterKernelCall()
	{
		unbind_field_texture(oldPosTex);
	}

	void Body3D::Body3D_copyFieldsToDevice()
	{
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dBody3DFields, &hBody3DFields, sizeof(Body3DFields)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(dBody3DParams, &hBody3DParams, sizeof(Body3DParams)) );
	}

	// ////////////////////////////////////////////////////////////////////////
	// Generic kernel calls

	// ////////////////////////////////////////////////////////////////////////
	// Custom kernel calls

}



