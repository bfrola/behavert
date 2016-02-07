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
// 03-09 bf: Created
//
// ----------------

#pragma once

#include "DeviceData.cuh"


/// Body PlugIn parameters list
struct SchoolingParams
{
	float theta;
	float delta;
	float r_r;
	float r_p;
	float eta;
	float w_a;
	float w_o;
	float s;

	// Mersenne Twister parameter
	// This parameters specifies how many random numbers the 
	// engine has to generate for each lane. The number of
	// lanes is fixed to 4096. NPerRng must be a multiple of 2
	int NPerRng;

	// Specifies the weight of the random direction
	float randomDeviationWeight;

};

share_struct(SchoolingParams);

/// Body PlugIn features declaration
struct SchoolingFields
{
	int randomValue;
};

share_struct(SchoolingFields);


// Kernel declarations
extern "C"
{
	namespace Schooling
	{
		// Kernel declarations
		BehaveRT_declareKernel(Schooling_animateSchool);
		BehaveRT_declareKernel(Schooling_attractionOrientation);
		BehaveRT_declareKernel(Schooling_repulsion);
		BehaveRT_declareKernel(Schooling_move);
		BehaveRT_declareKernel(Schooling_generateRandomValues);

		// Boundary functions
		void Schooling_beforeKernelCall();
		void Schooling_afterKernelCall();
		void Schooling_copyFieldsToDevice();

		// User-defined funcs (Mersenne Twister)
		void Schooling_loadMTGPU(const char *fname);
		void Schooling_seedMTGPU(unsigned int seed);
	}	
}

// --------------------------------------------------------
// --------------------------------------------------------
// MERSENNE TWISTER DATA AND PARAMS (imported from CUDA SDK)



// Data structure
typedef struct{
    unsigned int matrix_a;
    unsigned int mask_b;
    unsigned int mask_c;
    unsigned int seed;
} mt_struct_stripped;

// Parameters Random number generator
#define   MT_RNG_COUNT 4096
#define          MT_MM 9
#define          MT_NN 19
#define       MT_WMASK 0xFFFFFFFFU
#define       MT_UMASK 0xFFFFFFFEU
#define       MT_LMASK 0x1U
#define      MT_SHIFT0 12
#define      MT_SHIFTB 7
#define      MT_SHIFTC 15
#define      MT_SHIFT1 18


