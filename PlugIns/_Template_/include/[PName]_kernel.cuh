// ----------------------------------------------------------------------------
//
// Copyright (c) 2008-2009 ISISLab - University of Salerno
// Original author: Bernardino Frola <besaro@tin.it>
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

#pragma once

#include "DeviceData.cuh"


/// Body PlugIn parameters list
struct [PName]Params
{
	// Auto-generated code START
	[PParams]
	// Auto-generated code END
};

share_struct([PName]Params);

/// Body PlugIn features declaration
struct [PName]Fields
{
	// Auto-generated code START
	[PFeatures]
	// Auto-generated code END
};

share_struct([PName]Fields);


// Kernel declarations
extern "C"
{
	namespace [PName]
	{
		// Kernel declarations
		BehaveRT_declareKernel( [PName]Example_kernel );

		// Boundary functions
		void [PName]_beforeKernelCall();
		void [PName]_afterKernelCall();
		void [PName]_copyFieldsToDevice();
	}	
}

