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
//
// ----------------
// Change log
//
// 12-08 bf: Created
//
// ----------------

#pragma once

#include "vector_types.h"

namespace BehaveRT
{

/// Definition of the type unsigned int (in addition to vector_types)
typedef unsigned int uint;

// //////////////////////////////////////////////////////////////////////
// ////// STRUCTUREs
// //////////////////////////////////////////////////////////////////////

/**
	\brief Single element of the DeviceDataRepository, so-called "feature".  
	They are shared between host and device
*/
struct DataInfo {
	/// Deprecated
	void* textureRepositoryRef;

	/// Feature data reference on the device memory. Read version.
	void* inputDataPointer;

	/// Feature data reference on the device memory. Write version.
	void* outputDataPointer;

	/// Feature Vertex Buffer Object index (if enabled). Read version.
	uint inputVbo;

	/// Feature Vertex Buffer Object index (if enabled). Write version.
	uint outputVbo;

	/// Size of the feature
	int dataSize;

};



/**
	\brief Deprecated data type
*/
struct SupportedDataType
{
	char* name;
	void* textureRepositoryRef;
};


/**
	\brief Simulation parameters shared between host and device.
*/
struct SimParams {
	/// Deafault CUDA thread block size
	int commonBlockDim;

	/// Decides whether or not use thread syncronization after kernel call
	bool useThreadSync;
};

/**
	\brief Deprecated data type
*/
struct debugType 
{
	float sensing_cosAngle;
	float3 localspace_forward;
	int neighborhood_neighNum;
	uint sensing_neighNum;
	float3 velocity;
	float3 position;
	float3 smoothedAcceleration;
	float3 steeringForce;
};

/**
	\brief This struct contains the GPU information. 
	Used to tune the application.
*/
struct DeviceInfo
{
	char name[256];
	int multiProcessorCount;
	int totalGlobalMem;
	int sharedMemPerBlock;
	int regsPerBlock;
	int totalConstMem;
	int warpSize;
};

// //////////////////////////////////////////////////////////////////////
// ////// TYPEDEFs / CONSTANTs / VARs
// //////////////////////////////////////////////////////////////////////

/// 
#define DEVICEDATAREPOSITORY_SIZE 32

/// Generic kenel signature
typedef void (*genericKernelFuncPointer) ();

/// Kernel's boundary functions signature: before/after kernel call
typedef void (*kernelBoundaryFuncPointer) ();

/// DeviceDataRepository type
typedef DataInfo DeviceDataRepository[DEVICEDATAREPOSITORY_SIZE];

/// Deprecated
typedef SupportedDataType SupportedDataTypeList[DEVICEDATAREPOSITORY_SIZE];

}

/// DeviceDataRepository instance on host
extern "C" BehaveRT::DeviceDataRepository m_DeviceDataRepository;

// //////////////////////////////////////////////////////////////////////
// ////// MACROs
// //////////////////////////////////////////////////////////////////////

/// Deprecated - read an element of a feature from texture
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#define getInputFeatureCachedElement tex1Dfetch

/// Deprecated - read an element of a feature from texture and convert it to float 3
#define FETCH_FLOAT3(t, i) make_float3(FETCH(t, i))

/// This macro facilitates the kernel exportation in the .cu files
#define BehaveRT_exportKernel(kernel) 							\
	BehaveRT::genericKernelFuncPointer kernel##Ref() {	\
		return &kernel;									\
	}	

#define BehaveRT_exportKernelWithName(kernel, name) 			\
	BehaveRT::genericKernelFuncPointer name##Ref() {	\
		return &kernel;									\
	}

#define BehaveRT_getKernelRef(kernelName) kernelName##Ref() 

// ----------------------------------------------------------------------

/// Generic kernel declaration
#define BehaveRT_declareKernel(kernel)									\
	BehaveRT::genericKernelFuncPointer kernel##Ref()			\

/// Declare and initialize a device variable on global memory using 
///	the DeviceDataRepository (read mode)
#define declare_input(arrayName, type, dataInfoIndex)			\
	type* arrayName = (type*)									\
	deviceDataRepository[dataInfoIndex].inputDataPointer

/// Declare and initialize a device variable on global memory using 
///	the DeviceDataRepository (write mode)
#define declare_output(arrayName, type, dataInfoIndex)			\
	type* arrayName = (type*)									\
	deviceDataRepository[dataInfoIndex].outputDataPointer


// ----------------------------------------------------------------------

/// Export a varibale declared in CUDA enviroment to C++
#define share_struct(name)				\
	extern name h##name

/// Create the "host image" and the "device image" of a variable named <name>
/// d<name>: device image
/// h<name>: host image
#define share_device_struct(name)		\
	name h##name;						\
	__constant__ name d##name

// ----------------------------------------------------------------------

/// Bind a feature (identified with the DeviceDataRepository index)
/// on a sprcific texture reference
#define bind_field_texture(dataInfoIndex, textureReference)				\
	CUDA_SAFE_CALL(cudaBindTexture(0, textureReference,	\
		m_DeviceDataRepository[dataInfoIndex].inputDataPointer,			\
		m_DeviceDataRepository[dataInfoIndex].dataSize));			



/// Unbind a texture reference
#define unbind_field_texture(textureReference)						\
	CUDA_SAFE_CALL(cudaUnbindTexture(textureReference));

// ----------------------------------------------------------------------

/// Faciltates to read a parameter from config file
#define read_config_param(pluInName, paramName, type)								\
	h##pluInName##Params.##paramName## = BehaveRT::StringConverter::parse##type(	\
		m_CommonRes.getConfig()->													\
		getSetting(#paramName, ##pluInName##PlugIn::name()))

