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

#include "DeviceData.cuh"

template <class T>
	void 
	reduce(int size, int threads, int blocks, 
		   int whichKernel, T *d_idata, T *d_odata);

extern "C"
{

	

namespace BehaveRT
{
	
/**
	\brief Interface between c++ classes and CUDA kernels.
*/
class DeviceInterface
{

public:
	
	///Default constructor: intialize the CUDA enviroment and fill the DataInfo structure
	DeviceInterface();

	////////////////////////////////////////////////
	// CUDA common operations

	/// Initialize the CUDA environment
	void cudaInit(int argc, char **argv);

	/// Allocate devPtr on device
	void allocateArray(void **devPtr, int size);

	/// Deallocate devPtr on device
	void freeArray(void *devPtr);
	
	/// Syncronize the device threads
	void threadSync();	

	/**
		Copy data from device global memory.
		@param host destination
		@param device source
		@param vbo if != 0, the source is the device memory region referenced by the OpenGL VBO (Vertex Buffer Object)
		@param offset reading start point
		@param size number of bytes to copy
	*/
	void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int offset, int size);

	/**
		Copy data to device global memory (using OpenGL VBO).
		@param vbo destination: device memory region referenced by the VBO (Vertex Buffer Object)
		@param host source
		@param offset reading start point
		@param size number of bytes to copy
	*/
	void copyArrayToDeviceVbo(unsigned int vbo, const void* host, int offset, int size);

	/**
		Copy data to device global memory.
		@param device destination
		@param host source
		@param offset reading start point
		@param size number of bytes to copy
	*/
	void copyArrayToDevice(void* device, const void* host, int offset, int size);

	/**
		Copy data to device costant memory.
		@param device destination
		@param host source
		@param offset reading start point
		@param size number of bytes to copy
	*/
	void copyConstantToDevice(void* device, void* host, int size);
	
	/**
		Initialize an array on device global memory
		@param deviceData array reference
		@param value constant valut to put into the array
		@param size number of bytes to write
	*/
	void resetArray(void* deviceData, char value, int size);

	/**
		Initializes the OpenGL VBO (Vertex Buffer Object) memory region on device
		@param vbo Vertex Buffer Object index
	*/
	void registerGLBufferObject(unsigned int vbo);

	/**
		Finalizes the OpenGL VBO (Vertex Buffer Object) memory region on device
		@param vbo Vertex Buffer Object index
	*/
	void unregisterGLBufferObject(unsigned int vbo);

	/**
		Creates an association between an OpenGL VBO (Vertex Buffer Object) and an array on device global memory 
		@param device destination
		@param vbo Vertex Buffer Object index
	*/
	void mapGLBufferObject(void* device, unsigned int vbo);

	/**
		Deletes an association between an OpenGL VBO (Vertex Buffer Object) and an array on device global memory 
		@param vbo Vertex Buffer Object index
	*/
	void unmapGLBufferObject(unsigned int vbo);

	/**
		Copy both the hostSimParams and the DeviceDataRepository to the device constant memory
	*/
	void copySimParamsToDevice();

	/**
		Allocate a region of the device global memory on OpenGL VBO (Vertex Buffer Object)
		@param size allocation size
	*/
	unsigned int createVBO(unsigned int size);

	/**
		Deallocate the region of the device global memory associated to an OpenGL VBO (Vertex Buffer Object)
		@param vbo Vertex Buffer Object index
	*/
	void deleteVBO(unsigned int vbo);

	////////////////////////////////////////////////
	// DeviceDataRepository issues

	int addToDeviceDataRepository(
		void* inputDataPointer,
		void* outputDataPointer,
		unsigned int inputVbo,
		unsigned int outputVbo, 
		int dataSize,
		const char* typeName);

	void refreshDataRepository(
		int index,
		void* inputDataPointer,
		void* outputDataPointer,
		unsigned int inputVbo,
		unsigned int outputVbo);

	void addToSupportedDataTypeList(char* name, void* textureRepositoryRef) ;

	void mapVBOinDeviceDataRepository();
	void unmapVBOinDeviceDataRepository();
	
	void mapInputVBOinDeviceDataRepository();
	void mapOutputVBOinDeviceDataRepository();
	void unmapInputVBOinDeviceDataRepository();
	void unmapOutputVBOinDeviceDataRepository();

	////////////////////////////////////////////////
	// kernel calls wrappers

	/**
		Generic CUDA kernel call. The parameter blockSize is set to the defualt value.
		This function map and unmap the data referenced by the DeviceDataRepository.
		@param totalThreadNum number of threads you have to launch
		@param kernelRef reference (function pointer) to the kernel you have to execute
		@param beforeKernelRef reference (function pointer) to the function you have to execute immediatly before the kernel
		@param afterKenelRef reference (function pointer) to function you have to execute immediatly after the kernel
	*/
	void kernelCall(
		int totalThreadNum, 
		genericKernelFuncPointer kernelRef,
		kernelBoundaryFuncPointer beforeKernelRef = NULL,
		kernelBoundaryFuncPointer afterKenelRef = NULL);

	/**
		Generic CUDA kernel call. The parameter blockSize is set to the defualt value.
		This function map and unmap the data referenced by the DeviceDataRepository.
		@param totalThreadNum number of threads you have to launch
		@param blockSize dimension of the CUDA block. The GPU will launch exaclty totalThreadNum/blockSize blocks.
		@param kernelRef reference (function pointer) to the kernel you have to execute
		@param beforeKernelRef reference (function pointer) to the function you have to execute immediatly before the kernel
		@param afterKenelRef reference (function pointer) to function you have to execute immediatly after the kernel
	*/
	void kernelCall(
		int totalThreadNum,
		int blockSize, 
		genericKernelFuncPointer kernelRef,
		kernelBoundaryFuncPointer beforeKernelRef = NULL,
		kernelBoundaryFuncPointer afterKenelRef = NULL);

	/**
		Generic CUDA kernel call. The parameter blockSize is set to the defualt value.
		This function map and unmap the data referenced by the DeviceDataRepository. 
		@param fieldList array of indexes of the DeviceDataRepository to map/unmap
		@param fieldListSize the size of the array fieldList
		@param totalThreadNum number of threads you have to launch
		@param blockSize dimension of the CUDA block. The GPU will launch exaclty totalThreadNum/blockSize blocks.
		@param kernelRef reference (function pointer) to the kernel you have to execute
		@param beforeKernelRef reference (function pointer) to the function you have to execute immediatly before the kernel
		@param afterKenelRef reference (function pointer) to function you have to execute immediatly after the kernel
	*/
	void DeviceInterface::kernelCallUsingFields(
		int* fieldList, int fieldListSize,
		int totalThreadNum,
		int blockSize, 
		genericKernelFuncPointer kernelRef,
		kernelBoundaryFuncPointer beforeKernelRef = NULL,
		kernelBoundaryFuncPointer afterKenelRef = NULL);

	// ------------------------------

	// cuPrintf functions wrappers
	void DeviceInterface::devicePrintfInit();
	void DeviceInterface::devicePrintfDisplay();
	void DeviceInterface::devicePrintfEnd();
	
	////////////////////////////////////////////////
	// Get/Set methods

	/// @return device information
	DeviceInfo getDeviceInfo() { return m_DeviceInfo; }

	/// @return simulation parameters
	SimParams& getHostSimParams() { return m_HostSimParams; }

	/// @return a reference to the DeviceDataRepository
	DataInfo* getDeviceDataRepository() { return m_DeviceDataRepository; }

	/**
		@return an element of the DeviceDataRepository
		@param index element index
	*/
	DataInfo* getDeviceData(unsigned int index) { return &m_DeviceDataRepository[index]; }

	/// @return the last inserted element into the DeviceDataRepository
	DataInfo* getLastDeviceData() { return &m_DeviceDataRepository[m_DeviceDataRepositorySize]; }

	/// Augments the DeviceDataRepository size
	void incrementDataRepositorySize() { m_DeviceDataRepositorySize ++; }

	/// @return DeviceDataRepository size
	unsigned int getDeviceDataRepositorySize() { return m_DeviceDataRepositorySize; }

	void setDevicePrintfEnabled(bool devicePrintfEnabled) { m_DevicePrintfEnabled = devicePrintfEnabled; }

private:
	/// Device information
	DeviceInfo m_DeviceInfo;

	/// Simulation parameters
	SimParams m_HostSimParams;

	/// Data repository size
	unsigned int m_DeviceDataRepositorySize;

	/// Deprecated
	SupportedDataTypeList m_SupportedDataTypeList;

	/// Deprecated
	int m_SupportedDataTypeListSize;

	bool m_DevicePrintfEnabled;

};

} // Namespace

} // Extern "C"

