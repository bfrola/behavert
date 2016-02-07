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
// -----------------
// Change log
//
// 01-2009 bf: Created
// 05-2009 bf: Bug fix
// 09-2009 bf: Bug fix
// 05-2010 bf: Multidimentional data support - setHostArrayElement
// 08-2010 bf: Multidimentional data support - getHostArrayElementRef
//
// ----------------

#pragma once

#include "DeviceDataWrapper.h"
#include "DeviceInterface.cuh"


// ////////////////////////////////////////////////////////////
// Declaration

namespace BehaveRT
{

	/**
		\brief This class provides some utilities for host/device memory allocation.

		Helps the interaction between the device "image" and the host "image" of an array.

		Objects of this class are feature of the simulation engine.

		Similar to GPUArray entity included in the CUDA SDK 2.1.
	*/
	template <class Type>
	class DeviceArrayWrapper : public DeviceDataWrapper
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor

		/**
			Allocate and initialize arrays. Allocate on host and device size * dimension * sizeof(Type) bytes.
			Register the feature on the DeviceDataRepository.
			@param deviceInterface reference to DeviceInterface
			@param size number of elements
			@param dimension number of dimensions (1D, 2D, 3D, ...)
			@param useDoubleBufferingOnDevice double image on device, for doublebuffering
			@param useVbo OpenGL Vertex Buffer Object assotiation
		*/
		DeviceArrayWrapper(
			DeviceInterface* deviceInterface, 
			int size, int dimension = 1, 
			bool useDoubleBufferingOnDevice = true, bool useVbo = false);

		/// Deallocate host and device memory
		~DeviceArrayWrapper();
		
		
		// /////////////////////////////////////////////////////
		// Methods

		/// Returns the array on current read position
		Type* getReadDeviceArray(); 

		/// Returns the array on current write position
		Type* getWriteDeviceArray(); 

		/// Returns true if VBO is enabled
		bool useVbo() { return m_UseVbo; }

		/// Returns the VBO index on current read position
		int getReadVbo() { return m_vbo[m_CurrentPosRead]; }

		/// Returns the VBO index on current write position
		int getWriteVbo() { return m_vbo[m_CurrentPosWrite]; }

		/// Returns the size of one element: sizeof(Type) * dimension
		int getElementBytesCount() { return sizeof(Type) * m_Dimension; } // E.g. With Type = float4 returns 4*4=16

		/// Returns the size of the whole array: sizeof(Type) * dimension * size
		int getBytesCount() { return getElementBytesCount() * m_Size; }

		/// Returns the host "image" of the array
		Type* getHostArray() { return m_HostArray; }

		/// Returns the size of the array
		int getSize() { return m_Size; }

		/// Returns type information: a reference to typeid(Type)
		const type_info* getType() { return &typeid(Type); }

		/// If double-buffering is enabled, swaps the current read position with the current write position
		void swapPosReadWrite();
		
		/// Copies the whole array from host to device
		void copyArrayToDevice() { copyArrayToDevice(0, m_Size); };

		// UNDER DEBUG 27-08-2010 - added multidimentional support
		/// Copies one element of the array from host to device
		void copyArrayToDevice(int offsetElements) { copyArrayToDevice(offsetElements, m_Dimension); }

		/// Copies a window of the array from host to device
		void copyArrayToDevice(int offsetElements, int numElements);
		
		/// Copies the whole array from device to host
		void copyArrayFromDevice() { copyArrayFromDevice(0, m_Size); }

		// UNDER DEBUG 27-08-2010 - added multidimentional support
		/// Copies one element of the array from device to host
		void copyArrayFromDevice(int offsetElements) { copyArrayFromDevice(offsetElements, m_Dimension); }

		/// Copies a window of array from device to host
		void copyArrayFromDevice(int offsetElements, int numElements);

		/**
			@return an element of the host array
			@param index of the element
			@param copyFromDevice if true, copies the element from device
		*/
		Type getHostArrayElement(int index, bool copyFromDevice = false);

		/**
			@return the pointer of the host array element
			@param index of the element
			@param copyFromDevice if true, copies the element from device
		*/
		Type* getHostArrayElementRef(int index, bool copyFromDevice = false);

		/**
			Set an elemento of the host array
			@param index index of the element
			@param element data to write on the array element
			@param copyToDevice if true, copies the element to device
		*/
		void setHostArrayElement(int index, Type* element, bool copyToDevice = false);

		/**
			Create a link between the feature represented by the DeviceArrayWrapper and an element
			of the DeviceDataRepository.
		*/
		void bindToField(int& field) { field = m_DataInfoIndex; }

		/// Set all the element of the host array to val
		void reset(char val);

	protected:
		// /////////////////////////////////////////////////////
		// Fields
		int m_Size;			// The number of element 
		int m_Dimension;	// 1D or 2D or 3D or .. or nD
		Type* m_HostArray;
		Type* m_DeviceArray[2];

		int m_CurrentPosRead;
		int m_CurrentPosWrite;
		bool m_UseDoubleBufferingOnDevice; // XXX
		bool m_UseVbo; 
		DeviceInterface* m_DeviceInterface;

		// Double buffering on VBO
		int m_vbo[2];

		// SimParams.DataInfo assigned index
		int m_DataInfoIndex;

		// /////////////////////////////////////////////////////
		// Static fields
	};
}

// ////////////////////////////////////////////////////////////
// Definition

template <class Type>
BehaveRT::DeviceArrayWrapper<Type>::
DeviceArrayWrapper(
		DeviceInterface* deviceInterface, 
		int size, int dimension, 
		bool useDoubleBufferingOnDevice, 
		bool useVbo) :
	m_DeviceInterface(deviceInterface), 
	m_Size(size), 
	m_Dimension(dimension), 
	m_UseDoubleBufferingOnDevice(useDoubleBufferingOnDevice),
	m_UseVbo(useVbo)
{
	//m_UseDoubleBufferingOnDevice = false;

	// Host allocation
	m_HostArray = new Type[m_Size * m_Dimension];

	// If use VBO create the Vertex Vuffer Object and exit
	if (m_UseVbo)
	{
		m_vbo[0] = m_vbo[1] = m_DeviceInterface->createVBO(getBytesCount());
		
		if (m_UseDoubleBufferingOnDevice)
			m_vbo[1] = m_DeviceInterface->createVBO(getBytesCount()); 
	}
	else
	{
		m_vbo[0] = m_vbo[1] = 0;
	}

	// If do not use VBO exec the normal memory allocation
	if (!m_UseVbo)
		m_DeviceInterface->allocateArray((void**)&m_DeviceArray[0], getBytesCount());

	// Initialize duoble buffering positions and allocate data on device
	m_CurrentPosRead = m_CurrentPosWrite = 0;

	if (m_UseDoubleBufferingOnDevice)
	{
		if (!m_UseVbo)
			m_DeviceInterface->allocateArray((void**)&m_DeviceArray[1], getBytesCount());

		// Differs form m_CurrentPosRead only when double buffering is enalbled
		m_CurrentPosWrite = 1;
	}

	// >>>>>>>>>>>>>>>>>>>>>>>> 21-04-10
	//char msg[100];
	//sprintf(msg, "POINTER %d, %d -- %d, %d\n", 
	//	m_vbo[m_CurrentPosRead],
	//	m_vbo[m_CurrentPosWrite],
	//	m_DeviceArray[m_CurrentPosRead],
	//	m_DeviceArray[m_CurrentPosWrite]);
	//m_CommonRes.getLogger()->log("DAW", msg);
	// <<<<<<<<<<<<<<<<<<<<<<<<<

	// ------------------------
	// Initialize device repository
	// Put the references
	m_DataInfoIndex = 
		m_DeviceInterface->addToDeviceDataRepository(
			m_DeviceArray[m_CurrentPosRead],
			m_DeviceArray[m_CurrentPosWrite],
			m_vbo[m_CurrentPosRead],
			m_vbo[m_CurrentPosWrite],
			getBytesCount(), 
			typeid(Type).name());
}



// ---------------------------------------------------------------------

template <class Type>
BehaveRT::DeviceArrayWrapper<Type>::
~DeviceArrayWrapper()
{
	delete [] m_HostArray;

	if (m_UseVbo)
	{
		m_DeviceInterface->unmapGLBufferObject(m_vbo[0]);

		if (m_UseDoubleBufferingOnDevice)
			m_DeviceInterface->unmapGLBufferObject(m_vbo[1]);

		m_DeviceInterface->deleteVBO(m_vbo[0]);
		
		if (m_UseDoubleBufferingOnDevice)
			m_DeviceInterface->deleteVBO(m_vbo[1]);
	}

	// No VBO
	m_DeviceInterface->freeArray(m_DeviceArray[0]);
	if (m_UseDoubleBufferingOnDevice)
	{
		m_DeviceInterface->freeArray(m_DeviceArray[1]);
	}
}

// ---------------------------------------------------------------------

template <class Type>
void
BehaveRT::DeviceArrayWrapper<Type>::
swapPosReadWrite()
{
	if (!m_UseDoubleBufferingOnDevice)
		return;

	// Usefull only whether m_UseDoubleBufferingOnDevice = true
	//std::swap(m_CurrentPosRead, m_CurrentPosWrite);
	int temp = m_CurrentPosRead;
	m_CurrentPosRead = m_CurrentPosWrite;
	m_CurrentPosWrite = temp;
	
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>> 23-04-10
	// Invert mempory pointers
	// Fixes m_DeviceInterface->refreshDataRepository call
	void* tempPointer = 
		m_DeviceInterface->getDeviceDataRepository()[m_DataInfoIndex].inputDataPointer;

	m_DeviceInterface->getDeviceDataRepository()[m_DataInfoIndex].inputDataPointer =
		m_DeviceInterface->getDeviceDataRepository()[m_DataInfoIndex].outputDataPointer;
	
	m_DeviceInterface->getDeviceDataRepository()[m_DataInfoIndex].outputDataPointer =
		tempPointer;
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<< 23-04-10

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>> 26-04-10
	
	temp = 
		m_DeviceInterface->getDeviceDataRepository()[m_DataInfoIndex].inputVbo;

	m_DeviceInterface->getDeviceDataRepository()[m_DataInfoIndex].inputVbo =
		m_DeviceInterface->getDeviceDataRepository()[m_DataInfoIndex].outputVbo;
	
	m_DeviceInterface->getDeviceDataRepository()[m_DataInfoIndex].outputVbo =
		temp;

	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<

}

// ---------------------------------------------------------------------

template <class Type>
void
BehaveRT::DeviceArrayWrapper<Type>::
copyArrayToDevice(int offsetElements, int numElements)
{
	if (m_UseVbo)
	{
		m_DeviceInterface->copyArrayToDeviceVbo(
			m_vbo[m_CurrentPosWrite],
			m_HostArray,
			offsetElements * getElementBytesCount(), 
			numElements * getElementBytesCount());
		return;
	}

	m_DeviceInterface->copyArrayToDevice(
		m_DeviceArray[m_CurrentPosWrite],
		m_HostArray,
		offsetElements * getElementBytesCount(), 
		numElements * getElementBytesCount());	
}

// ---------------------------------------------------------------------

template <class Type>
void
BehaveRT::DeviceArrayWrapper<Type>::
copyArrayFromDevice(int offsetElements, int numElements)
{
	int vbo = 0;
	if (m_UseVbo)
	{
		vbo = m_vbo[m_CurrentPosRead];	
	}

	m_DeviceInterface->copyArrayFromDevice(
		m_HostArray,
		m_DeviceArray[m_CurrentPosRead],
		vbo,
		offsetElements * getElementBytesCount(), 
		numElements * getElementBytesCount());	
}

// ---------------------------------------------------------------------

template <class Type>
Type
BehaveRT::DeviceArrayWrapper<Type>::
getHostArrayElement(int index, bool copyFromDevice)
{
	//if (m_Dimension > 1)
		// THROW EXCEPTION

	if (copyFromDevice)
	{
		// Retrieve the element from the device
		// Warning: slow operation
		copyArrayFromDevice(index);
	}
	
	Type element;
	// Copy element from the host array
	memcpy(&element, &m_HostArray[index * m_Dimension],
		getElementBytesCount());

	return element;
}

template <class Type>
Type*
BehaveRT::DeviceArrayWrapper<Type>::
getHostArrayElementRef(int index, bool copyFromDevice)
{
	if (copyFromDevice)
	{
		// Retrieve the element from the device
		// Warning: slow operation
		copyArrayFromDevice(index);
	}
	
	//Type* element = new Type[getElementBytesCount()];
	// Copy element from the host array
	//memcpy(element, &m_HostArray[index * m_Dimension],
	//	getElementBytesCount());

	return &m_HostArray[index * m_Dimension];
}

// ---------------------------------------------------------------------

template <class Type>
void
BehaveRT::DeviceArrayWrapper<Type>::
setHostArrayElement(int index, Type* element, bool copyToDevice)
{
	for (int i = 0; i < m_Dimension; i ++)
	{
		m_HostArray[index * m_Dimension + i] = element[i];
	}

	if (copyToDevice)
	{
		// Put the element on device
		// Warning: slow operation
		copyArrayToDevice(index);
	}
}

// ---------------------------------------------------------------------
template <class Type>
Type* BehaveRT::DeviceArrayWrapper<Type>::
getReadDeviceArray()
{
	return m_DeviceArray[m_CurrentPosRead]; 
}

template <class Type>
Type* BehaveRT::DeviceArrayWrapper<Type>::
getWriteDeviceArray()
{ 
	return m_DeviceArray[m_CurrentPosWrite]; 
}

template <class Type>
void BehaveRT::DeviceArrayWrapper<Type>::
reset(char val)
{
	m_DeviceInterface->resetArray(
		 m_DeviceArray[m_CurrentPosWrite], 
		 val, getBytesCount());

	swapPosReadWrite();
}
