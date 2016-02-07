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
// 03-10 bf: Separated map/unmap methods
//
// ----------------

#include <cutil.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <cstdarg>

#include <GL/glew.h>
#include <GL/glut.h>

#include <GL/gl.h>

#include "DeviceInterface.cuh"
#include "DeviceData.cuh"

#include "common_resources.cu"
#include "cuPrintf.cu"

// 24-02-2011
#include "reduction_kernel.cu"

//extern "C" size_t textureOffset;

#include "Body/body3d_kernel.cu"
#include "EnvGrid3D/envgrid3d_kernel.cu"
#include "Proximity3D/Proximity3D_kernel.cu"
#include "OpenSteerWrapper/OpenSteerWrapper_kernel.cu"
#include "Drawable3D/drawable3d_kernel.cu"
#include "Building3D/Building3D_kernel.cu"
#include "Shapes3D/Shapes3D_kernel.cu"
#include "Schooling/Schooling_kernel.cu"
#include "Schooling/MersenneTwister_kernel.cu"
#include "BehaviorClustering/BehaviorClustering_kernel.cu"

#include <cudagl.h>
#include <cuda_gl_interop.h>

extern "C"
{
	

	namespace BehaveRT
	{

		DeviceInterface::DeviceInterface() 
		{
			m_DeviceDataRepositorySize = 0;

			m_SupportedDataTypeListSize = 0;

			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, 0);

			strcpy(m_DeviceInfo.name, deviceProp.name);
			m_DeviceInfo.multiProcessorCount = deviceProp.multiProcessorCount;
			m_DeviceInfo.totalGlobalMem = deviceProp.totalGlobalMem;
			m_DeviceInfo.sharedMemPerBlock = deviceProp.sharedMemPerBlock;
			m_DeviceInfo.regsPerBlock = deviceProp.regsPerBlock;
			m_DeviceInfo.totalConstMem = deviceProp.totalConstMem;
			m_DeviceInfo.warpSize = deviceProp.warpSize;

			m_DevicePrintfEnabled = false;

			
		}

		void DeviceInterface::cudaInit(int argc, char **argv)
		{   
			CUT_DEVICE_INIT(argc, argv);
		}

		void DeviceInterface::allocateArray(void **devPtr, int size)
		{
			CUDA_SAFE_CALL(cudaMalloc(devPtr, size));
		}

		void DeviceInterface::freeArray(void *devPtr)
		{
			CUDA_SAFE_CALL(cudaFree(devPtr));
		}

		void DeviceInterface::threadSync()
		{
			if (m_HostSimParams.useThreadSync)
				CUDA_SAFE_CALL(cudaThreadSynchronize());
		}
		
		void DeviceInterface::copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int offset, int size)
		{   
			if (vbo)
				CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&device, vbo));
			CUDA_SAFE_CALL(cudaMemcpy((char *) host + offset, (char *) device + offset, size, cudaMemcpyDeviceToHost));
			if (vbo)
				CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo));
		}

		void DeviceInterface::copyArrayToDevice(void* device, const void* host, int offset, int size)
		{
			CUDA_SAFE_CALL(cudaMemcpy((char *) device + offset, (char *) host + offset, size, cudaMemcpyHostToDevice));
		}

		void DeviceInterface::copyArrayToDeviceVbo(unsigned int vbo, const void* host, int offset, int size)
		{
			void* device;
			if (vbo)
				CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&device, vbo));
			CUDA_SAFE_CALL(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
			if (vbo)
				CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo));
		}

		void DeviceInterface::copyConstantToDevice(void* device, void* host, int size)
		{
			CUDA_SAFE_CALL( cudaMemcpyToSymbol(device, host, size) );
		}

		void DeviceInterface::resetArray(void* deviceData, char value, int size)
		{
			CUDA_SAFE_CALL(cudaMemset(deviceData, value, size));
		}


		// ----

		void DeviceInterface::registerGLBufferObject(uint vbo)
		{
			CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo));
		}
		void DeviceInterface::unregisterGLBufferObject(uint vbo)
		{
			CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo));
		}

		// ----

		void DeviceInterface::mapGLBufferObject(void* device, uint vbo)
		{
			CUDA_SAFE_CALL(cudaGLMapBufferObject(&device, vbo));
		}

		void DeviceInterface::unmapGLBufferObject(uint vbo)
		{
			CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo));
		}
			
		// ----
		
		void DeviceInterface::copySimParamsToDevice()
		{
			// copy parameters to constant memory
			CUDA_SAFE_CALL( cudaMemcpyToSymbol(params, &m_HostSimParams, sizeof(SimParams)) );
			CUDA_SAFE_CALL( cudaMemcpyToSymbol(deviceDataRepository, 
				&m_DeviceDataRepository, sizeof(DeviceDataRepository)) );
		}

		uint DeviceInterface::createVBO(uint size)
		{
			GLuint vbo;
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, size, NULL, GL_DYNAMIC_COPY);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			registerGLBufferObject(vbo);
			return vbo;
		}

		void DeviceInterface::deleteVBO(uint vbo)
		{
			unregisterGLBufferObject(vbo);
			glDeleteBuffers(2, (const GLuint*)&vbo);
		}
		

		void DeviceInterface::refreshDataRepository(
			int index,
			void* inputDataPointer,
			void* outputDataPointer,
			uint inputVbo,
			uint outputVbo)
		{
			m_DeviceDataRepository[index].inputDataPointer = inputDataPointer;
			m_DeviceDataRepository[index].outputDataPointer = outputDataPointer;
			m_DeviceDataRepository[index].inputVbo = inputVbo;
			m_DeviceDataRepository[index].outputVbo = outputVbo;
		}

		int DeviceInterface::addToDeviceDataRepository(
			void* inputDataPointer,
			void* outputDataPointer,
			uint inputVbo,
			uint outputVbo,
			int dataSize,
			const char* typeName)
		{
			int lastAssignedIndex = 
				m_DeviceDataRepositorySize;
			
			refreshDataRepository(lastAssignedIndex, 
				inputDataPointer, outputDataPointer, 
				inputVbo, outputVbo);

			m_DeviceDataRepository[lastAssignedIndex].dataSize = dataSize;

			// Assign the texture reference based on type name
			for (int i = 0; i < m_SupportedDataTypeListSize; i ++)
			{
				if (!strcmp(typeName, m_SupportedDataTypeList[i].name))
				{
					m_DeviceDataRepository[lastAssignedIndex].textureRepositoryRef = 
						m_SupportedDataTypeList[i].textureRepositoryRef;
					break;
				}
			}

			m_DeviceDataRepositorySize++;
			return lastAssignedIndex;
		}

		void DeviceInterface::addToSupportedDataTypeList(char* name, void* textureRepositoryRef)
		{
			m_SupportedDataTypeList[m_SupportedDataTypeListSize].name = name;
			m_SupportedDataTypeList[m_SupportedDataTypeListSize].textureRepositoryRef = textureRepositoryRef;
			m_SupportedDataTypeListSize ++;
		}

		// ------------------------------------------------------------------------------
		// ------------------------------------------------------------------------------
		// ------------------------------------------------------------------------------

//#define KERNELCALL_DEBUG

		void DeviceInterface::mapVBOinDeviceDataRepository()
		{
			GLint currentBuffer;
			glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &currentBuffer);

			for(uint i = 0; i < m_DeviceDataRepositorySize; i++)
			{				

				// Map vbo input array
				if (m_DeviceDataRepository[i].inputVbo > 0)
				{
					CUDA_SAFE_CALL(
						cudaGLMapBufferObject(
						(void**)&m_DeviceDataRepository[i].inputDataPointer, 
							m_DeviceDataRepository[i].inputVbo));
				}		
				
				// Map vbo output array
				if (m_DeviceDataRepository[i].outputVbo > 0 &&
					m_DeviceDataRepository[i].outputVbo != 
						m_DeviceDataRepository[i].inputVbo)
				{
					CUDA_SAFE_CALL(
						cudaGLMapBufferObject(
						(void**)&m_DeviceDataRepository[i].outputDataPointer, 
						m_DeviceDataRepository[i].outputVbo));
				} // if
			} // for

			glBindBuffer(GL_ARRAY_BUFFER, currentBuffer);
		} // mapVBO

		void DeviceInterface::mapInputVBOinDeviceDataRepository()
		{
			for(uint i = 0; i < m_DeviceDataRepositorySize; i++)
			{			
				// Map vbo input array
				if (m_DeviceDataRepository[i].inputVbo > 0)
				{
					CUDA_SAFE_CALL(
						cudaGLMapBufferObject(
						(void**)&m_DeviceDataRepository[i].inputDataPointer, 
							m_DeviceDataRepository[i].inputVbo));
				}		
			} // for
		} // mapVBO

		
		void DeviceInterface::mapOutputVBOinDeviceDataRepository()
		{
			for(uint i = 0; i < m_DeviceDataRepositorySize; i++)
			{			
				// Map vbo input array
				if (m_DeviceDataRepository[i].inputVbo > 0)
				{
					CUDA_SAFE_CALL(
						cudaGLMapBufferObject(
						(void**)&m_DeviceDataRepository[i].inputDataPointer, 
							m_DeviceDataRepository[i].inputVbo));
				}		
			} // for
		} // mapVBO

		void DeviceInterface::unmapVBOinDeviceDataRepository()
		{
			for(uint i = 0; i < m_DeviceDataRepositorySize; i++)
			{
				// Unmap vbo input array
				if (m_DeviceDataRepository[i].inputVbo > 0)
				{
					CUDA_SAFE_CALL(cudaGLUnmapBufferObject(
						m_DeviceDataRepository[i].inputVbo));
				}
				// Unmap vbo output array
				if (m_DeviceDataRepository[i].outputVbo > 0 &&
					m_DeviceDataRepository[i].outputVbo != 
						m_DeviceDataRepository[i].inputVbo)
				{
					CUDA_SAFE_CALL(cudaGLUnmapBufferObject(
						m_DeviceDataRepository[i].outputVbo));

					//return;
				}
			} // for
		} // unmapVBO

		void DeviceInterface::unmapInputVBOinDeviceDataRepository()
		{
			for(uint i = 0; i < m_DeviceDataRepositorySize; i++)
			{
				// Unmap vbo input array
				if (m_DeviceDataRepository[i].inputVbo > 0)
				{
					CUDA_SAFE_CALL(cudaGLUnmapBufferObject(
						m_DeviceDataRepository[i].inputVbo));
				}
			} // for
		} // unmapVBO

		void DeviceInterface::unmapOutputVBOinDeviceDataRepository()
		{
			for(uint i = 0; i < m_DeviceDataRepositorySize; i++)
			{
				// Unmap vbo output array
				if (m_DeviceDataRepository[i].outputVbo > 0 &&
					m_DeviceDataRepository[i].outputVbo != 
						m_DeviceDataRepository[i].inputVbo)
				{
					CUDA_SAFE_CALL(cudaGLUnmapBufferObject(
						m_DeviceDataRepository[i].outputVbo));

					//return;
				}
			} // for
		} // unmapVBO

		void DeviceInterface::kernelCall(
			int totalThreadNum,
			int blockSize, 
			genericKernelFuncPointer kernelRef,
			kernelBoundaryFuncPointer beforeKernelRef,
			kernelBoundaryFuncPointer afterKenelRef)
		{
			int numThreads, numBlocks;

#ifdef KERNELCALL_DEBUG
			// DEBUG - High CPU usage - 09/03/2010
			unsigned int timerCPU;
			cutilCheckError(cutCreateTimer(&timerCPU));
			cutStartTimer(timerCPU);
			printf("Kernel call DEBUG timings\n");
#endif

			// Update deviceDataRepository on device
			CUDA_SAFE_CALL( cudaMemcpyToSymbol(deviceDataRepository, 
				&m_DeviceDataRepository, sizeof(DeviceDataRepository)) );
			
			computeGridSize(
				totalThreadNum, blockSize, 
				numBlocks, numThreads);	

#ifdef KERNELCALL_DEBUG
			// DEBUG - High CPU usage - 09/03/2010
			cutStopTimer(timerCPU);
			printf("\tbind (%d): %f ms\n",
				m_DeviceDataRepositorySize, 
				cutGetTimerValue(timerCPU));
			cutResetTimer(timerCPU);
			cutStartTimer(timerCPU);
#endif

			if (beforeKernelRef != NULL)
				beforeKernelRef();

#ifdef KERNELCALL_DEBUG
			cutStopTimer(timerCPU);
			printf("\tbeforeOps: %f ms\n",
				cutGetTimerValue(timerCPU));
			cutResetTimer(timerCPU);
			cutStartTimer(timerCPU);
#endif

			kernelRef<<< numBlocks, numThreads >>>();

#ifdef KERNELCALL_DEBUG
			cutStopTimer(timerCPU);
			printf("\tkernelcall: %f ms\n",
				cutGetTimerValue(timerCPU));
			cutResetTimer(timerCPU);
			cutStartTimer(timerCPU);
#endif

			if (afterKenelRef != NULL)
				afterKenelRef();

#ifdef KERNELCALL_DEBUG
			cutStopTimer(timerCPU);
			printf("\tkafterOps: %f ms\n",
				cutGetTimerValue(timerCPU));
			cutResetTimer(timerCPU);
			cutStartTimer(timerCPU);
#endif
			
			//unmapVBOinDeviceDataRepository();
			
			threadSync();

#ifdef KERNELCALL_DEBUG
			cutStopTimer(timerCPU);
			printf("\tunbind: %f ms\n\n",
				cutGetTimerValue(timerCPU));
			cutResetTimer(timerCPU);
#endif

		}	
		// ------------------------------------------------------------------------------
		
		// WARNING: deleted map/unmap VBO operation
		void DeviceInterface::kernelCallUsingFields(
			int* fieldList, int fieldListSize,
			int totalThreadNum,
			int blockSize, 
			genericKernelFuncPointer kernelRef,
			kernelBoundaryFuncPointer beforeKernelRef,
			kernelBoundaryFuncPointer afterKenelRef)
		{
			int numThreads, numBlocks;

			//mapVBOinDeviceDataRepository();

			// Update deviceDataRepository on device
			CUDA_SAFE_CALL( cudaMemcpyToSymbol(deviceDataRepository, 
				&m_DeviceDataRepository, sizeof(DeviceDataRepository)) );
			
			computeGridSize(
				totalThreadNum, blockSize, 
				numBlocks, numThreads);	

			if (beforeKernelRef != NULL)
				beforeKernelRef();

			kernelRef<<< numBlocks, numThreads >>>();

			if (afterKenelRef != NULL)
				afterKenelRef();

			//unmapVBOinDeviceDataRepository();

			threadSync();
		}

		void DeviceInterface::devicePrintfInit()
		{
			if (!m_DevicePrintfEnabled)
				return; 

			cudaPrintfInit();
			//printf("DeviceInterface: WARNING, cuPrintf enabled (low performances)\n");
		}

		void DeviceInterface::devicePrintfDisplay()
		{
			if (!m_DevicePrintfEnabled)
				return; 

			cudaPrintfDisplay(stdout, false);
		}
		
		void DeviceInterface::devicePrintfEnd()
		{
			if (!m_DevicePrintfEnabled)
				return; 

			cudaPrintfEnd();
		}



	} // Class

}   // extern "C"


