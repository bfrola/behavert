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

#include "BehaveRT.h"
#include "Building3D_kernel.cuh"
#include "Building3DShaders.h"

#include <GL/glew.h>

// ----------------

namespace Building3D
{
	/**
		\brief Allow environment interaction.
		
	*/
	template <class Super>
	class Building3DPlugIn: public Super, public SimEnginePlugIn
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor

		/// XXX install/unistall plugin shoulde be automatic
		Building3DPlugIn() { SimEnginePlugIn::installPlugIn(); }
		~Building3DPlugIn() { SimEnginePlugIn::uninstallPlugIn(); }
		
		const std::string name() { return "Building3DPlugIn"; }	

		const DependenciesList plugInDependencies() 
		{ 
			DependenciesList dependencies;
			dependencies.push_back("Body3DPlugIn");
			dependencies.push_back("EnvGrid3DPlugIn");
			dependencies.push_back("Proximity3DPlugIn");
			dependencies.push_back("OpenSteerWrapperPlugIn");
			dependencies.push_back("Drawable3DPlugIn");
			return dependencies;	 
		}

		// ////////////////////////////////////////////////////
		// Methods
	private:
		/// @override
		void install();
		
		/// @override
		void uninstall();

	public:
		/// @override
		void reset();

		/// @override
		void update(const float elapsedTime);

		// Custom operations
	public:

		/// 
		void computeFloatingBehavior();
		void steerToFollowTerrain();
		void throwIndividuals();
		void manageSourceDestination();

		/// 
		void setBuildingBlock(int3 gridPos, BehaveRT::uint type);

		///
		void setBuildingBlock(float3 point, BehaveRT::uint type);

		///
		void setFlowField(int3 gridPos, float4 flow);

		///
		void setFlowField(float3 point, float4 flow);

		void buildindComplete();

		BehaveRT::DeviceArrayWrapper<float4>* getFlowField() { return m_FlowField; }

		// ////////////////////////////////////////////////////
		// Fields	
	protected:
		BehaveRT::DeviceArrayWrapper<BehaveRT::uint>* m_StateInfo;
		BehaveRT::DeviceArrayWrapper<BehaveRT::uint>* m_BlockInfo;
		BehaveRT::DeviceArrayWrapper<float4>* m_FlowField;

		//BehaveRT::DeviceArrayWrapper<float> m_TerrainData;
		

	};
}

using namespace Building3D;

// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
// Implementation
template <class Super>
void Building3DPlugIn<Super>::install()
{
	
	// Params

	read_config_param(Building3D, individualsSourcePos, Float3);
	read_config_param(Building3D, individualsDestPos, Float3);
	read_config_param(Building3D, individualsDestAttractionForce, Float);
	read_config_param(Building3D, individualsDestSize, Float);

	// Fields
	
	m_StateInfo = new DeviceArrayWrapper<BehaveRT::uint>(
		m_CommonRes.getDeviceInterface(), 
		hBody3DParams.numBodies);

	m_StateInfo->bindToField(hBuilding3DFields.stateInfo);

	m_BlockInfo = new DeviceArrayWrapper<BehaveRT::uint>(
		m_CommonRes.getDeviceInterface(), 
		hEnvGrid3DParams.numCells);

	m_BlockInfo->bindToField(hBuilding3DFields.blockInfo);

	m_FlowField = new DeviceArrayWrapper<float4>(
		m_CommonRes.getDeviceInterface(), 
		hEnvGrid3DParams.numCells);

	m_FlowField->bindToField(hBuilding3DFields.flowField);

	Building3D::Building3D_copyFieldsToDevice();


	// Set custom shaders
	//setBillboardsShaders(Buildings3DVertexShader, Buildings3DPixelShader);
}

// --------------------------------------------------------------

template <class Super>
void Building3DPlugIn<Super>::uninstall()
{
	// deletes
	delete m_StateInfo;
}

// --------------------------------------------------------------

template <class Super>
void Building3DPlugIn<Super>::reset()
{
	Super::reset(); // MANDATORY OPERATION

	// Initi the arrays

	// Initi the arrays
	for (BehaveRT::uint i = 0; i < hBody3DParams.numBodies; i ++ )
	{
		//float3 unitRand = unitRandFloat3();
		//float multipler = 1 - frand() * frand();

		float3 unitRand = make_float3(
			frand() - 0.5, frand() - 0.5, frand() - 0.5);
		float multipler = 2;

		float4 initPos = make_float4(
			(unitRand.x) * hEnvGrid3DParams.worldRadius.x * multipler + hEnvGrid3DParams.worldCenter.x, 
			//(unitRand.x) * hEnvGrid3DParams.worldRadius.x * multipler / 1.5 + 
			//	hEnvGrid3DParams.worldCenter.x + hEnvGrid3DParams.worldRadius.x * multipler / 1.5, 
			100, 
			(unitRand.z) * hEnvGrid3DParams.worldRadius.z * multipler + hEnvGrid3DParams.worldCenter.z, 
			1);

		m_Pos->setHostArrayElement(i, &initPos);
	}

	m_Pos->copyArrayToDevice();
	m_Pos->swapPosReadWrite();

	m_BlockInfo->reset(0);
	m_FlowField->reset(0);

}

// --------------------------------------------------------------

template <class Super>
void Building3DPlugIn<Super>::update(const float elapsedTime)
{
	Super::update(elapsedTime); // MANDATORY OPERATION

	// Insert here the default update operation
}

// --------------------------------------------------------------
// --------------------------------------------------------------



// Custom methods
template <class Super>
void Building3DPlugIn<Super>::computeFloatingBehavior()
{
	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		BehaveRT_getKernelRef(computeFloatingBehavior_kernel), 
		&Building3D_beforeKernelCall, 
		&Building3D_afterKernelCall);

	//m_SteerForce->swapPosReadWrite();
	m_SmoothedAcceleration->swapPosReadWrite();
	m_Forward->swapPosReadWrite();
	m_FlowField->swapPosReadWrite();

	m_CommonRes.getDeviceInterface()->threadSync();
}

// --------------------------------------------------------------
// --------------------------------------------------------------

// Custom methods
template <class Super>
void Building3DPlugIn<Super>::steerToFollowTerrain()
{
	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		BehaveRT_getKernelRef(followTerrain_kernel), 
		&Building3D_beforeKernelCall, 
		&Building3D_afterKernelCall);

	m_SteerForce->swapPosReadWrite();
	m_FlowField->swapPosReadWrite();

	m_CommonRes.getDeviceInterface()->threadSync();
}


// --------------------------------------------------------------

template <class Super>
void Building3DPlugIn<Super>::
setBuildingBlock(
			int3 gridPos,
			 BehaveRT::uint type)
{
	
	if (gridPos.x < 0 || gridPos.y < 0 || gridPos.z < 0 || 
			gridPos.x >= hEnvGrid3DParams.gridSize.x ||
			gridPos.y >= hEnvGrid3DParams.gridSize.y ||
			gridPos.z >= hEnvGrid3DParams.gridSize.z)
		return;

	BehaveRT::uint hash = calcGridHashH(gridPos, hEnvGrid3DParams.gridSize);

 	m_BlockInfo->setHostArrayElement(
		hash, &type);
}


// --------------------------------------------------------------

template <class Super>
void Building3DPlugIn<Super>::
setBuildingBlock(
			float3 point, 
			BehaveRT::uint type)
{
	float4 point4 = make_float4(
		point.x, point.y, point.z, 0);

	int3 gridPos = calcGridPosH(
		point4, hEnvGrid3DParams.worldOrigin, 
		hEnvGrid3DParams.cellSize);

	setBuildingBlock(gridPos, type);
}

// --------------------------------------------------------------

template <class Super>
void Building3DPlugIn<Super>::
setFlowField(int3 gridPos, float4 flow)
{
	BehaveRT::uint hash = calcGridHashH(gridPos, hEnvGrid3DParams.gridSize);

	m_FlowField->setHostArrayElement(
		hash, &flow);
}

// --------------------------------------------------------------

template <class Super>
void Building3DPlugIn<Super>::
setFlowField(
		float3 point, 
		float4 flow)
{
	float4 point4 = make_float4(
		point.x, point.y, point.z, 0);

	int3 gridPos = calcGridPosH(
		point4, hEnvGrid3DParams.worldOrigin, 
		hEnvGrid3DParams.cellSize);

	setFlowField(gridPos, flow);
}


// --------------------------------------------------------------

template <class Super>
void Building3DPlugIn<Super>::
buildindComplete()
{
	m_BlockInfo->copyArrayToDevice();
	m_BlockInfo->swapPosReadWrite();
	m_BlockInfo->copyArrayToDevice();
	m_BlockInfo->swapPosReadWrite();

	m_FlowField->copyArrayToDevice();
	m_FlowField->swapPosReadWrite();
	m_FlowField->copyArrayToDevice();
	m_FlowField->swapPosReadWrite();

}

// --------------------------------------------------------------

template <class Super>
void Building3DPlugIn<Super>::
throwIndividuals()
{
	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		throwIndividualsRef(), 
		&Building3D_beforeKernelCall, 
		&Building3D_afterKernelCall);

	m_Pos->swapPosReadWrite();
	m_Forward->swapPosReadWrite();

	m_CommonRes.getDeviceInterface()->threadSync();
}

// --------------------------------------------------------------

template <class Super>
void Building3DPlugIn<Super>::
manageSourceDestination()
{
	Building3D::Building3D_copyFieldsToDevice();

	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		BehaveRT_getKernelRef(manageSourceDestination_kernel), 
		&Building3D_beforeKernelCall, 
		&Building3D_afterKernelCall);

	m_Pos->swapPosReadWrite();
	m_SteerForce->swapPosReadWrite();

	m_CommonRes.getDeviceInterface()->threadSync();
}