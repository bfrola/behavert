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

#include "BehaveRT.h"
#include "proximity3d_kernel.cuh"

// ----------------

namespace Proximity3D
{
	/**
		\brief This plugIn provides a spactial DB based on m_CommonRes's grid.

		When an other plugIn extends this one, it inheritates the features contained in Proximity3DFields and parameters in Proximity3DParams
	*/
	template <class Super>
	class Proximity3DPlugIn: public Super, public SimEnginePlugIn
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor

		/// XXX install/unistall plugin shoulde be automatic
		Proximity3DPlugIn() { SimEnginePlugIn::installPlugIn(); }
		~Proximity3DPlugIn() { SimEnginePlugIn::uninstallPlugIn(); }
		
		const std::string name() { return "Proximity3DPlugIn"; }	

		const DependenciesList plugInDependencies() 
		{ 
			DependenciesList dependencies;
			dependencies.push_back("Body3DPlugIn");
			dependencies.push_back("EnvGrid3DPlugIn");
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

		/// Runs the neighbors searchig
		void computeNeighborhoods();

		BehaveRT::DeviceArrayWrapper<BehaveRT::uint>* getCellStart() {return m_CellStart; }

		// ////////////////////////////////////////////////////
		// Fields	
	protected:
		BehaveRT::DeviceArrayWrapper<BehaveRT::uint>* m_CellStart;
		BehaveRT::DeviceArrayWrapper<uint4>* m_NieghList;
		BehaveRT::DeviceArrayWrapper<BehaveRT::uint>* m_LastStepIndex;

		BehaveRT::DeviceArrayWrapper<BehaveRT::uint>* m_ExploreField;

		int m_CustomBlockSize;
		int m_ProximityIterationCounter;
		bool m_UseCellSizeAsSearchRadius;
	};
}

using namespace Proximity3D;

// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
// Implementation
template <class Super>
void Proximity3DPlugIn<Super>::install()
{
	char msg[100];

	// Params
	read_config_param(Proximity3D, searchDepth, Uint3);
	
	sprintf(msg, "Proximity3D: SeachDepth: %d %d %d\n", 
			hProximity3DParams.searchDepth.x,
			hProximity3DParams.searchDepth.y,
			hProximity3DParams.searchDepth.z);
	m_CommonRes.getLogger()->log(name(), msg);
	

	read_config_param(Proximity3D, useKnn, Bool);
	read_config_param(Proximity3D, maxNeighbors, Int);
	read_config_param(Proximity3D, useSplittedNeigCalc, Bool);

	m_CustomBlockSize = 
		BehaveRT::StringConverter::parseInt(m_CommonRes.getConfig()->getSetting("customBlockSize", Proximity3DPlugIn::name()));
	
	if (hProximity3DParams.useSplittedNeigCalc)
	{
		hProximity3DParams.exploreFieldSize =
			(hProximity3DParams.searchDepth.x + 2) * 
			(hProximity3DParams.searchDepth.y + 2) * 
			(hProximity3DParams.searchDepth.z + 2);
		
		sprintf(msg, "hProximity3DParams.exploreFieldSize: %d\n", 
			hProximity3DParams.exploreFieldSize);
		m_CommonRes.getLogger()->log("Proximity3DPlugIn", msg);
	}

	hProximity3DParams.numNeighWordsPerAgent =
		( (hProximity3DParams.maxNeighbors) / 4 ) + 1;

	sprintf(msg, "NeighWords#: %d\n", 
		hProximity3DParams.numNeighWordsPerAgent);
	m_CommonRes.getLogger()->log("Proximity3DPlugIn", msg);

	m_UseCellSizeAsSearchRadius = BehaveRT::StringConverter::parseBool(
			m_CommonRes.getConfig()->getSetting(
			"useCellSizeAsSearchRadius", 
			"Proximity3DPlugIn"));

	if (m_UseCellSizeAsSearchRadius)
		hProximity3DParams.commonSearchRadius = 
			hEnvGrid3DParams.cellSize.x;
	else
		read_config_param(Proximity3D, commonSearchRadius, Float);

	// Fields
	m_CellStart = new BehaveRT::DeviceArrayWrapper<BehaveRT::uint>(
			m_CommonRes.getDeviceInterface(), hEnvGrid3DParams.numCells);

	const BehaveRT::uint neighListCount = 
		hProximity3DParams.numNeighWordsPerAgent * hBody3DParams.numBodies;
	
	m_NieghList = new BehaveRT::DeviceArrayWrapper<uint4>(
			m_CommonRes.getDeviceInterface(), neighListCount);

	if (hProximity3DParams.useSplittedNeigCalc)
	{
		m_ExploreField = new DeviceArrayWrapper<BehaveRT::uint>(
			m_CommonRes.getDeviceInterface(), hBody3DParams.numBodies * hProximity3DParams.exploreFieldSize);
		m_ExploreField->bindToField(hProximity3DFields.exploreField);
	}

	m_CellStart->bindToField(hProximity3DFields.cellStart);
	m_NieghList->bindToField(hProximity3DFields.neighList);

	hProximity3DParams.useDiscreteApprox = true;
	hProximity3DParams.useDiscreteApproxThisFrame = false;
	hProximity3DParams.discreteApproxStep = 0;
	
	Proximity3D::Proximity3D_copyFieldsToDevice();

	m_ProximityIterationCounter = 0;
}

// --------------------------------------------------------------

template <class Super>
void Proximity3DPlugIn<Super>::uninstall()
{
	delete m_CellStart;
	delete m_NieghList;
}

// --------------------------------------------------------------

template <class Super>
void Proximity3DPlugIn<Super>::reset()
{
	Super::reset(); // MANDATORY OPERATION

	memset(m_CellStart->getHostArray(), 0xffffffff, m_CellStart->getBytesCount());
	m_CellStart->copyArrayToDevice();

	memset(m_NieghList->getHostArray(), 10, m_NieghList->getBytesCount());
	m_NieghList->copyArrayToDevice();
	m_NieghList->swapPosReadWrite();

	// Initi the arrays
	/*for (BehaveRT::uint i = 0; i < hBody3DParams.numBodies; i ++ )
	{
		BehaveRT::uint initLastStepIndex = i;
		m_LastStepIndex->setHostArrayElement(i, &initLastStepIndex);
	}

	m_LastStepIndex->copyArrayToDevice();
	m_LastStepIndex->swapPosReadWrite();*/

	if (hProximity3DParams.useSplittedNeigCalc)
	{
		for (BehaveRT::uint i = 0; i < hBody3DParams.numBodies * hProximity3DParams.exploreFieldSize; i ++ )
		{
			BehaveRT::uint initExploreField = 0;
			m_ExploreField->setHostArrayElement(i, &initExploreField);
		}

		m_ExploreField->copyArrayToDevice();
		m_ExploreField->swapPosReadWrite();
	}

	//m_ParamListGL->AddParam(
	//	new Param<float>("Search radius", 
	//		hProximity3DParams.commonSearchRadius, 
	//		1.0, 100.0, 1.0, 
	//		&hProximity3DParams.commonSearchRadius));
}

// --------------------------------------------------------------

template <class Super>
void Proximity3DPlugIn<Super>::update(const float elapsedTime)
{
	Super::update(elapsedTime); // MANDATORY OPERATION

	// Insert here the default update operation
}

// --------------------------------------------------------------
// --------------------------------------------------------------

template <class Super>
void Proximity3DPlugIn<Super>::computeNeighborhoods()
{
	// ----------------

	if (m_UseCellSizeAsSearchRadius)
		hProximity3DParams.commonSearchRadius = hEnvGrid3DParams.cellSize.x;

	Proximity3D::Proximity3D_copyFieldsToDevice();

	// ----------------

	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 256,
		genericFindCellStardDRef()); 

	if (hProximity3DParams.useSplittedNeigCalc)
	{
		// Two step: find area to explore and calc neighbors
		m_CommonRes.getDeviceInterface()->kernelCall(
				hBody3DParams.numBodies, 
				m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
				computeExploreFieldDRef(), 
				&Proximity3D_beforeKernelCall, 
				&Proximity3D_afterKernelCall);

		m_ExploreField->swapPosReadWrite();
		
		m_CommonRes.getDeviceInterface()->kernelCall(
				hBody3DParams.numBodies, 
				m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
				splittedFindNeighborhoodDRef(), 
				&Proximity3D_beforeKernelCall, 
				&Proximity3D_afterKernelCall);
	}
	else
	{
		// One-step neighborhood computing
		m_CommonRes.getDeviceInterface()->kernelCall(
			hBody3DParams.numBodies, 
			m_CustomBlockSize,
			calcNeighgborhoodDRef(), 
			&Proximity3D_beforeKernelCall, 
			&Proximity3D_afterKernelCall);
	}

	m_NieghList->swapPosReadWrite();

}


// --------------------------------------------------------------