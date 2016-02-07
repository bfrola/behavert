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
// 07-01-09 bf: Created
//
// ----------------

#pragma once

#include "BehaveRT.h"
#include "OpenSteerWrapper_kernel.cuh"

#include "OpenSteer/SimpleVehicle.h"

/// This Classes wrapper between BehaveRT and OpenSteer library. 
namespace OpenSteerWrapper
{
	OpenSteer::Vec3 float32Vec3(float3);
	OpenSteer::Vec3 float42Vec3(float4);
	OpenSteer::Vec3 int32Vec3(int3);
	OpenSteer::Vec3 uint32Vec3(uint3);
	float3 Vec32float3(OpenSteer::Vec3);
	float4 Vec32float4(OpenSteer::Vec3, float w);


	/**
		\brief Wrapper between BehaveRT and OpenSteer library.
	*/
	template <class Super>
	class OpenSteerWrapperPlugIn: public Super, public SimEnginePlugIn
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor

		/// XXX install/unistall plugin shoulde be automatic
		OpenSteerWrapperPlugIn() { SimEnginePlugIn::installPlugIn(); }
		~OpenSteerWrapperPlugIn() { SimEnginePlugIn::uninstallPlugIn(); }
		
		const std::string name() { return "OpenSteerWrapperPlugIn"; }	

		const DependenciesList plugInDependencies() 
		{ 
			DependenciesList dependencies;
			dependencies.push_back("EnvGrid3DPlugIn");
			dependencies.push_back("Proximity3DPlugIn");
			// Put here other dependecies
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
		void calcInverseHash();

		void computeCohesions();
		void computeSeparations();
		void computeAlignments();
		void computeSeekingsWorldCenter();

		void steerAndSlowToAvoidNeighbors();

		void steerForMoveAwayBaseTarget();

		void applySteeringForces(float elapsedTime);

		void resetSteerForce();

		BehaveRT::DeviceArrayWrapper<float4>* getForward() { return m_Forward; };

		// ////////////////////////////////////////////////////
		// Fields	
	protected:
		// PlugIn data

		BehaveRT::DeviceArrayWrapper<float4>* m_Forward;
		BehaveRT::DeviceArrayWrapper<float4>* m_Up;
		BehaveRT::DeviceArrayWrapper<float4>* m_SteerForce;
		BehaveRT::DeviceArrayWrapper<float4>* m_SmoothedAcceleration;
	};
}
// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
// Implementation

using namespace OpenSteerWrapper;

template <class Super>
void OpenSteerWrapperPlugIn<Super>::install()
{

	// Params
	read_config_param(OpenSteerWrapper, separationParams, Float3);
	read_config_param(OpenSteerWrapper, cohesionParams, Float3);
	read_config_param(OpenSteerWrapper, alignmentParams, Float3);
	read_config_param(OpenSteerWrapper, commonMass, Float);
	read_config_param(OpenSteerWrapper, commonMaxSpeed, Float);
	read_config_param(OpenSteerWrapper, commonMaxForce, Float);
	read_config_param(OpenSteerWrapper, commonSensingMinRadius, Float);
	read_config_param(OpenSteerWrapper, avoidBaseParams, Float3);
	read_config_param(OpenSteerWrapper, useForwardVBO, Bool);

	// Default values for avoidBaseParams
	if (hOpenSteerWrapperParams.avoidBaseParams.x == 0.0f)
	{
		hOpenSteerWrapperParams.avoidBaseParams.x = 500.0f;  // SteerForce
		hOpenSteerWrapperParams.avoidBaseParams.y = 500.0f;  // Acceleration
		hOpenSteerWrapperParams.avoidBaseParams.z = 1000.0f; // Radius
	}

	// Fields
	m_SmoothedAcceleration = new BehaveRT::DeviceArrayWrapper<float4>(
			m_CommonRes.getDeviceInterface(), hBody3DParams.numBodies);

	EnvGrid3DPlugIn::addToFeaturesToReorder(
		m_SmoothedAcceleration, m_SmoothedAcceleration->getType());

	m_SmoothedAcceleration->bindToField(hOpenSteerWrapperFields.smoothAccel);

	m_Forward = new BehaveRT::DeviceArrayWrapper<float4>(
			m_CommonRes.getDeviceInterface(), hBody3DParams.numBodies,
			1, true, 
			hOpenSteerWrapperParams.useForwardVBO); 
				// hOpenSteerWrapperParams.useForwardVBO = TRUE 
				// for openGL geometry instancing
			
	m_Forward->bindToField(hOpenSteerWrapperFields.forward);

	m_Up = new BehaveRT::DeviceArrayWrapper<float4>(
			m_CommonRes.getDeviceInterface(), hBody3DParams.numBodies);

	EnvGrid3DPlugIn::addToFeaturesToReorder(m_Forward, m_Forward->getType());

	m_SteerForce = new BehaveRT::DeviceArrayWrapper<float4>(
			m_CommonRes.getDeviceInterface(), hBody3DParams.numBodies);

	m_SteerForce->bindToField(hOpenSteerWrapperFields.steerForce);

	OpenSteerWrapper::OpenSteerWrapper_copyFieldsToDevice();
}

// --------------------------------------------------------------

template <class Super>
void OpenSteerWrapperPlugIn<Super>::uninstall()
{
	delete m_Forward;
	delete m_Up;
	delete m_SteerForce;
	delete m_SmoothedAcceleration;
}

template <class Super>
void OpenSteerWrapperPlugIn<Super>::reset()
{
	Super::reset(); // MANDATORY OPERATION

	for (int i = 0; i < hBody3DParams.numBodies; i ++ )
	{
		float3 unitRand = unitRandFloat3();
		float4 initVec = make_float4(
			unitRand.x,
			unitRand.y,
			unitRand.z, 
			hOpenSteerWrapperParams.commonMaxSpeed);

		m_Forward->setHostArrayElement(i, &initVec);

		initVec = make_float4(0, 0, 0, 0);
		m_SmoothedAcceleration->setHostArrayElement(i, &initVec);
		
	}

	m_Forward->copyArrayToDevice();
	m_Forward->swapPosReadWrite();
	m_Forward->copyArrayToDevice();
	m_Forward->swapPosReadWrite();

	m_SmoothedAcceleration->copyArrayToDevice();
	m_SmoothedAcceleration->swapPosReadWrite();
	m_SmoothedAcceleration->copyArrayToDevice();
	m_SmoothedAcceleration->swapPosReadWrite();

	// -----------------------------------

	m_ParamListGL->AddParam(
		new Param<float>("Max speed", 
			hOpenSteerWrapperParams.commonMaxSpeed, 
			0.1, 50.0, 1.0, 
			&hOpenSteerWrapperParams.commonMaxSpeed));

	m_ParamListGL->AddParam(
		new Param<float>("Max force", 
			hOpenSteerWrapperParams.commonMaxForce, 
			0.1, 50.0, 1.0, 
			&hOpenSteerWrapperParams.commonMaxForce));

	m_ParamListGL->AddParam(
		new Param<float>("Separation W", 
			hOpenSteerWrapperParams.separationParams.x, 
			0.0, 20.0, 0.5, 
			&hOpenSteerWrapperParams.separationParams.x));

	m_ParamListGL->AddParam(
		new Param<float>("Separation R", 
			hOpenSteerWrapperParams.separationParams.y, 
			0.0, 20.0, 0.5, 
			&hOpenSteerWrapperParams.separationParams.y));


	//m_ParamListGL->AddParam(
	//	new Param<float>("Cohesion W", 
	//		hOpenSteerWrapperParams.cohesionParams.x, 
	//		0.0, 5.0, 0.5, 
	//		&hOpenSteerWrapperParams.cohesionParams.x));

	//m_ParamListGL->AddParam(
	//	new Param<float>("Alignment W", 
	//		hOpenSteerWrapperParams.alignmentParams.x, 
	//		0.0, 5.0, 0.5, 
	//		&hOpenSteerWrapperParams.alignmentParams.x));
}

// --------------------------------------------------------------

template <class Super>
void OpenSteerWrapperPlugIn<Super>::update(const float elapsedTime)
{
	Super::update(elapsedTime); // MANDATORY OPERATION
	// Insert here the default update operation
}

// --------------------------------------------------------------
// --------------------------------------------------------------

template <class Super>
void OpenSteerWrapperPlugIn<Super>::calcInverseHash()
{
	m_CommonRes.getDeviceInterface()->calcInverseHash(
		(uint *) m_AgentHash->getReadDeviceArray(),
		(uint *) m_InverseAgentHash->getReadDeviceArray(),
		hBody3DParams.numBodies);
}


template <class Super>
void OpenSteerWrapperPlugIn<Super>::computeSeparations()
{
	if (hOpenSteerWrapperParams.separationParams.x == 0.0f)
		return;

	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		genericSeparationDRef(),
		&OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall,
		&OpenSteerWrapper::OpenSteerWrapper_afterKernelCall);
	
	m_SteerForce->swapPosReadWrite();
		
} // computeSeparations

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

template <class Super>
void OpenSteerWrapperPlugIn<Super>::steerAndSlowToAvoidNeighbors()
{
	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		steerToAvoidNeighborsRef(),
		&OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall,
		&OpenSteerWrapper::OpenSteerWrapper_afterKernelCall);
	
	m_SteerForce->swapPosReadWrite();
	m_Forward->swapPosReadWrite();

	m_CommonRes.getDeviceInterface()->threadSync();
} // steerAndSlowToAvoidNeighbors

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

template <class Super>
void OpenSteerWrapperPlugIn<Super>::computeAlignments()

{	
	if (hOpenSteerWrapperParams.alignmentParams.x == 0.0f)
		return;

	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		genericAlignmentDRef(),
		&OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall,
		&OpenSteerWrapper::OpenSteerWrapper_afterKernelCall);

	
	m_SteerForce->swapPosReadWrite();

	m_CommonRes.getDeviceInterface()->threadSync();

} // computeAlignments

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

template <class Super>
void OpenSteerWrapperPlugIn<Super>::computeCohesions()
{
	if (hOpenSteerWrapperParams.cohesionParams.x == 0.0f)
		return;
	
	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		genericCohesionDRef(),
		&OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall,
		&OpenSteerWrapper::OpenSteerWrapper_afterKernelCall);
		
	m_SteerForce->swapPosReadWrite();

	m_CommonRes.getDeviceInterface()->threadSync();

} //computeCohesions


// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------



template <class Super>
void OpenSteerWrapperPlugIn<Super>::computeSeekingsWorldCenter()
{
	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		genericSeekingWorldCenterDRef(),
		&OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall,
		&OpenSteerWrapper::OpenSteerWrapper_afterKernelCall);
	
	m_SteerForce->swapPosReadWrite();

	m_CommonRes.getDeviceInterface()->threadSync();
	
} //computeSeekingsWorldCenter

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

template <class Super>
void OpenSteerWrapperPlugIn<Super>::applySteeringForces(float elapsedTime)
{
	hOpenSteerWrapperDynParams.elapsedTime = elapsedTime;
	OpenSteerWrapper::OpenSteerWrapper_copyDynParams();

	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		genericApplySteeringForceDRef(),
		&OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall,
		&OpenSteerWrapper::OpenSteerWrapper_afterKernelCall);

	m_Pos->swapPosReadWrite();
	m_SmoothedAcceleration->swapPosReadWrite();
	m_Forward->swapPosReadWrite();

	m_CommonRes.getDeviceInterface()->threadSync();
	
}

template <class Super>
void OpenSteerWrapperPlugIn<Super>::resetSteerForce()
{
	m_SteerForce->reset(0);
}


template <class Super>
void OpenSteerWrapperPlugIn<Super>::steerForMoveAwayBaseTarget()
{
	OpenSteerWrapper::OpenSteerWrapper_copyFieldsToDevice();

	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		BehaveRT_getKernelRef(moveAwayFromTarget_kernel), 
		&OpenSteerWrapper::OpenSteerWrapper_beforeKernelCall,
		&OpenSteerWrapper::OpenSteerWrapper_afterKernelCall);

	m_SteerForce->swapPosReadWrite();
	m_SmoothedAcceleration->swapPosReadWrite();
}