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
#include "body3d_kernel.cuh"

/// Contains plugIns which provide a base for the extension chain
namespace Body
{
	/**
		\brief This class provieds the extention base for a simulation based on rigid bodies
		
		When an other plugIn extends this one, it inheritates the features contained in Body3DFields and parameters in Body3DParams

	*/
	template <class Super>
	class Body3DPlugIn: public Super, public SimEnginePlugIn
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor

		/// Defautl constructor: provides plugIn installation
		Body3DPlugIn() { SimEnginePlugIn::installPlugIn(); }

		/// Defautl constructor: provides plugIn uninstallation
		~Body3DPlugIn() { SimEnginePlugIn::uninstallPlugIn(); }
		
		const std::string name() { return "Body3DPlugIn"; }	

		/// Dependeciens: none
		const DependenciesList plugInDependencies() 
		{ 
			DependenciesList dependencies;
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

		// ////////////////////////////////////////////////////
		// Fields	
	protected:

		/// Only one feature: positions. Its type is float4 because texture mappings does not support
		/// type float3, which shold be the standard type for 3D positions
		BehaveRT::DeviceArrayWrapper<float4>* m_Pos;
	};
}

using namespace Body;

// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
// Implementation
template <class Super>
void Body3DPlugIn<Super>::install()
{
	int numBodies = BehaveRT::StringConverter::parseInt(
		m_CommonRes.getConfig()->getSetting("numBodies", "Body3DPlugIn"));

	if (BehaveRT::StringConverter::parseBool(
		m_CommonRes.getConfig()->getSetting("isBodyNumExponent", "Body3DPlugIn")))
	{
		hBody3DParams.numBodies = pow(2.0, numBodies);
	}
	else
		hBody3DParams.numBodies = numBodies;

	char msg[100];
	sprintf(msg, "Body3DPlugIn>> NumBodies: %d\n", hBody3DParams.numBodies);
	m_CommonRes.getLogger()->log("Body3DParams", msg);

	hBody3DParams.commonRadius = BehaveRT::StringConverter::parseFloat(
		m_CommonRes.getConfig()->getSetting("commonRadius", "Body3DPlugIn"));

	hBody3DParams.use2DProjection = BehaveRT::StringConverter::parseBool(
		m_CommonRes.getConfig()->getSetting("use2DProjection", "Body3DPlugIn"));

	Body3D::Body3D_copyFieldsToDevice();
}

// --------------------------------------------------------------

template <class Super>
void Body3DPlugIn<Super>::uninstall()
{
	
}

// --------------------------------------------------------------

template <class Super>
void Body3DPlugIn<Super>::reset()
{
	Super::reset(); // MANDATORY OPERATION
}

// --------------------------------------------------------------

template <class Super>
void Body3DPlugIn<Super>::update(const float elapsedTime)
{
	Super::update(elapsedTime); // MANDATORY OPERATION

	// Insert here the default update operation
}

// --------------------------------------------------------------
// --------------------------------------------------------------

// --------------------------------------------------------------