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

#include "InstallableEntity.h"

#include <vector>

namespace BehaveRT
{
	typedef std::vector<std::string> DependenciesList;

	/**
		\brief Represents the installable token associated to each plugIn.
		
		It resolves the following problem:
		\par
		Given an extension chain of plugIn: plugIn_1->plugIn_2->...->plugIn_2 each of which extends SimEnginePlugIn.
		\par
		I want to identify the extact pointer to the plugIn class from SimEnginePlugIn.
		\par
		The keyword "this" does not work well becouse it will refers to the top of the extension chain.
		\par
		The solution is to give an extension token to each installed plugIn which is out of the extension chain.
		
	*/
	class SimEnginePlugInInstallationToken : public InstallableEntity
	{
	public:

		/// Defualt costructor: it does nothing
		SimEnginePlugInInstallationToken() {}

		/// Initialize the field name
		SimEnginePlugInInstallationToken(const std::string name) : m_Name(name) {}

		/// Name identifies the SimEnginePlugIn membership
		std::string getName() const { return m_Name; }
	private:
		const std::string m_Name;
	};

	/**
		\brief Default SimEngine plugIn template.
		
		This template acts as an Java interface and contains the abstract methods which all the plugIns have to implement.
		
	*/
	class AbstractSimEnginePlugIn
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor

		// ////////////////////////////////////////////////////
		// Abstract methods

		/** PlugIn name. 
			Example:
			\code
			const std::string  name() 
			{
				return "PlugInCustomizedName"; 
			}
			\endcode
		*/
		const virtual std::string  name() = 0;

		/** List of dependecies (This will be removed)
			Example:
			\code
			DependenciesList plugInDependencies()
			{
				DependenciesList dependencies;
				dependencies.push_back("EnvGrid3DPlugIn");	// First dependency
				dependencies.push_back("ProximityDBPlugIn"); // Second dependency
				// Put here other dependecies
				return dependencies;
			}
			\endcode
		*/
		const virtual DependenciesList plugInDependencies() = 0;	

		/**
			This method must contain plugIn's parameters and features initialization
		*/
		virtual void reset() = 0;

		/**
			
		*/
		virtual void update(const float elapsedTime) = 0;
	protected:
		/**
			This method must allocate all device data. such as features (DeviceArrayWrapper)
			Example:
			\code
			m_SmoothedAcceleration = new BehaveRT::DeviceArrayWrapper<float4>(
				community.getDeviceInterface(), hBody3DParams.numBodies);
			\endcode
		*/
		virtual void install() = 0;

		/**
			This method must deallocate all device data. such as features (DeviceArrayWrapper)
			Example:
			\code
			delete m_SmoothedAcceleration
			\endcode
		*/
		virtual void uninstall() = 0;
	};

	/**
		\brief SimEngine plugIn concrete template

		A SimEngine plugIn has to extend this class and implement its virtual methods.

		It extends AbstractSimEnginePlugIn and provides some common operation, such as, installation and dependencies checking.
	*/
	class SimEnginePlugIn: public AbstractSimEnginePlugIn
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor
		SimEnginePlugIn();	
		~SimEnginePlugIn();	

	protected:
		// ////////////////////////////////////////////////////
		// Methods

		/// Add the plugIn to the list of installed plugIns.
		void addToRegistry();

		/// Check whether the dependencies expressed with the method plugInDependencies() are sodisfied.
		void checkForPlugInDependencies();

		/// Runs all the operation concerning the plugIn installation. Calls automatically the virtual method install().
		void installPlugIn();

		/// Runs all the operation concerning the plugIn installation. Calls automatically the virtual method uninstall().
		void uninstallPlugIn();

		// Get the setting from the global config file
		// Uses the nane of the plugIn as section
		std::string getPlugInSetting(std::string settingKey);
	
		// ////////////////////////////////////////////////////
		// Fields	
	private:
		SimEnginePlugInInstallationToken* m_InstallationToken;
	};


	/**
		\brief Default extension base for SimEngine chains

		All methods are empty.
	*/
	class DummyPlugIn : public SimEnginePlugIn
	{
	public:
		const std::string name() { return "PlugInRoot"; }
		const std::vector<std::string> plugInDependencies() { std::vector<std::string> dep; return dep; }
		void install() {};
		void uninstall() {};
		void reset() {};
		void update(const float elapsedTime) {};
	};
}