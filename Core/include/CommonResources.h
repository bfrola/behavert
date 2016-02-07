// ----------------------------------------------------------------------------
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


#include <vector>
#include "vector_functions.h"

#include "DeviceInterface.cuh"
#include "SimEnginePlugIn.h"

#include "UtilityConfigFile.h"
#include "UtilityLogger.h"
#include "DeviceArrayWrapper.h"
#include "InstallableEntity.h"

using namespace BehaveRT;

namespace BehaveRT
{

	/// List of installed entities
	typedef std::vector<InstallableEntity*> InstalledEntities;
	
	/// Enumeration of installable types
	typedef enum InstallableEntityId {
		CROWDENGINE_PLUGIN, 
	};

	/**
		\brief This class shares the GPU info througout the BehaveRT's classes.
	*/
	class CommonResources
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor

		/// Default constructor: instantiate the config file and the device interface
		CommonResources();

		/// Do nothing
		~CommonResources();

		// ////////////////////////////////////////////////////
		// Methods
	
		/// @return true whether the m_BehaviorEngine has already been initialized
		bool isInitialized() { return m_Initialized; }

		/// Reference to the device interface
		DeviceInterface* getDeviceInterface() { return m_DeviceInterface; }
		

		/// Shared config file reference
		UtilityConfigFile* getConfig() { return m_Config; }

		/// Shared logger
		UtilityLogger* getLogger() { return m_Logger; }
		
		/// Load the global configuration fron the config file
		void initialize(std::string configFileName);
		
		/// @return the list of installed entities witch referenced by the parameter id
		InstalledEntities& getInstalledEntities(InstallableEntityId id);

		/// Insert entity into the list witch is referenced by the parameter id
		void installEntity(InstallableEntityId id, InstallableEntity* entity);

		/// Delete the entity at position installationKey from the list witch is referenced by the parameter id
		void uninstallEntity(InstallableEntityId id, int installationKey);
	
		/// Insert entity into the list installedEntities
		void installEntity(InstalledEntities& installedEntities, InstallableEntity* arrayWrapper);

		/// Delete the entity at position installationKey from the list installedEntities
		void uninstallEntity(InstalledEntities& installedEntities, int installationKey);

		InstalledEntities& getSupportedDataTypes();
		int getDataTypeId(type_info type);

		// ////////////////////////////////////////////////////
		// Fields
	private:

		/// True if the m_BehaviorEngine is initialized
		bool m_Initialized;
	
		BehaveRT::UtilityConfigFile* m_Config;

		DeviceInterface* m_DeviceInterface;

		InstalledEntities m_InstalledPlugIns;

		UtilityLogger* m_Logger;
		
	};

}