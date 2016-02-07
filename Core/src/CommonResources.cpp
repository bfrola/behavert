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


#include "CommonResources.h"

#include "DeviceData.cuh"

#include "Utility.h"
#include "UtilityString.h"
#include "UtilityConfigFile.h"
#include "UtilityStringConverter.h"

#include "DeviceData.cuh"

#include <assert.h>

using namespace BehaveRT;

CommonResources::CommonResources() :
    m_Initialized(false)
{
	// Init the device interface
	m_DeviceInterface = new DeviceInterface();

	// Get configuration from config file
	m_Config = new BehaveRT::UtilityConfigFile();

	m_Logger = new BehaveRT::UtilityLogger(50);


	char* args = "BehaveRT";
	// Init CUDA
	m_DeviceInterface->cudaInit(1, &args);
}

CommonResources::~CommonResources()
{ }

// ------------------------------------------------------------


void
CommonResources::initialize(std::string configFileName)
{	
	m_Config->load(configFileName);
	
	// Simulation paramters
	m_DeviceInterface->getHostSimParams().commonBlockDim =
		BehaveRT::StringConverter::parseInt(m_Config->getSetting("BlockDim", "DeviceInterface"));

	m_DeviceInterface->getHostSimParams().useThreadSync =
		BehaveRT::StringConverter::parseBool(m_Config->getSetting("useThreadSync", "SimParams"));

	std::cout << "\n----------------------------------------------\n";
	std::cout << " BehaveRT library v0.1" << std::endl;
	std::cout << " http://isis.dia.unisa.it/projects/behavert/" << std::endl;
	std::cout << "----------------------------------------------\n";

	printf("\n\nCommonResources initialized successfully\n");

	m_Initialized = true;
} // initialize


// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

void CommonResources::installEntity(
	InstalledEntities& installedEntities, 
	InstallableEntity* entity)
{
	entity->setInstallationKey(installedEntities.size());
	installedEntities.push_back(entity);
}

// ------------------------------------------------------------------------------

void CommonResources::uninstallEntity(
	InstalledEntities& installedEntities, 
	int installationKey)
{
	installedEntities.erase(
		installedEntities.begin() + installationKey,
		installedEntities.begin() + installationKey + 1);
}

// ------------------------------------------------------------------------------

void CommonResources::installEntity(InstallableEntityId id, InstallableEntity* arrayWrapper)
{
	installEntity(getInstalledEntities(id), arrayWrapper);
}

// ------------------------------------------------------------------------------

void CommonResources::uninstallEntity(InstallableEntityId id, int installationKey)
{
	uninstallEntity(getInstalledEntities(id), installationKey);
}

// ------------------------------------------------------------------------------

InstalledEntities& CommonResources::getInstalledEntities(InstallableEntityId id)
{
	switch (id)
	{
		case InstallableEntityId::CROWDENGINE_PLUGIN:
			return m_InstalledPlugIns;
	}
}

// ------------------------------------------------------------------------------


// Define the shared object
CommonResources m_CommonRes;