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

#include "BehaveRT.h"
#include "stdarg.h"

SimEnginePlugIn::SimEnginePlugIn(void)
{
	
}

SimEnginePlugIn::~SimEnginePlugIn(void)
{
}

void SimEnginePlugIn::installPlugIn()
{
	install();
	
	printf("Installed plugIn: [%s]\n", name().c_str());
	
	m_InstallationToken = new SimEnginePlugInInstallationToken(name());
	m_CommonRes.installEntity(
		InstallableEntityId::CROWDENGINE_PLUGIN, m_InstallationToken);

	checkForPlugInDependencies();
}

void SimEnginePlugIn::uninstallPlugIn()
{
	uninstall();

	printf("Uninstalled plugIn: [%s]\n", name().c_str());
	
	m_CommonRes.uninstallEntity(
		InstallableEntityId::CROWDENGINE_PLUGIN, m_InstallationToken->getInstallationKey());
	delete m_InstallationToken;
}

void SimEnginePlugIn::checkForPlugInDependencies()
{
	DependenciesList dep = plugInDependencies();

	InstalledEntities installedPlugIns = m_CommonRes.getInstalledEntities(
		InstallableEntityId::CROWDENGINE_PLUGIN);

	if (installedPlugIns.size() == 0)
		return;

	printf("Checking dependencies for: [%s]", name().c_str());

	for (DependenciesList::const_iterator depIt = dep.begin();
			depIt != dep.end(); depIt ++)
	{
		const std::basic_string<char> currDep = (std::basic_string<char>) *depIt;
		//printf(" >>>> %s\n", currDep.c_str());

		std::string lastMissedDep = "";
		bool found = false;
		for(InstalledEntities::const_iterator pluginIt = installedPlugIns.begin();
			pluginIt != installedPlugIns.end(); pluginIt++)
		{
			printf(".");
			SimEnginePlugInInstallationToken* currToken = 
				(SimEnginePlugInInstallationToken*) *pluginIt;

			if (currDep == currToken->getName())
			{
				found = true;
				break;
			}
			else
			{
				lastMissedDep = currDep;
			}
		}
		
		if (!found)
		{
			printf("\nError! Missed plugIn dependency: [%s]\n", lastMissedDep.c_str());
			system("pause");
			exit(-1);
		}
	}

	printf(" Ok\n");
}

/*
void SimEnginePlugIn::plugInPrintf(const std::string name, const char *fmt, ...)
{
	printf("%s>> ", name.c_str());
	va_list ap;
	va_start(ap, fmt);
	ap += 4;
	printf(fmt, ap);
	va_end(ap);
}
*/

std::string SimEnginePlugIn::getPlugInSetting(std::string settingKey)
{
	m_CommonRes.getConfig()->getSetting(settingKey, m_InstallationToken->getName());
	return "";
}