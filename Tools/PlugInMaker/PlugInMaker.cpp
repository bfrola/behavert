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
//    04-09 bf: Created
// 11-10-10 bf: Update (bind/unbind)
//
// ----------------

#include "stdafx.h"
#include <iostream>
#include <algorithm>
#include <windows.h>
#include <fstream>

#include "UtilityConfigFile.h"


void outputSquare( std::string );
void parseConfig( BehaveRT::UtilityConfigFile* config );

void copyTemplate(std::string plugInName);
void replaceFileNames(std::string plugInName);
void replaceFileContent(std::string plugInName);


/**
	Classes envolved in the 
*/
namespace BehaveRT
{
	typedef std::pair<std::string, std::string> ReplaceListItem;
	typedef std::vector<ReplaceListItem> ReplaceList;

	// /////////////////////////////////////////////
	/**
		\brief Itearate plugIn names and sobstitutes their content.
	*/
	class ContentReplacer 
	{
	public:
		ContentReplacer(
			ReplaceList* replaceList, 
			std::string outDir, std::string outName, 
			std::string baseDir, std::string baseName )
		{
			m_ReplaceList = replaceList;
			m_OutDir = outDir;
			m_OutName = outName; 
			m_BaseDir = baseDir;
			m_BaseName = baseName;
		}

		ReplaceList* m_ReplaceList;
		std::string m_OutDir;
		std::string m_OutName;
		std::string m_BaseDir;
		std::string m_BaseName;

		/// Find and replace data
		void operator ()(std::string currStr)
		{
			std::string oldStr = currStr;
			
			findAndReplace(currStr, m_BaseName, m_OutName);

			std::string inputFileName = m_OutDir + oldStr;
			std::string outputFileName = m_OutDir + currStr;

			std::ifstream inputFile ( inputFileName.c_str(), std::ifstream::in );
			std::ofstream outputFile ( outputFileName.c_str(), std::ifstream::out );

			std::cout << inputFileName << std::endl;

			// Iterate over file lines
			std::string line;
			while (getline(inputFile, line))
			{
				
				findAndReplaceAll(line, m_BaseName, m_OutName);

				for(ReplaceList::const_iterator it = m_ReplaceList->begin(); 
					it != m_ReplaceList->end(); ++ it)
				{
					ReplaceListItem item = (ReplaceListItem) *it;
					findAndReplaceAll(line, item.first, item.second);
				}

				// Write the line
				outputFile << line << std::endl;
			}
				
			inputFile.close();
			outputFile.close();

			DeleteFileA(inputFileName.c_str());

		}

		/// Find and replace in str the first occuourence of oldStr with new Str
		bool findAndReplace(std::string& str, std::string oldStr, std::string newStr)
		{
			int occourrenceIndex = str.find(oldStr);
			// When threre are no more occourrences exits
			if (occourrenceIndex < 0)
				return false;

			str.replace(
				occourrenceIndex, 
				oldStr.size(),
				newStr);

			return true;
		}

		/// Find and replace in str the all occuourences of oldStr with new Str
		bool findAndReplaceAll(std::string& str, std::string oldStr, std::string newStr)
		{
			if (!findAndReplace(str, oldStr, newStr))
				return false;
			while (findAndReplace(str, oldStr, newStr));
			return true;
		}

	};


	// /////////////////////////////////////////////
	/**
		\brief PlugIn generation tool
	*/
	class PlugInMaker
	{
		
		// ///////////////////////////
		// Constructors/Descructors

	public:
		PlugInMaker()	  		
		{
			// Where is the template
			m_BaseDir = "..\\..\\..\\..\\..\\PlugIns\\";
			// The destination folder
			m_OutDir = "..\\..\\..\\..\\..\\PlugIns\\";
			m_TemplateName = "_Template_"; 
			m_StringToReplace = "[PName]";
		}

		~PlugInMaker() 
		{
			
		}

		// ///////////////////////////
		// Methods

		void parseConfig(BehaveRT::UtilityConfigFile* configFile)
		{
			m_ConfigFile = configFile;

			//BehaveRT::BRTStringVector features = 
			//	m_ConfigFile->getMultiSetting("featureName", "features");

			std::string plugInName = m_ConfigFile->getSetting("plugInName", "PlugInMaker");

			std::cout << "\nPlugIn name: <" << plugInName << ">" << std::endl << std::endl; 
			std::cout << "WARNING: If this plugIn already exists, it will be overwritten." << std::endl << std::endl; 
			system( "pause" );

			m_PlugInName = plugInName;

			copyTemplate();
			replaceFileNames();
			
		}

		// ----------------------------------------------------------

		void copyTemplate()
		{
			std::cout << " - Copy directory" << std::endl;

			std::string copyCmd = "copy ";

			std::string targetDir = m_OutDir + m_PlugInName;
			std::string srcDir = targetDir + "\\src";
			std::string includeDir = targetDir + "\\include";

			CreateDirectoryA(targetDir.c_str(), 0);
			CreateDirectoryA(srcDir.c_str(), 0);
			CreateDirectoryA(includeDir.c_str(), 0);

			std::string dirSource = m_BaseDir + m_TemplateName;
			
			std::string cmd = copyCmd + dirSource + " " + targetDir;
			system(cmd.c_str());

			cmd = copyCmd + dirSource  + "\\src " + srcDir;
			system(cmd.c_str());

			cmd = copyCmd + dirSource  + "\\include " + includeDir;
			system(cmd.c_str());

			std::cout << " - Copy directory" << cmd << std::endl;

		}

		// ----------------------------------------------------------

		ReplaceList generateReplaceList()
		{
			ReplaceList replaceList;

			BRTStringVector multiSetting;
			
			std::stringstream cuh_PFeatures;
			std::stringstream h_PFeaturesDeclaration;
			std::stringstream h_PFeaturesAllocation;
			std::stringstream h_PFeaturesDeallocation;
			std::stringstream cu_PFeaturesTextures;

			std::stringstream cu_PBindTextures;
			std::stringstream cu_PUnbindTextures;

			multiSetting = m_ConfigFile->getMultiSetting("feature", m_PlugInName);
			for(BRTStringVector::const_iterator it = multiSetting.begin(); 
				it != multiSetting.end(); ++ it)
			{
				std::string setting = (std::string) *it;
				BRTStringVector words = StringUtil::split(setting);
				
				if (words.size() != 2)
					continue;

				// Legend:
				// words[0] feature type
				// words[1] feature name

				cuh_PFeatures << 
					"\tint " << words[1] << ";" << std::endl;

				h_PFeaturesDeclaration << 
					"\t\tBehaveRT::DeviceArrayWrapper<" << words[0] << 
					">* m_" << words[1] << ";" << std::endl;

				h_PFeaturesAllocation << 
					"\tm_" << words[1] << " = new BehaveRT::DeviceArrayWrapper<" << words[0] << 
					">(\n\t\tm_CommonRes.getDeviceInterface(), hBody3DParams.numBodies);" << 
					std::endl;

				h_PFeaturesAllocation << "\tm_" << words[1] << "->bindToField(h" <<
					m_PlugInName << "." << words[1] << ")" << std::endl << std::endl;

				h_PFeaturesDeallocation << 
					"\tdelete m_" << words[1] << ";" << std::endl;

				cu_PFeaturesTextures << 
					"texture<" << words[0] << 
					", 1, cudaReadModeElementType> " << words[1] << "Cache;" << std::endl;

				// -------------
				// 11-10-2010
				cu_PBindTextures << 
					"bind_field_texture( h" << m_PlugInName << "Fields." << 
					words[1] << ", " << 
					words[1] << "Cache );" << std::endl;

				cu_PUnbindTextures << "unbind_field_texture( " << 
					words[1] << "Cache );" << std::endl;
				// -------------
			}
	
			// [PParams]

			std::stringstream cuh_PParams;

			multiSetting = m_ConfigFile->getMultiSetting("parameter", m_PlugInName);
			for(BRTStringVector::const_iterator it = multiSetting.begin(); 
				it != multiSetting.end(); ++ it)
			{
				std::string setting = (std::string) *it;
				cuh_PParams << "\t" << setting << ";" << std::endl;
			}

			std::stringstream h_PDependencies;
			std::stringstream cu_PDependenciesPaths;

			std::string lastDependecy = "";

			multiSetting = m_ConfigFile->getMultiSetting("dependency", m_PlugInName);
			for(BRTStringVector::const_iterator it = multiSetting.begin(); 
				it != multiSetting.end(); ++ it)
			{
				std::string setting = (std::string) *it;
				BRTStringVector words = StringUtil::split(setting);
				
				h_PDependencies << 
					"\t\t\tdependencies.push_back(\"" << words[0] << "PlugIn\");" << std::endl;

				cu_PDependenciesPaths << 
					"#include \"..\\" << ((words.size() == 2)?words[1]:words[0]) << 
					"\\include\\" << words[0] << "_kernel.cuh\""  << std::endl;
				cu_PDependenciesPaths << 
					"#include \"..\\" << ((words.size() == 2)?words[1]:words[0]) << 
					"\\" << words[0] << "_resources.cu\""  << std::endl << std::endl;

				// Get the last one
				lastDependecy = words[0];
				
			}

			replaceList.push_back(
				std::make_pair("[PFeatures]", cuh_PFeatures.str()));
			replaceList.push_back(
				std::make_pair("[PFeaturesDeclaration]", h_PFeaturesDeclaration.str()));
			replaceList.push_back(
				std::make_pair("[PFeaturesAllocation]", h_PFeaturesAllocation.str()));
			replaceList.push_back(
				std::make_pair("[PFeaturesDeallocation]", h_PFeaturesDeallocation.str()));
			replaceList.push_back(
				std::make_pair("[PFeaturesTextures]", cu_PFeaturesTextures.str()));

			// -------------
			// 11-10-2010
			replaceList.push_back(
				std::make_pair("[PFeaturesTexturesBinding]", cu_PBindTextures.str()));
			replaceList.push_back(
				std::make_pair("[PFeaturesTexturesUnbinding]", cu_PUnbindTextures.str()));

			// At least one dependency
			if (lastDependecy.length() > 0)
			{
				replaceList.push_back(
					std::make_pair("[PFeaturesDependencyBefoceCall]", 
					lastDependecy + "::" + lastDependecy + "_beforeKernelCall();" ));
				replaceList.push_back(
					std::make_pair("[PFeaturesDependencyAfterCall]", 
					lastDependecy + "::" + lastDependecy + "_afterKernelCall();" ));
			}

			// -------------
			
			replaceList.push_back(
				std::make_pair("[PParams]", cuh_PParams.str()));

			replaceList.push_back(
				std::make_pair("[PDependencies]", h_PDependencies.str()));
			replaceList.push_back(
				std::make_pair("[PDependenciesPaths]", cu_PDependenciesPaths.str()));
			

			return replaceList;
		}

		void replaceFileNames()
		{
			std::vector<std::string> fileInvolved;
			// Read from config what are files involved			
			fileInvolved = m_ConfigFile->getMultiSetting("fileInvolved", "PlugInMaker");
			
			std::cout << " - Replace file content" << std::endl;

			ReplaceList replaceList = generateReplaceList();
			
			// Iterate over all elements in the list
			std::for_each(fileInvolved.begin(), fileInvolved.end(), 
				ContentReplacer(
					&replaceList, 
					m_OutDir + m_PlugInName + "\\", 
					m_PlugInName, 
					m_BaseDir + m_PlugInName + "\\", 
					m_StringToReplace));

		}

		
		// ///////////////////////////
		// Fields

		std::string m_TemplateName;

	private:
		std::string m_PlugInName;
		BehaveRT::UtilityConfigFile* m_ConfigFile;
		std::string m_BaseDir;
		std::string m_OutDir;
		std::string m_StringToReplace;
	};

};



// ----------------------------------------------------------

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "Usage PlugInMaker <config file path>" << std::endl;
		return 0;
	}

	BehaveRT::UtilityConfigFile config;
	config.load( argv[1] );

	std::cout << "\n----------------------------------------------\n";
	std::cout << " PlugInMaker for BehaveRT v0.1" << std::endl;
	std::cout << " http://isis.dia.unisa.it/projects/behavert/" << std::endl;
	std::cout << "----------------------------------------------\n";

	BehaveRT::PlugInMaker maker;
	maker.parseConfig( &config );

	return 0;
}