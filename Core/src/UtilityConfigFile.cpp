/*
-----------------------------------------------------------------------------
This source file is part of OGRE
    (Object-oriented Graphics Rendering Engine)
For the latest info, see http://www.ogre3d.org/

Copyright (c) 2000-2006 Torus Knot Software Ltd
Also see acknowledgements in Readme.html

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59 Temple
Place - Suite 330, Boston, MA 02111-1307, USA, or go to
http://www.gnu.org/copyleft/lesser.txt.

You may alternatively use this source under the terms of a specific version of
the OGRE Unrestricted License provided you have obtained such a license from
Torus Knot Software Ltd.
-----------------------------------------------------------------------------
*/

#include "BehaveRT.h"

#include <fstream>

namespace BehaveRT {


    //-----------------------------------------------------------------------
    UtilityConfigFile::UtilityConfigFile()
    {
    }
	//-----------------------------------------------------------------------
	UtilityConfigFile::UtilityConfigFile(const std::string& filename, const std::string& separators, bool trimWhitespace)
	{
		load(filename, separators, trimWhitespace);
	}
    //-----------------------------------------------------------------------
    UtilityConfigFile::~UtilityConfigFile()
    {
        SettingsBySection::iterator seci, secend;
        secend = mSettings.end();
        for (seci = mSettings.begin(); seci != secend; ++seci)
        {
            //OGRE_DELETE_T(seci->second, SettingsMultiMap, MEMCATEGORY_GENERAL);
			delete seci->second;
        }
    }
    //-----------------------------------------------------------------------
    void UtilityConfigFile::clear(void)
    {
        for (SettingsBySection::iterator seci = mSettings.begin(); 
            seci != mSettings.end(); ++seci)
        {
             //OGRE_DELETE_T(seci->second, SettingsMultiMap, MEMCATEGORY_GENERAL);
			delete seci->second;
        }
        mSettings.clear();
    }
	//-----------------------------------------------------------------------
	void UtilityConfigFile::load(const std::string& filename, const std::string& separators, bool trimWhitespace)
    {
		/* Open the configuration file */
		std::ifstream stream;
        // Always open in binary mode
		stream.open(filename.c_str(), std::ios::in | std::ios::binary);
		
		if(!stream)
		{
			//TODO Exception
			printf("'%s' file not found!", filename.c_str());
			
		}

        /* Clear current settings map */
        clear();

        std::string currentSection = "";
        //SettingsMultiMap* currentSettings = OGRE_NEW_T(SettingsMultiMap, MEMCATEGORY_GENERAL)();
		
		SettingsMultiMap* currentSettings = new SettingsMultiMap();
        mSettings[currentSection] = currentSettings;
		
        /* Process the file line for line */
        std::string line, optName, optVal;
        while (!stream.eof())
        {
			//char buffer[256];
			std::getline(stream, line);
            /* Ignore comments & blanks */
            if (line.length() > 0 && line.at(0) != '#' && line.at(0) != '@')
            {
				const char* buf = line.c_str();

                if (buf[0] == '[' && buf[line.length()-2] == ']')
                {
					
                    // Section
                    currentSection = line.substr(1, line.length() - 3);

					SettingsBySection::const_iterator seci = mSettings.find(currentSection);
					if (seci == mSettings.end())
					{
						//currentSettings = OGRE_NEW_T(SettingsMultiMap, MEMCATEGORY_GENERAL)();
						currentSettings = new SettingsMultiMap(); 
						mSettings[currentSection] = currentSettings;
					}
					else
					{
						currentSettings = seci->second;
					} 
                }
                else
                {
                    /* Find the first seperator character and split the string there */
                    std::string::size_type separator_pos = line.find_first_of(separators, 0);
                    if (separator_pos != std::string::npos)
                    {
                        optName = line.substr(0, separator_pos);
                        /* Find the first non-seperator character following the name */
                        std::string::size_type nonseparator_pos = line.find_first_not_of(separators, separator_pos);
                        /* ... and extract the value */
                        /* Make sure we don't crash on an empty setting (it might be a valid value) */
                        optVal = (nonseparator_pos == std::string::npos) ? "" : line.substr(nonseparator_pos);
                        if (trimWhitespace)
                        {
                            StringUtil::trim(optVal);
                            StringUtil::trim(optName);
                        }
                        currentSettings->insert(std::multimap<std::string, std::string>::value_type(optName, optVal));
                    }
                }
            }
        }
    }

    //-----------------------------------------------------------------------
    //-----------------------------------------------------------------------
    std::string UtilityConfigFile::getSetting(const std::string& key, const std::string& section, const std::string& defaultValue) const
    {
        
        SettingsBySection::const_iterator seci = mSettings.find(section);
        if (seci == mSettings.end())
        {
			printf("WARNING: %s [%s] is empty\n", key.c_str(), section.c_str());
            return defaultValue;
        }
        else
        {
			SettingsMultiMap::const_iterator i = seci->second->find(key);
            if (i == seci->second->end())
            {
                return "";
            }
            else
            {
                return i->second;
            }
        }
    }
    //-----------------------------------------------------------------------
    BRTStringVector UtilityConfigFile::getMultiSetting(const std::string& key, const std::string& section) const
    {
        BRTStringVector ret;


        SettingsBySection::const_iterator seci = mSettings.find(section);
        if (seci != mSettings.end())
        {
            SettingsMultiMap::const_iterator i;

            i = seci->second->find(key);
            // Iterate over matches

            while (i != seci->second->end() && i->first == key)
            {
                ret.push_back(i->second);
                ++i;
            }
        }
        return ret;


    }
    
}
