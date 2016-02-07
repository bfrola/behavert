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

#include "UtilityString.h"
#include "vector_types.h"

#pragma once

namespace BehaveRT {

	/** \brief Class for converting the core Ogre data types to/from Strings.
    @remarks
        The code for converting values to and from strings is here as a separate
        class to avoid coupling std::string to other datatypes (and vice-versa) which reduces
        compilation dependency: important given how often the core types are used.
    @par
        This class is mainly used for parsing settings in text files. External applications
        can also use it to interface with classes which use the StringInterface template
        class.
    @par
        The std::string formats of each of the major types is listed with the methods. The basic types
        like int and Real just use the underlying C runtime library atof and atoi family methods,
        however custom types like Vector3, ColourValue and Matrix4 are also supported by this class
        using custom formats.
    @author
        Steve Streeting
    */
    class StringConverter
    {
    public:

        
        /** Converts a std::string to a float. 
        @returns
            0.0 if the value could not be parsed, otherwise the float version of the std::string.
        */
        static float parseFloat(const std::string& val);
        /** Converts a std::string to a whole number. 
        @returns
            0.0 if the value could not be parsed, otherwise the numeric version of the std::string.
        */
        static int parseInt(const std::string& val);
        /** Converts a std::string to a whole number. 
        @returns
            0.0 if the value could not be parsed, otherwise the numeric version of the std::string.
        */
        static unsigned int parseUnsignedInt(const std::string& val);
        /** Converts a std::string to a whole number. 
        @returns
            0.0 if the value could not be parsed, otherwise the numeric version of the std::string.
        */
        static long parseLong(const std::string& val);
        /** Converts a std::string to a whole number. 
        @returns
            0.0 if the value could not be parsed, otherwise the numeric version of the std::string.
        */
        static unsigned long parseUnsignedLong(const std::string& val);
        /** Converts a std::string to a boolean. 
        @remarks
            Returns true if case-insensitive match of the start of the string
			matches "true", "yes" or "1", false otherwise.
        */
        static bool parseBool(const std::string& val);
		/** Parses a float3 out of a std::string.
        @remarks
            Format is "x y z" ie. 3 float components, space delimited. Failure to parse returns
            a vector of zeros.
        */
        static float3 parseFloat3(const std::string& val);
		/** Parses a float4 out of a std::string.
        @remarks
            Format is "x y z" ie. 3 float components, space delimited. Failure to parse returns
            a vector of zeros.
        */
        static float4 parseFloat4(const std::string& val);
        /** Parses a uint3 out of a std::string.
        @remarks
            Format is "x y z" ie. 3 int components, space delimited. Failure to parse returns
            a vector of zeros.
        */
		static uint3 StringConverter::parseUint3(const std::string& val);
		/** Pareses a BRTStringVector from a string.
        @remarks
            Strings must not contain spaces since space is used as a delimiter in
            the output.
        */
        static BRTStringVector parseStringVector(const std::string& val);
        /** Checks the std::string is a valid number value. */
        static bool isNumber(const std::string& val);
    };


}


