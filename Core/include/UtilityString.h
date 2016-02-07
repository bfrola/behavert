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
#pragma once 

#include <iostream>
#include <vector>
#include <sstream>

namespace BehaveRT {
	typedef std::vector<std::string> BRTStringVector;
	
    /** \brief Utility class for manipulating Strings.  */
    class StringUtil
    {
	public:
        typedef std::ostringstream StrStreamType;

        /** Removes any whitespace characters, be it standard space or
            TABs and so on.
            @remarks
                The user may specify wether they want to trim only the
                beginning or the end of the std::string ( the default action is
                to trim both).
        */
        static void trim( std::string& str, bool left = true, bool right = true );

        /** Returns a StringVector that contains all the substrings delimited
            by the characters in the passed <code>delims</code> argument.
            @param
                delims A list of delimiter characters to split by
            @param
                maxSplits The maximum number of splits to perform (0 for unlimited splits). If this
                parameters is > 0, the splitting process will stop after this many splits, left to right.
        */
		static std::vector< std::string > split( const std::string& str, const std::string& delims = "\t\n ", unsigned int maxSplits = 0);

        /** Upper-cases all the characters in the string.
        */
        static void toLowerCase( std::string& str );

        /** Lower-cases all the characters in the string.
        */
        static void toUpperCase( std::string& str );


        /** Returns whether the string begins with the pattern passed in.
        @param pattern The pattern to compare with.
        @param lowerCase If true, the end of the string will be lower cased before
            comparison, pattern should also be in lower case.
        */
        static bool startsWith(const std::string& str, const std::string& pattern, bool lowerCase = true);

        /** Returns whether the string ends with the pattern passed in.
        @param pattern The pattern to compare with.
        @param lowerCase If true, the end of the string will be lower cased before
            comparison, pattern should also be in lower case.
        */
        static bool endsWith(const std::string& str, const std::string& pattern, bool lowerCase = true);

        /** Method for standardising paths - use forward slashes only, end with slash.
        */
        static std::string standardisePath( const std::string &init);

        /** Method for splitting a fully qualified filename into the base name
            and path.
        @remarks
            Path is standardised as in standardisePath
        */
        static void splitFilename(const std::string& qualifiedName,
            std::string& outBasename, std::string& outPath);

		/** Method for splitting a fully qualified filename into the base name,
		extension and path.
		@remarks
		Path is standardised as in standardisePath
		*/
		static void splitFullFilename(const std::string& qualifiedName, 
			std::string& outBasename, std::string& outExtention, 
			std::string& outPath);

		/** Method for splitting a filename into the base name
		and extension.
		*/
		static void splitBaseFilename(const std::string& fullName, 
			std::string& outBasename, std::string& outExtention);


        /** Simple pattern-matching routine allowing a wildcard pattern.
        @param str std::string to test
        @param pattern Pattern to match against; can include simple '*' wildcards
        @param caseSensitive Whether the match is case sensitive or not
        */
        static bool match(const std::string& str, const std::string& pattern, bool caseSensitive = true);





        /// Constant blank string, useful for returning by ref where local does not exist
        static const std::string BLANK;
    };


} // namespace BehaveRT

