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

#include "IndividualSelector.h"

#include <algorithm>
#include <functional>

// ---------------------------------------------------
// ---------------------------------------------------


IndividualSelector::IndividualSelector(int selectorsMaxSize, bool enableDeselection) :
	m_SelectorsMaxSize(selectorsMaxSize), m_EnableDeselection(enableDeselection)
{
	
}

// ---------------------------------------------------
// ---------------------------------------------------

IndividualSelector::~IndividualSelector()
{
}

// ---------------------------------------------------
// ---------------------------------------------------

bool IndividualSelector::selectIndividual(int individualIndex)
{
	// Check if individualIndex is already selected
	
	IndividualSelectorType::iterator selectorElement = 
		findIndividual(individualIndex);
	if (alreadySelected (selectorElement))
	{
		if (m_EnableDeselection)		
			m_Selectors.erase(selectorElement);		
		
		return false;
	}

	// Delete the first element
	// Do not allows more than m_SelectorsMaxSize selected items
	if (m_Selectors.size () == m_SelectorsMaxSize)
		m_Selectors.erase (m_Selectors.begin());

	m_Selectors.push_back (individualIndex);

	return true;

	
}

// ---------------------------------------------------
// ---------------------------------------------------

IndividualSelectorType::iterator 
	IndividualSelector::findIndividual (int individualIndex)
{
	return std::find(m_Selectors.begin(), m_Selectors.end(), individualIndex);
}

// ---------------------------------------------------
// ---------------------------------------------------

bool IndividualSelector::alreadySelected (int individualIndex)
{
	return findIndividual(individualIndex) != m_Selectors.end();
}

// ---------------------------------------------------
// ---------------------------------------------------

bool IndividualSelector::alreadySelected (IndividualSelectorType::iterator selectorElement)
{
	return selectorElement != m_Selectors.end();
}

// ---------------------------------------------------
// ---------------------------------------------------

bool IndividualSelector::selectHighlighted ()
{
	if (getHighlighted() == NULL)
		return false;

	return selectIndividual (getHighlighted ());
}

// ---------------------------------------------------
// ---------------------------------------------------
