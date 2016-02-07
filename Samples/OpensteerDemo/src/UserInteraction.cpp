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

#include "../Include/UserInteraction.h"

#include <algorithm>

// ----------------------------------------------------------
// ----------------------------------------------------------

UserInteraction::UserInteraction(IndividualPositionsType* positions) : 
	m_Positions(positions)
{
	m_ClusterMemerbership = NULL;
	
	changeMode(Merge);
	//changeMode(Split);
	//changeMode(Freeze);

	m_StateTimer = 0;
	m_CurrentState = Waiting;
}

// ----------------------------------------------------------
// ----------------------------------------------------------

UserInteraction::~UserInteraction()
{
	// TODO: free mem

	delete m_Selector;
}

// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::changeMode(UIMode mode)
{
	m_CurrentMode = mode;

	cout << "UI::Mode switched to: ";
	switch (mode)
	{
		case Merge: 
			cout << "Merge" << endl;
			m_Selector = new IndividualSelector(2);
			// TODO: init
			break;
		case Split:
			cout << "Split" << endl;
			m_Selector = new IndividualSelector(100, true);
			// TODO
			break;
		case Freeze:
			cout << "Freeze" << endl;
			m_Selector = new IndividualSelector(1, true);
			// TODO
			break;
	}
}

// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::changeState(UIState state, int timer = 0)
{
	m_CurrentState = state;
	m_LastStateChangetime = timeGetTime();
	m_StateTimer = timer;

	cout << "UI::State switched to: ";
	switch (m_CurrentState)
	{
		case Waiting: cout << "Waiting" << endl; break;
		case Running: cout << "Running" << endl; break;
	}
}

// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::update()
{
	//getSelector()->selectHighlighted();

	if (m_CurrentState == Waiting)
		return;

	// Checks the timer value
	int currentTime = timeGetTime();
	if (currentTime - m_LastStateChangetime > m_StateTimer)
		changeState(Waiting);
	
	// The behavior of continuos actions goes here
}

// ----------------------------------------------------------
// ----------------------------------------------------------

bool UserInteraction::isCurrentMode(UIMode checkCurrentMode)
{
	return m_CurrentMode == checkCurrentMode;
}

// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::highlightIndividual(Vec3 position, float radius, Vec3 color)
{
	drawCircleOrDisk(radius, Vec3(1, 0, 0), 
		position, color, 10, false, true);

	drawCircleOrDisk(radius, Vec3(0, 1, 0), 
		position, color, 10, false, true);
}

// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::draw()
{
	const bool show3Dtext = true;

	Vec3 highligthedPosition = OpenSteerWrapper::float42Vec3(
		m_Positions->getHostArrayElement(
				getSelector()->getHighlighted(), true));

	// Near mouse
	highlightIndividual(highligthedPosition - Vec3(0.01, 0.01, 0), 0.5f);

	if (show3Dtext)
	{
		ostringstream text;
		text << "   ID[" << getSelector()->getHighlighted() << "]";
		draw2dTextAt3dLocation(text, highligthedPosition, gBlack);
	}
		

	// Selected
	IndividualSelectorType selIndv = getSelector()->getSelectors();
	Vec3 previousPos = Vec3(0, 0, 0);

	// Highlight selected individuals
	for (IndividualSelectorType::iterator selected = selIndv.begin(); 
		selected != selIndv.end(); ++selected)
	{
		int current = *selected;
		Vec3 currentPos = OpenSteerWrapper::float42Vec3(
				m_Positions->getHostArrayElement(current, true));

		if (isCurrentMode(Split)) // Small circle
			highlightIndividual(currentPos, 0.4, gCyan);
		else // Large circle
			highlightIndividual(currentPos, 1.0f, gBlue);

		if (show3Dtext)
		{
			ostringstream text;
			text << "   ID[" << current << "]";
			draw2dTextAt3dLocation(text, currentPos, gBlack);
		}

		if (selected != selIndv.begin() && isCurrentMode(Merge))
		{
			drawLine(currentPos, previousPos, gBlack);
		}
		previousPos = currentPos;
	}
	
	// -------------------------------------

	// Needed sorted index of the cluster membership first element
	// That is the leader index
	if (m_ClusterMemerbership == NULL || m_Hash == NULL)
		return;

	// Highlight selected cluster memberships
	for (IndividualSelectorType::iterator selected = selIndv.begin(); 
		selected != selIndv.end(); ++selected)
	{
		int current = *selected;
		float3 clusterMembership = 
			m_ClusterMemerbership->getHostArrayElement(current, true);

		if (clusterMembership.x < 0)
			continue;


		//cout << clusterMembership.x << " ";
		//cout << clusterMembership.y << " ";
		//cout << clusterMembership.z << endl;

		int leaderIndex = 
			m_Hash->getHostArrayElement(
			clusterMembership.x, true).y;


		Vec3 leaderPos = OpenSteerWrapper::float42Vec3(
				m_Positions->getHostArrayElement(leaderIndex, true));

		if (show3Dtext)
		{
			ostringstream text;
			text << "\n   Leader[" << leaderIndex << "]";
			
			//if (m_CurrentState == Running)
			//{
			//	text << "\n   ACTION: MERGE";
			//}
	
			draw2dTextAt3dLocation(text, leaderPos, gBlack);
		}

		highlightIndividual(leaderPos, 1.0f, gGreen);
		
		if (m_CurrentState == Running)
		{
			// Additional visualization effects
			highlightIndividual(leaderPos - Vec3(0.01, 0.01, 0), 2.0f, gYellow);
		}
	} // for over cluster leaders

	// frozen leaders
	drawFrozenLeaders();
	
	
}
// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::drawFrozenLeaders()
{
	if (!isCurrentMode(Freeze) || m_FrozenLeaders.size() == 0)
		return; 


	for (IndividualSelectorType::iterator frozen = m_FrozenLeaders.begin(); 
		frozen != m_FrozenLeaders.end(); ++frozen)
	{
		int current = *frozen;
		float3 clusterMembership = 
			m_ClusterMemerbership->getHostArrayElement(current, true);
		Vec3 currentPos = OpenSteerWrapper::float42Vec3(
				m_Positions->getHostArrayElement(current, true));

		highlightIndividual(currentPos, 2.0f, gBlack);
	}
}
	

// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::mouseMotion(int x, int y)
{
	// Invalidate the click (mouse down->up)
	m_MotionFlag = true;

	if (isCurrentMode(Split) && sameClusterOfPreviousSelected(m_Selector->getHighlighted()))
	{
		m_Selector->changeEnableDeselection();
		if (m_Selector->selectHighlighted())
		{
			cout << "UI::Motion:Select: " << m_Selector->getHighlighted() << "\n";
		}
		m_Selector->changeEnableDeselection();
	}
}

// ----------------------------------------------------------
// ----------------------------------------------------------

bool UserInteraction::sameClusterOfPreviousSelected(int individualIndex)
{
	if (getSelector()->getSelectors().size() == 0)
		return true;

	int lastSelected = getSelector()->getSelectors()
		[ getSelector()->getSelectors().size() - 1 ]; // Last element

	int lastSelectedLeader = getLeaderIndex(lastSelected);

	if (lastSelectedLeader < 0) // Local propagation disabled
		return false; 

	if (getLeaderIndex(lastSelected) == getLeaderIndex(individualIndex))
		return true;

	// Different clusters
	return false;
}

void UserInteraction::mouseClick(int x, int y, int button, int state)
{
	if (state == 0)
		m_MotionFlag = false;

	// Down = 0; Up = 1;
	if (state != 1)
		return;
	
	// Do not allow selection during actions
	if (m_CurrentState != Waiting)
	{
		cout << "UI::Selection not allowed during actions running" << endl;
		return;
	}
	
	// If false, then additionalMotionFunc changed the flag
	// ==> The click is not valid
	if (m_MotionFlag == true)
		return;

	
	if (m_CurrentMode == Split)
	{
		// Exit if the selected item belongs to a different cluster
		if (!sameClusterOfPreviousSelected(m_Selector->getHighlighted()))
			return;
	}

	if (m_Selector->selectHighlighted()) 
		cout << "UI::[" << button << "] Select: " << m_Selector->getHighlighted() << "\n";
}

// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::keyboardKeyPressed(int key)
{
	switch (key)
	{
	case 'p':
		actionOneDispatcher();
		break;
	}
}

// ----------------------------------------------------------
// ----------------------------------------------------------

// Start Action I (current mode decides which one)
void UserInteraction::actionOneDispatcher()
{
	switch (m_CurrentMode)
	{
		case Merge:
			actionMerge();
			break;
		case Split:
			actionSplit();
			break;
		case Freeze:
			actionFreeze();
			break;
	}
}

// ----------------------------------------------------------
// ----------------------------------------------------------

int UserInteraction::getLeaderIndex(int individualIndex)
{
	int leaderIndex = m_ClusterMemerbership->getHostArrayElement(
			individualIndex, true).x; // Not sorted
	return leaderIndex;
}

// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::actionMerge()
{
	// Check for individual selction
	if (m_Selector->getSelectors().size() != 2)
	{
		cout << "UI::Select two individuals" << endl;
		return;
	}

	cout << "UI::Action: Merge" << endl;

	// Highlight selected individuals for 1 sec
	changeState(Running, 1000);

	int leaderIndex = getLeaderIndex( getSelector()->getSelectors()[0] );
	if (leaderIndex < 0)
	{
		cout << "UI::Enable cluster labels local propagation" << endl;
		return;
	}

	hBehaviorClusteringParams.mergeClusterLeaders[0] = leaderIndex;

	leaderIndex = getLeaderIndex( getSelector()->getSelectors()[1] );
	if (leaderIndex < 0)
	{
		cout << "UI::Enable cluster labels local propagation" << endl;
		hBehaviorClusteringParams.mergeClusterLeaders[0] = 0;
		return;
	}

	hBehaviorClusteringParams.mergeClusterLeaders[1] = leaderIndex;
	
} // actionMerge

// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::actionSplit()
{
	cout << "UI::Action: Split" << endl;
}

// ----------------------------------------------------------
// ----------------------------------------------------------

void UserInteraction::actionFreeze()
{
	// Get the selected index
	int leaderIndex = getLeaderIndex( getSelector()->getSelectors()[0] );
	if (leaderIndex < 0)
	{
		cout << "UI::Enable cluster labels local propagation" << endl;
		return;
	}

	// Add the unsorted val to the GPU datastructure
	hBehaviorClusteringParams.freezeClusterLeader = leaderIndex; 

	// Compute the sorted value
	leaderIndex = 
		m_Hash->getHostArrayElement(
		leaderIndex, true).y;

	IndividualSelectorType::iterator frozenLeadersIt = 
		std::find(m_FrozenLeaders.begin(), m_FrozenLeaders.end(), leaderIndex);

	// Un-freeze if into the list
	if (frozenLeadersIt != m_FrozenLeaders.end())
	{
		m_FrozenLeaders.erase(frozenLeadersIt);
		cout << "UI::Action: UN-Freeze-->Leader(" << leaderIndex << ")" << endl;
		return;
	}

	// Freeze the selected leader
	m_FrozenLeaders.push_back(leaderIndex);
	cout << "UI::Action: Freeze-->Leader(" << leaderIndex << ")" << endl;
	
}