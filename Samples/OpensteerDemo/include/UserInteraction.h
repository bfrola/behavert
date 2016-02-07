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

#pragma once

// Built on top of OpenSteer and OpenSteerWrapper plugIn of BehaveRT
#include "OpenSteer/SimpleVehicle.h"
#include "BehaveRT.h"

#include "GL/glew.h"

#include "OpenSteerWrapper/include/IndividualSelector.h"
#include "OpenSteerWrapper/include/OpenSteerWrapperPlugIn.h"

// For interaction with BehaviorClustering's kernels
#include "BehaviorClustering/include/BehaviorClustering_kernel.cuh"

using namespace OpenSteer;
using namespace BehaveRT;

namespace BehaveRT
{
	class UserInteraction
	{
		enum UIMode {Merge, Split, Freeze};
		enum UIState {Running, Waiting}; // State of actions

		typedef DeviceArrayWrapper<float4> IndividualPositionsType;
		typedef DeviceArrayWrapper<float3> ClusterMembershipType;
		typedef DeviceArrayWrapper<uint2> HashType;

	private:
		// Fields
		UIMode m_CurrentMode;
		UIState m_CurrentState;
		int m_StateTimer; // Timer for state changes
		 // after X secs set the state to WAITING
		int m_LastStateChangetime; // Holds the datetime of the last state change

		IndividualSelector* m_Selector;
		bool m_MotionFlag;

		// Action specigfic fields
		IndividualSelectorType m_FrozenLeaders; // vector<int>

		// Positions of individuals
		IndividualPositionsType* m_Positions;
		ClusterMembershipType* m_ClusterMemerbership;
		HashType* m_Hash; // For sorted indexes

		

		// Internal Methods
		void drawFrozenLeaders();
		void highlightIndividual(Vec3, float, Vec3 = gRed);

	public:
		// Constructor/Descructor
		UserInteraction(IndividualPositionsType*);
		~UserInteraction();

		// Getters/Setters
		IndividualSelector* getSelector() { return m_Selector; }
		void setClusterMembership(ClusterMembershipType* clusterMembership) 
		{ m_ClusterMemerbership = clusterMembership; }
		void setHash(HashType* hash) { m_Hash = hash; }
		
		// Methods
		void changeMode(UIMode);
		void changeState(UIState, int);
		
		void update();
		void draw();

		void mouseClick(int, int, int, int);
		void mouseMotion(int, int);
		void keyboardKeyPressed(int);

		int getLeaderIndex(int individualIndex);
		bool sameClusterOfPreviousSelected(int individualIndex);
		bool isCurrentState(UIState);
		bool isCurrentMode(UIMode);
		
		
		// Generic actions
		void actionOneDispatcher();
		
		// Specific actions
		void actionMerge();
		void actionSplit();
		void actionFreeze();
	};
}