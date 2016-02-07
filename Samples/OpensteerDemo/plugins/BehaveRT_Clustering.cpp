// ----------------------------------------------------------------------------
//
//
// OpenSteer -- Steering Behaviors for Autonomous Characters
//
// Copyright (c) 2002-2003, Sony Computer Entertainment America
// Original author: Craig Reynolds <craig_reynolds@playstation.sony.com>
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
//
// ----------------------------------------------------------------------------
//
//
// 
// 09-26-02 cwr: created 
//
//
// ----------------------------------------------------------------------------

#include <sstream>
#include "OpenSteer/OpenSteerDemo.h"
#include "OpenSteer/SimpleVehicle.h"

#include "BehaveRT.h"

#include "Body/include/Body3DPlugIn.h"
#include "EnvGrid3D/include/EnvGrid3DPlugIn.h"
#include "Proximity3D/include/Proximity3DPlugin.h"
#include "OpenSteerWrapper/include/OpenSteerWrapperPlugIn.h"
#include "Drawable3D/include/Drawable3DPlugIn.h"
#include "OpenSteerWrapper/include/OpenSteerWrapperPlugIn.h"
#include "Schooling/include/SchoolingPlugIn.h"
#include "BehaviorClustering/include/BehaviorClusteringPlugIn.h"

// Helpers
#include "OpenSteerWrapper/include/IndividualSelector.h"
#include "../Include/UserInteraction.h"

// Other

#include <fstream>

#include <cutil.h>
#include <cuda_gl_interop.h>


// Include names declared in the OpenSteer namespace into the namespaces to search to find names.
using namespace OpenSteer;

typedef 
	BehaviorClusteringPlugIn 
		<Drawable3DPlugIn
			<OpenSteerWrapperPlugIn	
				<Proximity3DPlugIn
					<EnvGrid3DPlugIn
						<Body3DPlugIn<DummyPlugIn>>>>>>
							BehaviorEngineClass;

#define BEHAVERT_CONFIG "Config//BehaveRT_Clustering.cfg"
#define PLUGIN_CONFIG "Config//PluginSettings.cfg"

// ----------------------------------------------------------------------------
// PlugIn for OpenSteerDemo

// Helper
// Global, so it is visible in additionalMotionFunc
//	and additionalMouseFunc
UserInteraction* m_UserInteraction;

bool m_MotionFlag = false;

void additionalMotionFunc(int x, int y)
{
	// Keep position on mouse motion
	OpenSteerDemo::mouseX = x;
	OpenSteerDemo::mouseY = y; 

	m_UserInteraction->mouseMotion(x, y);
}


void additionalMouseFunc(int x, int y, int button, int state)
{
	m_UserInteraction->mouseClick(x, y, button, state);
}

void additionalKeyboardFunc(int key, int, int)
{
	m_UserInteraction->keyboardKeyPressed(key);
}

class BehaveRT_Clustering : public PlugIn
{
public:
    
    const char* name (void) {return "BehaveRT_Clustering";}
    float selectionOrderSortKey (void) {return 0.03f;}
    virtual ~BehaveRT_Clustering() {} // be more "nice" to avoid a compiler warning


    void open (void)
    {
		drawBoids = true;
		updateBoids = true;

		m_ConfigFile = new UtilityConfigFile(PLUGIN_CONFIG);

		m_StopFrameCounterAt = StringConverter::parseInt(
			m_ConfigFile->getSetting("stopFrameCounterAt", "Benchmarking"));

		m_BlendingValue = StringConverter::parseFloat(
			m_ConfigFile->getSetting("blendingValue", "Benchmarking"));

		m_ShowDetailedTimingsReport = 
			//StringConverter::parseBool(
			//m_ConfigFile->getSetting("showDetailedTimingsReport", "Benchmarking"));
			false;

		m_ShowHudInfo = StringConverter::parseBool(
			m_ConfigFile->getSetting("showHudInfo", "Benchmarking"));
		
		m_CalcWorldRadiusAutomatically = StringConverter::parseBool(
			m_ConfigFile->getSetting("calcWorldRadiusAutomatically", "General"));

		m_UpdateElapsedTime = StringConverter::parseFloat(
			m_ConfigFile->getSetting("updateElapsedTime", "Benchmarking"));

		m_ComputeLocalPropagation = false;


        // initialize camera
        OpenSteerDemo::init3dCamera (*OpenSteerDemo::selectedVehicle);
		OpenSteerDemo::camera.mode = Camera::cmFixed;
        OpenSteerDemo::camera.fixedDistDistance = OpenSteerDemo::cameraTargetDistance;
        OpenSteerDemo::camera.fixedDistVOffset = 100;
        OpenSteerDemo::camera.lookdownDistance = 40;
        OpenSteerDemo::camera.aimLeadTime = 0.5;
        OpenSteerDemo::camera.povOffset.set (10, 10, 10);

		boidsDrawingStartTime = timeGetTime();
		boidsDrawingEndTime = timeGetTime();

		const float startElapsed = 0.0f;
		initializeElapsed = startElapsed;
		neighboroodSearchingElapsed = startElapsed;
		separationElapsed = startElapsed;
		alignmentElapsed = startElapsed;
		cohesionElapsed = startElapsed;
		boundaryHandlingElapsed = startElapsed;
		steeringElapsed = startElapsed;
		updateElapsed = startElapsed;
		coloringElapsed = startElapsed;
		updateDrawElapsed = startElapsed;

	}

	float getElapsedAndRestartGlobalTimer()
	{
		cutStopTimer(m_GlobalTimer);
		float timerVal = 
			cutGetTimerValue(m_GlobalTimer);

		cutResetTimer(m_GlobalTimer);
		cutStartTimer(m_GlobalTimer);

		return timerVal;
	}

    void update (const float currentTime, const float elapsedTime)
    {
		if (OpenSteerDemo::clock.getPausedState())
		{
			return;
		}

		if (frameCounter == 0)
			reset();

		frameCounter ++;

		if ( !updateBoids )
			return;

		m_UserInteraction->update();

		cutStartTimer(m_GlobalTimer);
		startingUpdateTime = timeGetTime();

		m_CommonRes.getDeviceInterface()->mapVBOinDeviceDataRepository();
		mapVBOsElapsed += getElapsedAndRestartGlobalTimer();
		
		m_BehaviorEngine->reorderSimData();
		initializeElapsed += getElapsedAndRestartGlobalTimer();
		
		m_BehaviorEngine->computeNeighborhoods();
		neighboroodSearchingElapsed += getElapsedAndRestartGlobalTimer();
		
		m_BehaviorEngine->resetSteerForce();
		m_BehaviorEngine->computeSeparations();
		separationElapsed += getElapsedAndRestartGlobalTimer();
		
		m_BehaviorEngine->computeAlignments();
		m_BehaviorEngine->computeCohesions();

		m_BehaviorEngine->steerForClustering();
		alignmentElapsed += getElapsedAndRestartGlobalTimer();
				
		if (m_ComputeLocalPropagation)
			m_BehaviorEngine->computeClusterMembership();
		cohesionElapsed += getElapsedAndRestartGlobalTimer();

		// Merge is active when the elements of the array
		// hBehaviorClusteringParams.mergeClusterLeaders hold different values
		m_BehaviorEngine->mergeClusters();
		

		m_BehaviorEngine->computeSeekingsWorldCenter();
		boundaryHandlingElapsed += getElapsedAndRestartGlobalTimer();

		m_BehaviorEngine->applySteeringForces(m_UpdateElapsedTime);
		steeringElapsed += getElapsedAndRestartGlobalTimer();

		m_BehaviorEngine->freezeClusters();
	
		m_CommonRes.getDeviceInterface()->unmapVBOinDeviceDataRepository();
		unmapVBOsElapsed += getElapsedAndRestartGlobalTimer();
		cutStopTimer(m_GlobalTimer);

		// Global timer
		updateElapsed += timeGetTime() - startingUpdateTime;

    }


	void showClusterMembershipStats() 
	{
		if (!m_ShowClusterMembershipStats)
			return; 

		m_BehaviorEngine->getClusterMembership()->copyArrayFromDevice();

		typedef std::vector<int> LabelsListType;
		LabelsListType clusterLabels;
		LabelsListType realLabelsCounter;
		typedef std::vector<std::vector<int>*> ContentListType;
		ContentListType clusterContentList;
		ContentListType clusterContentCounterList;

		// ---------------------------
		// Centralized local propagation analysis
		for (int i = 0; i < m_BehaviorEngine->getElementsNumber(); i ++)
		{
			float3 clusterMembership = 
				m_BehaviorEngine->getClusterMembership()->getHostArrayElement(i);

			int clusterLabel = clusterMembership.x;
			int realClusterLabel = clusterMembership.z;

			LabelsListType::iterator clusterLabelsIt = 
				find(clusterLabels.begin(), clusterLabels.end(), clusterLabel);

			if (clusterLabelsIt == clusterLabels.end())
			{
				clusterLabels.push_back( clusterLabel );

				LabelsListType* clusterContent = new LabelsListType();
				clusterContent->push_back( realClusterLabel );
				clusterContentList.push_back( clusterContent );

				LabelsListType* clusterContentCounter = new LabelsListType();
				clusterContentCounter->push_back( 1 );
				clusterContentCounterList.push_back( clusterContentCounter );
				continue;
			}

			// If the element has been found...

			int foundAt = distance( clusterLabels.begin(), clusterLabelsIt );

			LabelsListType* clusterContent = 
				clusterContentList[ foundAt ];

			LabelsListType* clusterContentCounter = 
				clusterContentCounterList[ foundAt ];

			clusterLabelsIt = 
				find(clusterContent->begin(), clusterContent->end(), realClusterLabel);

			if (clusterLabelsIt == clusterContent->end())
			{	
				clusterContent->push_back( realClusterLabel );
				clusterContentCounter->push_back( 1 );
				continue;
			}

			foundAt = distance( clusterContent->begin(), clusterLabelsIt );

			clusterLabelsIt = clusterContentCounter->begin() + foundAt;
			*clusterLabelsIt = clusterContentCounter->at( foundAt ) + 1;

		} // for i = 0 ..  getElementsNumber() - 1

		
		// ---------------------------
		// Print results
		//std::ostringstream clusterStatsText;	

		const int fontHeightPx = 16;
		const int fontWidthPx = 40;
		const int baseYPx = 30;
		const int baseXCaptionPx = 5;
		const int baseXContentPx = 45;

		int maxRealLabel = 0;
		
		for (int i = 0; i < clusterLabels.size(); i ++)
		{
			Vec3 clusterPosition = float42Vec3(
				m_BehaviorEngine->getPos()->getHostArrayElement(
					m_BehaviorEngine->getSortedIndex( clusterLabels[ i ] ), true));

			std::ostringstream clusterLabelText;
			clusterLabelText << "  <<<<< C" << clusterLabels[ i ] << std::ends;

			draw2dTextAt3dLocation(
				clusterLabelText, 
				clusterPosition, 
				gBlack);
			
			clusterLabelText.str("");
			clusterLabelText.clear();
			clusterLabelText << "> C" << clusterLabels[ i ] << std::ends;

			int confusionMatrixRowY = baseYPx + (i + 1) * fontHeightPx;

			draw2dTextAt2dLocation(
				clusterLabelText, 
				Vec3(baseXCaptionPx, confusionMatrixRowY, 0), 
				gBlack);
			
			LabelsListType* clusterContent = clusterContentList[ i ];
			LabelsListType* clusterContentCounter = clusterContentCounterList[ i ];
			
			for (int j = 0; j < clusterContent->size(); j ++)
			{
				std::ostringstream occurrencesStr;
				occurrencesStr << clusterContentCounter->at(j) << std::ends;

				draw2dTextAt2dLocation(
					occurrencesStr, 
					Vec3(
						baseXContentPx + (clusterContent->at(j) + 1) * fontWidthPx, 
						confusionMatrixRowY, 0), 
					gBlack);

				if (maxRealLabel < clusterContent->at(j))
					maxRealLabel = clusterContent->at(j);
			}

		} // for i = 0 .. clusterLabels.size() - 1

		//clusterStatsText << clusterLabels.size() << " clusters\n";

		//draw2dTextAt2dLocation(
		//	clusterStatsText, 
		//	Vec3(2, 30 + clusterLabels.size() * 16, 0), 
		//	gBlack);

		//clusterStatsText << std::ends;

		// Draw matrix caption [1..maxRealLabel]
		for (int clusterLabel = 1; clusterLabel <= maxRealLabel; clusterLabel ++)
		{

			std::ostringstream occurrencesStr;
			occurrencesStr << clusterLabel << std::ends;

			draw2dTextAt2dLocation(
				occurrencesStr, 
				Vec3(
					baseXContentPx + (clusterLabel + 1) * fontWidthPx,  // +1 for caption
					baseYPx + (clusterLabels.size() + 1) * fontHeightPx, 
					0), 
				gBlack);
		} // for clusterLabel = 1 .. maxRealLabel

		std::ostringstream clusterCountStr;
		clusterCountStr << clusterLabels.size() << " clusters" << std::ends;

		draw2dTextAt2dLocation(
			clusterCountStr, 
			Vec3(
				baseXCaptionPx, 
				baseYPx, 
				0), 
			gBlack);

		// ---------------------------
		// Esport result to file
		if (m_ExportClassificationResult && clusterLabels.size() == maxRealLabel )
		{
			
			for (int i = 0; i < m_BehaviorEngine->getElementsNumber(); i ++)
			{
				float3 clusterMembership = 
					m_BehaviorEngine->getClusterMembership()->getHostArrayElement(i);
				int clusterLabel = clusterMembership.x;

				m_ResultOutFile << clusterLabel;
				if ( i < m_BehaviorEngine->getElementsNumber() - 1)
					m_ResultOutFile << ", ";
			}
			m_ResultOutFile << "\n";

			std::ostringstream warnStr;
			warnStr << "WRITING RESULTS TO: " << m_ResultOutFileName << std::ends;
			draw2dTextAt2dLocation(
				warnStr, 
				Vec3(
					baseXCaptionPx + 200, 
					baseYPx, 
					0), 
				gRed);

		}

		
	} // showClusterMembershipStats


	void exportResultToFile()
	{
		
	}

	// ------------------------------------------------------------
	// ------------------------------------------------------------

	void drawHighlightAt(Vec3 position)
	{
		float circleSize = 0.5f;
		float definition = 8.0f;

		draw3dCircle(circleSize, position, 
			Vec3::up, gGreen, definition);
		draw3dCircle(circleSize, position, 
			Vec3(0, 0, 1), gRed, definition);

		draw3dCircle(circleSize * 2, position, 
			Vec3::up, gRed, definition);
		draw3dCircle(circleSize * 2, position, 
			Vec3(0, 0, 1), gGreen, definition);
	} // drawHighlightAt

	// ------------------------------------------------------------
	// ------------------------------------------------------------

	void printStatistics(
		std::ostringstream& mainStr, std::string label, 
		float elapsedTime, float totalElapsedTime, // elapsedTime in ms
		int frameCounter) 
	{
		float amountVal = ((float) elapsedTime / frameCounter );
		mainStr << label;
		mainStr << amountVal * 100 / totalElapsedTime << "% (" << amountVal << "ms)\n";
	} // printStatistics

	// ------------------------------------------------------------
	// ------------------------------------------------------------

	void selectNearestElementToMouse()
	{

		const Vec3 direction = 
			directionFromCameraToScreenPosition (
			OpenSteerDemo::mouseX, 
			OpenSteerDemo::mouseY);


	}
	
	// ------------------------------------------------------------
	// ------------------------------------------------------------

    void redraw (const float currentTime, const float elapsedTime)
    {

		// ------------------------------------------------------------------
		// Draw stage

		AbstractVehicle& selected = *OpenSteerDemo::selectedVehicle;
		 // update camera
		OpenSteerDemo::updateCamera (currentTime, elapsedTime, selected);

		hDrawable3DParams.mouseDirection = 
			OpenSteerWrapper::Vec32float3(
				directionFromCameraToScreenPosition(
					OpenSteerDemo::mouseX, 
					OpenSteerDemo::mouseY));

		hDrawable3DParams.cameraPosition = 
			OpenSteerWrapper::Vec32float3(
				OpenSteerDemo::camera.position());

		m_BehaviorEngine->computeMouseDistance();

		// DEBUG 
		//float mouseDistance = 
		//	m_BehaviorEngine->getMouseDistance()->getHostArrayElement(0, true);
			
		// DEBUG 
		//m_BehaviorEngine->getColor()->
		//	getSize() // numelements
		//	getBytesCount() // memsize

		// DEBUG 
		//float mouseDistance = 
		//	m_BehaviorEngine->getMouseDistance()->getHostArrayElement(0, false);
		
		//m_BehaviorEngine->getPos()->getHostArrayElement(i);

		// TODO: use getHostArrayElement(0, true); with reduction
		int highlightedElement = (int)
			m_BehaviorEngine->getMouseDistance()->getHostArrayElement(0, false);

		// DEBUG 
		//cout << selectedElement << " " << mouseDistance << endl;
		//cout << mouseDistance << " " << 
		//	m_BehaviorEngine->getMouseDistance()->getHostArrayElement(700, true) << endl;

		m_UserInteraction->getSelector()->setHighlighted( highlightedElement );
		//m_UserInteraction->getSelector()->selectHighlighted();
		m_UserInteraction->draw();
		

		m_BehaviorEngine->drawBodiesAsPointSprites();	

		if (m_ComputeLocalPropagation)
			m_BehaviorEngine->drawClusterMemberships();
		
		drawBoxOutline(LocalSpace(), 
			Vec3(hEnvGrid3DParams.worldRadius.x * 2, 
			hEnvGrid3DParams.worldRadius.y * 2, 
			hEnvGrid3DParams.worldRadius.z * 2),
			gGray40);

		selectNearestElementToMouse();
		

		// ------------------------------------------------------------------
		// Reporting stage

		if (m_ShowDetailedTimingsReport)
		{
		
			std::ostringstream updateTimes;
			updateTimes.precision( 2 );
			
			//const float totalElapsed = (float) ( positionUpdatingTime - startingUpdateTime );
			const float totalElapsed = ((float) updateElapsed / frameCounter );

			updateTimes << "FrameCounter: ";
			updateTimes << frameCounter << "\n";

			updateTimes << "Update: ";
			updateTimes << totalElapsed << "ms (" << 1 / totalElapsed * 1000 << "fps)\n";
			float amountVal;	

			printStatistics(updateTimes, "  mapVBOs:         ", mapVBOsElapsed, totalElapsed, frameCounter);
			printStatistics(updateTimes, "  reord.DataTime:  ", initializeElapsed, totalElapsed, frameCounter);
			printStatistics(updateTimes, "  neigh.Searching: ", neighboroodSearchingElapsed, totalElapsed, frameCounter);
			printStatistics(updateTimes, "  separation:      ", separationElapsed, totalElapsed, frameCounter);
			printStatistics(updateTimes, "  clustering:      ", alignmentElapsed, totalElapsed, frameCounter);
			printStatistics(updateTimes, "  labeling:        ", cohesionElapsed, totalElapsed, frameCounter);
			printStatistics(updateTimes, "  boundaryHandle:  ", boundaryHandlingElapsed, totalElapsed, frameCounter);
			printStatistics(updateTimes, "  steeringTime:    ", steeringElapsed, totalElapsed, frameCounter);
			printStatistics(updateTimes, "  unmapVBOs:       ", unmapVBOsElapsed, totalElapsed, frameCounter);
			
			const float h = drawGetWindowHeight ();
			updateTimes << std::ends;
			draw2dTextAt2dLocation (updateTimes, Vec3( drawGetWindowWidth() - 350, h - 40, 0 ), gBlack);

		}

		//if (/*m_ShowMinimalTimingsReport*/ true)
		if (false)
		{
			std::ostringstream updateTimes;
			updateTimes.precision( 3 );
			updateTimes << "Update: ";
			updateTimes << (float)updateElapsed  / frameCounter << "ms (" 
				<< 1000 / ((float)updateElapsed / frameCounter) 
						<< "fps)\n";
			updateTimes << std::ends;
			draw2dTextAt2dLocation (updateTimes, Vec3( drawGetWindowWidth() - 350, drawGetWindowHeight () - 40, 0 ), gBlack);
		}

		showClusterMembershipStats();
	
    }

	// ------------------------------------------------------------
	// ------------------------------------------------------------

    void close (void)
    {
		m_ResultOutFile.close();
		delete m_BehaviorEngine;
		delete m_UserInteraction;
    }

	// ------------------------------------------------------------
	// ------------------------------------------------------------



    void reset (void)
    {

		
     
        // reset camera position
        OpenSteerDemo::position3dCamera (*OpenSteerDemo::selectedVehicle);

        // make camera jump immediately to new position
        OpenSteerDemo::camera.doNotSmoothNextMove ();
				
		cudaGLSetGLDevice( /*cutGetMaxGflopsDeviceId()*/ 0 );
		m_CommonRes.initialize(BEHAVERT_CONFIG);

		// Create the custom CrowdEngine reference
		m_BehaviorEngine = new BehaviorEngineClass();

		// Reset all plugIns
		m_BehaviorEngine->reset();

		
		// User Interface helper
		m_UserInteraction = new UserInteraction(m_BehaviorEngine->getPos());
		m_UserInteraction->setClusterMembership(m_BehaviorEngine->getClusterMembership());
		m_UserInteraction->setHash(m_BehaviorEngine->getHash());
		
		// World Explorer
		glClearColor(1.0, 1.0, 1.0, 1.0);

		// Reset timer
		cutCreateTimer(&m_GlobalTimer);

		// Add mouse and keyboard additional listeners
		OpenSteer::setAdditionalMotionFunction(&Drawable3D::additionalMotionFunction);
		OpenSteer::setAdditionalMotionFunction(&additionalMotionFunc);
		
		OpenSteer::setAdditionalMouseFunction(&Drawable3D::additionalMouseFunction);
		OpenSteer::setAdditionalMouseFunction(&additionalMouseFunc);

		OpenSteer::setAdditionalSpecialFunction(Drawable3D::additionalSpecialFunction);
		OpenSteer::setAdditionalKeyboardFunction(additionalKeyboardFunc);

		std::stringstream fileName;
		std::string datasetName;


		std::string datasetPath = m_BehaviorEngine->getDatasetPath();

		datasetName = datasetPath.substr(
			datasetPath.find_last_of('\\') + 1, 
			datasetPath.size() - datasetPath.find_last_of('.') - 1);

		fileName << "R" << datasetName << 
			"_" << timeGetTime() << ".txt";

		m_ResultOutFileName = fileName.str();
		

		cout << "Clustering output file: " << m_ResultOutFileName << endl;

    }    


	// ------------------------------------------------------------
	// ------------------------------------------------------------


    void handleFunctionKeys (int keyNumber)
    {
		const int totalPD = 4; 

        switch (keyNumber)
        {
        case 1: 
			m_ShowClusterMembershipStats = !m_ShowClusterMembershipStats;
			break;
		
		case 2:
			m_BehaviorEngine->changeDisplayParamListGL();
			break;

		case 3:
			m_ShowDetailedTimingsReport = !m_ShowDetailedTimingsReport;
			break;

		case 4:
			m_ComputeLocalPropagation = !m_ComputeLocalPropagation;
			break;

		case 5:
			if ( !m_ExportClassificationResult )
			{
				m_ResultOutFile.open( m_ResultOutFileName.c_str(), std::ios::app );
			}
			else
			{
				m_ResultOutFile.close( );
			}
			m_ExportClassificationResult = !m_ExportClassificationResult;
			break;
        }
    }

	// ------------------------------------------------------------
	// ------------------------------------------------------------

    void printMiniHelpForFunctionKeys (void)
    {
        std::ostringstream message;
        message << "Function keys handled by ";
        message << '"' << name() << '"' << ':' << std::ends;
        OpenSteerDemo::printMessage (message);
        OpenSteerDemo::printMessage ("  No options.");
        OpenSteerDemo::printMessage ("");
    }

	// return an AVGroup containing each boid of the flock
    const AVGroup& allVehicles (void) {return (const AVGroup&)flock;}
	AVGroup flock;

	typedef DWORD TimeType;
	TimeType startingUpdateTime;
	TimeType boidsDrawingStartTime;
	TimeType boidsDrawingEndTime;

	typedef float ElapsedTimeType;
	ElapsedTimeType mapVBOsElapsed;
	ElapsedTimeType initializeElapsed;
	ElapsedTimeType neighboroodSearchingElapsed;
	ElapsedTimeType separationElapsed;
	ElapsedTimeType alignmentElapsed;
	ElapsedTimeType cohesionElapsed;
	ElapsedTimeType boundaryHandlingElapsed;
	ElapsedTimeType steeringElapsed;
	ElapsedTimeType updateElapsed;

	ElapsedTimeType unmapVBOsElapsed;

	ElapsedTimeType coloringElapsed;
	ElapsedTimeType updateDrawElapsed;

	// CUDA timer
	unsigned int m_GlobalTimer;

	float proximityDBDivNum;

	bool drawBoids;
	bool updateBoids;

	int frameCounter;
	float m_UpdateElapsedTime;
	
	bool m_CalcWorldRadiusAutomatically;

	int m_StopFrameCounterAt;
	float m_BlendingValue;
	bool m_ShowDetailedTimingsReport;
	bool m_ShowHudInfo;
	bool m_ComputeLocalPropagation;

	UtilityConfigFile* m_ConfigFile;

	BehaviorEngineClass* m_BehaviorEngine;

	bool m_ShowClusterMembershipStats;
	bool m_ExportClassificationResult;

	std::ofstream m_ResultOutFile;
	std::string m_ResultOutFileName;

	
};


BehaveRT_Clustering gBoidsPlugIn;

// ----------------------------------------------------------------------------









