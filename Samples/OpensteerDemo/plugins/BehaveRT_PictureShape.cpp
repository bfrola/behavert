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
#include "Shapes3D/include/Shapes3DPlugIn.h"

#include <fstream>

#include <cutil.h>
#include <cuda_gl_interop.h>


// Include names declared in the OpenSteer namespace into the namespaces to search to find names.
using namespace OpenSteer;

typedef 
	Shapes3DPlugIn
		<Drawable3DPlugIn
			<OpenSteerWrapperPlugIn	
				<Proximity3DPlugIn
					<EnvGrid3DPlugIn
						<Body3DPlugIn<DummyPlugIn>>>>>>
							BehaviorEngineType;

#define CS_CONFIG "Config//BehaveRT_PictureShape.cfg"
#define PLUGIN_CONFIG "Config//PluginSettings.cfg"

// ----------------------------------------------------------------------------
// PlugIn for OpenSteerDemo

class BehaveRT_PictureShape : public PlugIn
{
public:
    
    const char* name (void) {return "BehaveRT_PictureShape";}
    float selectionOrderSortKey (void) {return 0.03f;}
    virtual ~BehaveRT_PictureShape() {} // be more "nice" to avoid a compiler warning


    void open (void)
    {
		drawBoids = true;
		updateBoids = true;

		m_ConfigFile = new UtilityConfigFile(PLUGIN_CONFIG);

		m_StopFrameCounterAt = StringConverter::parseInt(
			m_ConfigFile->getSetting("stopFrameCounterAt", "Benchmarking"));

		m_BlendingValue = StringConverter::parseFloat(
			m_ConfigFile->getSetting("blendingValue", "Benchmarking"));

		m_ShowDetailedTimingsReport = StringConverter::parseBool(
			m_ConfigFile->getSetting("showDetailedTimingsReport", "Benchmarking"));

		m_ShowHudInfo = StringConverter::parseBool(
			m_ConfigFile->getSetting("showHudInfo", "Benchmarking"));
		
		m_CalcWorldRadiusAutomatically = StringConverter::parseBool(
			m_ConfigFile->getSetting("calcWorldRadiusAutomatically", "General"));

		m_UpdateElapsedTime = StringConverter::parseFloat(
			m_ConfigFile->getSetting("updateElapsedTime", "Benchmarking"));

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
		if (frameCounter > m_StopFrameCounterAt || OpenSteerDemo::clock.getPausedState())
		{
			return;
		}

		if (frameCounter == m_StopFrameCounterAt)
		{
			storeBenchmarkResume();
			OpenSteer::OpenSteerDemo::printMessage ("exit.");
            OpenSteer::OpenSteerDemo::exit (0);
		}

		if (frameCounter == 0)
			reset();

		frameCounter ++;

		if ( !updateBoids )
			return;

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
		alignmentElapsed += getElapsedAndRestartGlobalTimer();

		m_BehaviorEngine->computeCohesions();
		cohesionElapsed += getElapsedAndRestartGlobalTimer();

		m_BehaviorEngine->moveTowardsTarget();

		m_BehaviorEngine->computeSeekingsWorldCenter();
		boundaryHandlingElapsed += getElapsedAndRestartGlobalTimer();

		m_BehaviorEngine->applySteeringForces(m_UpdateElapsedTime);
		steeringElapsed += getElapsedAndRestartGlobalTimer();
	
		m_CommonRes.getDeviceInterface()->unmapVBOinDeviceDataRepository();
		unmapVBOsElapsed += getElapsedAndRestartGlobalTimer();
		cutStopTimer(m_GlobalTimer);

		// Global timer
		updateElapsed += timeGetTime() - startingUpdateTime;

    }

	void printStatistics(
		std::ostringstream& mainStr, std::string label, 
		float elapsedTime, float totalElapsedTime, // elapsedTime in ms
		int frameCounter) 
	{
		float amountVal = ((float) elapsedTime / frameCounter );
		mainStr << label;
		mainStr << amountVal * 100 / totalElapsedTime << "% (" << amountVal << "ms)\n";
	}

    void redraw (const float currentTime, const float elapsedTime)
    {

		// ------------------------------------------------------------------
		// Draw stage

        // selected vehicle (user can mouse click to select another)
		AbstractVehicle& selected = *OpenSteerDemo::selectedVehicle;
		 // update camera
		OpenSteerDemo::updateCamera (currentTime, elapsedTime, selected);

		boidsDrawingStartTime = timeGetTime();

		if(drawBoids)
		{
			m_BehaviorEngine->drawBodiesAsPointSprites();		
		}
		
		boidsDrawingEndTime = timeGetTime();

		coloringElapsed += boidsDrawingEndTime - boidsDrawingStartTime;
		

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
			printStatistics(updateTimes, "  flockingTime:    ", (separationElapsed + cohesionElapsed + alignmentElapsed), totalElapsed, frameCounter);
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
	
    }

    void close (void)
    {
		delete m_BehaviorEngine;
    }

    void reset (void)
    {
     
        // reset camera position
        OpenSteerDemo::position3dCamera (*OpenSteerDemo::selectedVehicle);

        // make camera jump immediately to new position
        OpenSteerDemo::camera.doNotSmoothNextMove ();
				
		cudaGLSetGLDevice( /*cutGetMaxGflopsDeviceId()*/ 0 );
		m_CommonRes.initialize(CS_CONFIG);

		// Create the custom CrowdEngine reference
		m_BehaviorEngine = new BehaviorEngineType();

		// Reset all plugIns
		m_BehaviorEngine->reset();
		
		// World Explorer
		glClearColor(1.0, 1.0, 1.0, 1.0);

		// Reset timer
		cutCreateTimer(&m_GlobalTimer);

    }    



    void handleFunctionKeys (int keyNumber)
    {
		const int totalPD = 4; 

        switch (keyNumber)
        {
			case 1: m_BehaviorEngine->setupNextShape();   break;
			case 3: m_BehaviorEngine->useShiftedIndexOnOff(); break;
        }
    }

    void printMiniHelpForFunctionKeys (void)
    {
        std::ostringstream message;
        message << "Function keys handled by ";
        message << '"' << name() << '"' << ':' << std::ends;
        OpenSteerDemo::printMessage (message);
        OpenSteerDemo::printMessage ("  No options.");
        OpenSteerDemo::printMessage ("");
    }

	void storeBenchmarkResume()
	{
		
		SYSTEMTIME dateTime;
        GetSystemTime(&dateTime);

		std::ostringstream datetimeStr;

		std::string directoryBase = m_ConfigFile->getSetting("directoryBase", "Benchmarking");

		datetimeStr << "csboids_" << dateTime.wYear << "-" << dateTime.wMonth << "-" << dateTime.wDay << "_" <<
			dateTime.wHour << "-" << dateTime.wMinute << "-" << dateTime.wSecond;

		std::string finalDirecectory = directoryBase + datetimeStr.str();

		CreateDirectory(finalDirecectory.c_str(), 0);

		CopyFile(CS_CONFIG, (finalDirecectory + "\\" + CS_CONFIG).c_str(), 0);
		CopyFile(PLUGIN_CONFIG, (finalDirecectory + "\\" + PLUGIN_CONFIG).c_str(), 0);
		
		std::ofstream outFile;
		outFile.open((finalDirecectory + "\\result.bench").c_str());

		outFile << "# CrowdSteer CSBoids demo benchmark \n";
		outFile << "# Datetime: "  << dateTime.wYear << "-" << dateTime.wMonth << "-" << dateTime.wDay << "_" <<
			dateTime.wHour << "-" << dateTime.wMinute << "-" << dateTime.wSecond << std::endl;
		
		outFile << std::endl;
		outFile << "[Configuration summary]" << std::endl;
		outFile << "BlockDim=" << m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim << std::endl;
		outFile << "numBodies=" << hBody3DParams.numBodies << std::endl;
		outFile << "use2DProjection=" << hBody3DParams.use2DProjection << std::endl;
		outFile << std::endl;

		outFile << "disableDrawing=" << hDrawable3DParams.disableDrawing << std::endl;
		outFile << "renderingType=" << hDrawable3DParams.renderingType << std::endl;
		outFile << "neighborhoodColoring=" << hDrawable3DParams.neighborhoodColoring << std::endl;
		outFile << std::endl;

		outFile << "worldRadius=" << hEnvGrid3DParams.worldRadius.x << std::endl;
		outFile << "cellSize=" << hEnvGrid3DParams.cellSize.x << " "<< 
			hEnvGrid3DParams.cellSize.y << " " << hEnvGrid3DParams.cellSize.z << std::endl;
		outFile << "gridSize=" << hEnvGrid3DParams.gridSize.x << " "<< 
			hEnvGrid3DParams.gridSize.y << " " << hEnvGrid3DParams.gridSize.z << std::endl;
		outFile << "numCells=" << hEnvGrid3DParams.numCells << std::endl;
		outFile << "maxBodiesPerCell=" << hEnvGrid3DParams.maxBodiesPerCell << std::endl;
		outFile << std::endl;

		outFile << "useKnn=" << hProximity3DParams.useKnn << std::endl;
		outFile << "maxNeighbors=" << hProximity3DParams.maxNeighbors << std::endl;
		outFile << std::endl;

		outFile << "[Device infos]" << std::endl;
		outFile << "name=" << 
			m_CommonRes.getDeviceInterface()->getDeviceInfo().name << std::endl;
		outFile << "multiProcessorCount=" << 
			m_CommonRes.getDeviceInterface()->getDeviceInfo().multiProcessorCount << std::endl;
		outFile << "totalGlobalMem=" << 
			m_CommonRes.getDeviceInterface()->getDeviceInfo().totalGlobalMem << std::endl;
		outFile << "sharedMemPerBlock=" << 
			m_CommonRes.getDeviceInterface()->getDeviceInfo().sharedMemPerBlock << std::endl;
		outFile << "regsPerBlock=" << 
			m_CommonRes.getDeviceInterface()->getDeviceInfo().regsPerBlock << std::endl;
		outFile << "totalConstMem=" << 
			m_CommonRes.getDeviceInterface()->getDeviceInfo().totalConstMem << std::endl;
		outFile << "warpSize=" << 
			m_CommonRes.getDeviceInterface()->getDeviceInfo().warpSize << std::endl;

		outFile << std::endl;

		outFile << "[Results] #(ms)" << std::endl;
		outFile << "initializeElapsed=" << (float) initializeElapsed / frameCounter << std::endl;
		outFile << "neighboroodSearchingElapsed=" << (float) neighboroodSearchingElapsed / frameCounter << std::endl;
		outFile << "separationElapsed=" << (float) separationElapsed / frameCounter << std::endl;
		outFile << "cohesionElapsed=" << (float) cohesionElapsed / frameCounter << std::endl;
		outFile << "alignmentElapsed=" << (float) alignmentElapsed / frameCounter << std::endl;
		outFile << "boundaryHandlingElapsed=" << (float) boundaryHandlingElapsed / frameCounter << std::endl;
		outFile << "steeringElapsed=" << (float) steeringElapsed / frameCounter << std::endl;
		outFile << "updateElapsed=" << (float) updateElapsed / frameCounter << std::endl;
		outFile << "coloringElapsed=" << (float) coloringElapsed / frameCounter << std::endl;
		
		outFile.close();

		outFile.open((finalDirecectory + "\\..\\overall_result.bench").c_str(), std::ios::app);

		outFile << 
			datetimeStr.str() << "\t" << 
			m_CommonRes.getDeviceInterface()->getDeviceInfo().name << "\t" << 
			m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim << "\t" << 
			hBody3DParams.numBodies << "\t" << 
			hBody3DParams.use2DProjection << "\t" << 
			hDrawable3DParams.disableDrawing << "\t" << 
			hDrawable3DParams.renderingType << "\t" << 
			hDrawable3DParams.neighborhoodColoring << "\t" << 
			hEnvGrid3DParams.worldRadius.x << "\t" << 
			hEnvGrid3DParams.cellSize.x << " "<< hEnvGrid3DParams.cellSize.y << " " << hEnvGrid3DParams.cellSize.z << "\t" << 
			hEnvGrid3DParams.gridSize.x << " "<< hEnvGrid3DParams.gridSize.y << " " << hEnvGrid3DParams.gridSize.z << "\t" << 
			hEnvGrid3DParams.maxBodiesPerCell << "\t" << 
			hProximity3DParams.maxNeighbors << "\t" << 
			(float) initializeElapsed / frameCounter << "\t" << 
			(float) neighboroodSearchingElapsed / frameCounter << "\t" << 
			(float) separationElapsed / frameCounter << "\t" << 
			(float) cohesionElapsed / frameCounter << "\t" << 
			(float) alignmentElapsed / frameCounter << "\t" << 
			(float) boundaryHandlingElapsed / frameCounter << "\t" << 
			(float) steeringElapsed / frameCounter << "\t" << 
			(float) updateElapsed / frameCounter << "\t" << 
			(float) coloringElapsed / frameCounter << "\t" << std::endl;

		printf("%d %d %d\n", 1000/(updateDrawElapsed / frameCounter), updateElapsed,
			updateDrawElapsed / frameCounter - updateElapsed / frameCounter);
		
		outFile.close();

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

	UtilityConfigFile* m_ConfigFile;

	BehaviorEngineType* m_BehaviorEngine;


};


BehaveRT_PictureShape gBoidsPlugIn;

// ----------------------------------------------------------------------------









