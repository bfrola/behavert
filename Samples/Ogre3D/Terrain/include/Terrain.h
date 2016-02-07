/*
-----------------------------------------------------------------------------
This source file is part of OGRE
    (Object-oriented Graphics Rendering Engine)
For the latest info, see http://www.ogre3d.org/

Copyright (c) 2000-2006 Torus Knot Software Ltd
Also see acknowledgements in Readme.html

You may use this sample code for anything you like, it is not covered by the
LGPL like the rest of the engine.
-----------------------------------------------------------------------------
*/

/**
    \file 
        Terrain.h
    \brief
        Specialisation of OGRE's framework application to show the
        terrain rendering plugin 
*/

#include "UtilityString.h"
#include "UtilityConfigFile.h"
#include "UtilityStringConverter.h"

#include "Body/include/Body3DPlugIn.h"
#include "EnvGrid3D/include/EnvGrid3DPlugIn.h"

#include "Proximity3D/include/Proximity3DPlugin.h"
#include "OpenSteerWrapper/include/OpenSteerWrapperPlugIn.h"
#include "Drawable3D/include/Drawable3DPlugIn.h"
#include "Shapes3D/include/Shapes3DPlugIn.h"
#include "Building3D/include/Building3DPlugIn.h"

#include "BehaveRT.h"
#include "ExampleApplication.h"

#include "BehaveRTOgreIntegration.h"

#include <GL/glew.h>
#include <GL/glut.h>


typedef 
	Building3DPlugIn
		<Drawable3DPlugIn
			<OpenSteerWrapperPlugIn
				<Proximity3DPlugIn
					<EnvGrid3DPlugIn
						<Body3DPlugIn<DummyPlugIn>>>>>>
							SimEngineFinal;


typedef enum MoveMode 
{
	FLY,
	WALK
};


RaySceneQuery* raySceneQuery = 0;
RenderQueueListener * mRenderSystemCommandsRenderQueueListener;
SimEngineFinal* m_SimEngine;
vector<SceneNode*> nodeList;
int nodeListSize = 0;
MoveMode mMoveMode;
SceneNode* mTargetNode;

typedef std::vector<SceneNode*> ObstacleList;
ObstacleList mObstacles;

typedef std::vector<float3> ObstacleListFloat3;
ObstacleListFloat3 mObstaclesPositions;


bool m_Initialized = false;
bool m_Pause = false;



// Event handler to add ability to alter curvature
class TerrainFrameListener : public ExampleFrameListener
{
public:
    TerrainFrameListener(RenderWindow* win, Camera* cam)
        : ExampleFrameListener(win, cam)
    {
        // Reduce move speed
        mMoveSpeed = 250;

    }

	void sampleSceneAndInitialize() 
	{	
		float2 startPoint;
		startPoint.x = hEnvGrid3DParams.worldOrigin.x + hEnvGrid3DParams.cellSize.x / 2;
		startPoint.y = hEnvGrid3DParams.worldOrigin.z + hEnvGrid3DParams.cellSize.z / 2;

		float upperY = hEnvGrid3DParams.worldOrigin.y + 
			hEnvGrid3DParams.cellSize.y * (hEnvGrid3DParams.gridSize.y - 1);


		// Terrain sampling
		int2 cellCounter;
		for (cellCounter.x = 0; cellCounter.x < hEnvGrid3DParams.gridSize.x; cellCounter.x ++)
		{
			for (cellCounter.y = 0; cellCounter.y < hEnvGrid3DParams.gridSize.z; cellCounter.y ++)
			{
				float2 currentPoint;
				currentPoint.x = startPoint.x + cellCounter.x * hEnvGrid3DParams.cellSize.x;
				currentPoint.y = startPoint.y + cellCounter.y * hEnvGrid3DParams.cellSize.z;

				// Check for terrain ray collision
				static Ray updateRay;
				updateRay.setOrigin(
					Vector3(currentPoint.x, upperY, currentPoint.y));
				updateRay.setDirection(Vector3::NEGATIVE_UNIT_Y);
				raySceneQuery->setRay(updateRay);
				RaySceneQueryResult& qryResult = raySceneQuery->execute();
				//RaySceneQueryResult::iterator i = qryResult.begin();
				
		        
				int counter = 0;
				for (RaySceneQueryResult::iterator i = qryResult.begin(); i != qryResult.end(); ++i)
				//if (i != qryResult.end())
				{

					int terrainDepth = 20;

					if (i->movable)
					{
						// Do not consider TerrainMipMaps, only static geometry and entities
						if (i->movable->getMovableType() != "Entity" &&
						i->movable->getMovableType() != "StaticGeometry")
							continue;

						terrainDepth = 1;
					}
					
					for (int depthIt = 0; depthIt < terrainDepth; depthIt ++)
					{
						float3 itersectionPoint = make_float3(currentPoint.x,
							(upperY - i->distance) - depthIt * hEnvGrid3DParams.cellSize.y,
							currentPoint.y);

						if (depthIt < terrainDepth)
						{
							m_SimEngine->setBuildingBlock(itersectionPoint, 1);
						}
						else
						{
							m_SimEngine->setBuildingBlock(itersectionPoint, 2);
						}
						
						
					} // for
					counter ++;

				} // if interect
			} // for cellcounter.y
		} // for cellcounter.x


		m_SimEngine->buildindComplete();
		m_Initialized = true;
	} // sampleSceneAndInitialize

	void clampCameraToTerrain()
	{
		static Ray updateRay;
		updateRay.setOrigin(mCamera->getPosition());
		updateRay.setDirection(Vector3::NEGATIVE_UNIT_Y);
		raySceneQuery->setRay(updateRay);
		RaySceneQueryResult& qryResult = raySceneQuery->execute();
		RaySceneQueryResult::iterator i = qryResult.begin();
        
		if (i != qryResult.end() && i->worldFragment)
		{
			mCamera->setPosition(mCamera->getPosition().x, 
				i->worldFragment->singleIntersection.y + 
					hBody3DParams.commonRadius * 5, 
				mCamera->getPosition().z);
		}
	}

	int mPeopleThrowCounter;


	void managePeopleThrowing() 
	{
		if (mKeyboard->isKeyDown(OIS::KC_E))
		{
			mPeopleThrowCounter += 4;
			if (mPeopleThrowCounter >= hBody3DParams.numBodies)
				mPeopleThrowCounter = 0;

			hBuilding3DParams.individualThrowingIndex = 
				mPeopleThrowCounter;
			hBuilding3DParams.individualThrowingPosition =
				hOpenSteerWrapperParams.avoidBase;


			Vector3 throwDir = 
				mCamera->getDirection() * 10 + Vector3(0, 1, 0); 

			hBuilding3DParams.individualThrowingDirection = 
				make_float3(throwDir.x, throwDir.y, throwDir.z);
					

			Building3D_copyFieldsToDevice();
			
			m_SimEngine->throwIndividuals();
			
		} 
	}

	void manageAvoidBase() 
	{
		Vector3 avoidBase = 
			mCamera->getPosition() + mCamera->getDirection() * mMouse->getMouseState().Z.abs / 10;
		hOpenSteerWrapperParams.avoidBase = make_float3(
			avoidBase.x, 
			avoidBase.y, 
			avoidBase.z);


		raySceneQuery->setRay(mCamera->getCameraToViewportRay(0.5, 0.5));
		RaySceneQueryResult& rsqResult = raySceneQuery->execute();
		
		int counter = 0;
		float minDist = 1000;
		//RaySceneQueryResult::iterator ri = rsqResult.begin();
		for (RaySceneQueryResult::iterator ri = rsqResult.begin();ri != rsqResult.end(); ++ri)
		//if (ri != rsqResult.end())
		{
			if (ri->movable)
			{
				if (ri->movable->getMovableType() != "Entity" &&
					ri->movable->getMovableType() != "StaticGeometry")
						continue;
			}

			RaySceneQueryResultEntry& res = *ri;
			
			if (res.distance < minDist)
				minDist = res.distance;

			counter ++;
		}
		mTargetNode->setPosition(avoidBase);

		int3 targetGridPos = calcGridPosH(make_float4(
					mTargetNode->getPosition().x, 
					mTargetNode->getPosition().y, 
					mTargetNode->getPosition().z, 0), 
				hEnvGrid3DParams.worldOrigin,
				hEnvGrid3DParams.cellSize);
	}


    bool frameRenderingQueued(const FrameEvent& evt)
    {
        if( ExampleFrameListener::frameRenderingQueued(evt) == false )
			return false;

		// -----------------------------------------
		// -----------------------------------------
		// Keyboard

		
		if (mKeyboard->isKeyDown(OIS::KC_X))
		{
			mMoveMode = MoveMode::FLY;
			mMoveSpeed = 300;
		} 
		else if (mKeyboard->isKeyDown(OIS::KC_C))
		{
			mMoveMode = MoveMode::WALK;
			mMoveSpeed = 40;
		}

		  // clamp to terrain on walking mode
		if ( mMoveMode == MoveMode::WALK )
		{
			clampCameraToTerrain();
		}

		if (mKeyboard->isKeyDown(OIS::KC_K))
		{
			m_SimEngine->setUseWireframe( true );
		} 
		else if (mKeyboard->isKeyDown(OIS::KC_L))
		{
			m_SimEngine->setUseWireframe( false );
		}
		// Must be before pause
		

		if (m_Pause)
		{
			if (mKeyboard->isKeyDown(OIS::KC_V))
			{
				m_Pause = false;
			} 
			return true;
		}

		if (mKeyboard->isKeyDown(OIS::KC_SPACE))
		{
			m_Pause = true;
		} 
		
		// --------------------------------------------------
		// --------------------------------------------------

		if (!m_Initialized)
		{
			sampleSceneAndInitialize();			
		}

		manageAvoidBase();

		m_CommonRes.getDeviceInterface()->setDevicePrintfEnabled( false );
		
		// -------------------------------
		// DEBUG
		// Device printf init
		// Uncomment for device printf (Together with device printf finalize)
		// Warning: Device printf is very slow on CC 1.0
		// 
		// m_CommonRes.getDeviceInterface()->devicePrintfInit();
		// -------------------------------

		// -----------------------------------------
		// -----------------------------------------
		// Crowd behavior update

		m_CommonRes.getDeviceInterface()->mapVBOinDeviceDataRepository();

		managePeopleThrowing();

		m_SimEngine->reorderSimData();

		m_SimEngine->computeNeighborhoods();

		m_SimEngine->resetSteerForce();
		m_SimEngine->steerAndSlowToAvoidNeighbors();
		m_SimEngine->computeAlignments();
		m_SimEngine->computeCohesions();

		// Uncomment for camera avoiding
		m_SimEngine->steerForMoveAwayBaseTarget();	

		//for (ObstacleListFloat3::iterator obstaclesIterator = mObstaclesPositions.begin(); 
		//	obstaclesIterator != mObstaclesPositions.end(); 
		//	obstaclesIterator ++)
		//{
		//	float3 obstaclePosition = *obstaclesIterator;
		//	hOpenSteerWrapperParams.avoidBase = obstaclePosition;
		//	m_SimEngine->steerForMoveAwayBaseTarget();			
		//}

		m_SimEngine->steerToFollowTerrain();
		m_SimEngine->computeSeekingsWorldCenter();
		m_SimEngine->computeFloatingBehavior();

		m_SimEngine->manageSourceDestination();
		
		//m_SimEngine->smoothColorD(); // DISABLED
		m_SimEngine->applySteeringForces(evt.timeSinceLastFrame * 3.0f );

		m_CommonRes.getDeviceInterface()->unmapVBOinDeviceDataRepository();

		// -------------------------------
		// DEBUG
		// Device printf finalize
		// Uncomment for device printf (Together with device printf init)
		// Warning: Device printf is very slow on CC 1.0
		// 
		// m_CommonRes.getDeviceInterface()->devicePrintfDisplay();
		// m_CommonRes.getDeviceInterface()->devicePrintfEnd();
		// -------------------------------

		// -----------------------------------------
		// -----------------------------------------
		// Overlays update

		OverlayElement* currOverlay;

		// Update the right logger
		//////////////////////////////////////////////////////////
		if ( m_CommonRes.getLogger()->isModified())
		{
			currOverlay = 
				OverlayManager::getSingleton().getOverlayElement("Core/LogPanel/Messages");
			
			currOverlay->setCaption( m_CommonRes.getLogger()->getMsgListString( "\n" ) );		
		}
		
		
        return true;
		
    }

	

};



class TerrainApplication : public ExampleApplication
{
	

public:
    TerrainApplication() {}

    ~TerrainApplication()
    {
        delete raySceneQuery;
		delete m_SimEngine;

		if (mRenderSystemCommandsRenderQueueListener)
		{
			mSceneMgr->removeRenderQueueListener(mRenderSystemCommandsRenderQueueListener);
			delete mRenderSystemCommandsRenderQueueListener;
			mRenderSystemCommandsRenderQueueListener = NULL;
		}

    }

protected:


    virtual void chooseSceneManager(void)
    {
        // Get the SceneManager, in this case a generic one
        mSceneMgr = mRoot->createSceneManager("TerrainSceneManager");
    }

    virtual void createCamera(void)
    {
        // Create the camera
        mCamera = mSceneMgr->createCamera("PlayerCam");

        // Position it at 500 in Z direction
		mCamera->setPosition(Vector3::ZERO);
        // Look back along -Z
        mCamera->setNearClipDistance( 1 );
        mCamera->setFarClipDistance( 1000 );

    }
   
    // Just override the mandatory create scene method
    void createScene(void)
    {
		new Ogre::Exception(1, "Test", "Init");

        Plane waterPlane;

        // Set ambient light
        mSceneMgr->setAmbientLight(ColourValue(0.5, 0.5, 0.5));

		mSceneMgr->setSkyBox(true, "Examples/CloudyNoonSkyBox", 1000);

        // Create a light
        Light* l = mSceneMgr->createLight("MainLight");
        // Accept default settings: point light, white diffuse, just set position
        // NB I could attach the light to a SceneNode if I wanted it to move automatically with
        //  other objects, but I don't
        l->setPosition(20,80,50);

        // Fog
        // NB it's VERY important to set this before calling setWorldGeometry 
        // because the vertex program picked will be different
        ColourValue fadeColour(0.70, 0.70, 0.85);
        mWindow->getViewport(0)->setBackgroundColour(fadeColour);

        std::string terrain_cfg("terrain.cfg");
        mSceneMgr -> setWorldGeometry( terrain_cfg );
		
        // Infinite far plane?
        if (mRoot->getRenderSystem()->getCapabilities()->hasCapability(RSC_INFINITE_FAR_PLANE))
        {
            mCamera->setFarClipDistance(0);
        }

        // Define the required skyplane
        Plane plane;
        // 5000 world units from the camera
        plane.d = 5000;
        // Above the camera, facing down
        plane.normal = -Vector3::UNIT_Y;

        // Set a nice viewpoint
        mCamera->setPosition(1200, 200, 20);
        //mCamera->setOrientation(Quaternion(-0.3486, 0.0122, 0.9365, 0.0329));
		mCamera->lookAt(
			Vector3(1200, 0, 1200));
		//mRoot->showDebugOverlay( true );

        raySceneQuery = mSceneMgr->createRayQuery(Ray()/*, Ogre::SceneManager::WORLD_GEOMETRY_TYPE_MASK*/);
		
		
		// -----------------------------------------
		// -----------------------------------------
		// Behavior engine initialization

		

		

		// -----------------------------------------
		// -----------------------------------------
		// Rendering

		ManualObject *manObj; // we will use this Manual Object as a reference point for the native rendering
        manObj = mSceneMgr->createManualObject("sampleArea");
		
        // Attach to child of root node, better for culling (otherwise bounds are the combination of the 2)
        mSceneMgr->getRootSceneNode()->createChildSceneNode()->attachObject(manObj);

		Ogre::String RenderSystemName = mSceneMgr->getDestinationRenderSystem()->getName();
		mRenderSystemCommandsRenderQueueListener = NULL;
		BehaveRTNativeRenderSystemCommandsRenderQueueListener<SimEngineFinal>*
			behavertRenderer = NULL;

		if ("OpenGL Rendering Subsystem" == RenderSystemName)
		{

			behavertRenderer = new 
				BehaveRTNativeRenderSystemCommandsRenderQueueListener<SimEngineFinal>(
					manObj, mCamera, mSceneMgr);
			mRenderSystemCommandsRenderQueueListener = behavertRenderer;
			mSceneMgr->addRenderQueueListener(mRenderSystemCommandsRenderQueueListener);
		}

		m_Pause = false;

		m_CommonRes.initialize("Config\\BuildingCF.cfg");

		// Create the custom CrowdEngine reference
		m_SimEngine = new SimEngineFinal();
		behavertRenderer->setBehaviorEngine(m_SimEngine);

		// Reset all plugIns
		m_SimEngine->reset();
		
		// -----------------------------------------
		// -----------------------------------------
		// Obstacles SceneNodes

		Entity* obstacleEntity = mSceneMgr->createEntity("ObstacleEntity", "sphere.mesh");

		const int numObstacles = 15; 
		for (int obstaleIndex = 0; obstaleIndex < numObstacles; obstaleIndex ++)
		{
			//SceneNode
			//mObstacles.push_back(
			m_CommonRes.getLogger()->log("Demo", "Added obstacle: " +   
				Ogre::StringConverter::toString(obstaleIndex) +
				Ogre::StringConverter::toString((float) (rand() - 0.5) * 2));

			float randX = hEnvGrid3DParams.worldCenter.x + 
				((float)rand()/RAND_MAX - 0.5) * hEnvGrid3DParams.worldRadius.x * 0.8;
			float randZ = hEnvGrid3DParams.worldCenter.z + 
				((float)rand()/RAND_MAX - 0.5) * hEnvGrid3DParams.worldRadius.z * 0.8;
			
			SceneNode* obstacleNode = mSceneMgr->
				getRootSceneNode()->createChildSceneNode(
					Vector3(randX, 40, randZ));

			obstacleNode->attachObject( obstacleEntity->clone(
				"ObstacleEntity" + Ogre::StringConverter::toString(obstaleIndex)) );
			obstacleNode->setScale(Vector3(1, 1, 1) * 0.2);
			mObstacles.push_back(obstacleNode);

			mObstaclesPositions.push_back(
				make_float3(
					obstacleNode->getPosition().x, 
					obstacleNode->getPosition().y, 
					obstacleNode->getPosition().z));
		}

		// --------------------------------------------------
		// --------------------------------------------------
		// Source Destination SceneNodes

		Entity* sourceDestEntity = mSceneMgr->createEntity("SourceDestEntity", "Cube.mesh");

		SceneNode* sourceDestNode = mSceneMgr->
			getRootSceneNode()->createChildSceneNode(
				Vector3(
					hBuilding3DParams.individualsSourcePos.x, 
					hBuilding3DParams.individualsSourcePos.y, 
					hBuilding3DParams.individualsSourcePos.z));

		sourceDestNode->attachObject( sourceDestEntity->clone("SourceEntity") );
		sourceDestNode->setScale(Vector3(1, 1, 1));

		sourceDestNode = mSceneMgr->
			getRootSceneNode()->createChildSceneNode(
				Vector3(
					hBuilding3DParams.individualsDestPos.x, 
					hBuilding3DParams.individualsDestPos.y, 
					hBuilding3DParams.individualsDestPos.z));

		sourceDestNode->attachObject( sourceDestEntity->clone("DestEntity") );
		sourceDestNode->setScale(Vector3(1, 1, 1) * 
			hBuilding3DParams.individualsDestSize * 0.02);


		// --------------------------------------------------
		// --------------------------------------------------

		Entity* targetEnt = mSceneMgr->createEntity("testray", "sphere.mesh");
        MaterialPtr mat = MaterialManager::getSingleton().create("targeter", 
            ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
        Pass* pass = mat->getTechnique(0)->getPass(0);
        TextureUnitState* tex = pass->createTextureUnitState();
        tex->setColourOperationEx(LBX_SOURCE1, LBS_MANUAL, LBS_CURRENT, 
            ColourValue::Red);
        pass->setLightingEnabled(false);
        pass->setSceneBlending(SBT_ADD);
        pass->setDepthWriteEnabled(false);


        targetEnt->setMaterialName("targeter");
        targetEnt->setCastShadows(false);
        targetEnt->setQueryFlags(0);
        mTargetNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
        mTargetNode->scale(0.05, 0.05, 0.05);
        mTargetNode->attachObject(targetEnt);

		mTargetNode->setVisible( false );


		
		


    }
    // Create new frame listener
    void createFrameListener(void)
    {
        mFrameListener= new TerrainFrameListener(mWindow, mCamera);
        mRoot->addFrameListener(mFrameListener);
    }


	

};
