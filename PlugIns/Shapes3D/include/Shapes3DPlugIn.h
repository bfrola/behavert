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
// 01-09 bf: Created
//
// ----------------

#pragma once

#include "BehaveRT.h"
#include "Shapes3D_kernel.cuh"
#include "Drawable3D\include\texture.h"
#include "Drawable3D\include\3dsloader.h"
#include "nvVector.h"

#include <GL/glew.h>

#include "math.h"

// ----------------

namespace Shapes3D
{

	#define MAX_OBJECTS 100

	typedef struct ObjectInfoType
	{
		int type;
		std::string path;
		float3 size;
		float3 color;
	};

	/**
		\brief This plugIn allow to create 2D images and 3D shapes using the simulated bodie
	*/
	template <class Super>
	class Shapes3DPlugIn: public Super, public SimEnginePlugIn
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor

		/// XXX install/unistall plugin shoulde be automatic
		Shapes3DPlugIn() { SimEnginePlugIn::installPlugIn(); }
		~Shapes3DPlugIn() { SimEnginePlugIn::uninstallPlugIn(); }
		
		const std::string name() { return "Shapes3DPlugIn"; }	

		const DependenciesList plugInDependencies() 
		{ 
			DependenciesList dependencies;
			dependencies.push_back("Body3DPlugIn");
			dependencies.push_back("EnvGrid3DPlugIn");
			dependencies.push_back("Proximity3DPlugIn");
			dependencies.push_back("OpenSteerWrapperPlugIn");
			return dependencies;	
		}

		// ////////////////////////////////////////////////////
		// Methods
	private:
		/// @override
		void install();
		
		/// @override
		void uninstall();

	public:
		/// @override
		void reset();

		/// @override
		void update(const float elapsedTime);

		// Custom operations
	public:

		void moveTowardsTarget();
		void setupShape();
		void setupSpecifiedShape(int objectIndex);
		void setupNextShape();
		void setTargetPoint(int index, nv::vec3<float> targetPoint);

		void useShiftedIndexOnOff() { m_UseShiftedIndex = !m_UseShiftedIndex; }
		bool getUseShiftedIndex() { return m_UseShiftedIndex; }
		ObjectInfoType getCurrentObjectInfoType() { return m_ObjectInfo[m_CurrentObject]; }
		

		obj_type m_Object_3ds;

		// ////////////////////////////////////////////////////
		// Fields	
	protected:
		BehaveRT::DeviceArrayWrapper<float4>* m_FinalTarget;

		int m_CurrentObject;
		
		


		
		ObjectInfoType m_ObjectInfo[MAX_OBJECTS];


		int m_FrameCounter;
		int m_NumObjects;
		
		bool m_UseShiftedIndex;

		int m_IndexShiftDelta;
	};
}

using namespace Shapes3D;

// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
// Implementation
template <class Super>
void Shapes3DPlugIn<Super>::install()
{
	
	// Params

	hShapes3DParams.targetBase = make_float3(0, 0, 0);

	m_IndexShiftDelta = BehaveRT::StringConverter::parseInt(
		m_CommonRes.getConfig()->getSetting("shiftIndexDelta", Shapes3DPlugIn::name()));

	m_NumObjects = BehaveRT::StringConverter::parseInt(
		m_CommonRes.getConfig()->getSetting("numObjects", Shapes3DPlugIn::name()));

	for (int i = 0; i < m_NumObjects; i ++)
	{
		std::stringstream keyName;
		keyName << "objectType_" << i;
		m_ObjectInfo[i].type = BehaveRT::StringConverter::parseInt(
			m_CommonRes.getConfig()->getSetting(keyName.str(), Shapes3DPlugIn::name()));

		keyName.str("");
		keyName << "objectPath_" << i;
		m_ObjectInfo[i].path = m_CommonRes.getConfig()->getSetting(keyName.str(), Shapes3DPlugIn::name());

		keyName.str("");
		keyName << "objectSize_" << i;
		m_ObjectInfo[i].size = BehaveRT::StringConverter::parseFloat3(
			m_CommonRes.getConfig()->getSetting(keyName.str(), Shapes3DPlugIn::name()));

		keyName.str("");
		keyName << "objectColor_" << i;
		m_ObjectInfo[i].color = BehaveRT::StringConverter::parseFloat3(
			m_CommonRes.getConfig()->getSetting(keyName.str(), Shapes3DPlugIn::name()));

	}



	// Fields
	
	m_FinalTarget = new DeviceArrayWrapper<float4>(
		m_CommonRes.getDeviceInterface(), 
		hBody3DParams.numBodies);

	m_FinalTarget->bindToField(hShapes3DFields.finalTarget);
	//EnvGrid3DPlugIn::addToFeaturesToReorder(m_FinalTarget, m_FinalTarget->getType());
	
	//m_Object_3ds.id_texture = 
	//	LoadBitmapTexture("data\\spaceshiptexture.bmp");
	
	m_UseShiftedIndex = false;

	Shapes3D::Shapes3D_copyFieldsToDevice();
}

// --------------------------------------------------------------

template <class Super>
void Shapes3DPlugIn<Super>::uninstall()
{
	// deletes
	delete m_FinalTarget;
}

// --------------------------------------------------------------

template <class Super>
void Shapes3DPlugIn<Super>::setupSpecifiedShape(int objectIndex)
{
	m_CurrentObject = objectIndex;
	setupShape();
}

// --------------------------------------------------------------

template <class Super>
void Shapes3DPlugIn<Super>::setupNextShape()
{
	m_CurrentObject++;

	if (m_CurrentObject >= m_NumObjects)
		m_CurrentObject = 0;

	setupShape();
}

template <class Super>
void Shapes3DPlugIn<Super>::
setTargetPoint(int index, nv::vec3<float> targetPoint)
{
	if (index >= hBody3DParams.numBodies)
		return;

	float3 worldCenter = EnvGrid3DPlugIn::getWorldCenter();
	float4 target = make_float4(
		worldCenter.x + targetPoint.x, 
		worldCenter.y + targetPoint.z, 
		worldCenter.z + targetPoint.y,
		0.4 + frand() / 3);
	m_FinalTarget->setHostArrayElement(index, &target);

	float initVal = 1 - (float) index / hBody3DParams.numBodies;
		float4 initColor = make_float4(
			(int) (0.5 + initVal + hDrawable3DParams.colorBase.x), 
			(int) (0.5 + initVal + hDrawable3DParams.colorBase.y), 
			(int) (0.5 + initVal + hDrawable3DParams.colorBase.z), 1);

	m_Color->setHostArrayElement(index, &initColor);
}

template <class Super>
void Shapes3DPlugIn<Super>::
setupShape()
{
	hShapes3DParams.indexShift = 0;

	if (m_ObjectInfo[m_CurrentObject].type > 0)
	{
		// --------------------------------------------
		// Picture flocking initialization

		float3 worldCenter = EnvGrid3DPlugIn::getWorldCenter();
		
		uint2 squareSize = make_uint2(sqrt((double)hBody3DParams.numBodies), sqrt((double)hBody3DParams.numBodies));
		
		int bodyIndex = 0;
				
		for (BehaveRT::uint i = 0; i < squareSize.x; i ++ )
		{
			for (BehaveRT::uint j = 0; j < squareSize.y; j ++ )
			{
				float4 taget;
			
				taget.x = j	 * hEnvGrid3DParams.worldRadius.x * 2 / 
					squareSize.x - hEnvGrid3DParams.worldRadius.x + worldCenter.x;
				
				taget.y = i * hEnvGrid3DParams.worldRadius.y * 2 / 
					squareSize.y - hEnvGrid3DParams.worldRadius.y + worldCenter.y;

				taget.z = //j * hEnvGrid3DParams.worldRadius.z  / 
					//squareSize.x - hEnvGrid3DParams.worldRadius.z / 2 + 
						worldCenter.z;
					// - i * hEnvGrid3DParams.worldRadius * 0.5 / 
					//squareSize.y - hEnvGrid3DParams.worldRadius; 
					//cos((float) i / 10 ) * 20;//i * hEnvGrid3DParams.worldRadius * 2 / 
					//squareSize.x - hEnvGrid3DParams.worldRadius;

				taget.w =  0.7;

				m_FinalTarget->setHostArrayElement(bodyIndex++, &taget);
			} // for
		} // for
		

		for (int i = bodyIndex; i < hBody3DParams.numBodies; i ++)
		{
			float4 taget;
			
			taget.x = - hEnvGrid3DParams.worldRadius.x;
				
			taget.y = - hEnvGrid3DParams.worldRadius.y;

			taget.z = - hEnvGrid3DParams.worldRadius.z / 2;

			m_FinalTarget->setHostArrayElement(bodyIndex++, &taget);			
		}

		m_FinalTarget->copyArrayToDevice();
		m_FinalTarget->swapPosReadWrite();
		m_FinalTarget->copyArrayToDevice();
		m_FinalTarget->swapPosReadWrite();


		// Load the image
		unsigned char* img = LoadBitmap(m_ObjectInfo[m_CurrentObject].path.c_str());

		// Fixme, retrieve from image file
		int imgIndex = 0;
		uint2 imageSize = make_uint2(256, 256);

		float2 interpRatio = make_float2(
			(float) imageSize.x / squareSize.x, 
			(float) imageSize.y / squareSize.y);

		//printf("%f %f\n", interpRatio.x, interpRatio.y);
		
		for (BehaveRT::uint i = 0; i < imageSize.y; i ++ )
		{
			for (BehaveRT::uint j = 0; j < imageSize.x; j ++ )
			{
				imgIndex = j * 4 + imageSize.y * 4 * i;

				float4 initColor = make_float4(
					(float) img[imgIndex] / 255, 
					(float) img[imgIndex + 1] / 255, 
					(float) img[imgIndex + 2] / 255, 
					1);

				if (j + squareSize.y * i >= hBody3DParams.numBodies)
					break;

				m_Color->setHostArrayElement(j + squareSize.y * i, &initColor);
			}
		}


		m_Color->copyArrayToDevice();
		m_Color->swapPosReadWrite();

		return;

	}

	//printf("Shape[%d] '%s'\n", m_CurrentObject, m_ObjectInfo[m_CurrentObject].path.c_str());

	

	// Load the model
	
	Load3DS (&m_Object_3ds, m_ObjectInfo[m_CurrentObject].path.c_str());

	nv::vec3<float> vertex[3];

	nv::vec3<float> meanVertex = nv::vec3<float>(m_Object_3ds.vertex[0].x, 
			m_Object_3ds.vertex[0].y, m_Object_3ds.vertex[0].z);

	float meanDist = 0;
	for(int i = 1; i < m_Object_3ds.vertices_qty; i ++)
	{
		nv::vec3<float> currVertex = nv::vec3<float>(
			m_Object_3ds.vertex[i].x, 
			m_Object_3ds.vertex[i].y,
			m_Object_3ds.vertex[i].z);

		meanVertex += currVertex;
	}

	// The center of the shape
	meanVertex /= m_Object_3ds.vertices_qty;

	for(int i = 1; i < m_Object_3ds.vertices_qty; i ++)
	{
		// Polarize vertexes
		m_Object_3ds.vertex[i].x -= meanVertex.x; 
		m_Object_3ds.vertex[i].y -= meanVertex.y; 
		m_Object_3ds.vertex[i].z -= meanVertex.z; 

		nv::vec3<float> currVertex = nv::vec3<float>(
			m_Object_3ds.vertex[i].x, 
			m_Object_3ds.vertex[i].y,
			m_Object_3ds.vertex[i].z);

		meanDist += nv::length(meanVertex - currVertex);
	}

	// Calculate the avg distance from the center
	meanDist /= m_Object_3ds.vertices_qty;


	int interpCountPerPoly = 0.5 * ((float)hBody3DParams.numBodies / (m_Object_3ds.polygons_qty));

	/*if (interpCountPerPoly < 100)
		interpCountPerPoly -= 2;
	else
		interpCountPerPoly -= 50;*/

	printf("Shapes3D>> Iterpolation: %d\n", interpCountPerPoly);

	printf("Shapes3D>> Vertexes: %d\n", m_Object_3ds.vertices_qty);
	

	float3 multipler = 
		make_float3(
			1/meanDist + m_ObjectInfo[m_CurrentObject].size.x, 
			1/meanDist + m_ObjectInfo[m_CurrentObject].size.y, 
			1/meanDist + m_ObjectInfo[m_CurrentObject].size.z);

	printf("Shapes3D>> Object size: %f %f %f\n", 
		m_ObjectInfo[m_CurrentObject].size.x,
		m_ObjectInfo[m_CurrentObject].size.y,
		m_ObjectInfo[m_CurrentObject].size.z);

	int bodyIndex = 0;
	int repeats = 0;
	while(bodyIndex < hBody3DParams.numBodies)
	{
		for (int l_index=0;l_index<m_Object_3ds.polygons_qty; l_index++)
		{				
			//printf("%d\n", l_index);
			nv::vec3<float> vertex[3];
			vertex[0] = nv::vec3<float>(
				m_Object_3ds.vertex[ m_Object_3ds.polygon[l_index].a ].x * hEnvGrid3DParams.worldRadius.x * multipler.x,
				m_Object_3ds.vertex[ m_Object_3ds.polygon[l_index].a ].y * hEnvGrid3DParams.worldRadius.y * multipler.y,
				m_Object_3ds.vertex[ m_Object_3ds.polygon[l_index].a ].z * hEnvGrid3DParams.worldRadius.z * multipler.z);

			vertex[1] = nv::vec3<float>(
				m_Object_3ds.vertex[ m_Object_3ds.polygon[l_index].b ].x * hEnvGrid3DParams.worldRadius.x * multipler.x,
				m_Object_3ds.vertex[ m_Object_3ds.polygon[l_index].b ].y * hEnvGrid3DParams.worldRadius.y * multipler.y,
				m_Object_3ds.vertex[ m_Object_3ds.polygon[l_index].b ].z * hEnvGrid3DParams.worldRadius.z * multipler.z);

			vertex[2] = nv::vec3<float>(
				m_Object_3ds.vertex[ m_Object_3ds.polygon[l_index].c ].x * hEnvGrid3DParams.worldRadius.x * multipler.x,
				m_Object_3ds.vertex[ m_Object_3ds.polygon[l_index].c ].y * hEnvGrid3DParams.worldRadius.y * multipler.y,
				m_Object_3ds.vertex[ m_Object_3ds.polygon[l_index].c ].z * hEnvGrid3DParams.worldRadius.z * multipler.z);

			if (repeats == 0)
			{
				setTargetPoint(bodyIndex++, vertex[0]);
				setTargetPoint(bodyIndex++, vertex[1]);
				setTargetPoint(bodyIndex++, vertex[2]);
			}

			//printf("BI: %d LI: %d INT: %d DIFF: %d\n", bodyIndex, l_index, interpCountPerPoly,
			//	hBody3DParams.numBodies - m_Object_3ds.polygons_qty);
			
			float dist1 = length(vertex[1] - vertex[2]);
			float dist2 = length(vertex[0] - vertex[1]);


			float interpThreshold = hBody3DParams.commonRadius * 4;

			if (l_index < hBody3DParams.numBodies - m_Object_3ds.polygons_qty 
					&& dist1 > interpThreshold && dist2 > interpThreshold
					&& dist1 < interpThreshold * 25 && dist2 < interpThreshold * 25)
			{

				int numInterps = 0;
				int numInterpolations = sqrt((float)interpCountPerPoly);
				for (int i = 0; i < numInterpolations; i ++)
				{
					for (int j = 0; j < numInterpolations; j ++)
					{
						if (numInterps > interpCountPerPoly)
							break;

						nv::vec3<float> interpVertex1 = 
							BehaveRT::interpolate(frand() * 0.99 + 0.005, vertex[0], vertex[1]);

						nv::vec3<float> interpVertex2 = 
							BehaveRT::interpolate(frand() * 0.99 + 0.005, vertex[2], interpVertex1);

						if (length(interpVertex2 - vertex[0]) < dist2)
							setTargetPoint(bodyIndex++, interpVertex2);
							
						numInterps ++;					
					}
				}

				//interpCountPerPoly = oldInterpCountPerPoly;

			} // if			
				
			

		}// for

		repeats ++;
		
	}// while

	
	for (BehaveRT::uint i = 0; i < hBody3DParams.numBodies; i ++ )
	{
		float initVal = 1 - (float) i / hBody3DParams.numBodies;
		float4 initColor = make_float4(
			initVal + m_ObjectInfo[m_CurrentObject].color.x, 
			initVal + m_ObjectInfo[m_CurrentObject].color.y, 
			initVal +m_ObjectInfo[m_CurrentObject].color.z, 1);

		m_Color->setHostArrayElement(i, &initColor);
	}

	m_Color->copyArrayToDevice();
	m_Color->swapPosReadWrite();

	m_FinalTarget->copyArrayToDevice();
	m_FinalTarget->swapPosReadWrite();
	m_FinalTarget->copyArrayToDevice();
	m_FinalTarget->swapPosReadWrite();
}

template <class Super>
void Shapes3DPlugIn<Super>::reset()
{
	Super::reset(); // MANDATORY OPERATION
	
	m_CurrentObject = -1;
	setupNextShape();
}

// --------------------------------------------------------------

template <class Super>
void Shapes3DPlugIn<Super>::update(const float elapsedTime)
{
	Super::update(elapsedTime); // MANDATORY OPERATION

	// Insert here the default update operation
}

// --------------------------------------------------------------
// --------------------------------------------------------------

// Custom methods
template <class Super>
void Shapes3DPlugIn<Super>::moveTowardsTarget()
{
	if (m_CurrentObject < 0)
		return;


	if (m_UseShiftedIndex)
	{
		hShapes3DParams.indexShift+=m_IndexShiftDelta;

		if (hShapes3DParams.indexShift >= hBody3DParams.numBodies)
			hShapes3DParams.indexShift = 0;
		
		Shapes3D::Shapes3D_copyFieldsToDevice();
	}

	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		moveTowardsTargetDRef(), 
		&Shapes3D_beforeKernelCall, 
		&Shapes3D_afterKernelCall);

	m_SteerForce->swapPosReadWrite();
	//m_Pos->swapPosReadWrite();
	m_FinalTarget->swapPosReadWrite();

	


}



// --------------------------------------------------------------

