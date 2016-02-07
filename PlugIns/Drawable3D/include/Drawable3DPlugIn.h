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
#include "drawable3d_kernel.cuh"
#include "shaders.h"
#include "math.h"

#include "texture.h"
#include "3dsloader.h"

#include <GL/glew.h>
#include <GL/glut.h>

#include "nvVector.h"

#include "paramgl.h"

//#include "cudpp.h"





// ----------------
// Change log

// Bernardino Frola
// 30-03-10: testing geometry instancing
// 30-03-10: geometry instancing global vars
// 28-04-10: wireframe support
// 02-06-10: pixel and vertex shader generalization
// 08-06-10: highLight positions

// ----------------

#define M_PI 3.14f

namespace Drawable3D
{
	/**
		\brief Simulation rendering using OpenGL VBOs
	*/
	template <class Super>
	class Drawable3DPlugIn: public Super, public SimEnginePlugIn
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor

		/// XXX install/unistall plugin shoulde be automatic
		Drawable3DPlugIn() { SimEnginePlugIn::installPlugIn(); }
		~Drawable3DPlugIn() { SimEnginePlugIn::uninstallPlugIn(); }
		
		const std::string name() { return "Drawable3DPlugIn"; }	

		const DependenciesList plugInDependencies() 
		{ 
			DependenciesList dependencies;
			dependencies.push_back("Body3DPlugIn");
			dependencies.push_back("EnvGrid3DPlugIn");
			dependencies.push_back("Proximity3DPlugIn");
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

		void createGeometry();
		void smoothColorD();
	public:

		// -----------------------------------
		// Operations
		void drawBodiesAsPointSprites();

		// Compute, for each individual the dinstance to the
		// camera position-->mouse direction line
		void computeMouseDistance();

		// DEBUG
		DeviceArrayWrapper<float>* getMouseDistance() { 
			return m_MouseDistance; }

		// -----------------------------------
		// Other

		void setWindowH(int windowH) { m_window_h = windowH; }
		void setFov(int fow) { m_fov = fow; }

		void setUseWireframe(bool useWireframe) { m_UseWireframe = useWireframe; }

		// PS and VS for billboards rendering
		void setBillboardsShaders(const char *vertexShader, const char *pixelShader) 
			{ m_BillboardsVertexShader = (char*) vertexShader; m_BillboardsPixelShader = (char*) pixelShader; }

		// PS and VS for geometry instancing rendering
		void setGIShaders(const char *vertexShader, const char *pixelShader) 
			{ m_GIVertexShader = (char*) vertexShader; m_GIPixelShader = (char*) pixelShader; }

		ParamListGL* getParamListGL() { return m_ParamListGL; }

		//void renderBillboards(GLuint positionVbo);
		void renderBillboards(GLuint positionsVbo, 
			int method = GL_POINTS,
			bool useProgram = true, 
			int numBodies = hBody3DParams.numBodies);

		DeviceArrayWrapper<float4>* getColor() { 
			return (m_UseSmoothedColor ? m_SmoothedColor : m_Color); }

		GLuint getColorReadVBO() { 
			return (m_UseSmoothedColor ? m_SmoothedColor->getReadVbo() : m_Color->getReadVbo()); }

		void changeDisplayParamListGL() {
			m_DisplayParamListGL = !m_DisplayParamListGL; }

		// Internal functions

		GLuint _compileProgram(const char *vsource, const char *fsource);

		// ////////////////////////////////////////////////////
		// Fields	
	protected:
		DeviceArrayWrapper<float4>* m_Color;
		DeviceArrayWrapper<float4>* m_SmoothedColor;
		DeviceArrayWrapper<float4>* m_Geometry; // CUDA Geometry - not efficient

		DeviceArrayWrapper<float4>* m_HighLightPositions;

		DeviceArrayWrapper<float>* m_MouseDistance;

		int m_window_h;
		float m_fov;
		GLuint m_program;
		GLuint m_InstancingProgram;

		int m_FloorTexIndex;
		int m_CeilTexIndex;
		int m_WreckTexIndex;
		float m_PointSpriteSizeMultipler;

		bool m_UseSmoothedColor;
		
		// Billboard texture
		GLuint g_textureID;
		GLuint m_VertexVBO;
		GLuint m_NormalVBO;

		GLint _positionTexUnit;          // a parameter to the fragment program
		GLint _forwardTexUnit;
		GLint _colorTexUnit;

		std::string m_MeshPath;

		GLint m_TimeUniform;
		float m_Time;

		obj_type m_MeshData;
		int m_VertexCount;

		int frameCounter;

		GLuint _TextureBufferInstances[2];	

		// -------------------
		// 30-03-10
		// Geometry instancing vertexes and normals
		GLfloat* vertexData;
		GLfloat* normalData;
		GLsizeiptr positionSize;

		// --------------------
		// 30-03-10
		bool m_UseWireframe;

		// --------------------
		// 02-06-10
		char *m_BillboardsVertexShader;
		char *m_BillboardsPixelShader;

		char *m_GIVertexShader;
		char *m_GIPixelShader;

		// --------------------
		// 03-06-10

		// m_ParamListGL: instantiated in install method.
		// Elements have to be added in reset method(s)
		
		bool m_DisplayParamListGL;

		float paramTestVal; // TEMP

		// ----------------------
		// 08-06-10

		bool m_HighLightPositionEnabled;

		

		
	};

	// FIX ME: set as class field
	// Global var
	ParamListGL* m_ParamListGL;

	void additionalMotionFunction(int x, int y)
	{
		m_ParamListGL->Motion(x, y);
		glutPostRedisplay();
	}

	
	void additionalMouseFunction(int x, int y, int button, int state)
	{
		m_ParamListGL->Mouse(x, y, button, state);
		glutPostRedisplay();
	}

	void additionalSpecialFunction(int k, int x, int y)
	{
		m_ParamListGL->Special(k, x, y);
		glutPostRedisplay();
	}
	
}



using namespace Drawable3D;

// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
// Implementation
template <class Super>
void Drawable3DPlugIn<Super>::install()
{
	if (!m_Pos->useVbo())
	{
		printf("Drawable3D: required pos on VBO");
		exit(-1);
	}

	// -----------------------------
	// Params

	read_config_param(Drawable3D, colorBase, Float3);
	read_config_param(Drawable3D, disableDrawing, Bool);
	read_config_param(Drawable3D, renderingType, Int);
	read_config_param(Drawable3D, neighborhoodColoring, Bool);
	read_config_param(Drawable3D, useCUDAGeometry, Bool);	// DEPRECATED
	read_config_param(Drawable3D, numVertexes, Int);		// DEPRECATED

	m_MeshPath = m_CommonRes.getConfig()->getSetting(
		"meshPath", "Drawable3DPlugIn");
	
	m_PointSpriteSizeMultipler = BehaveRT::StringConverter::parseFloat(
		m_CommonRes.getConfig()->getSetting("pointSpriteSizeMultipler", Drawable3DPlugIn::name()));

	m_UseSmoothedColor = BehaveRT::StringConverter::parseBool(
		m_CommonRes.getConfig()->getSetting("useSmoothedColor", Drawable3DPlugIn::name()));

	m_UseWireframe = BehaveRT::StringConverter::parseBool(
		m_CommonRes.getConfig()->getSetting("useWireframe", Drawable3DPlugIn::name()));
	
	m_DisplayParamListGL = BehaveRT::StringConverter::parseBool(
		m_CommonRes.getConfig()->getSetting("displayParamList", Drawable3DPlugIn::name()));

	// -----------------------------
	// Fields
	
	m_Color = new DeviceArrayWrapper<float4>(
		m_CommonRes.getDeviceInterface(), 
		hBody3DParams.numBodies, 
		hDrawable3DParams.numVertexes, 
		true, true); // XXX if smoothed color there is no need of VBO mapping

	m_Color->bindToField(hDrawable3DFields.color);

	if (m_UseSmoothedColor)
	{
		m_SmoothedColor = new DeviceArrayWrapper<float4>(
			m_CommonRes.getDeviceInterface(), 
			hBody3DParams.numBodies, 
			hDrawable3DParams.numVertexes, 
			true, true); 

		m_SmoothedColor->bindToField(hDrawable3DFields.smoothedColor);
	}

	m_HighLightPositionEnabled = true; // XXX must be a parameter

	if (m_HighLightPositionEnabled)
	{
		m_HighLightPositions = new DeviceArrayWrapper<float4>(
			m_CommonRes.getDeviceInterface(), 
			hBody3DParams.numBodies, 1, 
			true, true); 

		m_HighLightPositions->bindToField(hDrawable3DFields.highLightPositions);
	}

	m_MouseDistance = new DeviceArrayWrapper<float>(
		m_CommonRes.getDeviceInterface(), 
		hBody3DParams.numBodies);

	m_MouseDistance->bindToField(hDrawable3DFields.mouseDistance);

	// -------------------------
	
	m_window_h = m_fov = -1;
	
	Drawable3D::Drawable3D_copyFieldsToDevice();

	// Frame counter for generic purposes
	frameCounter = 0;

	// Default rendering shaders
	setBillboardsShaders(
		defaultBillboardsVertexShader,
		defaultBillboardsPixelShader);

	setGIShaders(
		defaultGIVertexShader,
		defaultGIPixelShader);

	// --------------------------

	m_ParamListGL = new ParamListGL("BehaveRT_paramList");
}

// --------------------------------------------------------------

template <class Super>
void Drawable3DPlugIn<Super>::uninstall()
{
	// deletes
	delete m_Color;
	delete m_SmoothedColor;
}

// --------------------------------------------------------------

nv::vec3<float> make_vec3(vertex_type vertex, float3 scale)
{
	return nv::vec3<float> (
		vertex.x * scale.x, 
		vertex.y * scale.y, 
		vertex.z * scale.z);
}

template <class Super>
void Drawable3DPlugIn<Super>::reset()
{
	Super::reset(); // MANDATORY OPERATION

	// REMOVE FROM HERE: add rendering init and update stages
	//if (hDrawable3DParams.renderingType == 2)
	//{
	//	m_FloorTexIndex = LoadBitmapTexture("data\\floor_tex.bmp");
	//	//m_FloorTexIndex = LoadBitmapTexture("data\\sand.bmp");
	//	m_CeilTexIndex = LoadBitmapTexture("data\\seaLevel.bmp");
	//	m_WreckTexIndex = LoadBitmapTexture("data\\wreck.bmp");
	//}

	// Enable GLSL program and colors
	if (hDrawable3DParams.renderingType == 1)
	{
		// Use attributes
		m_program = _compileProgram(
			m_BillboardsVertexShader,
			m_BillboardsPixelShader);

		if (m_window_h < 0)
			m_window_h = glutGet(GLUT_SCREEN_HEIGHT);

		if (m_fov < 0)
			m_fov = 1.0f;

		// Init color arrays
		for (BehaveRT::uint i = 0; i < hBody3DParams.numBodies; i ++ )
		{
			float initVal = 0.1 * (float) i / hBody3DParams.numBodies;
			float4 initColor = make_float4(
				initVal + hDrawable3DParams.colorBase.x, 
				initVal + hDrawable3DParams.colorBase.y, 
				initVal + hDrawable3DParams.colorBase.z, 1.0f);

			m_Color->setHostArrayElement(i, &initColor);

			initColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			if (m_UseSmoothedColor)
			{
				m_SmoothedColor->setHostArrayElement(i, &initColor);
			}

			if (m_HighLightPositionEnabled)
			{
				float4 initPos = m_Pos->getHostArrayElement(i);
				m_HighLightPositions->setHostArrayElement(i, &initPos);
			}
		}

		m_Color->copyArrayToDevice();
		m_Color->swapPosReadWrite();
		m_Color->copyArrayToDevice();
		m_Color->swapPosReadWrite();

		if (m_UseSmoothedColor)
		{
			m_SmoothedColor->copyArrayToDevice();
			m_SmoothedColor->swapPosReadWrite();
			m_SmoothedColor->copyArrayToDevice();
			m_SmoothedColor->swapPosReadWrite();
		}

		if (m_HighLightPositionEnabled)
		{
			m_HighLightPositions->copyArrayToDevice();
			m_HighLightPositions->swapPosReadWrite();
		}

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_Pos->getReadVbo());
		glVertexPointer(4, GL_FLOAT, 0, 0);
		glColorPointer(4, GL_FLOAT, 0, 0);
	}
		
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------

	// Enable GLSL program and model
	if (hDrawable3DParams.renderingType == 2)
	{

		stringstream stream;
		stream << 
			"#version 120\n" <<
			"#extension GL_EXT_gpu_shader4 : enable\n" <<
			"#extension GL_EXT_texture_buffer_object : enable\n" <<
			m_GIVertexShader;

		m_InstancingProgram = _compileProgram(
			stream.str().c_str(), 
			m_GIPixelShader);
		
		if (m_window_h < 0)
			m_window_h = glutGet(GLUT_SCREEN_HEIGHT);

		if (m_fov < 0)
			m_fov = 1.0f;

		

		m_TimeUniform = glGetUniformLocation(m_InstancingProgram, "time");
		m_Time = 0.0f;
		
		// -------------------------------------------------
		// VBO Binding
		// -------------------------------------------------
		Load3DS (&m_MeshData, m_MeshPath.c_str());

		m_VertexCount = m_MeshData.polygons_qty * 3;

		// Global vars
		vertexData = new GLfloat[m_VertexCount * 3];
		normalData = new GLfloat[m_VertexCount * 3];

		positionSize = m_VertexCount * 3 * sizeof(GLfloat);

		char msg[100];
		sprintf(msg, "Item 3D model info:\tPolys: %d\tVerts: %d", 
			m_MeshData.polygons_qty, m_MeshData.vertices_qty);
		m_CommonRes.getLogger()->log("Drawable3DPlugIn", msg);

		//float3 scale = make_float3(0.017, 0.017, 0.017);
		float3 scale = make_float3(
			m_PointSpriteSizeMultipler, 
			m_PointSpriteSizeMultipler, 
			m_PointSpriteSizeMultipler);

		for (int l_index=0;l_index<m_MeshData.polygons_qty; l_index++)
		{		
			nv::vec3<float> vertex[3];
				
			vertex[0] =make_vec3(m_MeshData.vertex[ m_MeshData.polygon[l_index].a ], scale);
			vertex[1] =make_vec3(m_MeshData.vertex[ m_MeshData.polygon[l_index].b ], scale);
			vertex[2] =make_vec3(m_MeshData.vertex[ m_MeshData.polygon[l_index].c ], scale);

			// Store the vertexes data 
			vertexData[l_index * 9] =	vertex[0].x;
			vertexData[l_index * 9 + 1] = vertex[0].y;
			vertexData[l_index * 9 + 2] = vertex[0].z;

			vertexData[l_index * 9 + 3] = vertex[1].x;
			vertexData[l_index * 9 + 4] = vertex[1].y;
			vertexData[l_index * 9 + 5] = vertex[1].z;

			vertexData[l_index * 9 + 6] = vertex[2].x;
			vertexData[l_index * 9 + 7] = vertex[2].y;
			vertexData[l_index * 9 + 8] = vertex[2].z;

			// Compute polygon normal
			nv::vec3<float> side1 = normalize(vertex[1] - vertex[0]); 
			nv::vec3<float> side2 = normalize(vertex[2] - vertex[0]);

			nv::vec3<float> normal = cross(side1, side2);

			// Store the normals data 
			normalData[l_index * 9] =	  normal.x;
			normalData[l_index * 9 + 1] = normal.y;
			normalData[l_index * 9 + 2] = normal.z;

			normalData[l_index * 9 + 3] = normal.x;
			normalData[l_index * 9 + 4] = normal.y;
			normalData[l_index * 9 + 5] = normal.z;

			normalData[l_index * 9 + 6] = normal.x;
			normalData[l_index * 9 + 7] = normal.y;
			normalData[l_index * 9 + 8] = normal.z;
		}

		// Bind vertexes list
		glGenBuffersARB(1, &m_VertexVBO);
		glBindBufferARB(GL_ARRAY_BUFFER, m_VertexVBO);

		glBufferData(GL_ARRAY_BUFFER, 
			positionSize, 
			vertexData, 
			GL_STREAM_DRAW);

		glVertexPointer(3, GL_FLOAT, 0, 0);

		// Bind normal list
		glGenBuffersARB(1, &m_NormalVBO);
		glBindBufferARB(GL_ARRAY_BUFFER, m_NormalVBO);

		glBufferData(GL_ARRAY_BUFFER, 
			positionSize, 
			vertexData, 
			GL_STREAM_DRAW);

		glNormalPointer(GL_FLOAT, 0, 0);

		glActiveTexture(GL_TEXTURE0);

		glColorPointer(4, GL_FLOAT, 0, 0);
		
	}

	// ---------------------------

	paramTestVal = 0.2f;

	//m_ParamListGL->AddParam(
	//	new Param<float>("time step", paramTestVal, 0.0, 1.0, 0.01, &paramTestVal));

	

}



// --------------------------------------------------------------

template <class Super>
void Drawable3DPlugIn<Super>::update(const float elapsedTime)
{
	Super::update(elapsedTime); // MANDATORY OPERATION

	// Insert here the default update operation
}

// --------------------------------------------------------------
// --------------------------------------------------------------


void drawQuad(float3 base, float3 size, float3 worldRadius)
{
	glBegin(GL_QUADS);

		glColor4f(1.0, 1.0, 1.0, 0.0);
		glVertex3f(  base.x-size.x, 
			base.y - size.y, base.z-size.z);

		glColor4f(1.0, 1.0, 1.0, 0.3);
        glVertex3f(  base.x-size.x, 
			base.y, base.z+size.z );

		glColor4f(1.0, 1.0, 1.0, 0.3);  
		glVertex3f(  base.x+size.x, 
			base.y, base.z+size.z );

		glColor4f(1.0, 1.0, 1.0, 0.0);
		glVertex3f( base.x+size.x, 
			base.y-size.y, base.z-size.z );

    glEnd();
}

template <class Super>
void Drawable3DPlugIn<Super>::renderBillboards(
	GLuint positionsVbo, 
	int method, 
	bool useProgram, 
	int numBodies)
{
	// Needed in Ogre integration, but it doesn't work with geometry instancing
	// Rebind buffers (for Ogre integration)
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, positionsVbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glColorPointer(4, GL_FLOAT, 0, 0);
	// -------------------------------


	if (useProgram)
	{
		glEnable(GL_POINT_SPRITE_ARB);
		glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);

		glUseProgram(m_program);

		glUniform1f( glGetUniformLocation(m_program, "pointScale"), 
			m_window_h / tanf(m_fov*0.5f*(float)M_PI/120.0f) );
		glUniform1f( glGetUniformLocation(m_program, "pointRadius"), 
			hBody3DParams.commonRadius * m_PointSpriteSizeMultipler );
	}

	glEnableClientState(GL_VERTEX_ARRAY);

	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Color management vbo REMOVED (see svn log for details)
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, getColorReadVBO());

	glColorPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawArrays(method, 0, numBodies);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	if (useProgram)
	{
		glDisable(GL_POINT_SPRITE_ARB);
		glUseProgram(0);
	}

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	glVertexPointer(4, GL_FLOAT, 0, 0);
}

// Custom methods
template <class Super>
void Drawable3DPlugIn<Super>::drawBodiesAsPointSprites()
{
	if (hDrawable3DParams.disableDrawing)
		return;

	if (hDrawable3DParams.neighborhoodColoring)
	{
		m_CommonRes.getDeviceInterface()->kernelCall(
			hBody3DParams.numBodies, 256,
			extractColorFromNeighborhoodDRef(), 
			&Drawable3D::Drawable3D_beforeKernelCall,
			&Drawable3D::Drawable3D_afterKernelCall);

		m_Color->swapPosReadWrite();
	}

	// ---------------------------------------
	// SECTION UNDER DEBUG - 08-06-2010

	//if (m_HighLightPositionEnabled)
	//{
	//	renderBillboards(m_HighLightPositions->getReadVbo());
	//}

	// ---------------------------------------

	if (hDrawable3DParams.renderingType == 1)
	{
		renderBillboards(m_Pos->getReadVbo());
	}

	// --------------------------------------------------------
	// --------------------------------------------------------

	if (hDrawable3DParams.renderingType == 2)
	{

		// ---------------------- 30-03-10
		
		// Rebind 3d model's vertexes and normals
		// Usefull with complex scenes

		// Bind vertexes list
		glBufferData(GL_ARRAY_BUFFER, 
			positionSize, 
			vertexData, 
			GL_STREAM_DRAW);

		glVertexPointer(3, GL_FLOAT, 0, 0);

		// Bind normal list
		glBufferData(GL_ARRAY_BUFFER, 
			positionSize, 
			vertexData, 
			GL_STREAM_DRAW);

		glNormalPointer(GL_FLOAT, 0, 0);

		// ----------------------

		// ----------------------  19-02-2010
		// Manages double buffering
		GLuint positionTextBufferObject;
		glActiveTexture(GL_TEXTURE10);
		glGenTextures(1, &positionTextBufferObject);
		glBindTexture(GL_TEXTURE_BUFFER_EXT, positionTextBufferObject);
		glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, m_Pos->getReadVbo());
		
		GLuint forwardTextBufferObject;
		glActiveTexture(GL_TEXTURE11);
		glGenTextures(1, &forwardTextBufferObject);
		glBindTexture(GL_TEXTURE_BUFFER_EXT, forwardTextBufferObject);
		glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, m_Forward->getReadVbo());

		GLuint colorTextBufferObject;
		glActiveTexture(GL_TEXTURE12);
		glGenTextures(1, &colorTextBufferObject);
		glBindTexture(GL_TEXTURE_BUFFER_EXT, colorTextBufferObject);
		glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, getColorReadVBO());
	
		_positionTexUnit = glGetUniformLocation(m_InstancingProgram, "positionSamplerBuffer");
		_forwardTexUnit = glGetUniformLocation(m_InstancingProgram, "forwardSamplerBuffer");
		_colorTexUnit = glGetUniformLocation(m_InstancingProgram, "colorSamplerBuffer");

		// ----------------------
	
		glClearDepth(1.0f);
		glUseProgram(m_InstancingProgram);

		int textureIndexStart = 10; // VALUE: to check
		// Number of 3D objects (?)

		// identify the bound texture unit as input to the filter
		glUniform1i(_positionTexUnit, textureIndexStart);
		glUniform1i(_forwardTexUnit, textureIndexStart + 1);
		glUniform1i(_colorTexUnit, textureIndexStart + 2);

		glUniform1f(m_TimeUniform, m_Time);
		m_Time += 0.01;

		// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
		// Color management vbo REMOVED (see svn log for details)
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, getColorReadVBO());
		
		glColorPointer(4, GL_FLOAT, 0, 0);
		glEnableClientState(GL_COLOR_ARRAY);
		
		// ----------
		// Vertex and normals

		glEnableClientState(GL_NORMAL_ARRAY);	
		glEnableClientState(GL_VERTEX_ARRAY);

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_VertexVBO);

		// ----------
		// Rendering
				
		GLint renderintType = m_UseWireframe ? GL_LINES : GL_TRIANGLES;

		// Use OpenGL geometry instancing
		glDrawArraysInstancedEXT(renderintType, 0, m_VertexCount, 
			hBody3DParams.numBodies);

		// ----------
		// Restoring client state

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
		glUseProgram(0);

		//glActiveTexture(0); // problems with BehavClustering plug-in (opensteerdemo)
	}

	 if (m_DisplayParamListGL) {
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
        m_ParamListGL->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

	return;
	
	// --------------------------------------------------------
	// --------------------------------------------------------

	//if (hDrawable3DParams.renderingType != 2)
	//	return;

	float texMultipler = 1.5;

	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Draw quad REMOVED (see svn log for details)
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	
	
	// Render the floor texture

	glBindTexture(GL_TEXTURE_2D, m_FloorTexIndex);

    glBegin(GL_QUADS);
        glTexCoord2f(0, 1); glVertex3f( -hEnvGrid3DParams.worldRadius.x * texMultipler, 
			-hEnvGrid3DParams.worldRadius.y-100, -hEnvGrid3DParams.worldRadius.z * texMultipler );
		
		glTexCoord2f(0, 0); glVertex3f( -hEnvGrid3DParams.worldRadius.x * texMultipler, 
			-hEnvGrid3DParams.worldRadius.y-100,  hEnvGrid3DParams.worldRadius.z * texMultipler );
		
        glTexCoord2f(1, 0); glVertex3f(  hEnvGrid3DParams.worldRadius.x * texMultipler, 
			-hEnvGrid3DParams.worldRadius.y-100,  hEnvGrid3DParams.worldRadius.z * texMultipler );
       		
		glTexCoord2f(1, 1); glVertex3f(  hEnvGrid3DParams.worldRadius.x * texMultipler, 
			-hEnvGrid3DParams.worldRadius.y-100, -hEnvGrid3DParams.worldRadius.z * texMultipler );
		

    glEnd();

	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Draw quad REMOVED (see svn log for details)
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	glDisable(GL_TEXTURE_2D);

	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Kernel for Sync REMOVED(see svn log for details)
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	m_CommonRes.getDeviceInterface()->threadSync();
	
}

// Custom methods
template <class Super>
void Drawable3DPlugIn<Super>::computeMouseDistance()
{
	// Copy mouse direction and camera position to device
	Drawable3D::Drawable3D_copyFieldsToDevice();


	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 
		m_CommonRes.getDeviceInterface()->getHostSimParams().commonBlockDim,
		BehaveRT_getKernelRef(computeMouseDistance_kernel), 
		&Drawable3D::Drawable3D_beforeKernelCall, 
		&Drawable3D::Drawable3D_afterKernelCall);


	m_MouseDistance->swapPosReadWrite();

	/*
	float* outputData =  m_MouseDistance->getWriteDeviceArray();
	float* inputData = m_MouseDistance->getReadDeviceArray();

	unsigned int numElements = m_MouseDistance->getSize();
    unsigned int memSize = m_MouseDistance->getBytesCount();

	
	CUDPPHandle   mScanPlan;

	CUDPPConfiguration scanConfig;
	scanConfig.algorithm = CUDPP_SCAN;
	scanConfig.datatype  = CUDPP_FLOAT;// Uno di questi non è OK
	scanConfig.op        = CUDPP_MIN;
	scanConfig.options   = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_FORWARD;
	cudppPlan(&mScanPlan, scanConfig, m_MouseDistance->getSize(), 1, 0);
	
	cudppScan(mScanPlan, outputData, inputData, m_MouseDistance->getSize());

	m_MouseDistance->swapPosReadWrite();

	m_MouseDistance->copyArrayFromDevice();
	for (int i = 1; i < 150; i ++)
	{
		cout << m_MouseDistance->getHostArrayElement(i) << ", ";
	}
	cout << endl;
	*/


	//reduce<float>(m_MouseDistance->getSize(), 
	//	numThreads, numBlocks,  
	//	1, inputData, outputData);

	//m_MouseDistance->swapPosReadWrite();

	//m_MouseDistance->copyTo

	// ==========================
	// No reduction - START

	m_MouseDistance->copyArrayFromDevice();
	float minValue = m_MouseDistance->getHostArrayElement(0);
	float minIndex = 0.0f;
	for (int i = 1; i < hBody3DParams.numBodies; i ++)
	{
		if (m_MouseDistance->getHostArrayElement(i) < minValue)
		{
			//cout << m_MouseDistance->getHostArrayElement(i) << endl;
			minIndex = (float) i;
			minValue = m_MouseDistance->getHostArrayElement(i);
		}

	}

	m_MouseDistance->setHostArrayElement(0, &minIndex);

	// No reduction - END
	// ==========================
}


// --------------------------------------------------------------

template <class Super>
GLuint Drawable3DPlugIn<Super>::_compileProgram(const char *vsource, const char *fsource)
{
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);


	if (vsource != NULL)
	{
		glShaderSource(vertexShader, 1, &vsource, 0);
		glCompileShader(vertexShader);
	}
    
	if (fsource != NULL)
	{
		glShaderSource(fragmentShader, 1, &fsource, 0);
		glCompileShader(fragmentShader);
	}

    GLuint program = glCreateProgram();

	if (vsource != NULL)
		glAttachShader(program, vertexShader);

	if (fsource != NULL)
		glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

template <class Super>
void Drawable3DPlugIn<Super>::
createGeometry()
{
	if (!hDrawable3DParams.useCUDAGeometry)
		return;

	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 256,
		createGeometryDRef(), 
		&Drawable3D::Drawable3D_beforeKernelCall,
		&Drawable3D::Drawable3D_afterKernelCall);

	m_Geometry->swapPosReadWrite();
	
}

template <class Super>
void Drawable3DPlugIn<Super>::
smoothColorD()
{
	if (!m_UseSmoothedColor)
		return;
	
	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 256,
		smoothColorDRef(), 
		&Drawable3D::Drawable3D_beforeKernelCall,
		&Drawable3D::Drawable3D_afterKernelCall);

	m_SmoothedColor->swapPosReadWrite();
	m_Color->swapPosReadWrite();
	
}