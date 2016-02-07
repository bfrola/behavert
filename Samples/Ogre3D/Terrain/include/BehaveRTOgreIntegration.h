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
// 06-10-10 bf: Created
//
// ----------------

#pragma once

#include "Ogre.h"

class NativeRenderSystemCommandsRenderQueueListener : public RenderQueueListener
{
protected:
	virtual void NativeRender() = 0;

};

template<class BehaviorEngine>
class BehaveRTNativeRenderSystemCommandsRenderQueueListener : public NativeRenderSystemCommandsRenderQueueListener
{
protected:
	MovableObject* mObject;		
	const Camera* mCamera;		
	SceneManager* mSceneMgr;
	BehaviorEngine* m_BehaviorEngine;

	void NativeRender()
	{
		glScalef(1.0, 1.0, 1.0);

		if (m_BehaviorEngine == NULL)
			return;
		
		m_BehaviorEngine->drawBodiesAsPointSprites();

	}
public:
	BehaveRTNativeRenderSystemCommandsRenderQueueListener(
			MovableObject* object, const Camera* camera, SceneManager* sceneMgr) :
	  mObject(object),
	  mCamera(camera),
	  mSceneMgr(sceneMgr)
	  {
		// display modes: RGB+Z and double buffered
		GLint mode = GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE;
		glutInitDisplayMode (mode);


		// ///////////////////////////////////
		// Add Support for VBO
		glewInit();

		if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
			fprintf(stderr, "Required OpenGL extensions missing.");
			system("pause");
			exit(-1);
		} 
	  
	}
	
	void setBehaviorEngine(BehaviorEngine* behaviorEngine) { m_BehaviorEngine = behaviorEngine;}
	
	virtual void renderQueueStarted(uint8 queueGroupId, const Ogre::String& invocation, 
		bool& skipThisInvocation) { }
	virtual void renderQueueEnded(uint8 queueGroupId, const Ogre::String& invocation, 
		bool& repeatThisInvocation)
	{
		// Set wanted render queue here - make sure there are - make sure that something is on
		// this queue - else you will never pass this if.
		if (queueGroupId != RENDER_QUEUE_MAIN) 
			return;

		// save matrices
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glMatrixMode(GL_TEXTURE);
		glPushMatrix();
		glLoadIdentity(); //Texture addressing should start out as direct.

		RenderSystem* renderSystem = mObject->_getManager()->getDestinationRenderSystem();
		Node* parentNode = mObject->getParentNode();
		renderSystem->_setWorldMatrix(parentNode->_getFullTransform());
		renderSystem->_setViewMatrix(mCamera->getViewMatrix());
		renderSystem->_setProjectionMatrix(mCamera->getProjectionMatrixRS());


		static Pass* clearPass = NULL;
		if (!clearPass)
		{
			MaterialPtr clearMat = MaterialManager::getSingleton().getByName("BaseWhite");
			clearPass = clearMat->getTechnique(0)->getPass(0);
		}
		//Set a clear pass to give the renderer a clear renderstate
		mSceneMgr->_setPass(clearPass, true, false);

		//GLboolean depthTestEnabled=glIsEnabled(GL_DEPTH_TEST);
		//glDisable(GL_DEPTH_TEST);
		GLboolean stencilTestEnabled = glIsEnabled(GL_STENCIL_TEST);
		glDisable(GL_STENCIL_TEST);

		// save attribs
		glPushAttrib(GL_ALL_ATTRIB_BITS);

		// call native rendering function
		//////////////////
		NativeRender();
		//////////////////

		// restore original state
		glPopAttrib();

		//if (depthTestEnabled)
		//{
			glEnable(GL_DEPTH_TEST);
		//}
		if (stencilTestEnabled)
		{
			glEnable(GL_STENCIL_TEST);
		}

		// restore matrices
		glMatrixMode(GL_TEXTURE);
		glPopMatrix();
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
	}
};