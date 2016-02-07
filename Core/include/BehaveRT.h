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
// 12-08 bf: Created
//
// ----------------

#pragma once

#include "CommonResources.h"
#include "Utility.h"
#include "UtilityConfigFile.h"
#include "UtilityStringConverter.h"
#include "UtilityLogger.h"

#include "DeviceArrayWrapper.h"
#include "DeviceDataWrapper.h"
#include "SimEnginePlugIn.h"

#include <assert.h>

//using namespace BehaveRT;

// Shared class Community
extern CommonResources m_CommonRes;

/** 
\mainpage

\image html behavertlogo.jpg
\li Project page: http://isis.dia.unisa.it/projects/behavert/
\li ISISLab: http://isis.dia.unisa.it/wiki/index.php/Main_Page


\par
\b BehaveRT is a framework (C++ library) that allows to create real-time massive simulations of 
behavioral models on Graphics Processing Unit (GPU). "RT" stands for Real-Time.
\par

\section system System description

\par 
The BehaveRT provides:
\par CommonResources
A static access point to the device (GPU) interface and the configuration.
\par SimEngine
Template for the simulation engine described below.

\par
CommonResources and SimEngine are described in detail in the next two subsections.

\subsection commonRes CommonResources

\par
By including the main .h file of the framework:

\code 
#include "BehaveRT.h"
\endcode

\par
the namespace of the application will contain \c m_CommonRes, which is an object of the class BehaveRT::CommonResources. 

\par 
As you can see in the class documentation, it manages the installation of plugIns and:

\par DeviceInterface
The interface to device, by using the method BehaveRT::CommonResources::getDeviceInterface. This method returns
a reference to the class BehaveRT::DeviceInterface. Thought the DeviceInterface, the SimEngine can access to the 
GPU. Between the device and the application there are two abstraction levels: 
\li CUDA: allows to use the programming model based on threads, and hide the complexity of the GPU. 
\li BehaveRT: adds somo facilities in order to run easly behavioral models on GPU. This facilities include a system
for call generic CUDA kernels without the need to specify any parameters. Also, the class BehaveRT::DeviceArrayWrapper 
uses the BehaveRT::DeviceInterface in order to provide a duplux data allocation on host (CPU) and device.

The DeviceInterface is directly used by the SimEngine, described below.

\image html deviceinterface.jpg

\par Configuration
The library includes the utility BehaveRT::UtilityConfigFile for read data from config file. This utility has been copied
from the source files of Ogre3D rendering engine and modified.
 




\subsection simengine SimEngine: Simulation engine
\par
The simulation engine is the earth of the system.
This entity manages:

\par Feature
Features are the data of the simulation entities. 

\par Functionalities
Each method provided by the SimEngine is a functionality. Functionality does operations using features.

The framework provides only a template class for the simulation engine. Features and functionalities are 
added to the simulation engine by plugIns. The class BehaveRT::SimEnginePlugIn speciefies the interface
that each plugIn has to implement.

\image html simengine.jpg

The SimEngine class is created into the user application by using the Mixing classes techinque. This operation
allow the user to choose the desired plugIns. By choosing desired plugIns, it is possibile to select the 
desired set of feature and functionality that the SimEngine must have. 

The next section describe how to mix plugIns in order to achieve the desired SimEngine class.

\section getting_started How to use BehaveRT

\par
The user application must contain the plugIn mixing section. The aim of this section is to create the 
SimEngine class by creating an extension chain of plugIn classes. Here an example:

\code
Proximity3DPlugIn
	<EnvGrid3DPlugIn
		<Body3DPlugIn<DummyPlugIn>>>
		SimEngine;
\endcode

\par
The class BehaveRT::DummyPlugIn is provided by the framework and it closes the chain.
The code in this example merges features and functionalities of Proximity3D::Proximity3DPlugIn, 
EnvGrid3D::EnvGrid3DPlugIn and Body::Body3DPlugIn.

\code
SimEngine m_BehaviorEngine;
\endcode

\par
The object \c m_BehaviorEngine have methods of all plugIns in the chain. For example, it have
the method Proximity3D::Proximity3DPlugIn::computeNeighborhoods and 
EnvGrid3D::EnvGrid3DPlugIn::reorderSimData. This methods belong to two different plugIns.

<br />
<hr /><hr />
\section casa09_demo Picture and Shape flocking demo

This demo allows to try the work presented ad CASA 09 http://hmi.ewi.utwente.nl/CASA09.

\subsection demo_gui GUI

\image html democasa09_screenshot.JPG

\par
Legend:

\par
\li Mode: two available modes: picture flocking/shape flocking.
\li Source: image or 3d model in use.
\li Compute neighborhood: if No both the neighborhood search and the flocking behavior of
each flock item are disabled. Set on No to accelerate the simulation. (Default value: Yes).
\li Follow the path: if Yes the 3d model points become checkpoints of a movement path.
(Default value: No).

\subsection demo_keys How to use

\par
\li Ctrl + mouse left click + mouse move: rotate camera.
\li Ctrl + mouse middle click + mouse move: change zoom.
\li F1: change mode/source.
\li F2: enable/disable item-item iterations.
\li F3: enable/disable follow the shape.
\li F4: enable/disable avoinding ball.
\li F5: avoinding ball follows the mouse pointer.
\li Space: stop/resume simualtion.
\li Esc: exit simulation.

\subsection demo_sw_conf Software requirements

\par
\li Nvidia CUDA v2.1/2.2/2.3
\li Microsoft Windows XP (to be checked for Vista and Seven).

\subsection demo_hw_conf Hardware requirements

\par
\li Any CUDA-enabled device.

Tested on GeForce 8800GTS, 8800GTX, GTX260, GTX285.

\subsection demo_conffile Configuration file

\par
Demo configuration can be changed by means of editing the configuration file config/CSShapes.cfg.
The configuration is a set key-value pairs. 
The file format is the same of Ogre3D configuration files.

\par
Here is a portion of the conf file:

\code 
[Drawable3DPlugIn]
neighborhoodColoring=false
colorBase=0.0f 0.2f 0.4f
disableDrawing=false
renderingType=1
meshPath=Data\\pyramid.3DS
useCUDAGeometry=false
numVertexes=1
pointSpriteSizeMultipler=0.01
useSmoothedColor=true
\endcode

\par
The key renderingType incates the single flock item rendering system.
By default the rendering system is 1 = billboard rendering.

\code
renderingType=1 
\endcode

\par
An other possibile value of renderingType is 2 = OpenGL instancing of the 3D model.

\code
renderingType=2 
\endcode

\par
The key meshPath indicates the 3D model source.


<br />
<hr /><hr />

\author Bernardino Frola

*/

// Sections to add
// \section create_plugin How to create a plugIn

