////////////////////////////////////////////////////////////
////							////
////  BehaveRT Tools					////
////  http://isis.dia.unisa.it/projects/behavert/	////
////							////
////////////////////////////////////////////////////////////

------------------------------------------------------------
--- PlugInMaker
------------------------------------------------------------

Create and run a new BehaveRT PlugIn. 

1. Setup plugIn name and other settings in "PlugInDescription.cfg"
	Path: <BehaveRT Root>\Tools\PlugInMaker\bin\win32\release\Config\

2. Execute the script "RunPlugInMaker.bat"
	Path: <BehaveRT Root>\Tools\PlugInMaker\bin\win32\release\

	After that, a new folder with name "<PlugInName>" will be created in:
	<BehaveRT Root>\PlugIns	

3. Add your plugIn Visual Studio project to the BehaveRT solution:
	Path: <BehaveRT Root>\PlugIns\<PlugInName>\

4. Add the following line to DeviceInterface.cu (<BehaveRT Root>\Core):
	#include <PlugInName>/<PlugInName>_kernel.cu

	This enable the CUDA-section of your plugIn.

5. Open the Property Page of DeviceInterface.cu. Add the following line 
	../PlugIns/<PlugInName>/<PlugInName>_kernel.cu;

	In "CUDA Build Rule vx.x.xx -> General -> Source Dependecies"

After these five steps, the library can be compiled. The CUDA section of the whole library will be recompiled every time the CUDA section of your plugIn is modified.



