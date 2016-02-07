#pragma once

#include "DeviceData.cuh"
#include "common_resources.cu"
#include "include\[PName]_kernel.cuh"

// Other plugIn dependencies
// Auto-generated code START
[PDependenciesPaths]
// Auto-generated code END

// Structures sharing
share_device_struct([PName]Params);
share_device_struct([PName]Fields);

// Textures
// Auto-generated code START
[PFeaturesTextures]
// Auto-generated code END

// Device functions

namespace [PName]
{
	
	__device__ bool deviceFuncExample( int individualIndex )
	{
		// ...
	}

}
