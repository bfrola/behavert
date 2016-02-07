#pragma once

#include "DeviceData.cuh"
#include "common_resources.cu"
#include "include\[PName]_kernel.cuh"

// Other plugIn dependencies
[PDependenciesPaths]

// Structures sharing
share_device_struct([PName]Params);
share_device_struct([PName]Fields);

// Textures
[PFeaturesTextures]

// Device functions

