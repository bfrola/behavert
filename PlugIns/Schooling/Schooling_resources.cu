#pragma once

#include "DeviceData.cuh"
#include "common_resources.cu"
#include "include\Schooling_kernel.cuh"

// Other plugIn dependencies
#include "..\Body\include\Body3D_kernel.cuh"
#include "..\Body\Body3D_resources.cu"

#include "..\EnvGrid3D\include\EnvGrid3D_kernel.cuh"
#include "..\EnvGrid3D\EnvGrid3D_resources.cu"

#include "..\Proximity3D\include\Proximity3D_kernel.cuh"
#include "..\Proximity3D\Proximity3D_resources.cu"

#include "..\OpenSteerWrapper\include\OpenSteerWrapper_kernel.cuh"
#include "..\OpenSteerWrapper\OpenSteerWrapper_resources.cu"



// Structures sharing
share_device_struct(SchoolingParams);
share_device_struct(SchoolingFields);

// Textures

texture<float4, 1, cudaReadModeElementType> randomValueTex;


// Device functions


__device__ bool inAttractionOrientationZone(
	float3 myPos, 
	float3 otherPos, 
	float3 myForward)
{
	float3 offset;
	offset.x = otherPos.x - myPos.x;
	offset.y = otherPos.y - myPos.y;
	offset.z = otherPos.z - myPos.z;

	float dist = length(offset);

	if ( dist > dSchoolingParams.r_p )
		return false;

	// Check angular data
	float cosAngle = dot(myForward, normalize(offset));
	return cosAngle > dSchoolingParams.eta;
	
}


__device__ void checkForAttractionOrientation(
	float3 myPos, 
	float3 otherPos, 
	float3 myForward,
	float3 otherForward,
	float3& attraction,
	float3& orientation)
{
	if (inAttractionOrientationZone(myPos, otherPos, myForward))
	{
		float3 offset =  otherPos - myPos;
		attraction += offset / length(offset);

		orientation += otherForward / length(otherForward);
	}
}

