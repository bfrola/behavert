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

#include "OpenSteerWrapperPlugIn.h"
using namespace OpenSteer;


OpenSteer::Vec3 OpenSteerWrapper::float32Vec3(float3 vec)
{
	return Vec3(vec.x, vec.y, vec.z);
}

OpenSteer::Vec3 OpenSteerWrapper::float42Vec3(float4 vec)
{
	return Vec3(vec.x, vec.y, vec.z);
}
OpenSteer::Vec3 OpenSteerWrapper::int32Vec3(int3 vec)
{
	return Vec3(vec.x, vec.y, vec.z);
}
OpenSteer::Vec3 OpenSteerWrapper::uint32Vec3(uint3 vec)
{
	return Vec3(vec.x, vec.y, vec.z);
}

float3 OpenSteerWrapper::Vec32float3(OpenSteer::Vec3 vec)
{
	return make_float3(vec.x, vec.y, vec.z);
}

float4 OpenSteerWrapper::Vec32float4(OpenSteer::Vec3 vec, float w)
{
	return make_float4(vec.x, vec.y, vec.z, w);
}
