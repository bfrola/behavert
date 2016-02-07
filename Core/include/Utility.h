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
//
// ----------------
// Change log
//
// 12-08 bf: Created
//
// ----------------

#pragma once
#include "math.h"

namespace BehaveRT
{
	// ----------------------------------------------------------------------------
    /// Generic interpolation

	template<class T> inline T interpolate (float alpha, const T& x0, const T& x1)
    {
        return x0 + ((x1 - x0) * alpha);
    }

	// ----------------------------------------------------------------------------
    /** Constrain a given value (x) to be between two (ordered) bounds: min
     and max.  Returns x if it is between the bounds, otherwise returns
     the nearer bound.
	*/

    inline float clip (const float x, const float min, const float max)
    {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

	// ----------------------------------------------------------------------------
    // Generic interpolation
	
	// ----------------------------------------------------------------------------
    /** blends new values into an accumulator to produce a smoothed time series
    @par
     Modifies its third argument, a reference to the float accumulator holding
     the "smoothed time series."
	@par
     The first argument (smoothRate) is typically made proportional to "dt" the
     simulation time step.  If smoothRate is 0 the accumulator will not change,
     if smoothRate is 1 the accumulator will be set to the new value with no
     smoothing.  Useful values are "near zero".
    
	@usage
     Usage:
             blendIntoAccumulator (dt * 0.4f, currentFPS, smoothedFPS);
	*/

    template<class T>
    inline void blendIntoAccumulator (const float smoothRate,
                                      const T& newValue,
                                      T& smoothedAccumulator)
    {
        smoothedAccumulator = interpolate (clip (smoothRate, 0, 1),
                                           smoothedAccumulator,
                                           newValue);
    }

	// ------------------------------------------------------------------------

	inline float frand()
	{
		return rand() / (float) RAND_MAX;
	}

	// ------------------------------------------------------------------------

	// dot product
	inline float mdot(float3 a, float3 b)
	{ 
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	// ------------------------------------------------------------------------

	// length
	inline float mlength(float3 v)
	{
		return sqrtf(mdot(v, v));
	}
	
	// ------------------------------------------------------------------------

	// normalize
	inline float3 mnormalize(float3 v)
	{
		float invLen = 1.0f / sqrtf(mdot(v, v));
		return make_float3(
			v.x * invLen,
			v.y * invLen,
			v.z * invLen);
	}

	// ------------------------------------------------------------------------

	inline float3 unitRandFloat3()
	{
		float3 unitRand = 
			make_float3(frand() - 0.5, 
				frand() - 0.5, 
				frand() - 0.5);
		return mnormalize(unitRand);
	}

	inline float4 float32float4(float3 v, float w)
	{
		return make_float4(
			v.x, v.y, v.z, w);
	}
}