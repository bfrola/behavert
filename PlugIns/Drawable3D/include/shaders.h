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

extern const char *vertexShader;
extern const char *spherePixelShader;

#define STRINGIFY(A) #A

// Default billboards vertex shader
const char *defaultBillboardsVertexShader = STRINGIFY(
uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels
uniform float densityScale;
uniform float densityOffset;
void main()
{
    // calculate window-space point size
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

    gl_FrontColor = gl_Color;
}
);

// Default billboards pixel shader for rendering points as shaded spheres
const char *defaultBillboardsPixelShader = STRINGIFY(
void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    //N.xy = gl_TexCoord[0].xy*vec2(2.0, -0.5) + vec2(-1.0, 1.0); // semicircle
	N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0); // circle
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);

	// calculate lighting
    float diffuse = max(0.25, dot(lightDir, N));

    gl_FragColor = gl_Color * diffuse;
	gl_FragColor.w = 1.0;
}
);

// -------------------------------------------------------------------
// -------------------------------------------------------------------

// Default geometry instancing vertex shader
const char *defaultGIVertexShader = STRINGIFY(

varying vec4 vertexPos;
varying float intensity;

uniform samplerBuffer positionSamplerBuffer;
uniform samplerBuffer forwardSamplerBuffer;

uniform float time;

void main(void)
{
	vec3 itemPosition = texelFetchBuffer(positionSamplerBuffer, gl_InstanceID).xyz;
	vec3 itemForward = texelFetchBuffer(forwardSamplerBuffer, gl_InstanceID).xyz;

	vec3 unitY = vec3(0.0, 1.0, 0.0);
	vec3 direction = normalize(itemForward);

	vec3 sideDirection = normalize(cross(unitY, direction));
	vec3 upDirection = normalize(cross(direction, sideDirection));

	mat3 rotation = mat3(
		sideDirection, upDirection, direction );

	vec3 rotatedPos = rotation * gl_Vertex.xyz;
	vec3 pos = rotatedPos + itemPosition.xyz;

	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1.0);

	// Vertex color
	// Color

	const vec3 lightDir = normalize( vec3(0, 0.2, -0.7) );	

	vec3 rotatedNormal = rotation * gl_Normal.xyz;

	//rotatedNormal = normalize ( rotatedNormal );
	//rotatedPos = normalize( rotatedPos );

	float lightDot = dot(lightDir, rotatedNormal);	 
	float diffuse = lightDot * lightDot * lightDot;

	vec3 baseColor = vec3(0.3, 0.5, 0.2);

	gl_FrontColor = 
		vec4(baseColor * diffuse, 1.0);

}
);

// Default geometry instancing pixel shader
const char *defaultGIPixelShader = STRINGIFY(
varying vec4 vertexPos;
varying float intensity;

void main(void)
{
	gl_FragColor = gl_Color;
}
);