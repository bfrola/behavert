#pragma once

const char *Buildings3DVertexShader = STRINGIFY(

varying vec4 vertexPos;
varying float intensity;

uniform samplerBuffer positionSamplerBuffer;
uniform samplerBuffer forwardSamplerBuffer;
uniform samplerBuffer colorSamplerBuffer;

uniform float time;

void main(void)
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

const char *Buildings3DPixelShader = STRINGIFY(
varying vec4 vertexPos;
varying float intensity;

void main(void)
{
	const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -0.5) + vec2(-1.0, 1.0); // semicircle
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);

	// calculate lighting
    float diffuse = max(0.25, dot(lightDir, N));

    gl_FragColor = gl_Color * diffuse;
	gl_FragColor.w = 1.0;
}
);