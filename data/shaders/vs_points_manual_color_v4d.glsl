#version 330

uniform mat4 modelview_matrix;
uniform mat4 projection_matrix;

uniform float radius;
uniform float viewport_width;

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 velocity;
layout (location = 2) in vec4 color;	

out block
{
	vec3 mv_pos;
	vec3 velocity;
	vec3 color;
}
Out;

void main()
{
    vec4 mv_pos = modelview_matrix * vec4(position.xyz, 1.0);
    vec4 proj = projection_matrix * vec4(radius, 0.0, mv_pos.z, mv_pos.w);
    gl_PointSize = viewport_width * proj.x / proj.w;

	Out.mv_pos = mv_pos.xyz;
	Out.velocity = velocity.xyz;
	Out.color = color.xyz;
    gl_Position = projection_matrix * mv_pos;  
}
