#pragma once
#include<GL/glut.h>


class Light
{
public:
	GLfloat Color[4] = { 0.1, 0.1, 0.1};
	GLfloat Ambient[4] = { 1, 1, 1, 1 };
	GLfloat Diffuse[4] = { 1, 1, 1, 1 };
	GLfloat Specular[4] = { 1, 1, 1, 1 };
	GLfloat Position[4] = { 10.0f, 10.0f, 10.0f, 1 };

	Light() {}
};
