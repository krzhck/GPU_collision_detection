#pragma once
#include<GL/glut.h>


class Light
{
public:
	GLfloat Color[4] = { 0.1, 0.1, 0.1}; //背景颜色
	GLfloat Ambient[4] = { 1, 1, 1, 1 }; //环境光
	GLfloat Diffuse[4] = { 1, 1, 1, 1 }; //漫反射
	GLfloat Specular[4] = { 1, 1, 1, 1 }; //镜面反射
	GLfloat Position[4] = { 10.0f, 10.0f, 10.0f, 1 }; //镜面指数

	Light() {}
};
