#pragma once
#include<GL/glut.h>


class Light
{
public:
	GLfloat Color[4] = { 0, 0, 0, 0 }; //背景颜色
	GLfloat Ambient[4] = { 0, 0, 0, 0 }; //环境光
	GLfloat Diffuse[4] = { 0, 0, 0, 0 }; //漫反射
	GLfloat Specular[4] = { 0, 0, 0, 0 }; //镜面反射
	GLfloat Position[4] = { 0, 0, 0, 0 }; //镜面指数

	Light() {}

	//初始化
	void Init(GLfloat color[], GLfloat ambient[], GLfloat diffuse[], GLfloat specular[], GLfloat position[])
	{
		for (int i = 0; i < 3; i++)
		{
			Color[i] = color[i];
			Ambient[i] = ambient[i];
			Diffuse[i] = diffuse[i];
			Specular[i] = specular[i];
			Position[i] = position[i];
		}
		//透明度：1
		Color[3] = 1.0;
		Ambient[3] = 1.0;
		Diffuse[3] = 1.0;
		Specular[3] = 1.0;

		//无限远处平行光
		Position[3] = 1.0; 
	}
};
