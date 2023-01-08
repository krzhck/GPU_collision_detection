#pragma once
#include<GL/glut.h>

using namespace std;


class Shader
{
public:
	GLfloat Color[4] = { 0, 0, 0, 0 }; //颜色
	GLfloat Ambient[4] = { 0, 0, 0, 0 }; //环境光
	GLfloat Diffuse[4] = { 0, 0, 0, 0 }; //漫反射
	GLfloat Specular[4] = { 0, 0, 0, 0 }; //镜面反射
	GLfloat Shininess[1] = { 0 }; //镜面指数

	Shader() {}

	Shader(GLfloat color[], GLfloat ambient[], GLfloat diffuse[], GLfloat specular[], GLfloat shininess)
	{
		for (int i = 0; i < 4; i++)
		{
			Color[i] = color[i];
			Ambient[i] = ambient[i];
			Diffuse[i] = diffuse[i];
			Specular[i] = specular[i];
		}
		Shininess[0] = shininess;
	}

	Shader(const Shader& shader)
	{
		for (int i = 0; i < 4; i++)
		{
			Color[i] = shader.Color[i];
			Ambient[i] = shader.Ambient[i];
			Diffuse[i] = shader.Diffuse[i];
			Specular[i] = shader.Specular[i];
		}
		Shininess[0] = shader.Shininess[0];
	}

	void SetColor(GLfloat color[])
	{
		for (int i = 0; i < 4; i++)
		{
			Color[i] = color[i];
		}
	}

	void SetAmbient(GLfloat ambient[])
	{
		for (int i = 0; i < 4; i++)
		{
			Ambient[i] = ambient[i];
		}
	}

	void SetDiffuse(GLfloat diffuse[])
	{
		for (int i = 0; i < 4; i++)
		{
			Diffuse[i] = diffuse[i];
		}
	}

	void SetSpecular(GLfloat specular[])
	{
		for (int i = 0; i < 4; i++)
		{
			Specular[i] = specular[i];
		}
	}

	void SetShininess(GLfloat shininess)
	{
		Shininess[0] = shininess;
	}

	void SetValue(GLfloat color[], GLfloat ambient[], GLfloat diffuse[], GLfloat specular[], GLfloat shininess)
	{
		for (int i = 0; i < 4; i++)
		{
			Color[i] = color[i];
			Ambient[i] = ambient[i];
			Diffuse[i] = diffuse[i];
			Specular[i] = specular[i];
		}
		Shininess[0] = shininess;
	}
};