#pragma once
#include<math.h>
#include<GL/glut.h>
#include<vector>
#include"coord.hpp"
#include "shader.hpp"
using namespace std;

#define BALL_SLICE 50

class Ball
{
public:
	//位置，速度信息
	Point CurrentPlace;
	float Radius;
	Point CurrentSpeed;
	float Weight;
	
	Shader BallShader;

	Ball(){}

	//初始化位置，速度信息
	void Init(Point pos, Point speed, float radius, Shader shader)
	{
		CurrentPlace = pos;
		CurrentSpeed = speed;
		Radius = radius;
		Weight = radius * radius * radius;

		BallShader = shader;
	}

	//绘制一个小球
	void DrawSelf()
	{
		//设置纹理，材质等信息
		glColor3f(BallShader.Color[0], BallShader.Color[1], BallShader.Color[2]);
		glMaterialfv(GL_FRONT, GL_AMBIENT, BallShader.Ambient);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, BallShader.Diffuse);
		glMaterialfv(GL_FRONT, GL_SPECULAR, BallShader.Specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, BallShader.Shininess);

		//平移到坐标原点，绘制，恢复坐标
		glPushMatrix();
		glTranslatef(CurrentPlace.x, CurrentPlace.y, CurrentPlace.z);
		glutSolidSphere(Radius, BALL_SLICE, BALL_SLICE);
		glPopMatrix();
	}
};
