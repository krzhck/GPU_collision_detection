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
	Coord Pos;
	float Radius;
	Coord Speed;
	float Weight;
	
	Shader BallShader; // surface info

	Ball(){}

	// init position, speed, surface
	void Init(Coord pos, Coord speed, float radius, Shader shader)
	{
		Pos = pos;
		Speed = speed;
		Radius = radius;
		Weight = radius * radius * radius;

		BallShader = shader;
	}

	void RenderBall()
	{
		// set surface info
		glColor3f(BallShader.Color[0], BallShader.Color[1], BallShader.Color[2]);
		glMaterialfv(GL_FRONT, GL_AMBIENT, BallShader.Ambient);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, BallShader.Diffuse);
		glMaterialfv(GL_FRONT, GL_SPECULAR, BallShader.Specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, BallShader.Shininess);

		// render and reposition
		glPushMatrix();
		glTranslatef(Pos.x, Pos.y, Pos.z);
		glutSolidSphere(Radius, BALL_SLICE, BALL_SLICE);
		glPopMatrix();
	}
};
