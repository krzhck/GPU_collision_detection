#pragma once
#include<math.h>
#include<GL/glut.h>
#include<vector>
#include<time.h>
#include"coord.hpp"
#include"ball.hpp"
#include"collision.cuh"
using namespace std;

class BallSet
{
public:
	Ball* balls;
	float XRange;
	float ZRange;
	float Height;
	int Cols;
	int NBalls;
	float MaxRadius;
	float RefreshInterval;
	float GridSize;
	int GridX, GridY, GridZ;

	BallSet(float x, float y, float z, int cols, float max_radius, float refresh_interval)
	{
		XRange = x;
		ZRange = z;
		Height = y;
		Cols = cols;
		MaxRadius = max_radius;
		RefreshInterval = refresh_interval;
		NBalls = cols * cols * cols;
		balls = new Ball[NBalls];
		GridSize = max_radius * 1.5;
		GridX = ceil(XRange * 2 / GridSize);
		GridY = ceil(Height / GridSize);
		GridZ = ceil(ZRange * 2 / GridSize);
	}

	void InitBalls()
	{
		//小球的纹理，材质，�?�色
		GLfloat color[4] = { 0, 0, 0, 1 };
		GLfloat ambient[4] = { 0.2, 0.4, 0.7, 1 };
		GLfloat diffuse[4] = { 0.5, 0.5, 0.5, 1 };
		GLfloat specular[4] = { 0.5, 0.5, 0.5 , 1};
		GLfloat shininess = 30;

		Shader shader(color, ambient, diffuse, specular, shininess);

		float diff_x = (2 * XRange - 2 * MaxRadius) / (Cols - 1);
		float diff_z = (2 * ZRange - 2 * MaxRadius) / (Cols - 1);
		float diff_y = (Height - 2 * MaxRadius) / (Cols - 1);

		for (int i = 0; i < Cols; i++)
		{
			for (int j = 0; j < Cols; j++)
			{
				for (int k = 0; k < Cols; k++)
				{	
					int index = i * Cols * Cols + j * Cols + k;

					float place_x = diff_x * i + MaxRadius - XRange;
					float place_z = diff_z * j + MaxRadius - ZRange;
					float place_y = diff_y * k + MaxRadius;
					Coord pos(place_x, place_y, place_z);
					
					float speed_x = ((rand() % 201) / 100.0f - 1.0f) * 10;
					float speed_y = ((rand() % 201) / 100.0f - 1.0f) * 10;
					float speed_z = ((rand() % 201) / 100.0f - 1.0f) * 10;
					Coord speed(speed_x, speed_y, speed_z);

					float radius = ((rand() % 51) / 100.0f + 0.5f) * MaxRadius;
					
					balls[index].Init(pos, speed, radius, shader);
				}
			}

		}
	}

	/*
		描述：绘制所有球
		参数：无
		返回：无
	*/
	void RenderBalls()
	{
		for (int i = 0; i < NBalls; i++)
		{
			balls[i].RenderBall();
		}
	}


	//更新球的运动--主函�?
	void UpdateBalls()
	{
		// CUDA collision detection
		CollisionDetection(balls, RefreshInterval, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ, NBalls);
	}
};
