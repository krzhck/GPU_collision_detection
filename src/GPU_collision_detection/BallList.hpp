#pragma once
#include<math.h>
#include<GL/glut.h>
#include<vector>
#include<time.h>
#include"point.hpp"
#include"board.hpp"
#include"ball.hpp"
#include"collision.cuh"
using namespace std;

class BallList
{
public:
	Ball* balls;
	BallList() {}
	float XRange;
	float ZRange;
	float Height;
	int Num;
	int NBalls;
	float MaxRadius;
	float TimeOnce;
	float GridSize;
	int GridX, GridY, GridZ;


	/*
	��������ʼ��λ����Ϣ
	������x��Χ��ʵ����-x��x����y��Χ��0��y����z��Χ��-z��z����ÿ�������������ʵ��num���������򣩣������뾶��ģʽ
	*/
	void Init(float x, float y, float z, int num, float max_radius, float time_once)
	{
		XRange = x;
		ZRange = z;
		Height = y;
		Num = num;
		MaxRadius = max_radius;
		TimeOnce = time_once;
		NBalls = num * num * num;
		balls = new Ball[NBalls];
		GridSize = max_radius * 1.5;
		GridX = ceil(XRange * 2 / GridSize);
		GridY = ceil(Height / GridSize);
		GridZ = ceil(ZRange * 2 / GridSize);

	}

	void InitBalls()
	{
		//С����������ʣ���ɫ
		GLfloat color[3] = { 1.0, 0.0, 0.0 };
		GLfloat ambient[3] = { 0.4, 0.2, 0.2 };
		GLfloat diffuse[3] = { 1, 0.8, 0.8 };
		GLfloat specular[3] = { 0.5, 0.3, 0.3 };
		GLfloat shininess = 10;
		int complexity = 40;

		float diff_x = (2 * XRange - 2 * MaxRadius) / (Num - 1);
		float diff_z = (2 * ZRange - 2 * MaxRadius) / (Num - 1);
		float diff_y = (Height - 2 * MaxRadius) / (Num - 1);

		for (int i = 0; i < Num; i++)
		{
			for (int j = 0; j < Num; j++)
			{
				for (int k = 0; k < Num; k++)
				{	
					
					float place_x = diff_x * i + MaxRadius - XRange;
					float place_z = diff_z * j + MaxRadius - ZRange;
					float place_y = diff_y * k + MaxRadius;
					
					int index = i * Num * Num + j * Num + k;
					balls[index].InitColor(color, ambient, diffuse, specular, shininess, complexity);
					float speed_x = ((rand() % 201) / 100.0f - 1.0f) * 10;
					float speed_y = ((rand() % 201) / 100.0f - 1.0f) * 10;
					float speed_z = ((rand() % 201) / 100.0f - 1.0f) * 10;
					float radius = ((rand() % 51) / 100.0f + 0.5f) * MaxRadius;
					balls[index].InitPlace(place_x, place_y, place_z, radius, speed_x, speed_y, speed_z);
				}
			}

		}
	}

	/*
		����������������
		��������
		���أ���
	*/
	void DrawBalls()
	{
		for (int i = 0; i < NBalls; i++)
		{
			balls[i].DrawSelf();
		}
	}

	/*
		�������ж��������Ƿ���ײ
		��������a����b
		���أ���1����0
	*/
	bool JudgeCollision(Ball& a, Ball& b)
	{
		float dist = (a.CurrentPlace - b.CurrentPlace).Dist();
		if (dist < a.Radius + b.Radius)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}

	/*
		������������ײ������ٶ�
		��������a����b
		���أ���
	*/
	void ChangeSpeed(Ball& a, Ball& b)
	{
		//�����ٶȰ����������任�������ٶȲ���
		Point diff = b.CurrentPlace - a.CurrentPlace;
		float dist = diff.Dist();
		
		//���򣬷����ٶ�
		Point speed_collide_a = diff * (a.CurrentSpeed * diff / dist / dist);
		Point speed_collide_b = diff * (b.CurrentSpeed * diff / dist / dist);
		Point unchanged_a = a.CurrentSpeed - speed_collide_a;
		Point unchanged_b = b.CurrentSpeed - speed_collide_b;
		
		//����b������aײb���������߾����ٶ�
		Point speed_collide_new_a = (speed_collide_a * (a.Weight - b.Weight) + speed_collide_b * (2 * b.Weight)) / (a.Weight + b.Weight);
		Point speed_collide_new_b = (speed_collide_a * (2 * a.Weight) + speed_collide_b * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
		Point speed_new_a = speed_collide_new_a + unchanged_a;
		Point speed_new_b = speed_collide_new_b + unchanged_b;
		a.CurrentSpeed = speed_new_a;
		b.CurrentSpeed = speed_new_b;
	}

	//��������˶�--������
	void UpdateBalls()
	{
		static int total_num = 0;
		static float total_time = 0;

		clock_t start, end;
		start = clock();

		// CUDA collision detection
		UpdateBallsGridGPU(balls, TimeOnce, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ, NBalls);

		end = clock();
		float duration = float(end - start) / CLOCKS_PER_SEC * 1000;
		total_num++;
		total_time += duration;
		if (total_num == 10000)
		{
			float average_time = total_time / total_num;
			cout << total_num << "����ײ���ƽ����ʱ" << average_time << "ms" << endl;
			total_num = 0;
			total_time = 0;
		}
	}

};