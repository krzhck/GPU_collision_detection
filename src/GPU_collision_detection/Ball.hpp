#pragma once
#include<math.h>
#include<GL/glut.h>
#include<vector>
#include"point.hpp"
#include"board.hpp"
using namespace std;

#pragma once
class Ball
{
public:
	//λ�ã��ٶ���Ϣ
	Point CurrentPlace;
	float Radius;
	Point CurrentSpeed;
	int BallComplexity;
	float Weight;

	//���ʣ�������ɫ��Ϣ
	GLfloat Color[3] = { 0, 0, 0 }; //��ɫ
	GLfloat Ambient[4] = { 0, 0, 0, 0 }; //������
	GLfloat Diffuse[4] = { 0, 0, 0, 0 }; //������
	GLfloat Specular[4] = { 0, 0, 0, 0 }; //���淴��
	GLfloat Shininess[4] = { 0 }; //����ָ��
public:
	Ball(){}

	//��ʼ��λ�ã��ٶ���Ϣ
	void InitPlace(float x, float y, float z, float radius, float speed_x, float speed_y, float speed_z)
	{
		CurrentPlace.SetPlace(x, y, z);
		CurrentSpeed.SetPlace(speed_x, speed_y, speed_z);
		Radius = radius;
		Weight = radius * radius * radius;
	}

	//��ʼ����ɫ������������Ϣ
	void InitColor(GLfloat color[], GLfloat ambient[], GLfloat diffuse[], GLfloat specular[], GLfloat shininess, int complexity)
	{
		for (int i = 0; i < 3; i++)
		{
			Color[i] = color[i];
			Ambient[i] = ambient[i];
			Diffuse[i] = diffuse[i];
			Specular[i] = specular[i];
		}
		//͸���ȣ�1
		Ambient[3] = 1.0;
		Diffuse[3] = 1.0;
		Specular[3] = 1.0;
		Shininess[0] = shininess;
		BallComplexity = complexity;
	}

	//����һ��С��
	void DrawSelf()
	{
		//�����������ʵ���Ϣ
		glMaterialfv(GL_FRONT, GL_AMBIENT, Ambient);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, Diffuse);
		glMaterialfv(GL_FRONT, GL_SPECULAR, Specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, Shininess);

		//ƽ�Ƶ�����ԭ�㣬���ƣ��ָ�����
		glPushMatrix();
		glTranslatef(CurrentPlace.x, CurrentPlace.y, CurrentPlace.z);
		glutSolidSphere(Radius, BallComplexity, BallComplexity);
		glPopMatrix();
	}

	/*
		����������С�������˶�����߽���ײ
		�����������˶�ʱ�䣬X��Χ��-X, X), Z��Χ(-Z, Z), Y��Χ(0, Y)
		���أ���
	*/
	void Move(float time, float XRange, float ZRange, float Height)
	{
		CurrentPlace = CurrentPlace + CurrentSpeed * time;
		HandleCollisionBoard(XRange, ZRange, Height);
	}
	/*
		������������߽���ײ
		������X��Χ��-X, X), Z��Χ(-Z, Z), Y��Χ(0, Y)
		���أ���
	*/
	void HandleCollisionBoard(float XRange, float ZRange, float Height)
	{
		if (CurrentPlace.x - Radius < -XRange)
		{
			CurrentPlace.x = -XRange + Radius;
			CurrentSpeed.x = -CurrentSpeed.x;
		}
		else if (CurrentPlace.x + Radius > XRange)
		{
			CurrentPlace.x = XRange - Radius;
			CurrentSpeed.x = -CurrentSpeed.x;
		}
		if (CurrentPlace.z - Radius < -ZRange)
		{
			CurrentPlace.z = -ZRange + Radius;
			CurrentSpeed.z = -CurrentSpeed.z;
		}
		else if (CurrentPlace.z + Radius > ZRange)
		{
			CurrentPlace.z = ZRange - Radius;
			CurrentSpeed.z = -CurrentSpeed.z;
		}
		if (CurrentPlace.y - Radius < 0)
		{
			CurrentPlace.y = Radius;
			CurrentSpeed.y = -CurrentSpeed.y;
		}
		else if (CurrentPlace.y + Radius > Height)
		{
			CurrentPlace.y = Height - Radius;
			CurrentSpeed.y = -CurrentSpeed.y;
		}
	}

	
};


