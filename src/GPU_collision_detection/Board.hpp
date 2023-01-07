#pragma once
#include<math.h>
#include"point.hpp"
using namespace std;

class Board
{
public:
	Point PointList[4];
	Point Normal;
	//���ʣ�������ɫ��Ϣ
	GLfloat Color[3] = { 0, 0, 0 }; //��ɫ
	GLfloat Ambient[4] = { 0, 0, 0, 0 }; //������
	GLfloat Diffuse[4] = { 0, 0, 0, 0 }; //������
	GLfloat Specular[4] = { 0, 0, 0, 0 }; //���淴��
	GLfloat Shininess[1] = { 0 }; //����ָ��


	Board(){}
	void InitPlace(Point a, Point b, Point c, Point d)
	{
		PointList[0] = a;
		PointList[1] = b;
		PointList[2] = c;
		PointList[3] = d;
		GetNorm();
	}

	//��ʼ����ɫ������������Ϣ
	void InitColor(GLfloat color[], GLfloat ambient[], GLfloat diffuse[], GLfloat specular[], GLfloat shininess)
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
	}

	//��ƽ�淨����(����ָ����ࣩ
	void GetNorm()
	{
		Point v1 = PointList[0];
		Point v2 = PointList[1];
		Point v3 = PointList[2];
		float na = (v2.y - v1.y)*(v3.z - v1.z) - (v2.z - v1.z)*(v3.y - v1.y);
		float nb = (v2.z - v1.z)*(v3.x - v1.x) - (v2.x - v1.x)*(v3.z - v1.z);
		float nc = (v2.x - v1.x)*(v3.y - v1.y) - (v2.y - v1.y)*(v3.x - v1.x);
		float norm = sqrt(na * na + nb * nb + nc * nc);
		na /= norm;
		nb /= norm;
		nc /= norm;
		if (na * v1.x + nb * v1.y + nc * v1.z < 0)
		{
			na = -na;
			nb = -nb;
			nc = -nc;
		}
		Normal.SetPlace(na, nb, nc);
		
	}

	//��㵽ƽ�����
	float GetDist(Point p)
	{
		GetNorm();
		float dist = abs(Normal.x * p.x + Normal.y * p.y + Normal.z * p.z);
		float norm = sqrt(Normal.x * Normal.x + Normal.y * Normal.y + Normal.z * Normal.z);
		return dist / norm;
	}

	
};