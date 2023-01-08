#pragma once
#include <math.h>
#include "point.hpp"
#include "shader.hpp"
using namespace std;

class Board
{
public:
	Point PointList[4];
	Point Normal;
	//材质，纹理，颜色信息

	Shader WallShader;


	Board(){}
	void InitPlace(Point a, Point b, Point c, Point d)
	{
		PointList[0] = a;
		PointList[1] = b;
		PointList[2] = c;
		PointList[3] = d;
		GetNorm();
	}

	//初始化颜色，纹理，材质信息
	void InitColor(Shader & shader)
	{
		WallShader = shader;
	}

	//求平面法向量(方向指向外侧）
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

	//求点到平面距离
	float GetDist(Point p)
	{
		GetNorm();
		float dist = abs(Normal.x * p.x + Normal.y * p.y + Normal.z * p.z);
		float norm = sqrt(Normal.x * Normal.x + Normal.y * Normal.y + Normal.z * Normal.z);
		return dist / norm;
	}
};
