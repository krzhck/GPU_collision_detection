#pragma once
#include <math.h>
#include "coord.hpp"
#include "shader.hpp"
using namespace std;

class Wall
{
public:
	Coord Vertexes[4];
	Coord Normal; // normal vector
	Shader WallShader;


	Wall(){}
	void InitPos(Coord a, Coord b, Coord c, Coord d)
	{
		Vertexes[0] = a;
		Vertexes[1] = b;
		Vertexes[2] = c;
		Vertexes[3] = d;
		GetNorm();
	}

	void GetNorm()
	{
		Coord v1 = Vertexes[0];
		Coord v2 = Vertexes[1];
		Coord v3 = Vertexes[2];
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
		Normal.SetPos(na, nb, nc);
		
	}

	// distance between point p and wall
	float GetDist(Coord p)
	{
		GetNorm();
		float dist = abs(Normal.x * p.x + Normal.y * p.y + Normal.z * p.z);
		float norm = sqrt(Normal.x * Normal.x + Normal.y * Normal.y + Normal.z * Normal.z);
		return dist / norm;
	}
};
