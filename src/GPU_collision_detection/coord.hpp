#pragma once

#include<math.h>
using namespace std;

#define PI 3.14159265

// 3D坐标�?
class Coord
{
public:
	float x;
	float y;
	float z;
	Coord()
	{
		x = 0;
		y = 0;
		z = 0;
	}
	Coord(float tx, float ty, float tz)
	{
		x = tx;
		y = ty;
		z = tz;
	}

	Coord(const Coord& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
	}

	void SetPos(float tx, float ty, float tz)
	{
		x = tx;
		y = ty;
		z = tz;
	}
	Coord operator+(const Coord & b)
	{
		Coord c;
		c.x = x + b.x;
		c.y = y + b.y;
		c.z = z + b.z;
		return c;
	}
	Coord operator-(const Coord & b)
	{
		Coord c;
		c.x = x - b.x;
		c.y = y - b.y;
		c.z = z - b.z;
		return c;
	}
	Coord operator*(const float & b)
	{
		Coord c;
		c.x = x * b;
		c.y = y * b;
		c.z = z * b;
		return c;
	}
	Coord operator/(const float & b)
	{
		Coord c;
		c.x = x / b;
		c.y = y / b;
		c.z = z / b;
		return c;
	}
	float operator*(const Coord & b)
	{
		float sum = 0;
		sum += x * b.x;
		sum += y * b.y;
		sum += z * b.z;
		return sum;
	}

	float Dist()
	{
		float sum = 0;
		sum += x * x;
		sum += y * y;
		sum += z * z;
		sum = sqrt(sum);
		return sum;
	}
};

