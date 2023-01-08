#pragma once
#include<math.h>
#include"coord.hpp"
using namespace std;

class Camera
{
public:
	Coord Pos;
	Coord LookCenter;
	float R_xOz;
	float Arc_xOz; // 0 - 2pi
	float H_y; // height
	int MouseX;
	int MouseY;
	const float v_xOz = 0.01; // mouse horizontal speed
	const float v_y = 0.05; // mouse vertical speed
	const float v_key = 0.5; // keyboard speed

	Camera(float R, float height)
	{
		R_xOz = R;
		Arc_xOz = PI / 4;
		H_y = height;
		ResetPos();
		MouseX = -1;
		MouseY = -1;
		LookCenter.SetPos(0, 0, 0);
	}

	void ResetPos()
	{
		float x = R_xOz * cos(Arc_xOz) + LookCenter.x;
		float y = H_y + LookCenter.y;
		float z = R_xOz * sin(Arc_xOz) + LookCenter.z;
		Pos.SetPos(x, y, z);
	}

	// mouse click
	void MouseDown(int x, int y)
	{
		MouseX = x;
		MouseY = y;
	}

	// mouse move
	void MouseMove(int x, int y)
	{
		int dx = x - MouseX;
		int dy = y - MouseY;
		Arc_xOz = Arc_xOz + dx * v_xOz;
		while (Arc_xOz < 0) Arc_xOz += 2.0 * PI;
		while (Arc_xOz >= 2.0 * PI) Arc_xOz -= 2.0 * PI;
		H_y += dy * v_y;
		ResetPos();
		MouseX = x;
		MouseY = y;
	}

	// wasd key move
	void KeyboardMove(int type)
	{
		float change_x = 0;
		float change_z = 0;
		
		if (type == 0)
		{
			change_x = -cos(Arc_xOz) * v_key;
			change_z = -sin(Arc_xOz) * v_key;
		}
		else if (type == 1)
		{
			change_x = -sin(Arc_xOz) * v_key;
			change_z = cos(Arc_xOz) * v_key;
		}
		else if (type == 2)
		{
			change_x = cos(Arc_xOz) * v_key;
			change_z = sin(Arc_xOz) * v_key;
		}
		else if (type == 3)
		{
			change_x = sin(Arc_xOz) * v_key;
			change_z = -cos(Arc_xOz) * v_key;
		}
		LookCenter.x += change_x;
		LookCenter.z += change_z;
		ResetPos();
	}
};
