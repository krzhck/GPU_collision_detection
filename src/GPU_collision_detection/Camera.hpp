#pragma once
#include<math.h>
#include"point.hpp"
using namespace std;


class Camera
{
public:
	Point CurrentPlace; //当前相机所在位置
	Point LookCenter; //当前的视点中心，y坐标一定是0
	float R_Horizontal; //XOZ平面的半径
	float Arc_Horizontal; //XOZ平面的弧度（0-2pi）
	float H_Vertical; //Y轴高度
	int MouseX; //上次鼠标的位置
	int MouseY;
	const float K_Horizontal = 0.002; //水平移动速度
	const float K_Vertical = 0.03; //垂直移动速度
	const float K_Translate = 0.2; //平移速度
public:
	Camera(){}

	void Init(float R, float start_height)
	{
		R_Horizontal = R;
		Arc_Horizontal = 0;
		H_Vertical = start_height;
		ResetCurrentPlace();
		MouseX = -1;
		MouseY = -1;
		LookCenter.SetPlace(0, 0, 0);
	}



	void ResetCurrentPlace()
	{
		float x = R_Horizontal * cos(Arc_Horizontal) + LookCenter.x;
		float y = H_Vertical + LookCenter.y;
		float z = R_Horizontal * sin(Arc_Horizontal) + LookCenter.z;
		CurrentPlace.SetPlace(x, y, z);
	}

	//处理按下鼠标事件
	void MouseDown(int x, int y)
	{
		MouseX = x;
		MouseY = y;
	}

	//处理鼠标移动事件:横向移动改角度，纵向移动改高度
	void MouseMove(int x, int y)
	{
		int dx = x - MouseX;
		int dy = y - MouseY;
		Arc_Horizontal = Arc_Horizontal + dx * K_Horizontal;
		while (Arc_Horizontal < 0) Arc_Horizontal += 2.0 * PI;
		while (Arc_Horizontal >= 2.0 * PI) Arc_Horizontal -= 2.0 * PI;
		H_Vertical += dy * K_Vertical;
		ResetCurrentPlace();
		MouseX = x;
		MouseY = y;
	}

	//处理键盘移动事件，更改水平位置和视点中心
	void KeyboardMove(int type)
	{
		float change_x = 0;
		float change_z = 0;
		
		//0123代表WASD
		if (type == 0)
		{
			change_x = -cos(Arc_Horizontal) * K_Translate;
			change_z = -sin(Arc_Horizontal) * K_Translate;
		}
		else if (type == 1)
		{
			change_x = -sin(Arc_Horizontal) * K_Translate;
			change_z = cos(Arc_Horizontal) * K_Translate;
		}
		else if (type == 2)
		{
			change_x = cos(Arc_Horizontal) * K_Translate;
			change_z = sin(Arc_Horizontal) * K_Translate;
		}
		else if (type == 3)
		{
			change_x = sin(Arc_Horizontal) * K_Translate;
			change_z = -cos(Arc_Horizontal) * K_Translate;
		}
		LookCenter.x += change_x;
		LookCenter.z += change_z;
		ResetCurrentPlace();
	}
	
};