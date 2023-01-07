#pragma once
#include<math.h>
#include<GL/glut.h>
#include<vector>
#include<time.h>
#include"Point.hpp"
#include"Board.hpp"
#include"Ball.hpp"
#include"Collision.cuh"
using namespace std;

#define NAIVE_CPU 0
#define NAIVE_GPU 1
#define FAST_CPU 2
#define FAST_GPU 3
#define HOME_CELL 0
#define PHANTOM_CELL 1

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
	int Mode;
	float GridSize;
	int GridX, GridY, GridZ;


	/*
	描述：初始化位置信息
	参数：x范围（实际是-x到x），y范围（0到y），z范围（-z到z），每个轴上球个数（实际num的立方个球），球最大半径，模式
	*/
	void Init(float x, float y, float z, int num, float max_radius, float time_once, int mode)
	{
		XRange = x;
		ZRange = z;
		Height = y;
		Num = num;
		MaxRadius = max_radius;
		TimeOnce = time_once;
		Mode = mode;
		NBalls = num * num * num;
		balls = new Ball[NBalls];
		GridSize = max_radius * 1.5;
		GridX = ceil(XRange * 2 / GridSize);
		GridY = ceil(Height / GridSize);
		GridZ = ceil(ZRange * 2 / GridSize);

	}

	void InitBalls()
	{
		//小球的纹理，材质，颜色
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
		描述：绘制所有球
		参数：无
		返回：无
	*/
	void DrawBalls()
	{
		for (int i = 0; i < NBalls; i++)
		{
			balls[i].DrawSelf();
		}
	}

	/*
		描述：判断两个球是否相撞
		参数：球a，球b
		返回：是1，否0
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
		描述：两球相撞后更新速度
		参数：球a，球b
		返回：无
	*/
	void ChangeSpeed(Ball& a, Ball& b)
	{
		//径向速度按照质量做变换，法向速度不变
		Point diff = b.CurrentPlace - a.CurrentPlace;
		float dist = diff.Dist();
		
		//求径向，法向速度
		Point speed_collide_a = diff * (a.CurrentSpeed * diff / dist / dist);
		Point speed_collide_b = diff * (b.CurrentSpeed * diff / dist / dist);
		Point unchanged_a = a.CurrentSpeed - speed_collide_a;
		Point unchanged_b = b.CurrentSpeed - speed_collide_b;
		
		//假设b不动，a撞b，更新两者径向速度
		Point speed_collide_new_a = (speed_collide_a * (a.Weight - b.Weight) + speed_collide_b * (2 * b.Weight)) / (a.Weight + b.Weight);
		Point speed_collide_new_b = (speed_collide_a * (2 * a.Weight) + speed_collide_b * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
		Point speed_new_a = speed_collide_new_a + unchanged_a;
		Point speed_new_b = speed_collide_new_b + unchanged_b;
		a.CurrentSpeed = speed_new_a;
		b.CurrentSpeed = speed_new_b;
	}

	//更新球的运动--主函数
	void UpdateBalls()
	{
		static int total_num = 0;
		static float total_time = 0;

		clock_t start, end;
		start = clock();

		if (Mode == NAIVE_CPU)
		{
			CollisionNaive();
			UpdateBallsMove();
		}
		else if (Mode == NAIVE_GPU)
		{
			UpdateBallsNaiveGPU(balls, TimeOnce, XRange, ZRange, Height, NBalls);
		}
		else if (Mode == FAST_CPU)
		{
			CollisionGrid();
			UpdateBallsMove();
		}
		else if (Mode == FAST_GPU)
		{
			UpdateBallsGridGPU(balls, TimeOnce, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ, NBalls);
		}
		end = clock();
		float duration = float(end - start) / CLOCKS_PER_SEC * 1000;
		total_num++;
		total_time += duration;
		if (total_num == 10000)
		{
			float average_time = total_time / total_num;
			cout << total_num << "次碰撞检测平均耗时" << average_time << "ms" << endl;
			total_num = 0;
			total_time = 0;
		}
	}

	/*
		描述：在球之间的碰撞检测完成后，处理球的运动以及和边界的碰撞（串行）
		参数：无
		返回：无
	*/
	void UpdateBallsMove()
	{
		for (int i = 0; i < NBalls; i++)
		{
			balls[i].Move(TimeOnce, XRange, ZRange, Height);
		}
	}


	/*
	描述：球之间碰撞检测（n^2算法，串行）
	参数：无
	返回：无
	*/
	void CollisionNaive()
	{
		for (int i = 0; i < NBalls - 1; i++)
		{
			for (int j = i + 1; j < NBalls; j++)
			{
				if (JudgeCollision(balls[i], balls[j]))
				{
					ChangeSpeed(balls[i], balls[j]);
				}
			}
		}
	}


	/*
	描述：获取球对应的grid home坐标
	参数：球
	返回：home坐标
	*/
	int GetHomeNum(Ball& ball)
	{
		int place_x = floor((ball.CurrentPlace.x + XRange) / GridSize);
		int place_y = floor(ball.CurrentPlace.y / GridSize);
		int place_z = floor((ball.CurrentPlace.x + ZRange) / GridSize);
		int num = place_x * GridY * GridZ + place_y * GridZ + place_z;
		return num;
	}

	/*
	描述：获取球对应的grid坐标
	参数：球, home坐标
	返回：phantom坐标
	*/
	vector<int> GetPhantomNums(Ball& ball, int home_num)
	{
		vector<int> phantom_nums;
		phantom_nums.clear();
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					int current_num = home_num + i * GridY * GridZ + j * GridZ + k;
					if (current_num < 0 || current_num >= GridX * GridY * GridZ)
					{
						continue;
					}
					int home_x = home_num / (GridY * GridZ);
					int home_y = (home_num - home_x * GridY * GridZ) / GridZ;
					int home_z = home_num % GridZ;

					Point relative;
					if (i == 0)
					{
						relative.x = ball.CurrentPlace.x;
					}
					else if (i == -1)
					{
						relative.x = home_x * GridSize - XRange;
					}
					else
					{
						relative.x = (home_x + 1) * GridSize - XRange;
					}

					if (k == 0)
					{
						relative.z = ball.CurrentPlace.z;
					}
					else if (k == -1)
					{
						relative.z = home_z * GridSize - ZRange;
					}
					else
					{
						relative.z = (home_z + 1) * GridSize - ZRange;
					}

					if (j == 0)
					{
						relative.y = ball.CurrentPlace.y;
					}
					else if (j == -1)
					{
						relative.y = home_y * GridSize;
					}
					else
					{
						relative.y = (home_y + 1) * GridSize;
					}

					float dist = (ball.CurrentPlace - relative).Dist();
					if (dist > ball.Radius)
					{
						phantom_nums.push_back(current_num);
					}
				}
			}
		}
		return phantom_nums;
	}

	struct cell
	{
		int object_num;
		int cell_type;
		int cell_num;
	};

	/*
	描述：球之间碰撞检测（网格加速，串行）
	参数：无
	返回：无
	*/
	void CollisionGrid()
	{
		vector<cell> cell_list;
		vector<int> cell_list_index; //记录第i个物体的开始
		cell_list.clear();
		cell_list_index.clear();

		vector<vector<cell>> sorted_cells;
		sorted_cells.clear();
		for (int i = 0; i < GridX * GridY * GridZ; i++)
		{
			vector<cell> nova_grid_home;
			vector<cell> nova_grid_phantom;

			nova_grid_home.clear();
			nova_grid_phantom.clear();
			sorted_cells.push_back(nova_grid_home);
			sorted_cells.push_back(nova_grid_phantom);

		}

		//建立grid cell
		for (int i = 0; i < NBalls; i++)
		{
			cell_list_index.push_back(cell_list.size());
			int home_num = GetHomeNum(balls[i]);
			vector<int> phantom_nums = GetPhantomNums(balls[i], home_num);

			cell new_cell;
			new_cell.cell_type = HOME_CELL;
			new_cell.cell_num = home_num;
			new_cell.object_num = i;
			cell_list.push_back(new_cell);

			for (int j = 0; j < phantom_nums.size(); j++)
			{
				cell new_cell;
				new_cell.cell_type = PHANTOM_CELL;
				new_cell.cell_num = phantom_nums[j];
				new_cell.object_num = i;
				cell_list.push_back(new_cell);
			}
		}
		cell_list_index.push_back(cell_list.size());

		//基数排序
		for (int i = 0; i < cell_list.size(); i++)
		{
			if (cell_list[i].cell_type == HOME_CELL)
			{
				sorted_cells[2 * cell_list[i].cell_num].push_back(cell_list[i]);
			}
			else
			{
				sorted_cells[2 * cell_list[i].cell_num + 1].push_back(cell_list[i]);
			}
		}

		//碰撞检测
		for (int i = 0; i < GridX * GridY * GridZ; i++)
		{

			for (int j = 0; j < sorted_cells[2 * i].size(); j++)
			{
				//先比较home和home，无需判重
				for (int k = j + 1; k < sorted_cells[2 * i].size(); k++)
				{
					int object_a = sorted_cells[2 * i][j].object_num;
					int object_b = sorted_cells[2 * i][k].object_num;
					if (JudgeCollision(balls[object_a], balls[object_b]))
					{
						ChangeSpeed(balls[object_a], balls[object_b]);
					}

				}

				//再比较A的home和B的phantom，需要处理两类重复：
				//1是之前A的home和B的home比较过，2是之前A的phantom和B的home比较过
				//判断标准：B的home标号小于等于A的home标号，而且A的home+phantom里能找到和Bhome标号一致的
				//理论复杂度O（1），常数时间
				for (int k = 0; k < sorted_cells[2 * i + 1].size(); k++)
				{
					int object_a = sorted_cells[2 * i][j].object_num;
					int object_b = sorted_cells[2 * i + 1][k].object_num;

					int home_a = i;
					int home_b = cell_list[cell_list_index[object_b]].cell_num;
					bool repeat = 0;
					if (home_b <= home_a)
					{
						int start = cell_list_index[object_a];
						int end = cell_list_index[object_a + 1];
						for (int ii = start; ii < end; ii++)
						{
							int current_cell = cell_list[ii].cell_num;
							if (current_cell == home_b)
							{
								repeat = 1;
								break;
							}
						}
					}
					if (repeat == 0)
					{
						if (JudgeCollision(balls[object_a], balls[object_b]))
						{
							ChangeSpeed(balls[object_a], balls[object_b]);
						}
					}
					
				}
			}
		}

	}

};