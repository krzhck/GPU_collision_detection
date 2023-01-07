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
	��������ʼ��λ����Ϣ
	������x��Χ��ʵ����-x��x����y��Χ��0��y����z��Χ��-z��z����ÿ�������������ʵ��num���������򣩣������뾶��ģʽ
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
			cout << total_num << "����ײ���ƽ����ʱ" << average_time << "ms" << endl;
			total_num = 0;
			total_time = 0;
		}
	}

	/*
		����������֮�����ײ�����ɺ󣬴�������˶��Լ��ͱ߽����ײ�����У�
		��������
		���أ���
	*/
	void UpdateBallsMove()
	{
		for (int i = 0; i < NBalls; i++)
		{
			balls[i].Move(TimeOnce, XRange, ZRange, Height);
		}
	}


	/*
	��������֮����ײ��⣨n^2�㷨�����У�
	��������
	���أ���
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
	��������ȡ���Ӧ��grid home����
	��������
	���أ�home����
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
	��������ȡ���Ӧ��grid����
	��������, home����
	���أ�phantom����
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
	��������֮����ײ��⣨������٣����У�
	��������
	���أ���
	*/
	void CollisionGrid()
	{
		vector<cell> cell_list;
		vector<int> cell_list_index; //��¼��i������Ŀ�ʼ
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

		//����grid cell
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

		//��������
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

		//��ײ���
		for (int i = 0; i < GridX * GridY * GridZ; i++)
		{

			for (int j = 0; j < sorted_cells[2 * i].size(); j++)
			{
				//�ȱȽ�home��home����������
				for (int k = j + 1; k < sorted_cells[2 * i].size(); k++)
				{
					int object_a = sorted_cells[2 * i][j].object_num;
					int object_b = sorted_cells[2 * i][k].object_num;
					if (JudgeCollision(balls[object_a], balls[object_b]))
					{
						ChangeSpeed(balls[object_a], balls[object_b]);
					}

				}

				//�ٱȽ�A��home��B��phantom����Ҫ���������ظ���
				//1��֮ǰA��home��B��home�ȽϹ���2��֮ǰA��phantom��B��home�ȽϹ�
				//�жϱ�׼��B��home���С�ڵ���A��home��ţ�����A��home+phantom�����ҵ���Bhome���һ�µ�
				//���۸��Ӷ�O��1��������ʱ��
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