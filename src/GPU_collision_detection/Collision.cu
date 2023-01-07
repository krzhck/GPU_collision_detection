#include"collision.cuh"
#include "ball.hpp"


//通用函数

__device__ float Dist(float x, float y, float z)
{
	return sqrt(x * x + y * y + z * z);
}

__device__ float Dist(Point& p)
{
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

__device__ float Multiply(Point& a, Point& b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

/*
	描述：处理与边界相撞
	参数：X范围（-X, X), Z范围(-Z, Z), Y范围(0, Y)
	返回：无
*/
__device__ void HandleCollisionBoard(Ball& ball, float XRange, float ZRange, float Height)
{
	if (ball.CurrentPlace.x - ball.Radius < -XRange)
	{
		ball.CurrentPlace.x = -XRange + ball.Radius;
		ball.CurrentSpeed.x = -ball.CurrentSpeed.x;
	}
	else if (ball.CurrentPlace.x + ball.Radius > XRange)
	{
		ball.CurrentPlace.x = XRange - ball.Radius;
		ball.CurrentSpeed.x = -ball.CurrentSpeed.x;
	}
	if (ball.CurrentPlace.z - ball.Radius < -ZRange)
	{
		ball.CurrentPlace.z = -ZRange + ball.Radius;
		ball.CurrentSpeed.z = -ball.CurrentSpeed.z;
	}
	else if (ball.CurrentPlace.z + ball.Radius > ZRange)
	{
		ball.CurrentPlace.z = ZRange - ball.Radius;
		ball.CurrentSpeed.z = -ball.CurrentSpeed.z;
	}
	if (ball.CurrentPlace.y - ball.Radius < 0)
	{
		ball.CurrentPlace.y = ball.Radius;
		ball.CurrentSpeed.y = -ball.CurrentSpeed.y;
	}
	else if (ball.CurrentPlace.y + ball.Radius > Height)
	{
		ball.CurrentPlace.y = Height - ball.Radius;
		ball.CurrentSpeed.y = -ball.CurrentSpeed.y;
	}
}


/*
	描述：处理小球自行运动和与边界碰撞
	参数：单次运动时间，X范围（-X, X), Z范围(-Z, Z), Y范围(0, Y)
	返回：无
*/
__device__ void BallMove(Ball& ball, float time, float XRange, float ZRange, float Height)
{

	ball.CurrentPlace.x = ball.CurrentPlace.x + ball.CurrentSpeed.x * time;
	ball.CurrentPlace.y = ball.CurrentPlace.y + ball.CurrentSpeed.y * time;
	ball.CurrentPlace.z = ball.CurrentPlace.z + ball.CurrentSpeed.z * time;
	HandleCollisionBoard(ball, XRange, ZRange, Height);
}

/*
	描述：判断两个球是否相撞
	参数：球a，球b
	返回：是1，否0
*/
__device__ bool JudgeCollision(Ball& a, Ball& b)
{
	float dist = 0;
	float dist_x = a.CurrentPlace.x - b.CurrentPlace.x;
	float dist_y = a.CurrentPlace.y - b.CurrentPlace.y;
	float dist_z = a.CurrentPlace.z - b.CurrentPlace.z;
	dist = Dist(dist_x, dist_y, dist_z);
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
__device__ void ChangeSpeed(Ball& a, Ball& b)
{
	//径向速度按照质量做变换，法向速度不变
	float dist = 0;
	float diff_x = b.CurrentPlace.x - a.CurrentPlace.x;
	float diff_y = b.CurrentPlace.y - a.CurrentPlace.y;
	float diff_z = b.CurrentPlace.z - a.CurrentPlace.z;
	dist = Dist(diff_x, diff_y, diff_z);

	//求径向，法向速度
	float rate_collide_a = (a.CurrentSpeed.x * diff_x + a.CurrentSpeed.y * diff_y + a.CurrentSpeed.z * diff_z) / dist / dist;
	float speed_collide_a_x = diff_x * rate_collide_a;
	float speed_collide_a_y = diff_y * rate_collide_a;
	float speed_collide_a_z = diff_z * rate_collide_a;

	float rate_collide_b = (b.CurrentSpeed.x * diff_x + b.CurrentSpeed.y * diff_y + b.CurrentSpeed.z * diff_z) / dist / dist;
	float speed_collide_b_x = diff_x * rate_collide_b;
	float speed_collide_b_y = diff_y * rate_collide_b;
	float speed_collide_b_z = diff_z * rate_collide_b;

	float unchanged_a_x = a.CurrentSpeed.x - speed_collide_a_x;
	float unchanged_a_y = a.CurrentSpeed.y - speed_collide_a_y;
	float unchanged_a_z = a.CurrentSpeed.z - speed_collide_a_z;

	float unchanged_b_x = b.CurrentSpeed.x - speed_collide_b_x;
	float unchanged_b_y = b.CurrentSpeed.y - speed_collide_b_y;
	float unchanged_b_z = b.CurrentSpeed.z - speed_collide_b_z;


	//假设b不动，a撞b，更新两者径向速度
	float speed_collide_new_a_x = (speed_collide_a_x * (a.Weight - b.Weight) + speed_collide_b_x * (2 * b.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_a_y = (speed_collide_a_y * (a.Weight - b.Weight) + speed_collide_b_y * (2 * b.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_a_z = (speed_collide_a_z * (a.Weight - b.Weight) + speed_collide_b_z * (2 * b.Weight)) / (a.Weight + b.Weight);

	float speed_collide_new_b_x = (speed_collide_a_x * (2 * a.Weight) + speed_collide_b_x * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_b_y = (speed_collide_a_y * (2 * a.Weight) + speed_collide_b_y * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_b_z = (speed_collide_a_z * (2 * a.Weight) + speed_collide_b_z * (b.Weight - a.Weight)) / (a.Weight + b.Weight);

	a.CurrentSpeed.x = speed_collide_new_a_x + unchanged_a_x;
	a.CurrentSpeed.y = speed_collide_new_a_y + unchanged_a_y;
	a.CurrentSpeed.z = speed_collide_new_a_z + unchanged_a_z;

	b.CurrentSpeed.x = speed_collide_new_b_x + unchanged_b_x;
	b.CurrentSpeed.y = speed_collide_new_b_y + unchanged_b_y;
	b.CurrentSpeed.z = speed_collide_new_b_z + unchanged_b_z;
}

/*
描述：在球之间的碰撞检测完成后，处理球的运动以及和边界的碰撞（并行）
参数：球列表，一次的时间，X范围(-X,X),Z范围(-Z,Z),Y范围(0,Y)，球个数
返回：无，但是更新球列表
*/
__global__ void UpdateBallsMove(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N)
{
	// 获取全局索引
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// 步长
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride)
	{
		BallMove(balls[i], TimeOnce, XRange, ZRange, Height);
	}

}


//暴力算法相关函数
/*
描述：暴力算法处理碰撞检测和速度更新
参数：球列表，N个球
返回：无，但是更新球列表
*/
__global__ void HandleCollisionNaive(Ball* balls, int N)
{
	// 获取全局索引
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// 步长
	int stride = blockDim.x * gridDim.x;
	for (int k = index; k < N * N; k += stride)
	{
		int i = k / N;
		int j = k % N;
		if(i < j)
		{
			if (JudgeCollision(balls[i], balls[j]))
			{
				ChangeSpeed(balls[i], balls[j]);
			}
		}
	}
}



/*
描述：GPU碰撞检测+运动更新主函数（暴力算法）
参数：球列表，一次的时间，X范围(-X,X),Z范围(-Z,Z),Y范围(0,Y)，球个数
返回：无，但是更新球列表
*/
void UpdateBallsNaiveGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N)
{
	// 申请托管内存
	int nBytes = N * sizeof(Ball);
	Ball* balls_gpu;
	cudaMallocManaged((void**)&balls_gpu, nBytes);

	// 初始化数据
	cudaMemcpy((void*)balls_gpu, (void*)balls, nBytes, cudaMemcpyHostToDevice);

	// 定义kernel的执行配置
	dim3 blockSize(256);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

	// 执行kernel
	HandleCollisionNaive << < gridSize, blockSize >> > (balls_gpu, N);
	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();

	// 执行kernel
	UpdateBallsMove <<< gridSize, blockSize >>> (balls_gpu, TimeOnce, XRange, ZRange, Height, N);
	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();

	// 记录结果
	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyDeviceToHost);

	// 释放内存
	cudaFree(balls_gpu);
}


//空间划分算法相关函数
/*
描述：初始化cells，objects数组，前者记录物体所在的格子信息（格子x，y，z的id，home还是phantom），后者记录物体id和home/phantom
参数：空的cell，phantom；球列表和个数，还有各种格子信息
返回：更新cells，objects数组和cell_num
*/
__global__ void InitCellKernel(uint32_t *cells, uint32_t *objects, Ball* balls, int N,
	float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ) 
{
	unsigned int count = 0;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
	{
		int current_cell_id = i * 8; //每个球最多在8个格子内
		int cell_info = 0;
		int object_info = 0;
		int current_count = 0;
		float x = balls[i].CurrentPlace.x;
		float y = balls[i].CurrentPlace.y;
		float z = balls[i].CurrentPlace.z;
		float radius = balls[i].Radius;

		//找到home cell
		int hash_x = (x + XRange) / GridSize;
		int hash_y = (y) / GridSize;
		int hash_z = (z + ZRange) / GridSize;
		cell_info = hash_x << 17 | hash_y << 9 | hash_z << 1 | HOME_CELL;
		object_info = i << 1 | HOME_OBJECT;
		cells[current_cell_id] = cell_info;
		objects[current_cell_id] = object_info;
		current_cell_id++;
		count++;
		current_count++;

		//找phantom
		for (int dx = -1; dx <= 1; dx++)
		{
			for (int dy = -1; dy <= 1; dy++)
			{
				for (int dz = -1; dz <= 1; dz++)
				{
					int new_hash_x = hash_x + dx;
					int new_hash_y = hash_y + dy;
					int new_hash_z = hash_z + dz;

					//自己不考虑
					if (dx == 0 && dy == 0 && dz == 0)
					{
						continue;
					}

					//越界不考虑
					if (new_hash_x < 0 || new_hash_x >= GridX ||
						new_hash_y < 0 || new_hash_y >= GridY ||
						new_hash_z < 0 || new_hash_z >= GridZ)
					{
						continue;
					}

					float relative_x = 0;
					float relative_y = 0;
					float relative_z = 0;
					if (dx == 0)
					{
						relative_x = x;
					}
					else if (dx == -1)
					{
						relative_x = hash_x * GridSize - XRange;
					}
					else
					{
						relative_x = (hash_x + 1) * GridSize - XRange;
					}

					if (dz == 0)
					{
						relative_z = z;
					}
					else if (dz == -1)
					{
						relative_z = hash_z * GridSize - ZRange;
					}
					else
					{
						relative_z = (hash_z + 1) * GridSize - ZRange;
					}

					if (dy == 0)
					{
						relative_y = y;
					}
					else if (dy == -1)
					{
						relative_y = hash_y * GridSize;
					}
					else
					{
						relative_y = (hash_y + 1) * GridSize;
					}

					relative_x -= x;
					relative_y -= y;
					relative_z -= z;

					float dist = Dist(relative_x, relative_y, relative_z);
					if (dist < radius)
					{
						int cell_info = new_hash_x << 17 | new_hash_y << 9 | new_hash_z << 1 | PHANTOM_CELL;
						int object_info = i << 1 | PHANTOM_OBJECT;
						cells[current_cell_id] = cell_info;
						objects[current_cell_id] = object_info;
						current_cell_id++;
						count++;
						current_count++;
					}
				}
			}
		}

		//补齐
		while (current_count < 8)
		{

			cells[current_cell_id] = UINT32_MAX;
			objects[current_cell_id] = i << 2;
			current_cell_id++;
			current_count++;
		}

	}

}

/*
描述：初始化cells， objects数组的主函数
参数：空的cell，phantom；球列表和个数，还有各种格子信息，线程信息
返回：更新cells，objects数组和cell_num
 */
void InitCells(uint32_t *cells, uint32_t *objects, Ball* balls, int N,
	 float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ, 
	 unsigned int num_blocks, unsigned int threads_per_block) {
	 InitCellKernel << <num_blocks, threads_per_block,
		 threads_per_block * sizeof(unsigned int) >> > (
			 cells, objects, balls, N, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ);
 }


/*
描述：计算前i和的算法
参数：原始数组，个数n
返回：原始数组变成前i个和数组
*/
__device__ void PrefixSum(uint32_t *values, unsigned int n) 
{
	int offset = 1;
	int a;
	uint32_t temp;

	//reduction
	for (int d = n / 2; d; d /= 2) 
	{
		__syncthreads();

		if (threadIdx.x < d) 
		{
			a = (threadIdx.x * 2 + 1) * offset - 1;
			values[a + offset] += values[a];
		}

		offset *= 2;
	}

	if (!threadIdx.x) 
	{
		values[n - 1] = 0;
	}

	//reverse
	for (int d = 1; d < n; d *= 2) 
	{
		__syncthreads();
		offset /= 2;

		if (threadIdx.x < d) 
		{
			a = (threadIdx.x * 2 + 1) * offset - 1;
			temp = values[a];
			values[a] = values[a + offset];
			values[a + offset] += temp;
		}
	}
}

/*
描述：对cells求前缀和
参数：cells，待更新前缀和，N个cell，偏移量
返回：更新前缀和
*/
__global__ void GetRadixSum(uint32_t *cells, uint32_t *radix_sums, int N, int shift)
 {
	 int index = threadIdx.x + blockIdx.x * blockDim.x;
	 int stride = blockDim.x * gridDim.x;
	 int num_indices = 1 << RADIX_LENGTH;


	 //初始化
	 for (int i = index; i < num_indices; i++)
	 {
		 radix_sums[i] = 0;
	 }
	 __syncthreads();


	 //求和
	 for (int i = index; i < N; i += stride)
	 {
		 //非常重要，不这样做无法有效求和
		 for (int j = 0; j < blockDim.x; j++)
		 {
			 if (threadIdx.x % blockDim.x == j)
			 {
				 int current_radix_num = (cells[i] >> shift) & (num_indices - 1);
				 radix_sums[current_radix_num] ++;
			 }
		 }

	 }
	 __syncthreads();
	 //求前缀和
	 PrefixSum(radix_sums, num_indices);
	 __syncthreads();
}

 /*
 描述：重新分配元素
 参数：cells，object数组，他们待更新的分配结果temp，前缀和数组，N个元素，偏移量，每个线程处理几个cell
 返回：更新前缀和
 */
__global__ void RearrangeCell(uint32_t *cells, uint32_t *objects, uint32_t *cells_temp, uint32_t *objects_temp, 
	 uint32_t *radix_sums, int N, int shift)
 {
	 int index = threadIdx.x + blockIdx.x * blockDim.x;
	 int stride = blockDim.x * gridDim.x;
	 int num_radices = 1 << RADIX_LENGTH;

	 if (index != 0) return;
	 //分配
	 for (int i = 0; i < N; i ++ )
	 {
		int current_radix_num = (cells[i] >> shift) & (num_radices - 1);
		cells_temp[radix_sums[current_radix_num]] = cells[i];
		objects_temp[radix_sums[current_radix_num]] = objects[i];
		radix_sums[current_radix_num] ++;
	 }
 }

/*
描述：获取排序后数组的index（cell变化的位置）
参数：cell，cell个数N,待更新的indice，待更新的indice个数
返回：无，但是更新indice数组和indice个数
*/
__global__ void GetCellIndex(uint32_t *cells, int N, uint32_t* indices, uint32_t* num_indices)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	//只能串行
	if (index != 0) return;
	num_indices[0] = 0;
	uint32_t mask = (1 << 24) - 1;
	uint32_t previous = UINT32_MAX;
	uint32_t current = UINT32_MAX;
	for (int i = 0; i < N; i++)
	{
		current = mask & (cells[i] >> 1);
		if (previous == UINT32_MAX)
		{
			previous = current;
		}
		if (previous != current)
		{
			indices[num_indices[0]] = i;
			num_indices[0]++;
		}
		previous = current;
	}
	indices[num_indices[0]] = N;
	num_indices[0]++;
}


/*
描述：对cell，object做基数排序，并且获取index（cell变化的位置）
参数：cell，object数组；他们的temp形式用于排序；待求的前缀和数组；cell个数；待求的index数组和长度；线程情况
返回：无，但是更新cell，object数组，还有index数组和其长度
*/
void SortCells(uint32_t *cells, uint32_t *objects, uint32_t *cells_temp, uint32_t *objects_temp,
	uint32_t *radix_sums, int N, uint32_t* indices, uint32_t* num_indices,
	unsigned int num_blocks, unsigned int threads_per_block)
{
	uint32_t *cells_swap;
	uint32_t *objects_swap;
	for (int i = 0; i < 32; i += RADIX_LENGTH)
	{
		//求前缀和
		GetRadixSum <<< num_blocks, threads_per_block >>> (cells, radix_sums, N, i);

		//用前缀和重新分配
		RearrangeCell << < num_blocks, threads_per_block >> > (cells, objects, cells_temp, objects_temp,
			radix_sums, N, i);
		
		//交换原始和temp
		cells_swap = cells;
		cells = cells_temp;
		cells_temp = cells_swap;
		objects_swap = objects;
		objects = objects_temp;
		objects_temp = objects_swap;
	}
	GetCellIndex << < num_blocks, threads_per_block >> > (cells, N, indices, num_indices);
}

/*
描述：cuda碰撞检测和处理函数
参数：cell和object数组，ball数组，球和cell的个数，index数组和个数，线程信息，场景的各种限制和格子信息
返回：无，但是进行碰撞检测和处理
*/
__global__ void HandleCollisionCuda(uint32_t *cells, uint32_t *objects, Ball* balls, int num_balls, int num_cells,
	uint32_t* indices, uint32_t num_indices, unsigned int group_per_thread,
	float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int group_num = 0; group_num < group_per_thread; group_num++)
	{
		//判断是否越界，找到处理的start，end
		int cell_id = index * group_per_thread + group_num;
		if (cell_id >= num_indices)
		{
			break;
		}
		int end = indices[cell_id];
		int start = 0;
		if (cell_id == 0)
		{
			start = 0;
		}
		else
		{
			start = indices[cell_id - 1];
		}

		//找其中home的个数
		int home_num = 0;
		for (int i = start; i < end; i++)
		{
			int type = cells[i] & 1;
			if (type == HOME_CELL)
			{
				home_num++;
			}
			else
			{
				break;
			}
		}

		//遍历碰撞检测
		for (int i = start; i < start + home_num; i++)
		{
			if (cells[i] == UINT32_MAX) break;
			int ball_i = (objects[i] >> 1) & 65535;
			
			for (int j = i + 1; j < end; j++)
			{
				if (cells[j] == UINT32_MAX) break;
				int ball_j = (objects[j] >> 1) & 65535;

				//2个home，直接碰撞检测
				if (j < start + home_num)
				{
					if (JudgeCollision(balls[ball_i], balls[ball_j]))
					{
						ChangeSpeed(balls[ball_i], balls[ball_j]);
					}
				}

				//home和phantom，需要判重
				else
				{
					int home_i = (cells[i] >> 1) & ((1 << 24) - 1);
					int j_x = (balls[ball_j].CurrentPlace.x + XRange) / GridSize;
					int j_y = balls[ball_j].CurrentPlace.y / GridSize;
					int j_z = (balls[ball_j].CurrentPlace.z + ZRange) / GridSize;
					int home_j = j_x << 16 | j_y << 8 | j_z;

					//只有这样才可以
					if(home_i < home_j)
					{
						if (JudgeCollision(balls[ball_i], balls[ball_j]))
						{
							ChangeSpeed(balls[ball_i], balls[ball_j]);
						}
					}
				}
			}
		}

	}



}

/*
描述：碰撞检测和处理函数
参数：cell和object数组，ball数组，球和cell的个数，index数组和个数，线程信息，场景的各种限制和格子信息
返回：无，但是进行碰撞检测和处理
*/
void HandleCollision(uint32_t *cells, uint32_t *objects, Ball* balls, int num_balls, int num_cells,
	uint32_t* indices, uint32_t num_indices, unsigned int num_blocks, unsigned int threads_per_block,
	float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ)
{
	unsigned int threads_total = num_blocks * threads_per_block;
	unsigned int group_per_thread = num_indices / threads_total + 1;
	HandleCollisionCuda << <num_blocks, threads_per_block >> > (cells, objects, balls, num_balls, num_cells,
		indices, num_indices, group_per_thread,
		XRange, ZRange, Height, GridSize, GridX, GridY, GridZ);
}

/*
描述：空间划分算法处理碰撞检测和速度更新（主函数）
参数：球列表，X范围(-X,X),Z范围(-Z,Z),Y范围(0,Y)，格子大小，X格子个数，Y格子个数，Z格子个数，N个球
返回：无，但是更新球列表
*/
void HandleCollisionGrid(Ball* balls, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N,
	unsigned int num_blocks, unsigned int threads_per_block)
{

	//申请内存
	unsigned int cell_size = N * 8 * sizeof(uint32_t);

	uint32_t *cells_gpu;
	uint32_t *cells_gpu_temp;
	uint32_t *objects_gpu;
	uint32_t *objects_gpu_temp;
	uint32_t *indices_gpu;
	uint32_t *indices_num_gpu;
	uint32_t *radix_sums_gpu;

	int num_radices = 1 << RADIX_LENGTH;

	cudaMalloc((void **)&cells_gpu, cell_size);
	cudaMalloc((void **)&cells_gpu_temp, cell_size);
	cudaMalloc((void **)&objects_gpu, cell_size);
	cudaMalloc((void **)&objects_gpu_temp, cell_size);
	cudaMalloc((void **)&indices_gpu, cell_size);
	cudaMalloc((void **)&indices_num_gpu, sizeof(uint32_t));
	cudaMalloc((void **)&radix_sums_gpu, num_radices * sizeof(uint32_t));


	
	//初始化cell和object
	InitCells(cells_gpu, objects_gpu, balls, N,
		XRange, ZRange, Height, GridSize, GridX, GridY, GridZ,
		num_blocks, threads_per_block);


	//基数排序
	SortCells(cells_gpu, objects_gpu, cells_gpu_temp, objects_gpu_temp, radix_sums_gpu, 
		8 * N, indices_gpu, indices_num_gpu, num_blocks, threads_per_block);
	


	uint32_t indices_num;
	cudaMemcpy((void*)&indices_num, (void*)indices_num_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	
	HandleCollision(cells_gpu, objects_gpu, balls, N, 8 * N, indices_gpu, indices_num,
		num_blocks, threads_per_block,
		XRange, ZRange, Height, GridSize, GridX, GridY, GridZ);
	

	cudaFree(cells_gpu);
	cudaFree(cells_gpu_temp);
	cudaFree(objects_gpu);
	cudaFree(objects_gpu_temp);
	cudaFree(indices_gpu);
	cudaFree(indices_num_gpu);
	cudaFree(radix_sums_gpu);
}


/*
描述：GPU碰撞检测+运动更新主函数（空间划分算法）
参数：球列表，一次的时间，X范围(-X,X),Z范围(-Z,Z),Y范围(0,Y)，一个格子大小，X,Y,Z的格子个数，球个数
返回：无，但是更新球列表
*/
void UpdateBallsGridGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N)
{
	//设置，计算需要多少block和thread
	unsigned int num_blocks = 128;
	unsigned int threads_per_block = 512;
	unsigned int object_size = (N - 1) / threads_per_block + 1;
	if (object_size < num_blocks) {
		num_blocks = object_size;
	}

	Ball* balls_gpu;
	unsigned int nBytes = N * sizeof(Ball);
	cudaMalloc((void**)&balls_gpu, nBytes);


	// 初始化数据
	cudaMemcpy((void*)balls_gpu, (void*)balls, nBytes, cudaMemcpyHostToDevice);

	// 执行kernel
	HandleCollisionGrid(balls_gpu, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ, N, num_blocks, threads_per_block);
	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();

	// 执行kernel
	UpdateBallsMove << < num_blocks, threads_per_block>> > (balls_gpu, TimeOnce, XRange, ZRange, Height, N);
	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();

	// 记录结果
	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyDeviceToHost);

	// 释放内存
	cudaFree(balls_gpu);
}