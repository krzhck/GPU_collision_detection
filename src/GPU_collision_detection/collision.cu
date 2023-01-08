#include"collision.cuh"
#include "ball.hpp"


// utilities

__device__ float Dist(float x, float y, float z)
{
	return sqrt(x * x + y * y + z * z);
}

__device__ float Dist(Coord & p)
{
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

__device__ float Multiply(Coord & a, Coord& b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

// ball and wall collision
__device__ void WallCollision(Ball& ball, float XRange, float ZRange, float Height)
{
	if (ball.Pos.x - ball.Radius < -XRange)
	{
		ball.Pos.x = -XRange + ball.Radius;
		ball.Speed.x = -ball.Speed.x;
	}
	else if (ball.Pos.x + ball.Radius > XRange)
	{
		ball.Pos.x = XRange - ball.Radius;
		ball.Speed.x = -ball.Speed.x;
	}
	if (ball.Pos.z - ball.Radius < -ZRange)
	{
		ball.Pos.z = -ZRange + ball.Radius;
		ball.Speed.z = -ball.Speed.z;
	}
	else if (ball.Pos.z + ball.Radius > ZRange)
	{
		ball.Pos.z = ZRange - ball.Radius;
		ball.Speed.z = -ball.Speed.z;
	}
	if (ball.Pos.y - ball.Radius < 0)
	{
		ball.Pos.y = ball.Radius;
		ball.Speed.y = -ball.Speed.y;
	}
	else if (ball.Pos.y + ball.Radius > Height)
	{
		ball.Pos.y = Height - ball.Radius;
		ball.Speed.y = -ball.Speed.y;
	}
}

// if 2 balls collide
__device__ bool IsCollision(Ball& a, Ball& b)
{
	float dist = 0;
	float dist_x = a.Pos.x - b.Pos.x;
	float dist_y = a.Pos.y - b.Pos.y;
	float dist_z = a.Pos.z - b.Pos.z;
	dist = Dist(dist_x, dist_y, dist_z);
	if (dist < a.Radius + b.Radius)
	{
		return true;
	}
	else
	{
		return false;
	}
}

// single ball position update
__device__ void UpdateSingleBall(Ball& ball, float time, float XRange, float ZRange, float Height)
{

	ball.Pos.x = ball.Pos.x + ball.Speed.x * time;
	ball.Pos.y = ball.Pos.y + ball.Speed.y * time;
	ball.Pos.z = ball.Pos.z + ball.Speed.z * time;
	WallCollision(ball, XRange, ZRange, Height);
}

// update speed of 2 balls after ball and ball collision
__device__ void UpdateSpeed(Ball& a, Ball& b)
{
	float dist = 0;
	float diff_x = b.Pos.x - a.Pos.x;
	float diff_y = b.Pos.y - a.Pos.y;
	float diff_z = b.Pos.z - a.Pos.z;
	dist = Dist(diff_x, diff_y, diff_z);

	float rate_collide_a = (a.Speed.x * diff_x + a.Speed.y * diff_y + a.Speed.z * diff_z) / dist / dist;
	float speed_collide_a_x = diff_x * rate_collide_a;
	float speed_collide_a_y = diff_y * rate_collide_a;
	float speed_collide_a_z = diff_z * rate_collide_a;

	float rate_collide_b = (b.Speed.x * diff_x + b.Speed.y * diff_y + b.Speed.z * diff_z) / dist / dist;
	float speed_collide_b_x = diff_x * rate_collide_b;
	float speed_collide_b_y = diff_y * rate_collide_b;
	float speed_collide_b_z = diff_z * rate_collide_b;

	float unchanged_a_x = a.Speed.x - speed_collide_a_x;
	float unchanged_a_y = a.Speed.y - speed_collide_a_y;
	float unchanged_a_z = a.Speed.z - speed_collide_a_z;

	float unchanged_b_x = b.Speed.x - speed_collide_b_x;
	float unchanged_b_y = b.Speed.y - speed_collide_b_y;
	float unchanged_b_z = b.Speed.z - speed_collide_b_z;

	float speed_collide_new_a_x = (speed_collide_a_x * (a.Weight - b.Weight) + speed_collide_b_x * (2 * b.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_a_y = (speed_collide_a_y * (a.Weight - b.Weight) + speed_collide_b_y * (2 * b.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_a_z = (speed_collide_a_z * (a.Weight - b.Weight) + speed_collide_b_z * (2 * b.Weight)) / (a.Weight + b.Weight);

	float speed_collide_new_b_x = (speed_collide_a_x * (2 * a.Weight) + speed_collide_b_x * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_b_y = (speed_collide_a_y * (2 * a.Weight) + speed_collide_b_y * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_b_z = (speed_collide_a_z * (2 * a.Weight) + speed_collide_b_z * (b.Weight - a.Weight)) / (a.Weight + b.Weight);

	a.Speed.x = speed_collide_new_a_x + unchanged_a_x;
	a.Speed.y = speed_collide_new_a_y + unchanged_a_y;
	a.Speed.z = speed_collide_new_a_z + unchanged_a_z;

	b.Speed.x = speed_collide_new_b_x + unchanged_b_x;
	b.Speed.y = speed_collide_new_b_y + unchanged_b_y;
	b.Speed.z = speed_collide_new_b_z + unchanged_b_z;
}

// update all balls position (including wall collision) after ball collision detection 
__global__ void UpdateAllBalls(Ball* balls, float RefreshInterval, float XRange, float ZRange, float Height, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride)
	{
		UpdateSingleBall(balls[i], RefreshInterval, XRange, ZRange, Height);
	}
}


// spatial subdivision related
__global__ void InitCellKernel(uint32_t *cells, uint32_t *objects, Ball* balls, int N, float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ) 
{
	unsigned int count = 0;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
	{
		int current_cell_id = i * 8; //每个球最多在8�?格子�?
		int cell_info = 0;
		int object_info = 0;
		int current_count = 0;
		float x = balls[i].Pos.x;
		float y = balls[i].Pos.y;
		float z = balls[i].Pos.z;
		float radius = balls[i].Radius;

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

					//�?己不考虑
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

__global__ void GetRadixSum(uint32_t *cells, uint32_t *radix_sums, int N, int shift)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int num_indices = 1 << RADIX_LENGTH;


	//初�?�化
	for (int i = index; i < num_indices; i++)
	{
		radix_sums[i] = 0;
	}
	__syncthreads();


	 //求和
	for (int i = index; i < N; i += stride)
	{
		//非常重�?�，不这样做无法有效求和
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
	//求前缀�?
	PrefixSum(radix_sums, num_indices);
	__syncthreads();
}

__global__ void RearrangeCell(uint32_t *cells, uint32_t *objects, uint32_t *cells_temp, uint32_t *objects_temp, uint32_t *radix_sums, int N, int shift)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
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

__global__ void GetCellIndex(uint32_t *cells, int N, uint32_t* indices, uint32_t* num_indices)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	//�?能串�?
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


// radix sort
void SortCells(uint32_t *cells, uint32_t *objects, uint32_t *cells_temp, uint32_t *objects_temp,
	uint32_t *radix_sums, int N, uint32_t* indices, uint32_t* num_indices,
	unsigned int num_blocks, unsigned int threads_per_block)
{
	uint32_t *cells_swap;
	uint32_t *objects_swap;
	for (int i = 0; i < 32; i += RADIX_LENGTH)
	{
		GetRadixSum <<< num_blocks, threads_per_block >>> (cells, radix_sums, N, i);

		RearrangeCell <<< num_blocks, threads_per_block >>> (cells, objects, cells_temp, objects_temp, radix_sums, N, i);
		
		cells_swap = cells;
		cells = cells_temp;
		cells_temp = cells_swap;
		objects_swap = objects;
		objects = objects_temp;
		objects_temp = objects_swap;
	}
	GetCellIndex <<< num_blocks, threads_per_block >>> (cells, N, indices, num_indices);
}

// handle ball collision using cuda, called by spatial subdivision
__global__ void HandleCollision(uint32_t *cells, uint32_t *objects, Ball* balls, int num_balls, int num_cells,
	uint32_t* indices, uint32_t num_indices, unsigned int group_per_thread,
	float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int group_num = 0; group_num < group_per_thread; group_num++)
	{
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

		for (int i = start; i < start + home_num; i++)
		{
			if (cells[i] == UINT32_MAX) break;
			int ball_i = (objects[i] >> 1) & 65535;
			
			for (int j = i + 1; j < end; j++)
			{
				if (cells[j] == UINT32_MAX) break;
				int ball_j = (objects[j] >> 1) & 65535;

				if (j < start + home_num)
				{
					if (IsCollision(balls[ball_i], balls[ball_j]))
					{
						UpdateSpeed(balls[ball_i], balls[ball_j]);
					}
				}
				else
				{
					int home_i = (cells[i] >> 1) & ((1 << 24) - 1);
					int j_x = (balls[ball_j].Pos.x + XRange) / GridSize;
					int j_y = balls[ball_j].Pos.y / GridSize;
					int j_z = (balls[ball_j].Pos.z + ZRange) / GridSize;
					int home_j = j_x << 16 | j_y << 8 | j_z;

					if(home_i < home_j)
					{
						if (IsCollision(balls[ball_i], balls[ball_j]))
						{
							UpdateSpeed(balls[ball_i], balls[ball_j]);
						}
					}
				}
			}
		}

	}
}

// algorithm main body
void SpatialSubdivision(Ball* balls, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N,
	unsigned int num_blocks, unsigned int threads_per_block)
{
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

	// initialize cells and objects
	InitCellKernel <<< num_blocks, threads_per_block, threads_per_block * sizeof(unsigned int) >>> (cells_gpu, objects_gpu, balls, N, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ);

	// radix sort
	SortCells(cells_gpu, objects_gpu, cells_gpu_temp, objects_gpu_temp, radix_sums_gpu, 
		8 * N, indices_gpu, indices_num_gpu, num_blocks, threads_per_block);
	
	uint32_t indices_num;
	cudaMemcpy((void*)&indices_num, (void*)indices_num_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	
	unsigned int threads_total = num_blocks * threads_per_block;
	unsigned int group_per_thread = indices_num / threads_total + 1;
	HandleCollision <<< num_blocks, threads_per_block >>> (cells_gpu, objects_gpu, balls, N, 8 * N, indices_gpu, indices_num, group_per_thread, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ);
	
	cudaFree(cells_gpu);
	cudaFree(cells_gpu_temp);
	cudaFree(objects_gpu);
	cudaFree(objects_gpu_temp);
	cudaFree(indices_gpu);
	cudaFree(indices_num_gpu);
	cudaFree(radix_sums_gpu);
}

// entry function
// collision detection + movement update
void CollisionDetection(Ball* balls, float RefreshInterval, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N)
{
	unsigned int num_blocks = 128;
	unsigned int threads_per_block = 512;
	unsigned int object_size = (N - 1) / threads_per_block + 1;
	if (object_size < num_blocks) {
		num_blocks = object_size;
	}

	Ball* balls_gpu;
	unsigned int nBytes = N * sizeof(Ball);
	cudaMalloc((void**)&balls_gpu, nBytes);

	cudaMemcpy((void*)balls_gpu, (void*)balls, nBytes, cudaMemcpyHostToDevice);

	// update status for all balls
	UpdateAllBalls <<< num_blocks, threads_per_block >>> (balls_gpu, RefreshInterval, XRange, ZRange, Height, N);
	cudaDeviceSynchronize();

	// collision detection using spatial subdivison on GPU
	SpatialSubdivision(balls_gpu, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ, N, num_blocks, threads_per_block);
	cudaDeviceSynchronize();

	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyDeviceToHost);

	cudaFree(balls_gpu);
}