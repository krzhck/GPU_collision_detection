#include"collision.cuh"
#include "ball.hpp"


//ͨ�ú���

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
	������������߽���ײ
	������X��Χ��-X, X), Z��Χ(-Z, Z), Y��Χ(0, Y)
	���أ���
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
	����������С�������˶�����߽���ײ
	�����������˶�ʱ�䣬X��Χ��-X, X), Z��Χ(-Z, Z), Y��Χ(0, Y)
	���أ���
*/
__device__ void BallMove(Ball& ball, float time, float XRange, float ZRange, float Height)
{

	ball.CurrentPlace.x = ball.CurrentPlace.x + ball.CurrentSpeed.x * time;
	ball.CurrentPlace.y = ball.CurrentPlace.y + ball.CurrentSpeed.y * time;
	ball.CurrentPlace.z = ball.CurrentPlace.z + ball.CurrentSpeed.z * time;
	HandleCollisionBoard(ball, XRange, ZRange, Height);
}

/*
	�������ж��������Ƿ���ײ
	��������a����b
	���أ���1����0
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
	������������ײ������ٶ�
	��������a����b
	���أ���
*/
__device__ void ChangeSpeed(Ball& a, Ball& b)
{
	//�����ٶȰ����������任�������ٶȲ���
	float dist = 0;
	float diff_x = b.CurrentPlace.x - a.CurrentPlace.x;
	float diff_y = b.CurrentPlace.y - a.CurrentPlace.y;
	float diff_z = b.CurrentPlace.z - a.CurrentPlace.z;
	dist = Dist(diff_x, diff_y, diff_z);

	//���򣬷����ٶ�
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


	//����b������aײb���������߾����ٶ�
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
����������֮�����ײ�����ɺ󣬴�������˶��Լ��ͱ߽����ײ�����У�
���������б�һ�ε�ʱ�䣬X��Χ(-X,X),Z��Χ(-Z,Z),Y��Χ(0,Y)�������
���أ��ޣ����Ǹ������б�
*/
__global__ void UpdateBallsMove(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N)
{
	// ��ȡȫ������
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// ����
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride)
	{
		BallMove(balls[i], TimeOnce, XRange, ZRange, Height);
	}

}


//�����㷨��غ���
/*
�����������㷨������ײ�����ٶȸ���
���������б�N����
���أ��ޣ����Ǹ������б�
*/
__global__ void HandleCollisionNaive(Ball* balls, int N)
{
	// ��ȡȫ������
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// ����
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
������GPU��ײ���+�˶������������������㷨��
���������б�һ�ε�ʱ�䣬X��Χ(-X,X),Z��Χ(-Z,Z),Y��Χ(0,Y)�������
���أ��ޣ����Ǹ������б�
*/
void UpdateBallsNaiveGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N)
{
	// �����й��ڴ�
	int nBytes = N * sizeof(Ball);
	Ball* balls_gpu;
	cudaMallocManaged((void**)&balls_gpu, nBytes);

	// ��ʼ������
	cudaMemcpy((void*)balls_gpu, (void*)balls, nBytes, cudaMemcpyHostToDevice);

	// ����kernel��ִ������
	dim3 blockSize(256);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

	// ִ��kernel
	HandleCollisionNaive << < gridSize, blockSize >> > (balls_gpu, N);
	// ͬ��device ��֤�������ȷ����
	cudaDeviceSynchronize();

	// ִ��kernel
	UpdateBallsMove <<< gridSize, blockSize >>> (balls_gpu, TimeOnce, XRange, ZRange, Height, N);
	// ͬ��device ��֤�������ȷ����
	cudaDeviceSynchronize();

	// ��¼���
	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyDeviceToHost);

	// �ͷ��ڴ�
	cudaFree(balls_gpu);
}


//�ռ仮���㷨��غ���
/*
��������ʼ��cells��objects���飬ǰ�߼�¼�������ڵĸ�����Ϣ������x��y��z��id��home����phantom�������߼�¼����id��home/phantom
�������յ�cell��phantom�����б�͸��������и��ָ�����Ϣ
���أ�����cells��objects�����cell_num
*/
__global__ void InitCellKernel(uint32_t *cells, uint32_t *objects, Ball* balls, int N,
	float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ) 
{
	unsigned int count = 0;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
	{
		int current_cell_id = i * 8; //ÿ���������8��������
		int cell_info = 0;
		int object_info = 0;
		int current_count = 0;
		float x = balls[i].CurrentPlace.x;
		float y = balls[i].CurrentPlace.y;
		float z = balls[i].CurrentPlace.z;
		float radius = balls[i].Radius;

		//�ҵ�home cell
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

		//��phantom
		for (int dx = -1; dx <= 1; dx++)
		{
			for (int dy = -1; dy <= 1; dy++)
			{
				for (int dz = -1; dz <= 1; dz++)
				{
					int new_hash_x = hash_x + dx;
					int new_hash_y = hash_y + dy;
					int new_hash_z = hash_z + dz;

					//�Լ�������
					if (dx == 0 && dy == 0 && dz == 0)
					{
						continue;
					}

					//Խ�粻����
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

		//����
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
��������ʼ��cells�� objects�����������
�������յ�cell��phantom�����б�͸��������и��ָ�����Ϣ���߳���Ϣ
���أ�����cells��objects�����cell_num
 */
void InitCells(uint32_t *cells, uint32_t *objects, Ball* balls, int N,
	 float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ, 
	 unsigned int num_blocks, unsigned int threads_per_block) {
	 InitCellKernel << <num_blocks, threads_per_block,
		 threads_per_block * sizeof(unsigned int) >> > (
			 cells, objects, balls, N, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ);
 }


/*
����������ǰi�͵��㷨
������ԭʼ���飬����n
���أ�ԭʼ������ǰi��������
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
��������cells��ǰ׺��
������cells��������ǰ׺�ͣ�N��cell��ƫ����
���أ�����ǰ׺��
*/
__global__ void GetRadixSum(uint32_t *cells, uint32_t *radix_sums, int N, int shift)
 {
	 int index = threadIdx.x + blockIdx.x * blockDim.x;
	 int stride = blockDim.x * gridDim.x;
	 int num_indices = 1 << RADIX_LENGTH;


	 //��ʼ��
	 for (int i = index; i < num_indices; i++)
	 {
		 radix_sums[i] = 0;
	 }
	 __syncthreads();


	 //���
	 for (int i = index; i < N; i += stride)
	 {
		 //�ǳ���Ҫ�����������޷���Ч���
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
	 //��ǰ׺��
	 PrefixSum(radix_sums, num_indices);
	 __syncthreads();
}

 /*
 ���������·���Ԫ��
 ������cells��object���飬���Ǵ����µķ�����temp��ǰ׺�����飬N��Ԫ�أ�ƫ������ÿ���̴߳�����cell
 ���أ�����ǰ׺��
 */
__global__ void RearrangeCell(uint32_t *cells, uint32_t *objects, uint32_t *cells_temp, uint32_t *objects_temp, 
	 uint32_t *radix_sums, int N, int shift)
 {
	 int index = threadIdx.x + blockIdx.x * blockDim.x;
	 int stride = blockDim.x * gridDim.x;
	 int num_radices = 1 << RADIX_LENGTH;

	 if (index != 0) return;
	 //����
	 for (int i = 0; i < N; i ++ )
	 {
		int current_radix_num = (cells[i] >> shift) & (num_radices - 1);
		cells_temp[radix_sums[current_radix_num]] = cells[i];
		objects_temp[radix_sums[current_radix_num]] = objects[i];
		radix_sums[current_radix_num] ++;
	 }
 }

/*
��������ȡ����������index��cell�仯��λ�ã�
������cell��cell����N,�����µ�indice�������µ�indice����
���أ��ޣ����Ǹ���indice�����indice����
*/
__global__ void GetCellIndex(uint32_t *cells, int N, uint32_t* indices, uint32_t* num_indices)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	//ֻ�ܴ���
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
��������cell��object���������򣬲��һ�ȡindex��cell�仯��λ�ã�
������cell��object���飻���ǵ�temp��ʽ�������򣻴����ǰ׺�����飻cell�����������index����ͳ��ȣ��߳����
���أ��ޣ����Ǹ���cell��object���飬����index������䳤��
*/
void SortCells(uint32_t *cells, uint32_t *objects, uint32_t *cells_temp, uint32_t *objects_temp,
	uint32_t *radix_sums, int N, uint32_t* indices, uint32_t* num_indices,
	unsigned int num_blocks, unsigned int threads_per_block)
{
	uint32_t *cells_swap;
	uint32_t *objects_swap;
	for (int i = 0; i < 32; i += RADIX_LENGTH)
	{
		//��ǰ׺��
		GetRadixSum <<< num_blocks, threads_per_block >>> (cells, radix_sums, N, i);

		//��ǰ׺�����·���
		RearrangeCell << < num_blocks, threads_per_block >> > (cells, objects, cells_temp, objects_temp,
			radix_sums, N, i);
		
		//����ԭʼ��temp
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
������cuda��ײ���ʹ�����
������cell��object���飬ball���飬���cell�ĸ�����index����͸������߳���Ϣ�������ĸ������ƺ͸�����Ϣ
���أ��ޣ����ǽ�����ײ���ʹ���
*/
__global__ void HandleCollisionCuda(uint32_t *cells, uint32_t *objects, Ball* balls, int num_balls, int num_cells,
	uint32_t* indices, uint32_t num_indices, unsigned int group_per_thread,
	float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int group_num = 0; group_num < group_per_thread; group_num++)
	{
		//�ж��Ƿ�Խ�磬�ҵ������start��end
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

		//������home�ĸ���
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

		//������ײ���
		for (int i = start; i < start + home_num; i++)
		{
			if (cells[i] == UINT32_MAX) break;
			int ball_i = (objects[i] >> 1) & 65535;
			
			for (int j = i + 1; j < end; j++)
			{
				if (cells[j] == UINT32_MAX) break;
				int ball_j = (objects[j] >> 1) & 65535;

				//2��home��ֱ����ײ���
				if (j < start + home_num)
				{
					if (JudgeCollision(balls[ball_i], balls[ball_j]))
					{
						ChangeSpeed(balls[ball_i], balls[ball_j]);
					}
				}

				//home��phantom����Ҫ����
				else
				{
					int home_i = (cells[i] >> 1) & ((1 << 24) - 1);
					int j_x = (balls[ball_j].CurrentPlace.x + XRange) / GridSize;
					int j_y = balls[ball_j].CurrentPlace.y / GridSize;
					int j_z = (balls[ball_j].CurrentPlace.z + ZRange) / GridSize;
					int home_j = j_x << 16 | j_y << 8 | j_z;

					//ֻ�������ſ���
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
��������ײ���ʹ�����
������cell��object���飬ball���飬���cell�ĸ�����index����͸������߳���Ϣ�������ĸ������ƺ͸�����Ϣ
���أ��ޣ����ǽ�����ײ���ʹ���
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
�������ռ仮���㷨������ײ�����ٶȸ��£���������
���������б�X��Χ(-X,X),Z��Χ(-Z,Z),Y��Χ(0,Y)�����Ӵ�С��X���Ӹ�����Y���Ӹ�����Z���Ӹ�����N����
���أ��ޣ����Ǹ������б�
*/
void HandleCollisionGrid(Ball* balls, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N,
	unsigned int num_blocks, unsigned int threads_per_block)
{

	//�����ڴ�
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


	
	//��ʼ��cell��object
	InitCells(cells_gpu, objects_gpu, balls, N,
		XRange, ZRange, Height, GridSize, GridX, GridY, GridZ,
		num_blocks, threads_per_block);


	//��������
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
������GPU��ײ���+�˶��������������ռ仮���㷨��
���������б�һ�ε�ʱ�䣬X��Χ(-X,X),Z��Χ(-Z,Z),Y��Χ(0,Y)��һ�����Ӵ�С��X,Y,Z�ĸ��Ӹ����������
���أ��ޣ����Ǹ������б�
*/
void UpdateBallsGridGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N)
{
	//���ã�������Ҫ����block��thread
	unsigned int num_blocks = 128;
	unsigned int threads_per_block = 512;
	unsigned int object_size = (N - 1) / threads_per_block + 1;
	if (object_size < num_blocks) {
		num_blocks = object_size;
	}

	Ball* balls_gpu;
	unsigned int nBytes = N * sizeof(Ball);
	cudaMalloc((void**)&balls_gpu, nBytes);


	// ��ʼ������
	cudaMemcpy((void*)balls_gpu, (void*)balls, nBytes, cudaMemcpyHostToDevice);

	// ִ��kernel
	HandleCollisionGrid(balls_gpu, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ, N, num_blocks, threads_per_block);
	// ͬ��device ��֤�������ȷ����
	cudaDeviceSynchronize();

	// ִ��kernel
	UpdateBallsMove << < num_blocks, threads_per_block>> > (balls_gpu, TimeOnce, XRange, ZRange, Height, N);
	// ͬ��device ��֤�������ȷ����
	cudaDeviceSynchronize();

	// ��¼���
	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyDeviceToHost);

	// �ͷ��ڴ�
	cudaFree(balls_gpu);
}