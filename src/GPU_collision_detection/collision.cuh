#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ball.hpp"


#define HOME_CELL 0x00
#define PHANTOM_CELL 0x01
#define HOME_OBJECT 0x01
#define PHANTOM_OBJECT 0x00

#define RADIX_LENGTH 8
#define NUM_BLOCKS 16
#define GROUPS_PER_BLOCK 12
#define THREADS_PER_GROUP 16
#define PADDED_BLOCKS 16
#define PADDED_GROUPS 256

void UpdateBallsGridGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N);