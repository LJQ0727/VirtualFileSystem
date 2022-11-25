#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

#define SUPERBLOCK_SIZE 4096 //32K/8 bits = 4 K 
// 1024KB/32B = 2^15 data Blocks, which needs 2^15 bits to indicate the status, which is 4KB in exact
#define FCB_SIZE 32 //32 bytes per FCB, including file name, file size, file blocks. in total 1024 FCBs
#define FCB_ENTRIES 1024	// maximum 1024 stored files, so the fcb part is 32kb
#define VOLUME_SIZE 1085440 //4096+32768+1048576 	volume size is 1060kb
#define STORAGE_BLOCK_SIZE 32	// the storage block size is 32b, so in total 2^15 Blocks

#define MAX_FILENAME_SIZE 20
#define MAX_FILE_NUM 1024
#define MAX_FILE_SIZE 1048576	// max file size is 1024kb

#define FILE_BASE_ADDRESS 36864 //4096+32768=36kb	//start address of file storage area


// data input and output
__device__ __managed__ uchar input[MAX_FILE_SIZE];
__device__ __managed__ uchar output[MAX_FILE_SIZE];

// volume (disk storage)
__device__ __managed__ uchar volume[VOLUME_SIZE];

// expand the FCB space in bonus
__device__ __managed__ FCB fcb_entry[FCB_ENTRIES];


__device__ void user_program(FileSystem *fs, uchar *input, uchar *output);

__global__ void mykernel(uchar *input, uchar *output) {

  // Initilize the file system	
  FileSystem fs;
  fs_init(&fs, volume, SUPERBLOCK_SIZE, FCB_SIZE, FCB_ENTRIES, 
			VOLUME_SIZE,STORAGE_BLOCK_SIZE, MAX_FILENAME_SIZE, 
			MAX_FILE_NUM, MAX_FILE_SIZE, FILE_BASE_ADDRESS, fcb_entry);

  // user program the access pattern for testing file operations
  user_program(&fs, input, output);
}

__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}

__host__ int load_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "rb");

	if (!fp)
	{
		printf("***Unable to open file %s***\n", fileName);
		exit(1);
	}

	//Get file length
	fseek(fp, 0, SEEK_END);
	int fileLen = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (fileLen > bufferSize)
	{
		printf("****invalid testcase!!****\n");
		printf("****software warrning: the file: %s size****\n", fileName);
		printf("****is greater than buffer size****\n");
		exit(1);
	}

	//Read file contents into buffer
	fread(buffer, fileLen, 1, fp);
	fclose(fp);
	return fileLen;
}

int main() {
  cudaError_t cudaStatus;
  load_binaryFile(DATAFILE, input, MAX_FILE_SIZE);

//   printf("sizeof is %d\n", sizeof(FCB));

  // Launch to GPU kernel with single thread
  mykernel<<<1, 1>>>(input, output);

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "mykernel launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return 0;
  }

  cudaDeviceSynchronize();
  cudaDeviceReset();

  write_binaryFile(OUTFILE, output, MAX_FILE_SIZE);


  return 0;
}
