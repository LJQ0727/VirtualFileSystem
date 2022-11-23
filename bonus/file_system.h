#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <iostream>

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;

#define G_WRITE 1
#define G_READ 0

// for fs_gsys
#define LS_D 0
#define LS_S 1
#define RM 2
#define MKDIR 3
#define CD 4
#define CD_P 5
#define RM_RF 6
#define PWD 7

// File control block
struct FCB {
	char filename[21];	// maximum size of filename is 20 bytes
	u32 size;	// the size of the file **in bytes**
	u16 modified_time;	// the last modified time
	u16 creation_time;
	u16 start_block_idx;	// the index of the first of its contiguous blocks
	bool is_on;

	// additional information
	bool is_dir;	// true if it is a directory
	int parent_dir_idx;	// the index of the parent directory, of the dir the file is in
	int dir_idx;	// the index of the directory that the file is in; for a dir, it is its own idx

	// only set if the FCB is a dir, which points to the first file in the subdir
	FCB * dir_files;	// a doubly linked list to all files in the dir; 

	// a doubly-linked list connecting all in the current directory
	// the first file should point back to the directory file
	FCB * next;
	FCB * prev;
};


struct FileSystem {
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
	int STORAGE_BLOCK_COUNT;

	uchar *start_of_superblock;
	FCB *start_of_fcb;
	uchar *start_of_contents;
	int cwd;	// current working directory's fcb index, **not** block index
};


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS,
	FCB *start_of_fcb);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);


#endif