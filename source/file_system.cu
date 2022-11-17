#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  // init some custom pointers
  fs->start_of_fcb = (FCB*)(volume+SUPERBLOCK_SIZE);
  fs->start_of_superblock = volume;
  fs->start_of_contents = volume + FILE_BASE_ADDRESS;

  // initialize volume
  for (int i = 0; i < VOLUME_SIZE; i++)
    volume[i] = 0;

}


__device__ bool strmatch(char *start1, char* start2) {
  // match two strings, return true if they are the same
  int idx = 0;
  while (start1[idx] != '\0' && start2[idx] != '\0')
  {
    if (start1[idx] != start2[idx])
    {
      return false;
    }
    idx++;
  }
  return true;
}

__device__ void mark_block_used(FileSystem *fs, int block_idx) {
  // mark a block as used in the superblock
  // operate on only one block at a time
  uchar bitmap = fs->start_of_superblock[block_idx/8];
  uchar mask = 1 << (block_idx % 8);
  fs->start_of_superblock[block_idx/8] = bitmap | mask;
}

__device__ char * alloc_new_blocks(FileSystem *fs, int target_block_size) {
  // allocate contiguous blocks with `target_block_size`, register it in the bitmap
  // return the pointer to the start of the first block
  // if no enough contiguous blocks, have to manage the fragmentation
}




__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  // s ends with '\0'
  // op: open mode, G_READ or G_WRITE
  // returns the file pointer, which is the index of the FCB entry

  // find if the specific file already exists in the FCB
  bool file_exists = false;
  int fcb_idx = 0;
  for (int i = 0; i < fs->FCB_SIZE; i++)
  {
    FCB target_fcb = fs->start_of_fcb[i];
    if (target_fcb.is_on && strmatch(target_fcb.filename, s))
    {
      file_exists = true;
      fcb_idx = i;
      break;
    }
  }
  
  switch (op)
  {
    case G_READ:
      // find file with the filename among all files, returns the index of the FCB
      if (file_exists) {
        return fcb_idx;
      }
      assert(0);  // file not found
      break;
    case G_WRITE:
      // create the directory for the new file, if it's not already there; returns the address of the new FCB
      if (file_exists) {
        // have to empty the file in the next write operation
        // in which we will check the `size` attribute, if it's not 0, we will free the blocks
        return fcb_idx;
      } else {  // file not exists
        // allocate a new fcb index for the newly-created file
        for (int i = 0; i < fs->FCB_SIZE; i++)
        {
          FCB target_fcb = fs->start_of_fcb[i];
          if (!target_fcb.is_on)
          {
            // mark the FCB as on
            fs->start_of_fcb[i].is_on = true;
            // copy the filename
            int idx = 0;
            while (s[idx] != '\0')
            {
              fs->start_of_fcb[i].filename[idx] = s[idx];
              idx++;
            }
            fs->start_of_fcb[i].filename[idx] = '\0';

            fs->start_of_fcb[i].size = 0;
            return i;
          }
        }
        assert(0);  // no empty FCB
      }
      break;

    default:
      assert(0);  // no such option
      break;
    return fcb_idx;
  }
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  // fp the index of the FCB
  uchar *start = fs->start_of_contents + fs->start_of_fcb[fp].start_block_idx * fs->STORAGE_BLOCK_SIZE;
  // read `size` bytes to buffer `output`
  for (u32 i = 0; i < size; i++)
  {
    output[i] = start[i];
  } 
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
  // fp the index of the FCB
  gtime++;
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
}
