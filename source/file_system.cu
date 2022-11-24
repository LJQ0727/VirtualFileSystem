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
  fs->STORAGE_BLOCK_COUNT = (fs->STORAGE_SIZE - fs->SUPERBLOCK_SIZE - (fs->FCB_SIZE * fs->FCB_ENTRIES)) / fs->STORAGE_BLOCK_SIZE;

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
  while (!(start1[idx] == '\0' || start2[idx] == '\0'))
  {
    if (start1[idx] != start2[idx])
    {
      return false;
    }
    idx++;
  }
  if (start1[idx] != start2[idx])
  {
    return false;
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

__device__ void mark_block_unused(FileSystem *fs, int block_idx) {
  // mark a block as unused in the superblock
  // operate on only one block at a time
  uchar bitmap = fs->start_of_superblock[block_idx/8];
  uchar mask = 1 << (block_idx % 8);
  fs->start_of_superblock[block_idx/8] = bitmap & ~mask;
}

__device__ bool check_block_used(FileSystem *fs, int block_idx) {
  uchar bitmap = fs->start_of_superblock[block_idx/8];
  uchar mask = 1 << (block_idx % 8);
  return bitmap & mask;
}





__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  // s ends with '\0'
  // op: open mode, G_READ or G_WRITE
  // returns the file pointer, which is the index of the FCB entry
  gtime++;

  // find if the specific file already exists in the FCB
  bool file_exists = false;
  int fcb_idx = 0;
  for (int i = 0; i < fs->FCB_ENTRIES; i++)
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
        // printf("fs_open file %s exists, index %d\n", s, fcb_idx);
        return fcb_idx;
      } else {  // file not exists
        // allocate a new fcb index for the newly-created file
        for (int i = 0; i < fs->FCB_ENTRIES; i++)
        { // find an unused fcb
          FCB target_fcb = fs->start_of_fcb[i];
          if (!target_fcb.is_on)
          {
            // mark the FCB as on
            fs->start_of_fcb[i].is_on = true;
            fs->start_of_fcb[i].modified_time = gtime;
            fs->start_of_fcb[i].size = 0;  // size at creation
            fs->start_of_fcb[i].creation_time = gtime;  // time at creation
            fs->start_of_fcb[i].start_block_idx = 0;
            // copy the filename
            int idx = 0;
            while (s[idx] != '\0')
            {
              fs->start_of_fcb[i].filename[idx] = s[idx];
              idx++;
            }
            fs->start_of_fcb[i].filename[idx] = '\0';

            fs->start_of_fcb[i].size = 0; // no content for this file for now
            
            // printf("fs_open new fcb %s, index %d\n", s, i);
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

__device__ u32 block_of_bytes(FileSystem *fs, u32 bytes) {
  // returns how many blocks the `bytes` information will occupy
  u32 ret = bytes / fs->STORAGE_BLOCK_SIZE;
  if ((bytes % fs->STORAGE_BLOCK_SIZE) != 0) {
    ret++;
  }
  return ret;
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  // fp the index of the FCB
  assert(fs->start_of_fcb[fp].is_on);
  uchar *start = fs->start_of_contents + fs->start_of_fcb[fp].start_block_idx * fs->STORAGE_BLOCK_SIZE;
  FCB fcb = fs->start_of_fcb[fp];   // the fcb for this file

  // printf("fs_read %d bytes from %s\n", size, fcb.filename);
  
  // read `size` bytes to buffer `output`
  for (u32 i = 0; i < size; i++)
  {
    output[i] = start[i];
  } 
}

__device__ void block_move(FileSystem *fs, int target_block_idx, int source_block_idx) {
  // printf("moving block %d to %d\n", source_block_idx, target_block_idx);

  uchar *target_start = fs->start_of_contents + target_block_idx * fs->STORAGE_BLOCK_SIZE;
  uchar *source_start = fs->start_of_contents + source_block_idx * fs->STORAGE_BLOCK_SIZE;
  for (int i = 0; i < fs->STORAGE_BLOCK_SIZE; i++)
  {
    target_start[i] = source_start[i];
    source_start[i] = 0;
  }
  mark_block_unused(fs, source_block_idx);
  mark_block_used(fs, target_block_idx);
}

__device__ u16 alloc_new_blocks(FileSystem *fs, int target_block_size) {
  // allocate contiguous blocks with `target_block_size`, register it in the bitmap
  // return the index of the first block
  // if no enough contiguous blocks, have to manage the fragmentation
  // printf("allocating %d blocks in alloc_new_blocks\n", target_block_size);
  int current_block_idx = 0;
  int block_count = 0;
  while (current_block_idx < fs->STORAGE_BLOCK_COUNT)
  {
    if (check_block_used(fs, current_block_idx)) {
      // this block is used, reset the counter
      block_count = 0;
    } else {
      block_count++;
      if (block_count == target_block_size) {
        // found enough contiguous blocks
        // printf("contiguous block found, returning block %d, span%d\n", current_block_idx - target_block_size + 1, target_block_size);
        // mark blocks as used
        for (int i = 0; i < target_block_size; i++)
        {
          mark_block_used(fs, current_block_idx - target_block_size + 1 + i);
        }
        
        return current_block_idx - target_block_size + 1;
      }
    }
    current_block_idx++;
  }

  // printf("No enough contiguous blocks, have to manage the fragmentation\n");
  // not enough contiguous space, have to manage the fragmentation
  // compation algorithm
  int first_unused_block_idx = 0;
  while (true)
  {
    // find the first unused block idx
    while (first_unused_block_idx < fs->STORAGE_BLOCK_COUNT)
    {
      if (!check_block_used(fs, first_unused_block_idx)) {
        break;
      }
      first_unused_block_idx++;
    }
    
    
    current_block_idx = first_unused_block_idx+1;
    // find the next used block idx
    while (current_block_idx < fs->STORAGE_BLOCK_COUNT)
    {
      if (check_block_used(fs, current_block_idx)) {
        break;
      }
      current_block_idx++;
    }

    if (current_block_idx >= fs->STORAGE_BLOCK_COUNT) {
      // no more used blocks
      break;
    }

    // swap the two blocks
    block_move(fs, first_unused_block_idx, current_block_idx);
    // reassign the associated fcbs of the moved block
    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      FCB *fcb = fs->start_of_fcb + i;
      if (fcb->start_block_idx == current_block_idx)
      {
        // printf("reassigning fcb block %d to %d\n", fcb->start_block_idx, first_unused_block_idx);
        fcb->start_block_idx = first_unused_block_idx;
        break;
      }
    }
    
    first_unused_block_idx++;
    current_block_idx++;
    if (current_block_idx >= fs->STORAGE_BLOCK_COUNT) {
      // no more used blocks
      break;
    }
  }

  // reallocate
  // printf("reallocating %d blocks in alloc_new_blocks\n", target_block_size);
  current_block_idx = 0;
  block_count = 0;
  while (current_block_idx < fs->STORAGE_BLOCK_COUNT)
  {
    if (check_block_used(fs, current_block_idx)) {
      // this block is used, reset the counter
      block_count = 0;
    } else {
      block_count++;
      if (block_count == target_block_size) {
        // found enough contiguous blocks
        // printf("contiguous block found, returning block %d, span%d\n", current_block_idx - target_block_size + 1, target_block_size);
        // mark blocks as used
        for (int i = 0; i < target_block_size; i++)
        {
          mark_block_used(fs, current_block_idx - target_block_size + 1 + i);
        }
        
        return current_block_idx - target_block_size + 1;
      }
    }
    current_block_idx++;
  }
  assert(0);  // fail to reallocate

}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
  // fp the index of the FCB
  gtime++;


  uchar *start = fs->start_of_contents + fs->start_of_fcb[fp].start_block_idx * fs->STORAGE_BLOCK_SIZE; // the initial byte of the file content
  FCB *fcb = fs->start_of_fcb+fp;   // the fcb for this file
  u16 start_block_idx = fcb->start_block_idx;
  
  // printf("fs_write %d bytes into %s\n", size, fcb->filename);
  // printf("start_block_idx %d\n", start_block_idx);
  // printf("fcb->size %d\n", fcb->size);
  // printf("check used: %d\n", check_block_used(fs, start_block_idx));

  // if the file already exists, we have to free the blocks 
  for (u32 i = 0; i < block_of_bytes(fs, fcb->size); i++)
  {
    mark_block_unused(fs, start_block_idx+i);
  }
  // empty the bytes, replace by 0
  for (u32 i = 0; i < fcb->size; i++)
  {
    start[i] = 0;
  }


  // begin writing to new file
  bool can_directly_write = true;
  for (u32 i = 0; i < block_of_bytes(fs, size); i++)
  {
    if (check_block_used(fs, start_block_idx+i))
    {
      can_directly_write = false;
      break;
    }
  }
  
  if (can_directly_write)
  {
    // printf("directly writing %d blocks starting from block %d\n", block_of_bytes(fs, size), start_block_idx);
    // directly write to it
    for (u32 i = 0; i < size; i++)
    {
      start[i] = input[i];
    }
    for (u32 i = 0; i < block_of_bytes(fs, size); i++)
    {
      mark_block_used(fs, start_block_idx+i);
    }
    fcb->size = size;
    fcb->modified_time = gtime;
    // printf("modified time of %s: %d\n", fcb->filename ,gtime);
    
    return size;
    
  } else {
    // cannot directly write, need to fix fragmentation, then directly write
    fcb->start_block_idx = alloc_new_blocks(fs, block_of_bytes(fs, size));
    // printf("cannot directly write, resetting start_block_idx to %d\n", fcb->start_block_idx);
    // perform write
    start = fs->start_of_contents + fcb->start_block_idx * fs->STORAGE_BLOCK_SIZE; // the initial byte of the file content
    for (u32 i = 0; i < size; i++)
    {
      start[i] = input[i];
    }
    for (u32 i = 0; i < block_of_bytes(fs, size); i++)
    {
      mark_block_used(fs, fcb->start_block_idx+i);
    }

    fcb->size = size;
    fcb->modified_time = gtime;
    return size;
  }
  
  
  
  
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
  // count number of files, needed by both
  int file_count = 0;
  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    FCB fcb = fs->start_of_fcb[i];
    if (fcb.is_on)
    {
      file_count++;
    }
  }
  // printf("number of files: %d\n", file_count);

	/* Implement LS_D and LS_S operation here */
  switch (op)
  {
  case LS_D:
  {

    // list the file name and order by modified time of files

    printf("===sort by modified time===\n");

    
    int last_item_time = (1<<15); // trace the time of last printed file
    // print the most recent modified file before the last item
    for (int i = 0; i < file_count; i++)
    {
      int latest_modified_time = 0;
      FCB latest_fcb;
      for (int j = 0; j < fs->FCB_ENTRIES; j++)
      {
        FCB fcb = fs->start_of_fcb[j];
        if (fcb.is_on && (fcb.modified_time > latest_modified_time) && (fcb.modified_time < last_item_time))
        {
          latest_fcb = fcb;
          latest_modified_time = fcb.modified_time;
        }
        
      }
      last_item_time = latest_fcb.modified_time;
      printf("%s\n", latest_fcb.filename);
      // printf("%s   time%d\n", latest_fcb.filename, last_item_time);
    }
    

    break;
  }


  case LS_S:
  {
      
    printf("===sort by file size===\n");
    // If there are several files with the same size, then first create first print.

    
    u32 last_item_size = (u32)(1<<31); // the distinct size of the last printed file
    int print_count = 0;

    while (print_count < file_count)
    {
      int largest_file_size = 0;
      for (int i = 0; i < fs->FCB_ENTRIES; i++)
      {
        FCB fcb = fs->start_of_fcb[i];
        if (fcb.is_on && (fcb.size >= largest_file_size) && (fcb.size < last_item_size))
        {
          largest_file_size = fcb.size;
        }
      }
      last_item_size = largest_file_size;

      // printf("last item size: %d\n", last_item_size);
      // printf("largest file size: %d\n", largest_file_size);

      // count the number of files with the size of largest_file_size
      int largest_file_count = 0;
      for (int i = 0; i < fs->FCB_ENTRIES; i++)
      {
        FCB fcb = fs->start_of_fcb[i];
        if (fcb.is_on && (fcb.size == largest_file_size))
        {
          largest_file_count++;
        }
      }
      // printf("largest file size: %d, count: %d\n", largest_file_size, largest_file_count);

      u16 last_item_time = 0;
      for (int i = 0; i < largest_file_count; i++)
      {
        u16 earliest_created_time = (1<<15);
        FCB earliest_fcb;
        for (int i = 0; i < fs->FCB_ENTRIES; i++)
        {
          // find the file with the file size of largest_file_size and the earliest created time among all unprinted items
          FCB fcb = fs->start_of_fcb[i];
          if (fcb.is_on && (fcb.size == largest_file_size) && (fcb.creation_time < earliest_created_time) && (fcb.creation_time > last_item_time))
          {
            earliest_fcb = fcb;
            earliest_created_time = fcb.creation_time;
          }
        }
        last_item_time = earliest_fcb.creation_time;
        printf("%s %d\n", earliest_fcb.filename, earliest_fcb.size);
      }
      print_count += largest_file_count;
      
    }
    break;
  }
  default:
    break;  // no such option
  } // end of switch
  
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  // find the specific file in the FCB
  bool file_exists = false;
  int fcb_idx = 0;
  FCB *target_fcb;

  for (int i = 0; i < fs->FCB_ENTRIES; i++)
  {
    target_fcb = &fs->start_of_fcb[i];
    if (target_fcb->is_on && strmatch(target_fcb->filename, s))
    {
      file_exists = true;
      fcb_idx = i;
      break;
    }
  }

	/* Implement rm operation here */
  if (op == RM)
  {
    // delete the specific file
    if (!file_exists)
    {
      assert(0);  // file not found
    } else {
      target_fcb->is_on = false;

      // free the content memory
      uchar *start = fs->start_of_contents + target_fcb->start_block_idx * fs->STORAGE_BLOCK_SIZE; // the initial byte of the file content
      
      // printf("fs_delete removing %d bytes of %s, start from block %d span %d\n", target_fcb->size, target_fcb->filename, target_fcb->start_block_idx, block_of_bytes(fs, target_fcb->size));

      // free the blocks  
      for (u32 i = 0; i < block_of_bytes(fs, target_fcb->size); i++)
      {
        mark_block_unused(fs, target_fcb->start_block_idx+i);
      }
      // empty the bytes, replace by 0
      for (u32 i = 0; i < target_fcb->size; i++)
      {
        start[i] = 0;
      }
    }
    
  } else {
    printf("no such option in delete\n");
    assert(0);
  }
  
}
