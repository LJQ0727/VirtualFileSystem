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
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS,
              FCB *start_of_fcb)
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
  fs->start_of_fcb = start_of_fcb;
  fs->start_of_superblock = volume;
  fs->start_of_contents = volume + FILE_BASE_ADDRESS;
  fs->cwd = -1;   // the root directory will have parent dir index -1

  // initialize volume
  for (int i = 0; i < VOLUME_SIZE; i++)
    volume[i] = 0;

  // initialize fcb
  for (int i = 0; i < FCB_ENTRIES; i++) {
    fs->start_of_fcb[i].is_dir = false;
    fs->start_of_fcb[i].is_on = false;
    fs->start_of_fcb[i].parent_dir_idx = -1;
  }

  // make root directory and cd to it
  fs_gsys(fs, MKDIR, "/\0");
  fs->cwd = 0;  // root will be created at idx 0
}

__device__ bool strmatch(char *start1, char* start2);

__device__ uchar * get_content(FileSystem *fs, int block_idx, int byte_offset) {
  // given a block index, get the pointer to the content of a file (or directory)
  return fs->start_of_contents + block_idx * fs->STORAGE_BLOCK_SIZE + byte_offset;
}

__device__ void set_gtime_recursive(FileSystem *fs, int fcb_idx, u32 gtime) {
  // recursively set the gtime of the fcb and propagate this change to all parent dirs
  FCB *fcb = fs->start_of_fcb + fcb_idx;
  fcb->modified_time = gtime;
  printf("modified gtime of '%s' to %d, isdir: %d, parent_dir_idx: %d\n", fcb->filename , gtime,fcb->is_dir, fcb->parent_dir_idx);
  if (fcb->dir_idx != -1 && !fcb->is_dir) {
    // set gtime for the file's dir
    set_gtime_recursive(fs, fcb->dir_idx, gtime);
  }
  if (fcb->parent_dir_idx != -1) {
    set_gtime_recursive(fs, fcb->parent_dir_idx, gtime);
  }
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


__device__ void my_memcpy(char *destination, char *source, int size) {
  // this will not automatically add '\0' to the string
  for (int i = 0; i < size; i++)
  {
    destination[i] = source[i];
  }
  
}

__device__ int my_strlen(char *s) {
  // find the length of a string, **including '\0'**
  int idx = 0;
  while (s[idx] != '\0')
  {
    idx++;
  }
  return idx+1;
}

__device__ FCB * get_file_in_curr_dir(FileSystem *fs, char* s) {
  // return the pointer to the file or subdir if the file exists in the current directory
  // return nullptr if the file does not exist
  if (fs->cwd == -1)
  {
    return nullptr;
  }
  
  FCB dir = fs->start_of_fcb[fs->cwd];
  FCB *current = dir.dir_files;
  while (current != nullptr)
  {
    if (strmatch(current->filename, s))
    {
      // found
      return current;
    }
    current = current->next;
  }
  return nullptr;  // no such file in this directory
}

__device__ bool file_exists_in_curr_dir(FileSystem *fs, char* s) {
  return get_file_in_curr_dir(fs, s) != nullptr;
}

__device__ int get_fcb_idx(FileSystem *fs, FCB* fcb) {
  // get the index of the fcb in the fcb array
  return (int)(fcb - fs->start_of_fcb);
}

__device__ bool isdirname(char *s) {
  // check if a string is a directory name
  // a directory name ends with '/'
  int len = my_strlen(s);
  return s[len-2] == '/';
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  // s ends with '\0'
  // op: open mode, G_READ or G_WRITE
  // returns the file pointer, which is the index of the FCB entry
  gtime++;
  bool file_exists = file_exists_in_curr_dir(fs, s);
  FCB * target_fcb = get_file_in_curr_dir(fs, s);
  if (isdirname(s))
  {
    file_exists = false;
  }
  

  switch (op)
  {
    case G_READ:
      // find file with the filename among all files, returns the index of the FCB
      if (file_exists) {
        return get_fcb_idx(fs, target_fcb);
      }
      assert(0);  // file not found
      break;
    case G_WRITE:
      // create the directory for the new file, if it's not already there; returns the address of the new FCB
      if (file_exists) {
        // have to empty the file in the next write operation
        // in which we will check the `size` attribute, if it's not 0, we will free the blocks
        int fcb_idx = get_fcb_idx(fs, target_fcb);
        set_gtime_recursive(fs, fcb_idx, gtime);
        printf("fs_open file %s exists, index %d\n", s, fcb_idx);
        return fcb_idx;
      } else {  // file not exists
        // allocate a new fcb index for the newly-created file
        for (int i = 0; i < fs->FCB_ENTRIES; i++)
        { // find an unused fcb
          target_fcb = fs->start_of_fcb + i;
          if (!target_fcb->is_on)
          {
            // mark the FCB as on and set its attrs
            target_fcb->size = 0;  // size at creation
            target_fcb->modified_time = gtime;
            target_fcb->creation_time = gtime;  // time at creation
            target_fcb->start_block_idx = 0;
            target_fcb->is_on = true;

            target_fcb->is_dir = false;
            target_fcb->parent_dir_idx = fs->start_of_fcb[fs->cwd].parent_dir_idx;
            target_fcb->dir_idx = fs->cwd;

            target_fcb->dir_files = nullptr;


            bool is_dir_name = isdirname(s);
            if (is_dir_name)
            {
              // this is a directory
              target_fcb->is_dir = true;
              target_fcb->parent_dir_idx = fs->cwd;
              target_fcb->dir_idx = i;
              s[my_strlen(s)-2] = '\0';  // remove the last '/'
              printf("fs_open dir %s created, index %d\n", s, i);
              printf("Parent dir idx %d\n", target_fcb->parent_dir_idx);
            }
            
            // copy the filename
            {
              int idx = 0;
              while (s[idx] != '\0')
              {
                fs->start_of_fcb[i].filename[idx] = s[idx];
                idx++;
              }
              fs->start_of_fcb[i].filename[idx] = '\0';
            }

            if (is_dir_name && (target_fcb->parent_dir_idx == -1))
            {
              // this is the root directory
              return target_fcb->dir_idx;
            }
            

            // append the new file to the current directory's doubly linked list
            {
              FCB *curr_dir = &fs->start_of_fcb[fs->cwd];
              if (curr_dir->dir_files == nullptr) {
                // the current directory is empty
                curr_dir->dir_files = target_fcb;
                target_fcb->prev = curr_dir;
                target_fcb->next = nullptr;
              } else {
                // the current directory is not empty
                FCB *last_file = curr_dir->dir_files;
                while (last_file->next != nullptr)
                {
                  last_file = last_file->next;
                }
                last_file->next = target_fcb;
                target_fcb->prev = last_file;
                target_fcb->next = nullptr;
              }
              
            }


            // add the filename to the directory file content
            {
              uchar * cwd_content = get_content(fs, fs->start_of_fcb[fs->cwd].start_block_idx, 0);
              int cwd_curr_size = fs->start_of_fcb[fs->cwd].size;
              uchar * input = new uchar[cwd_curr_size + my_strlen(s)];

              my_memcpy((char*)input, (char*)cwd_content, cwd_curr_size);
              my_memcpy((char*)(input+cwd_curr_size), (char*)s, my_strlen(s));
              
              printf("fs_open new fcb %s, index %d\n", s, i);
              
              fs_write(fs, input, cwd_curr_size + my_strlen(s), fs->cwd);

              delete[] input;
            }
            set_gtime_recursive(fs, i, gtime);

            return i;
          }
        }
        assert(0);  // no empty FCB
      }
      break;

    default:
      assert(0);  // no such option
      break;
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

  printf("fs_read %d bytes from %s\n", size, fcb.filename);
  
  // read `size` bytes to buffer `output`
  for (u32 i = 0; i < size; i++)
  {
    output[i] = start[i];
  } 
}

__device__ void block_move(FileSystem *fs, int target_block_idx, int source_block_idx) {
  printf("moving block %d to %d\n", source_block_idx, target_block_idx);

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
  printf("allocating %d blocks in alloc_new_blocks\n", target_block_size);
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
        printf("contiguous block found, returning block %d, span %d\n", current_block_idx - target_block_size + 1, target_block_size);
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

  printf("No enough contiguous blocks, have to manage the fragmentation\n");
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
        printf("reassigning fcb block %d to %d\n", fcb->start_block_idx, first_unused_block_idx);
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
  printf("reallocating %d blocks in alloc_new_blocks\n", target_block_size);
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
        printf("contiguous block found, returning block %d, span %d\n", current_block_idx - target_block_size + 1, target_block_size);
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
	// write bytes to the file
  // fp the index of the FCB
  gtime++;
  set_gtime_recursive(fs, fp, gtime);

  uchar *start = fs->start_of_contents + fs->start_of_fcb[fp].start_block_idx * fs->STORAGE_BLOCK_SIZE; // the initial byte of the file content
  FCB *fcb = fs->start_of_fcb+fp;   // the fcb for this file
  u16 start_block_idx = fcb->start_block_idx;
  
  printf("fs_write %d bytes into %s\n", size, fcb->filename);
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
    printf("directly writing %d blocks starting from block %d\n", block_of_bytes(fs, size), start_block_idx);
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
    
    return size;
    
  } else {
    // cannot directly write, need to fix fragmentation, then directly write
    fcb->start_block_idx = alloc_new_blocks(fs, block_of_bytes(fs, size));
    printf("resetting start_block_idx to %d\n", fcb->start_block_idx);
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
    return size;
  }
}

__device__ void pwd_helper(FileSystem *fs, int fcb_idx) {
  // to print cwd, this should be called `pwd_helper(fs, fs->cwd)`

  // recursively print the path of the current directory
  FCB *fcb = fs->start_of_fcb + fcb_idx;
  if (fcb->parent_dir_idx == -1) {
    // root directory
    return;
  }
  pwd_helper(fs, fcb->parent_dir_idx);
  printf("/%s", fcb->filename);
}


// ls_d, ls_s, cd_p, pwd goes here
__device__ void fs_gsys(FileSystem *fs, int op)
{
  FCB cwd_fcb = fs->start_of_fcb[fs->cwd];
  uchar *cwd_content = get_content(fs, cwd_fcb.start_block_idx, 0);
  // get the number of files and subdirectories **in the current directory**
  int file_count = 0;
  for (int i = 0; i < cwd_fcb.size; i++)
  {
    if (cwd_content[i] == '\0')
    {
      file_count++;
    }
  }
  
  printf("number of files or dirs: %d\n", file_count);

	/* Implement ls_d, ls_s, cd_p, pwd operation here */
  switch (op)
  {
  case PWD:
  {
    printf("printing pwd\n");
    // printf("fs.cwd: %d\n", fs->cwd);
    if (cwd_fcb.parent_dir_idx == -1) {
      // root directory
      printf("/\n");
      break;
    } else {
      pwd_helper(fs, fs->cwd);
      printf("\n");
      break;
    }
  }
  case CD_P:
  {
    // cd to parent dir
    if (cwd_fcb.parent_dir_idx == -1) {
      // root directory
      printf("already in root directory\n");
      break;
    } else {
      printf("changed directory to parent directory\n");
      fs->cwd = cwd_fcb.parent_dir_idx;
      break;
    }
  }
  case LS_D:
  {

    // list the file name and order by modified time of files

    printf("===sort by modified time===\n");

    int last_item_time = (1<<15); // trace the time of last printed file
    // print the most recent modified file before the last item
    for (int i = 0; i < file_count; i++)  // print each file once
    {
      int latest_modified_time = 0;
      FCB latest_fcb;

      // tokenize the file name in the cwd content
      uchar current_byte;
      int token_start_idx = 0;

      // find the most recently modified file or subdir
      for (int j = 0; j < cwd_fcb.size; j++)
      {
        current_byte = *get_content(fs, cwd_fcb.start_block_idx, j);
        if (current_byte == '\0')
        {
          // get this full token
          char token[21];
          my_memcpy(token, (char*)get_content(fs, cwd_fcb.start_block_idx, token_start_idx), 21);
          // get the fcb
          FCB *fcb = get_file_in_curr_dir(fs, token);
          if (fcb->is_on && (fcb->modified_time > latest_modified_time) && (fcb->modified_time < last_item_time))
          {
            latest_fcb = *fcb;
            latest_modified_time = fcb->modified_time;
          }
          
          token_start_idx = j+1;
        }
        
      }

      last_item_time = latest_fcb.modified_time;
      if (latest_fcb.is_dir)
      {
        printf("%s d\n", latest_fcb.filename);
      } else {
        printf("%s\n", latest_fcb.filename);
      }
      
      // printf("%s   time%d\n", latest_fcb.filename, last_item_time);
    }
    break;
  }

  case LS_S:
  {
      
    printf("===sort by file size===\n");
    // If there are several files with the same size, then first create first print.

    
    u32 last_item_size = (1<<31); // the distinct size of the last printed file
    int print_count = 0;

    while (print_count < file_count)
    {

      int largest_file_size = -1;
      // get the largest file size less than `last_item_size`
      FCB * curr = cwd_fcb.dir_files;
      while (curr != nullptr)
      {
        if (curr->is_on && (curr->size < last_item_size) && ((int)(curr->size) > largest_file_size))
        {
          largest_file_size = curr->size;
        }
        curr = curr->next;
      }
      
      // print all files with the same size
      last_item_size = largest_file_size;

      // printf("largest file size: %d\n", largest_file_size);

      // count the number of files with the size of largest_file_size

      int largest_file_count = 0;

      curr = cwd_fcb.dir_files;
      while (curr != nullptr)
      {
        if (curr->is_on && (curr->size == largest_file_size))
        {
          largest_file_count++;
        }
        curr = curr->next;
      }

      // printf("largest_file_count: %d\n", largest_file_count);

      // find the file with the file size of largest_file_size and the earliest created time among all unprinted items
      u16 last_item_time = 0;
      while (largest_file_count > 0)
      {
        u16 earliest_created_time = (1<<15);
        FCB *earliest_fcb;
        curr = cwd_fcb.dir_files;
        
        while (curr != nullptr)
        {
          assert(curr->is_on);
          if ((curr->size == largest_file_size) && (curr->creation_time < earliest_created_time) && (curr->creation_time > last_item_time))
          {
            earliest_created_time = curr->creation_time;
            earliest_fcb = curr;
            // printf("earliest_created_time: %d\n", curr->creation_time);
          }
          curr = curr->next;
        }
        last_item_time = earliest_created_time;
        largest_file_count--;
        print_count++;
        if (earliest_fcb->is_dir)
        {
          printf("%s %d d\n", earliest_fcb->filename, earliest_fcb->size);
        } else {
          printf("%s %d\n", earliest_fcb->filename, earliest_fcb->size);
        }
        
      }
    }
    break;
  }
  default:
    assert(0);
    break;  // no such option
  } // end of switch
  
}


// rm, cd, mkdir, rm_rf goes here
__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  // find the specific file in the FCB
  bool file_exists = file_exists_in_curr_dir(fs, s);
  FCB *target_fcb = get_file_in_curr_dir(fs, s);
  

	/* Implement rm operation here */
  if (op == RM)
  {
    // delete the specific file
    if (!file_exists)
    {
      assert(0);  // file not found
    } else {
      target_fcb->is_on = false;

      // remove the item in the directory's linked list
      assert(target_fcb->prev != nullptr);
      target_fcb->prev->next = target_fcb->next;
      if (target_fcb->prev->is_dir)
      {
        target_fcb->prev->dir_files = target_fcb->next;
      }
      
      if (target_fcb->next != nullptr)
      {
        target_fcb->next->prev = target_fcb->prev;
      }
           
      // remove the item in the parent dir's content
      {
        printf("removing %s in dir %s\n", target_fcb->filename, fs->start_of_fcb[fs->cwd].filename);

        uchar * cwd_content = get_content(fs, fs->start_of_fcb[fs->cwd].start_block_idx, 0);
        int cwd_curr_size = fs->start_of_fcb[fs->cwd].size;
        
        uchar *new_input = new uchar[cwd_curr_size];
        my_memcpy((char*)new_input, (char*)cwd_content, cwd_curr_size);

        // find the position of this filename
        uchar current_byte;
        int token_start_idx = 0;
        for (int j = 0; j < cwd_curr_size; j++)
        {
          current_byte = new_input[j];
          // printf("I am examining byte %c\n", current_byte);
          if (current_byte == '\0')
          {
            // printf("token starts with %c\n", *(new_input+token_start_idx));
            if (strmatch(target_fcb->filename, (char*)(new_input+token_start_idx)))
            {
              // printf("match\n");
              my_memcpy((char*)(new_input+token_start_idx), (char*)(new_input+j+1), my_strlen(target_fcb->filename));
              break;
            }
            token_start_idx = j+1;
          }
        }

        fs_write(fs, new_input, cwd_curr_size-my_strlen(target_fcb->filename), fs->cwd);
        delete[] new_input;
      }

      // free the content memory
      uchar *start = get_content(fs, target_fcb->start_block_idx, 0); // the initial byte of the file content
      printf("fs_delete removing %d bytes of %s, start from block %d span %d\n", target_fcb->size, target_fcb->filename, target_fcb->start_block_idx, block_of_bytes(fs, target_fcb->size));

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
    
  } else if (op == MKDIR)
  {
    // create a new directory
    if (file_exists) {
      assert(0);  // directory already exists
    } else {
      // append '/' at the end of s
      int len = my_strlen(s);
      char tmp[21];
      my_memcpy(tmp, s, len);
      tmp[len-1] = '/';
      tmp[len] = '\0';
      u32 fp = fs_open(fs, tmp, G_WRITE);
    }
  } else if (op == CD) {
    assert(file_exists);  // if assertion failed, the directory does not exist
    fs->cwd = target_fcb->dir_idx;
    printf("change directory to %s, index %d\n", s, fs->cwd);
  } else if (op == RM_RF) {
    // Remove the app directory and all its subdirectories and files recursively
    assert(file_exists);  // if assertion failed, the directory or file does not exist
    printf("RM_RF called, to remove %s\n", s);
    
    // locate the additional fcb
    if (!target_fcb->is_dir)
    {
      // remove that regular file only
      printf("rm_rf removing regular file %s\n", s);
      fs_gsys(fs, RM, s);
    } else {
      // recursively remove the directory, its subdirectories and files

      // first CD into the target dir, after it has been removed CD back
      fs_gsys(fs, CD, s);

      FCB *cwd_fcb = fs->start_of_fcb + fs->cwd;
      FCB *curr = cwd_fcb->dir_files;
      // move to the last one and move forward
      while (curr->next != nullptr)
      {
        curr = curr->next;
      }
      while (curr != cwd_fcb)
      {
        FCB *previous = curr->prev;
        if (!curr->is_dir)
        {
          printf("rm_rf removing regular file %s\n", curr->filename);
          fs_gsys(fs, RM, curr->filename);
        } else {
          printf("rm_rf removing dir %s\n", curr->filename);
          fs_gsys(fs, RM_RF, curr->filename);
        }
        curr = previous;
      }
      // CD back
      fs_gsys(fs, CD_P);

      // since we have emptied the dir's subunits, we can now remove it as a regular file
      fs_gsys(fs, RM, target_fcb->filename);

    }
    
  }
  
  
}
