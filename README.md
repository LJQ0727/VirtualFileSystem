# Assignment 4 Project Report

Li Jiaqi  120090727

# Overview

In this assignment, I have completed both the basic single-directory File System and the bonus task of tree-structured directory File System. The following content will describe the relevant information of these two tasks I completed.

# Environment

## OS Version

I use the university’s High Performance Cluster (HPC) for testing and running the CUDA program. The nodes run on a CentOS version 7.5.1804. 

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled.png)

## Kernel version

This is the kernel version of the HPC. Other versions should also be OK.

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%201.png)

## CUDA Version

I use the CUDA compiler version 11.7 for compiling the CUDA program. 

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%202.png)

## GPU Info

For each node in the HPC, it is equipped with a Quadro RTX 4000 GPU. Each time the program only runs on one allocated node. I have also tested my programs on a RTX3090 GPU.

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%203.png)

# Running the program

## Basic task compilation and running

To compile: inside the `source/` folder, there is a file named `slurm.sh`. On the HPC with slurm installed, we can directly use the shell script to compile and run the executable:

```
sbatch slurm.sh
```

On a device without slurm, one can first compile using 

```
nvcc --relocatable-device-code=true main.cu user_program.cu file_system.cu -o test
```

and run `./test` to run the program (might need `srun` in the cluster).

## Basic task sample outputs

On the first test program: (The first 36 lines are compiler warnings)

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%204.png)

On the second test program: (The first 36 lines are compiler warnings)

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%205.png)

On the third test program: (there are too many lines and we only display the begin, middle and end)

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%206.png)

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%207.png)

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%208.png)

One the fourth test case: (there are too many lines and we only screenshot the beginning and end)

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%209.png)

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2010.png)

I checked that for test case 4, the `snapshot.bin` is the same as `data.bin`:

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2011.png)

For test case 3, the `snapshot.bin` is the same as expected, where 

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2012.png)

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2013.png)

For test case 2 and 1, with the offset, the snapshots behave as expected:

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2014.png)

## Bonus Task Compilation and running

Because the bonus task shares the same template structure with the basic task, the compilation and running steps are exactly the same as above, which means we could use:

```
sbatch slurm.sh
```

## Bonus task sample output

For test case 1 and 2, I have checked that the output as well as the snapshot comparison is the same as the basic task, please refer to the previous results.

For the bonus test case, the output is shown below, the first 91 lines are compiler warnings.

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2015.png)

# Program Design

## Basic task design

In this CUDA program, we implement a single-directory file system using a limited GPU memory pool. The memory usage strictly obeys the one in the instruction, that no extra global memory is maintained or used. Temporary usage for function stack is limited. 

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2016.png)

For the task, we allocate a volume of 1060kb with the 4kb volume control block using bitmap, 32kb for 1024 FCBs, each FCB is 32 bytes. The content of files use 1024kb, divided into storage blocks each 32 bytes.

### FCB Structure

Here is the FCB structure I used. Note that this does not create extra memory space, we just turn the specific portion of volume, originally `uchar*` to `FCB*` for better information storage and retrieval. I have tested that the `sizeof(FCB)` is 32, which is exactly the size of desired FCB. The attributes are self-explanatory.

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2017.png)

### Allocation strategy

The maximum file size allowed is the total content size of files, 1024 KB. I use dynamic, contiguous allocation together with compaction algorithm to maintain the FS. The dynamic scheme is to allow storing this maximum size of file. 

The contiguous allocation is one such that adjacent blocks store the file content sequentially, as illustrated in the figure: 

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2018.png)

The compaction algorithm is utilized when there is fragmentation and a newly written file cannot file enough space. In my compaction algorithm implementation, I maintain a pointer to the first unused block and the first used block moving forward together. They are constantly swapped, which in effect “compacts” all the used blocks to the front. In the mean time, we update the FCB’s content block attribute.

![Compaction algorithm](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2019.png)

Compaction algorithm

For the superblock, I use the bitmap, which uses one bit for one content block to indicate on or off. 

### Designing the APIs

All the required APIs `fs_open`, `fs_read`, `fs_write`, `fs_gsys` (including LS_S, LS_D, rm operations) are implemented. The `fs_open` returns an fp, which is the index of the FCB in the FCB array. Another interesting point is for the LS_D and LS_S operations. I did not use external storage for these two sorting operations. Instead, my implementation is simple, which is in each time, traverse all files and find the largest element that is not printed. This does not need to re-place the blocks. The following figure shows my implementation of `LS_D`. 

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2020.png)

The LS_S is more tricky but still uses the above idea. We do three traverses in total. First we traverse each item to get the largest unprinted size of files. Then we traverse to get the count of the largest size files. Then find the file with the file size of largest_file_size and the **earliest created time** among all unprinted items.

## Bonus task design

The bonus task is based upon the basic task with modification to add files for directories.

Firstly, we need to add a new attribute to trace the current working directory. I add to the `fs` struct.

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2021.png)

Then, the FCB should be different to record each file’s directory index. 

We squeeze the `is_on` and `is_dir` 1-bit attribute to the first two bits of `u32 size`. So this is 32 bytes again.

![Untitled](Assignment%204%20Project%20Report%20df997535fd09484fbe8700aea427099a/Untitled%2022.png)

My implementation does not use extra global memory. Other implementations are similar to basic task. Because we record the file contents or subdirectory names in the content of directory, we need to traverse and check the filename match and the `dir_idx` match the cwd.

All required operations are supported, including the extra command `MKDIR`,  `PWD`, `CD` , `RM_RF` and `CD_P`, in addition to the ones in basic task.

My implementation also supports absolute addressing to increase robustness.

# Project reflection and conclusion

Several problems I met in this assignment gave me valuable experience in solving them. The first is 

about data structure used to implement FCB in bonus. At first I thought 32 bytes is not enough and want to implement a doubly linked list in the bonus. However, later I found that I could squeeze the FCB on/off bits, and can use a filename match traversing strategy instead of linked list. This allows mroe efficient storage utilization. 

I think this project is a valuable experience for learning the FS, including dynamic allocation, contiguous allocation, compaction. I also learn the technique of writing CUDA programs, which are somewhat like C/C++ but have restricted access to some standard library routines, like `memcpy`.