# 2 CUDA Programming Model

## 引入CUDA编程模型 INTRODUCING THE CUDA PROGRAMMING MODEL

编程模型是对计算机架构的抽象，它是应用（Application）与底层硬件的桥梁。  
通信抽象是程序与编程模型实现之间的边界，它通过编译器或库使用特权硬件原语（privileged hardware primitives）和操作系统来实现。

![重要层抽象](pic/1%20重要层的抽象.png "the important layers of abstraction")

相比于一般的并行编程模型，CUDA编程模型提供了两个特殊功能：
* 组织线程
* 访问显存

从编程者视角，我们可以从不同层级去看待编程模型：
* Domain level
  * 当设计程序和算法时，我们关心的是如何分解数据和函数，这就是domain level；
* Logic level
  * 在编程阶段，我们关心如何组织线程，此为logic level；
* Hardware level
  * 需要理解线程如何映射到核上，这样可以帮助改进程序。

### CUDA Programming Structure
* Unified Memory
  * CUDA 6开始出现，它可以使编程人员通过单个指针访问CPU和GPU中的数据。

CUDA编程一个关键组成部分为内核（kernel），内核指的是在GPU上运行的代码。kernel可以看作一个串行程序，CUDA将会将kernel分配到线程上。  
当kernel运行时，控制权立马返回给host。

一个典型的CUDA程序处理流为：
1. 将CPU的数据复制到GPU；
2. 调用内核，在GPU存储的数据上进行计算；
3. 将数据从GPU复制回CPU。

### Managing Memory
CUDA runtime提供了管理device内存的函数，它们和C语言相似，如表格所示：

|STANDARD C FUNCTIONS|CUDA C FUNCTIONS|
|:--:|:--:|
|malloc|cudaMalloc|
|memcpy|cudaMemcpy|
|memset|cudaMemset|
|free|cudaFree|

* `cudaError_t cudaMalloc (void** devPtr, size_t size)`  
关于void**，首先，void\*为一个通用指针类型，可以用它指向GPU的地址。而通过void**，函数在GPU上开辟空间后，可以在存储void指针的地址，直接使void指针指向GPU中开辟出的区域。  
这个函数会在device内存上分配线性空间。

* `cudaError_t cudaMemcpy (void* dst, const void* src, size_t count, cudaMemcpyKind kind)`  
kind可以有四种：1. cudaMemcpyHostToHost; 2. cudaMemcpyHostToDevice, 3. cudaMemcpyDeviceToHost; 4.cudaMemcpyDeviceToDevice。

以上这两个函数对于host都是同步函数，这些函数运行时，host会阻塞，等到它们返回结果后才会继续运行。实际上，除了kernel函数，其它所有CUDA函数都会返回枚举类型`cudaError_t`。如，若GPU的内存成功开辟，则返回：  
`cudaSuccess`  
否则会返回
`cudaErrorMemoryAllocation`  
若要将错误代码转化为可读信息，则可以使用函数：  
`char* cudaGetErrorString(cudaError_t error)`

在GPU的存储层次结构中，全局内存（global memory）和共享内存（shared memory）是两个很重要的存储类型。以CPU为参照，全局内存相当于CPU的内存层次，而共享内存相当于cache。GPU的共享内存可以由kernel直接控制。

![内存管理](pic/2%20内存管理.png "managing memory")

对于cuda内存控制函数应用示例：

``` c
float *d_A, *d_B, *d_C;
cudaMalloc((float**)&d_A, nBytes);
cudaMalloc((float**)&d_B, nBytes);
cudaMalloc((float**)&d_C, nBytes);
```
``` c
cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);
```
``` c
cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
```
``` c
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```
### Organizing Threads
由单个kernel启动所产生的所有线程统称为grid，grid里所有线程共享全局内存，一个grid由多个线程块组成；  
一个线程块（thread block）由一组线程组成，这些线程可以通过   
(1) 块内同步  
(2) 块内共享内存  
来进行协作。

Thread可以通过：  
(1) blockIdx（一个grid中对block的索引）  
(2) threadIdx（一个block中对thread的索引）  
来将自身与其它线程区别开来。这两个是内置且初始化好了的变量，可以被内核函数直接获取。  
这两个定位变量的类型为uint3，它是CUDA内置的向量类型，由最基本的整型派生而成。该结构包含3个无符号整型，分别对应x、y、z。即：  
```
  blockIdx.x
  blockIdx.y
  blockIdx.z

  threadIdx.x
  threadIdx.y
  threadIdx.z
```
grid和block三个维度的大小由两个变量指定：  
* blockDim
* gridDim
这两个变量的类型为dim3，它是基于uint3的整型向量类型，用于指定维度。在定义int3变量时，任何没有赋值的部分将会被初始化为1。dim3三部分依旧和x、y、z分别对应，如：
```
blockDim.x
blockDim.y
blockDim.z
```
下面程序示例kernel函数中如何获取线程的坐标。

``` c
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void) {
    printf(
        "threadIdx: (%d, %d, %d)   blockIdx: (%d, %d, %d)   blockDim: (%d, %d, %d)   "
        "gridDim: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, 
        blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, 
        gridDim.x, gridDim.y, gridDim.z
    );
}

int main() {
    int nElem = 6;

    dim3 block(3);
    dim3 grid ((nElem+block.x-1)/block.x);

    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);


    checkIndex<<<grid, block>>>();

    cudaDeviceReset();

    return(0);
}
```
在用nvcc进行编译时，如果没有用-arch参数指定虚拟架构类型，那么默认虚拟架构为当前nvcc所支持的最低架构。

已知数据大小，决定grid和block维度的一般步骤为：
* 决定block维度大小
* 依据block大小和数据大小计算grid维度

为了决定block维度，需要考虑：
* kernel的性能特性
* GPU的资源限制
### Launching a CUDA Kernel
CUDA kernal调用是对C函数语法的直接扩展，它增加了三角括号，对kernel的执行进行配置：   
`kernal_name <<<grid, block>>>(argument list)`
通过对grid和block进行配置，我们可以决定：   
* 为内核分配多少线程
* 线程布局

由于数据在全局内存中线性存储，我们可以使用内置变量blockIdx和threadIdx：
* 标识grid中的每一个线程
* 建立线程和data元素的映射

调用kernel后，控制权将会立刻回到host中，要等待kernel执行结束，可以使用：  
`cudaError_t cudaDeviceSynchronize(void);`  
而一些CUDA runtime APIs在主机和设备之间是同步的。如cudaMemcpy，完成复制之前host必须等待。

### Writing Your Kernel
用__global__来定义kernel:  
`__global__ void kernel_name(argument list)`  
kernel的返回类型必须为void。
实际上，函数类型声明（如__global__）指定
(1) 函数在host还是device被调用，以及(2) 函数被host还是device调用。__device__和__host__限定符可以一起被使用，如此函数编译后，在device和host均可以运行。

|限定符|执行位置|可调用的设备|标注|
|:---:|:---:|:---:|:---:|
|\_\_global\_\_|device|host和device|返回类型必须为void|
|\_\_device\_\_|host|device||
|\_\_host\_\_|host|host|可以省略|

总结，kernel函数的限制有：
* 只能访问device内存
* 返回类型必须为void
* 不支持可变数量的参数
* 不支持静态变量
* 不支持函数指针
* 不支持异步行为

### Verifying Your Kernel
有两个基本方法来验证内核函数：  
1. 使用printf；
2. 设置grid和block为`<<<1,1>>>`，此时程序将会线性执行。

### Handling Errors
用宏实现：
``` c
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} \
```
call传入的是cudaError_t的实例。

## Timing Your Kernel
### Timing with CPU Timer
我们可以使用CPU时间去测量。首先包含sys/time.h头文件，编写获取时间的函数：
``` c
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
```
使用方式可以如下所示：
``` c
  iStart = cpuSecond();
  sumArraysOnGPU <<<grid, block>>> (d_A, d_B, d_C, nElem);
  cudaDeviceSynchronize();
  iElaps = cpuSecond() - iStart;
```

### Timing with nvprof
可以使用命令行分析工具，叫nvprof，可以搜集程序信息。使用方法为：  
`nvprof [nvprof_args] <application> [application_args]`  
如：
`nvprof ./sumArraysOnGPU-timer`

nvprof获得的时间参数会比host统计的时间更加准确，host统计时间还包括nvprof的开销。

我们可以将App的表现与理论限制相比。

## Organizing Parallel Threads
这一章，我们将以各种形式的grid和block来对矩阵进行计算。此处，我们矩阵在内存中的表现实际上也是一个一维数组，一个$6 \times 8$的矩阵如图所示进行表示：

![矩阵存储](./pic/3%20矩阵存储.png "矩阵存储")

在之后展现的内核函数中，每一个线程通常只处理一个数据元素。一般，我们在kernel中需要处理得到的信息有：  
* 线程索引和块索引
* 数据点在矩阵中对应的坐标（表现为二元组）
* 数据点在全局内存中（表现为一维数组）的存储位置  

![block和grid与矩阵](./pic/4%20block和grid与矩阵.png "block和grid与矩阵")

线程索引、块索引可分别用threadIdx、blockIdx来获取。数据点在矩阵中的坐标可以通过：  
$ix = threadIdx.x + blockIdx.x * blockDim.x$  
$iy = threadIdx.y + blockIdx.y * blockDim.y$   
来获取，而全局内存的坐标，通过  
$idx = ix + iy * nx$    
进行计算，其中 nx 是矩阵每一行中元素个数

### Summing Matrices with a 2D Grid and 2D Blocks

当grid和block均为二维时，矩阵被划分的方式为前图所示，在kernel中对矩阵进行计算的代码可以为：
``` c
__global__ void sumMatrixOnGPU2D(float* MatA, float* MatB, float* MatC, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}
```
通过对block进行调整，程序计算时间将会有所不同：
```
int dimx = 32;
int dimy = 32;
Using Device 0: NVIDIA GeForce GTX 1650
Matrix size: nx 16384 ny 16384
sumMatrixOnGPU2D <<<(512,512), (32,32)>>> elapsed 0.023089 sec

int dimx = 32;
int dimy = 16;
Using Device 0: NVIDIA GeForce GTX 1650
Matrix size: nx 16384 ny 16384
sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> elapsed 0.022963 sec

int dimx = 16;
int dimy = 16;
Using Device 0: NVIDIA GeForce GTX 1650
Matrix size: nx 16384 ny 16384
sumMatrixOnGPU2D <<<(1024,1024), (16,16)>>> elapsed 0.022974 sec
```
结果组织为如下表格：
|kernel配置|kernel计算时间|block数目|
|:---:|:---:|:---:|
|(32, 32)|0.023089 sec|512 x 512|
|(32, 16)|0.022963 sec|512 x 1024|
|(16, 16)|0.022974 sec|1024 x 1024|

### Summing Matrices with a 1D Grid and 1D Blocks

在这个情况下，block和grid的组织情况为(block_size, 1, 1)和(grid_size, 1, 1)。此时，对于block、grid在矩阵上的布局可以采取如图所示的策略：

![1D blcok 1D grid和矩阵](./pic/5%201D%20blcok%201D%20grid和矩阵.png "1D blcok 1D grid和矩阵")

每个线程处理ny（即一列）个数据，每个block处理的数据量即为block_size * ny个。此时，内核可以按如下方式编写：
``` c
__global__ void sumMatrixOnGPU1D(float* MatA, float* MatB, float* MatC, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;

    if (ix < nx) {
        for (int iy = 0;iy < ny;++iy) {
            int idx = iy*nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}
```
内核中，对于每一个ix，我们都算了对应的一列数据，一共ny个。
调整其block大小，可以获得如下结果：
``` 
Using Device 0: NVIDIA GeForce GTX 1650
Matrix size: nx 16384 ny 16384
sumMatrixOnGPU1D <<<(512,1), (32,1)>>> elapsed 0.040942 sec

Using Device 0: NVIDIA GeForce GTX 1650
Matrix size: nx 16384 ny 16384
sumMatrixOnGPU1D <<<(128,1), (128,1)>>> elapsed 0.039020 sec
```

### Summing Matrices with a 2D Grid and 1D Blocks
相比于1D grid和1D block，grid多了y维度，在此处我们可以用来定位矩阵的y维度坐标。可以堪称这个y维度将上图中每个block中覆盖的矩阵的y维度进行切割。我们可以令grid.y = ny，这样每个block只会处理一行的一部分数据，如下图表示：

![1D block 2D grid和矩阵](./pic/6%201D%20block%202D%20grid和矩阵.png "1D block 2D grid和矩阵")

对应的kernel实现为：
``` c
__global__ void sumMatrixOnGPUMix(float* MatA, float* MatB, float* MatC, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}
```
调整其kernel配置，获得的运行速度如下所示：
```
Using Device 0: NVIDIA GeForce GTX 1650
Matrix size: nx 16384 ny 16384
sumMatrixOnGPUMix <<<(512,16384), (32,1)>>> elapsed 0.030278 sec

Using Device 0: NVIDIA GeForce GTX 1650
Matrix size: nx 16384 ny 16384
sumMatrixOnGPUMix <<<(64,16384), (256,1)>>> elapsed 0.022409 sec
```

统计一下，每种block、grid组织方式，在我们进行的配置中，最好的运行时间和对应的配置如下表所示：

|内核|运行配置|时间|
|:---:|:---:|:---:|
|sumMatrixOnGPU1D|(512,1024), (32,16)|0.022963 sec|
|sumMatrixOnGPU1D|(128,1), (128,1)|0.039020 sec|
|sumMatrixOnGPU1D|(64,16384), (256,1)|0.022409 sec|

从中我们可以发现：
* 改变配置会影响运行表现
* 只有简单的内核运行（没有对配置进行探究）常常不会有最好的运行效果
* 对于一个给定的内核，尝试不同维度的grid和block会找到更好的表现

## Managing Devices

NVIDIA提供了一系列方法用于查询、管理GPU。有两个基本方法：  
1. CUDA runtime API函数
2. NVIDIA Systems Management Interface(nvidia-smi) command-line utility

### 使用Runtime API查询GPU信息
可以使用以下函数对GPU进行查询：  
`cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);`  
CUDA信息返回于cudaDeviceProp结构体，关于该结构体可以于https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp查询。以下是该函数相关使用代码：

``` c
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {

    printf("%s Starting...\n", argv[0]);

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);    

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n -> %prints\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("THrer are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("    CUDA Driver Version / Runtime Version           %d.%d / %d.%d\n",
        driverVersion/1000, (driverVersion % 100) / 10,
        runtimeVersion/1000, (runtimeVersion % 100) / 10
    );
    printf("    CUDA Capability Major/Minor version number:     %d.%d\n",
        deviceProp.major, deviceProp.minor
    );
    printf("    Total amount of global memory:                  %.2f GBytes (%llu bytes)\n",
        (float)deviceProp.totalGlobalMem/(pow(1024.0, 3)),
        (unsigned long long) deviceProp.totalGlobalMem
    );
    printf("    GPU Clock rate:                                 %.0f MHz (%0.2f GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6
    );
    printf("    Memory Bus Width:                               %d-bit\n",
        deviceProp.memoryBusWidth
    );

    if (deviceProp.l2CacheSize) {
        printf("    L2 Cache Size:                                  %d bytes\n",
            deviceProp.l2CacheSize
        );
    }

    printf("    Max Texture Dimension Size (x, y, z)            "
    "1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
        deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
        deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]
    );

    printf("    Max Layered Texture Size (dim) x layers         1D=(%d) x %d, 2D=(%d,%d) x %d\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]
    );
    printf("    Total amount of constant memory                 %lu bytes\n",
        deviceProp.totalConstMem
    );
    printf("    Total amount of shared memory per block:        %lu bytes\n",
        deviceProp.sharedMemPerBlock
    );
    printf("    Total number of registers available per block:  %d\n",
        deviceProp.regsPerBlock
    );
    printf("    Warp size:                                      %d\n",
        deviceProp.warpSize
    );
    printf("    Maximun number of threads per multiprocessor:   %d\n",
        deviceProp.maxThreadsPerMultiProcessor
    );
    printf("    Maximun number of threads per block:            %d\n",
        deviceProp.maxThreadsPerBlock
    );
    printf("    Maximum sizes of each dimension of a block:     %d x %d x %d\n",
        deviceProp.maxThreadsDim[0],
        deviceProp.maxThreadsDim[1],
        deviceProp.maxThreadsDim[2]
    );
    printf("    Maximum sizes of each dimension of a grid:      %d x %d x %d\n",
        deviceProp.maxGridSize[0],
        deviceProp.maxGridSize[1],
        deviceProp.maxGridSize[2]
    );
    printf("    Maximum memory pitch:                           %lu bytes\n",
        deviceProp.memPitch
    );

    exit(EXIT_SUCCESS);
}
```

若有多个GPU，要决定使用哪个GPU，可以以GPU的multiprocessor数目为依据进行选择。

### 使用nvidia-smi进行查询
若要查询电脑上有多少GPU，可以使用：  
`nvidia-smi -L`  

若要查询GPU0的所有信息，可以使用：  
`nvidia-smi -q -i 0`  

当然，也可以用-d参数指定所要获取的信息。-d参数可以接收的内容如下：  
➤ MEMORY  
➤ UTILIZATION  
➤ ECC  
➤ TEMPERATURE  
➤ POWER  
➤ CLOCK  
➤ COMPUTE  
➤ PIDS  
➤ PERFORMANCE  
➤ SUPPORTED_CLOCKS  
➤ PAGE_RETIREMENT  
➤ ACCOUNTING  
如，可以使用
`nvidia-smi -q -i 0 -d MEMORY`  
来获取内存的信息。