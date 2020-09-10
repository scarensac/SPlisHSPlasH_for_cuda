
#include "DFSPH_cuda_basic.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <iomanip>

#include "DFSPH_define_cuda.h"
#include "DFSPH_macro_cuda.h"
#include "DFSPH_static_variables_structure_cuda.h"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DFSPH_c_arrays_structure.h"
#include "cub.cuh"


#include <curand.h>
#include <curand_kernel.h>

#include "basic_kernels_cuda.cuh"


#include <sstream>


/*
AFTER THIS ARE ONLY THE TEST FUNCTION TO HAVE CUDA WORKING ...
*/


inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}

inline __host__ __device__ float4 operator*(float4& a, RealCuda b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, 0);
}

inline __host__ __device__ float3 operator*(float3& a, RealCuda b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}



template<typename T>
__global__ void test_vector_type_kernel(T* v1, T* v2, RealCuda factor, int count_elem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count_elem) { return; }

    v1[i]+=v2[i]*factor;
}

void compare_vector3_struct_speed(){
    RealCuda factor=0.001;
    int count_elem= 1000000;
    Vector3d* v1_v3d;
    Vector3d* v2_v3d;
    float3* v1_f3;
    float3* v2_f3;
    float4* v1_f4;
    float4* v2_f4;
    Vector3d* v1_v3d_2=new Vector3d[count_elem];
    Vector3d* v2_v3d_2=new Vector3d[count_elem];
    float3* v1_f3_2=new float3[count_elem];
    float3* v2_f3_2=new float3[count_elem];
    float4* v1_f4_2=new float4[count_elem];
    float4* v2_f4_2=new float4[count_elem];
    cudaMalloc(&(v1_v3d), count_elem * sizeof(Vector3d));
    cudaMalloc(&(v2_v3d), count_elem * sizeof(Vector3d));

    cudaMalloc(&(v1_f3), count_elem * sizeof(float3));
    cudaMalloc(&(v2_f3), count_elem * sizeof(float3));

    cudaMalloc(&(v1_f4), count_elem * sizeof(float4));
    cudaMalloc(&(v2_f4), count_elem * sizeof(float4));

    for (int i=0;i<count_elem;++i){
        v1_v3d_2[i]=i;
        v2_v3d_2[i]=i;
        v1_f3_2[i]=make_float3(i);
        v2_f3_2[i]=make_float3(i);
        v1_f4_2[i]=make_float4(i);
        v2_f4_2[i]=make_float4(i);
    }

    gpuErrchk(cudaMemcpy(v1_v3d, v1_v3d_2,count_elem * sizeof(Vector3d),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v2_v3d, v2_v3d_2,count_elem * sizeof(Vector3d),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v1_f3, v1_f3_2,count_elem * sizeof(float3),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v2_f3, v2_f3_2,count_elem * sizeof(float3),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v1_f4, v1_f4_2,count_elem * sizeof(float4),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v2_f4, v2_f4_2,count_elem * sizeof(float4),cudaMemcpyHostToDevice));

    int numBlocks = (count_elem + BLOCKSIZE - 1) / BLOCKSIZE;
    gpuErrchk(cudaDeviceSynchronize());

    float avg0=0;
    float avg1=0;
    float avg2=0;

    int iter=10;
    for (int i=0;i<iter;++i){
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        test_vector_type_kernel<float3> << <numBlocks, BLOCKSIZE >> > (v1_f3, v2_f3, factor, count_elem);
        gpuErrchk(cudaDeviceSynchronize());

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        test_vector_type_kernel<Vector3d> << <numBlocks, BLOCKSIZE >> > (v1_v3d, v2_v3d, factor, count_elem);
        gpuErrchk(cudaDeviceSynchronize());

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        test_vector_type_kernel<float4> << <numBlocks, BLOCKSIZE >> > (v1_f4, v2_f4, factor, count_elem);
        gpuErrchk(cudaDeviceSynchronize());

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

        float time_0 = std::chrono::duration_cast<std::chrono::nanoseconds> (t1 - t0).count() / 1000000.0f;
        float time_1 = std::chrono::duration_cast<std::chrono::nanoseconds> (t2 - t1).count() / 1000000.0f;
        float time_2 = std::chrono::duration_cast<std::chrono::nanoseconds> (t3 - t2).count() / 1000000.0f;

        printf("comparison between vector data struct  (float3, Vector3d, float4): %f   %f   %f\n", time_0, time_1, time_2);

        if (iter>0){
            avg0+=time_0;
            avg1+=time_1;
            avg2+=time_2;
        }
    }
    iter--;

    printf("comparison between vector data struct Global (float3, Vector3d, float4): %f   %f   %f\n",
           avg0/iter, avg1/iter, avg2/iter);

}







cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
//*
__global__ void addKernel(Vector3d* vect)
{
    int i = threadIdx.x;
    vect[i].z = vect[i].x + vect[i].y;
}

__global__ void setVectkernel(Vector3d& vect)
{
    vect.x = 5;
    vect.y = 6;
    vect.z = 7;
}
//*/
int test_cuda()
{
    //DFSPHCData* data;
    std::cout << "start cuda test basic" << std::endl;

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    //*
    Vector3d* vect;
    cudaMallocManaged(&vect, arraySize * sizeof(Vector3d));
    for (int i = 0; i < arraySize; ++i) {
        vect[i].x = a[i];
        vect[i].y = b[i];
    }
    //*/*

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }


    printf("macro val: %d, %d, %d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
           c[0], c[1], c[2], c[3], c[4]);

    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }


    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, arraySize >> > (vect);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    printf("with vects {1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
           (int)(vect[0].z), (int)(vect[1].z), (int)(vect[2].z), (int)(vect[3].z), (int)(vect[4].z));

    cudaFree(vect);



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    /*
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
    }
    //*/

    printf("Finished test cuda\n");


    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}



void test_particleshift() {
    compare_vector3_struct_speed();

#define count_pos 100
    RealCuda pos[count_pos];
    pos[0] = 0;

    auto normalize_string = [](std::string s, int length) {
        int to_add = length - s.length();
        std::ostringstream oss;
        oss << s;
        for (int i = 0; i < to_add; ++i) {
            oss << " ";
        }
        return oss.str();
    };

#define W(p1,p2) (MAX_MACRO_CUDA(1-abs(p2-p1),0))
#define W_SIGNED(p1,p2)  W(p1,p2)*((p2>p1)?1:-1)
#define W_GRAD(p1,p2) ((W(p1,p2)>0)?((p2-p1)<0?-1:1):0)

    RealCuda mass[count_pos];
    RealCuda density[count_pos];
    RealCuda c[count_pos];
    RealCuda nabla_c[count_pos];
    for (int i = 0; i < count_pos; ++i) {
        mass[i] = 1;

        if (i > 0 && i < 33) {
            pos[i] = pos[i - 1] + 0.5;
        }else if(i>32){
            pos[i] = pos[i - 1] + 0.25;
        }

       // std::cout << "pos: " << i << "  " <<  pos[i] << std::endl;

    }

    std::vector<std::vector<RealCuda>> hist_pos;
    std::vector<std::vector<RealCuda>> hist_den;

    //concentration based
    if(false){
        for (int repeat = 0; repeat < 5; ++repeat) {

            for (int i = 0; i < count_pos; ++i) {
                density[i] = 0;
                for (int j = 0; j < count_pos; ++j) {
                    density[i] += mass[j]*W(pos[j],pos[i]);
                    if (i == 33) {
                        //std::cout << "W/grad: " <<j<<"  "<< W(pos[j], pos[i]) <<"  "<< W_GRAD(pos[j], pos[i]) << std::endl;
                    }
                }
            }


            for (int i = 0; i < count_pos; ++i) {
                c[i] = 0;
                for (int j = 0; j < count_pos; ++j) {
                    if (i != j) {
                        c[i] += (mass[j] / density[j]) * W(pos[j], pos[i]);
                    }
                }
            }

            for (int i = 0; i < count_pos; ++i) {
                nabla_c[i] = 0;
                for (int j = 0; j < count_pos; ++j) {
                    if (i != j) {
                        nabla_c[i] += (c[j]-c[i]) * (mass[j] / density[j]) * W_GRAD(pos[j], pos[i]);
                    }
                }
            }


            /*
            
            for (int i = 0; i < count_pos; ++i) {
                std::ostringstream oss;
                oss << std::setprecision(2) << pos[i];
                std::cout << normalize_string(oss.str(), 6) << "   ";
            }
            std::cout << std::endl;

            for (int i = 0; i < count_pos; ++i) {
                std::ostringstream oss;
                oss << std::setprecision(2) << density[i];
                std::cout << normalize_string(oss.str(), 6) << "   ";
            }
            std::cout << std::endl;

            for (int i = 0; i < count_pos; ++i) {
                std::ostringstream oss;
                oss << std::setprecision(2) << c[i];
                std::cout << normalize_string(oss.str(), 6) << "   ";
            }
            std::cout << std::endl;

            for (int i = 0; i < count_pos; ++i) {
                std::ostringstream oss;
                oss << std::setprecision(2) << nabla_c[i];
                std::cout << normalize_string(oss.str(), 6) << "   ";
            }
            std::cout << std::endl;
            //*/
            std::vector<RealCuda> vect_pos;
            std::vector<RealCuda> vect_den;
            for (int i = 0; i < count_pos; ++i) {
                vect_pos.push_back(pos[i]);
                vect_den.push_back(density[i]);
            }
            hist_pos.push_back(vect_pos);
            hist_den.push_back(vect_den);


            RealCuda pos2[count_pos];
            RealCuda density2[count_pos];

            RealCuda factor = 0.5;
            for (int i = 0; i < count_pos; ++i) {
                pos2[i] = pos[i] + nabla_c[i] * factor;
            }

            for (int i = 0; i < count_pos; ++i) {
                density2[i] = 0;
                for (int j = 0; j < count_pos; ++j) {
                    density2[i] += mass[j] * W(pos2[j], pos2[i]);
                }
            }


            for (int i = 0; i < count_pos; ++i) {
                std::cout << pos[i] << "   " << density[i] << "   " << c[i] << "   " << nabla_c[i] << "   " << pos2[i] << "   " << density2[i] << "   ";
                std::cout << std::endl;
            }

            for (int i = 0; i < count_pos; ++i) {
                pos[i] = pos2[i];
            }

        }
    }

    //simply densty based
    //not much better
    for (int repeat = 0; repeat < 5; ++repeat) {
        for (int i = 0; i < count_pos; ++i) {
            density[i] = 0;
            for (int j = 0; j < count_pos; ++j) {
                density[i] += mass[j] * W(pos[j], pos[i]);
            }
        }

        for (int i = 0; i < count_pos; ++i) {
            nabla_c[i] = 0;
            for (int j = 0; j < count_pos; ++j) {
                if (i != j) {
                    nabla_c[i] += (density[j] - density[i])* (mass[j]/density[j]) * W_SIGNED(pos[j], pos[i]);
                }
            }
        }

        for (int i = 0; i < count_pos; ++i) {
            //std::cout << pos[i] << "   " << density[i] << "   " << c[i] << "   " << nabla_c[i];
            {
                std::ostringstream oss;
                oss << std::setprecision(2) << pos[i];
                std::cout << normalize_string(oss.str(), 6) << "   ";
            }
            {
                std::ostringstream oss;
                oss << std::setprecision(2) << density[i];
                std::cout << normalize_string(oss.str(), 6) << "   ";
            }
            {
                std::ostringstream oss;
                oss << std::setprecision(2) << nabla_c[i];
                std::cout << normalize_string(oss.str(), 6) << "   ";
            }
            std::cout << std::endl;
        }

        std::vector<RealCuda> vect_pos;
        std::vector<RealCuda> vect_den;
        for (int i = 0; i < count_pos; ++i) {
            vect_pos.push_back(pos[i]);
            vect_den.push_back(density[i]);
        }
        hist_pos.push_back(vect_pos);
        hist_den.push_back(vect_den);


        for (int i = 0; i < count_pos; ++i) {
            if (abs(nabla_c[i]) > 0.15) {
                nabla_c[i] *= 0.15 / abs(nabla_c[i]);
            }
            pos[i] += nabla_c[i];
        }
    }
    

    //ok so let's trys with a regular space sampling


    /*
    std::cout << "end results" << std::endl;
    std::cout << "end results" << std::endl;
    std::cout << "end results" << std::endl;
    std::ostringstream oss;
    for (int k = 0; k < hist_pos.size(); ++k) {
        oss << "pos_gen_"<< k << "  " << "den_gen_" << k << "  ";
    }
    oss << std::endl;
    for (int i = 0; i < count_pos; ++i) {
        for (int k = 0; k < hist_pos.size(); ++k) {
            oss << hist_pos[k][i] << "  " << hist_den[k][i] << "  ";
        }
        oss << std::endl;
    }
    std::cout <<oss.str();
    //*/

#undef W
#undef W_SIGNED
#undef W_GRAD
}