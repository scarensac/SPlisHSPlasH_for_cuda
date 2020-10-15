#ifndef __BasicTypes_h__
#define __BasicTypes_h__

//#define USE_FLOAT
#ifndef USE_FLOAT
#define USE_DOUBLE
#endif

#define SPLISHSPLASH_FRAMEWORK

#ifdef USE_DOUBLE
typedef double Real;

#define MAX_MACRO(x,y) fmax(x,y)
#define MIN_MACRO(x,y) fmin(x,y)

#define REAL_MAX DBL_MAX
#define REAL_MIN DBL_MIN

#pragma warning( disable : 4244 4305 )  

#else
typedef float Real;

#define MAX_MACRO(x,y) fmaxf(x,y)
#define MIN_MACRO(x,y) fminf(x,y)

#define REAL_MAX FLT_MAX
#define REAL_MIN FLT_MIN

#pragma warning( disable : 4244 4305 )  
#endif


//#define USE_DOUBLE_CUDA

#ifdef USE_DOUBLE_CUDA
typedef double RealCuda;

#define MAX_MACRO_CUDA(x,y) (((x)<(y))?(y):(x))
#define MIN_MACRO_CUDA(x,y) (((x)<(y))?(x):(y))

#define SQRT_MACRO_CUDA(x) sqrt (x)
#define ABS_MACRO_CUDA(x) abs(x)

#define GL_FORMAT GL_DOUBLE

#pragma warning( disable : 4244 4305 )  

#else
typedef float RealCuda;

#define MAX_MACRO_CUDA(x,y) fmaxf(x,y)
#define MIN_MACRO_CUDA(x,y) fminf(x,y)

#define SQRT_MACRO_CUDA(x) sqrtf (x)
//#define ABS_MACRO_CUDA(x) abs(x)
#define ABS_MACRO_CUDA(x) (((x) > 0)?(x):-(x))


#define GL_FORMAT GL_FLOAT

#pragma warning( disable : 4244 4305 )  
#endif



#if defined(WIN32) || defined(_WIN32) || defined(WIN64)	   
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline))
#endif

#endif
