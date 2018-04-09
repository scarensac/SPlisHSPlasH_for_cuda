#ifndef __BasicTypes_h__
#define __BasicTypes_h__

#ifndef USE_FLOAT
#define USE_DOUBLE
#endif

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

#define MAX_MACRO_CUDA(x,y) fmax(x,y)
#define MIN_MACRO_CUDA(x,y) fmin(x,y)

#define GL_FORMAT GL_DOUBLE

#pragma warning( disable : 4244 4305 )  

#else
typedef float RealCuda;

#define MAX_MACRO_CUDA(x,y) fmaxf(x,y)
#define MIN_MACRO_CUDA(x,y) fminf(x,y)

#define GL_FORMAT GL_FLOAT

#pragma warning( disable : 4244 4305 )  
#endif



#if defined(WIN32) || defined(_WIN32) || defined(WIN64)	   
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline))
#endif

#endif
