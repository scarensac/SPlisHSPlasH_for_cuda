#ifndef __Vector_h__
#define __Vector_h__

#include <cmath>
#include <string>
#include "SPlisHSPlasH\BasicTypes.h"
#include "DFSPH\DFSPH_define_c.h"

#ifdef __NVCC__
#define FUNCTION __host__ __device__
#define SQRT_MACRO(x) SQRT_MACRO_CUDA(x)
#define ABS_MACRO(x) ABS_MACRO_CUDA(x)
#else
#define FUNCTION 
#define SQRT_MACRO(x) SQRT_MACRO_CUDA(x)
#define ABS_MACRO(x) ABS_MACRO_CUDA(x)
//should be that but using the one for cuda will not harms since they are pure c
//#define SQRT_MACRO(x) std::sqrt(x)
//#define ABS_MACRO(x) std::abs(x)
#endif

/*
#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif
//*/





namespace SPH
{

	template <typename T>
	class /*MY_ALIGN(16)*/ Vector3 {
	public:
		T x, y, z;

#ifdef USE_PADDING_FOR_MEMORY_ALIGNMENT
		T w;
#endif


	 	FUNCTION Vector3(T a, T b, T c) { x = a; y = b; z = c; }





		FUNCTION Vector3(T val) { x = val; y = val; z = val; }
		//init from array without check
		template<typename T2>
		FUNCTION Vector3(T2 val[]) { x = val[0]; y = val[1]; z = val[2]; }

		template<typename T2>
		FUNCTION Vector3(const Vector3<T2>& v) { 
			x = v.x; y = v.y; z = v.z; 
#ifdef USE_VECTOR_PADDING_FOR_STORAGE
			w = v.w;
#endif
		}

		FUNCTION Vector3() { setZero(); }

		FUNCTION ~Vector3() {}

		FUNCTION inline void setZero() { x = 0; y = 0; z = 0; }
		FUNCTION inline bool isZero() { return (x == 0) && (y == 0) && (z == 0); }
		FUNCTION inline static Vector3 Zero() { return Vector3(0, 0, 0); }

		template<typename T2>
		FUNCTION inline Vector3& operator = (const Vector3<T2> &o) { 
			x = o.x; y = o.y; z = o.z; 
#ifdef USE_VECTOR_PADDING_FOR_STORAGE
			w = o.w;
#endif
			return *this; 
		}

		template<typename T2>
		FUNCTION inline friend bool operator == (const Vector3& lhs, const Vector3<T2>& rhs) {
			return ((lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z));
		}

		template<typename T2>
		FUNCTION inline friend bool operator != (const Vector3& lhs, const Vector3<T2>& rhs) { return !(lhs == rhs); }

		template<typename T2>
		FUNCTION inline Vector3& operator-= (const Vector3<T2> &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
		FUNCTION inline Vector3& operator-= (const T val) { x -= val; y -= val; z -= val; return *this; }
		template<typename T2>
		FUNCTION inline friend Vector3 operator- (const Vector3& v1, const Vector3<T2> &v2) { return Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }
		FUNCTION inline friend Vector3 operator- (const Vector3& v1, const T val) { return Vector3(v1.x - val, v1.y - val, v1.z - val); }
		FUNCTION inline friend Vector3 operator- (const T val, const Vector3& v1) { return Vector3(v1.x - val, v1.y - val, v1.z - val); }

		template<typename T2>
		FUNCTION inline Vector3& operator+= (const Vector3<T2> &o) { x += o.x; y += o.y; z += o.z; return *this; }
		FUNCTION inline Vector3& operator+= (const T val) { x += val; y += val; z += val; return *this; }
		template<typename T2>
		FUNCTION inline friend Vector3 operator+ (const Vector3& v1, const Vector3<T2> &v2) { return Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }
		FUNCTION inline friend Vector3 operator+ (const Vector3& v1, const T val) { return Vector3(v1.x + val, v1.y + val, v1.z + val); }
		FUNCTION inline friend Vector3 operator+ (const T val, const Vector3& v1) { return Vector3(v1.x + val, v1.y + val, v1.z + val); }

		template<typename T2>
		FUNCTION inline Vector3& operator*= (const Vector3<T2> &o) { x *= o.x; y *= o.y; z *= o.z; return *this; }
		FUNCTION inline Vector3& operator*= (const T val) { x *= val; y *= val; z *= val; return *this; }
		template<typename T2>
		FUNCTION inline friend Vector3 operator* (const Vector3& v1, const Vector3<T2> &v2) { return Vector3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z); }
		FUNCTION inline friend Vector3 operator* (const Vector3& v1, const T val) { return Vector3(v1.x * val, v1.y * val, v1.z * val); }
		FUNCTION inline friend Vector3 operator* (const T val, const Vector3& v1) { return Vector3(v1.x * val, v1.y * val, v1.z * val); }

		template<typename T2>
		FUNCTION inline Vector3& operator/= (const Vector3<T2> &o) { x /= o.x; y /= o.y; z /= o.z; return *this; }
		FUNCTION inline Vector3& operator/= (const T val) { x /= val; y /= val; z /= val; return *this; }
		template<typename T2>
		FUNCTION inline friend Vector3 operator/ (const Vector3& v1, const Vector3<T2> &v2) { return Vector3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z); }
		FUNCTION inline friend Vector3 operator/ (const Vector3& v1, const T val) { return Vector3(v1.x / val, v1.y / val, v1.z / val); }
		FUNCTION inline friend Vector3 operator/ (const T val, const Vector3& v1) { return Vector3(v1.x / val, v1.y / val, v1.z / val); }


		template<typename T2>
		FUNCTION inline friend bool operator < (const Vector3& lhs, const Vector3<T2>& rhs) { return (lhs.x < rhs.x) || (lhs.y < rhs.y) || (lhs.z < rhs.z); }
		FUNCTION inline friend bool operator < (const Vector3& lhs, const T val) { return (lhs.x < val) || (lhs.y < val) || (lhs.z < val); }
		template<typename T2>
		FUNCTION inline friend bool operator > (const Vector3& lhs, const Vector3<T2>& rhs) { return (lhs.x > rhs.x) || (lhs.y > rhs.y) || (lhs.z > rhs.z); }
		FUNCTION inline friend bool operator > (const Vector3& lhs, const T val) { return (lhs.x > val) || (lhs.y > val) || (lhs.z > val); }


		FUNCTION inline T squaredNorm() { return x*x + y*y + z*z; }
		FUNCTION inline T squaredNorm() const { return x*x + y*y + z*z; }
		FUNCTION inline T norm() { return SQRT_MACRO(squaredNorm()); }
		FUNCTION inline T norm() const { return SQRT_MACRO(squaredNorm()); }

		FUNCTION inline Vector3& toUnit() { (*this) /= norm(); return *this; }
		FUNCTION inline Vector3 unit() { Vector3 v = (*this); ; return v/ norm(); }

		template<typename T2>
		FUNCTION inline T dot(const Vector3<T2> &o) const { return x * o.x + y * o.y + z * o.z; }
		template<typename T2>
		FUNCTION inline Vector3 cross(const Vector3<T2> &o) const { return Vector3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x); }

		FUNCTION inline T avg() { return (x + y + z) / 3.0; }
		FUNCTION inline Vector3& toAvg() { (*this) = Vector3(avg()); return *this; }

		FUNCTION inline Vector3 abs() { return Vector3((x > 0) ? x : -x, (y > 0) ? y : -y, (z > 0) ? z : -z); }
		FUNCTION inline Vector3& toAbs() { if (x < 0)x *= -1; if (y < 0)y *= -1; if (z < 0)z *= -1; return *this; }

   

        FUNCTION inline Vector3& toFloor() { x = (int)x - ((((int)x)!=x)?(x < 0):0); y = (int)y - ((((int)y)!=y)?(y < 0):0);
                                             z = (int)z - ((((int)z)!=z)?(z < 0):0); return *this; }

		FUNCTION inline Vector3& toMin(const T val) { if (x > val)x = val; if (y > val)y = val; if (z > val)z = val; return *this; }
		template<typename T2>
        FUNCTION inline Vector3& toMin(const Vector3<T2> &o) { if (x > o.x)x = o.x; if (y > o.y)y = o.y; if (z > o.z)z = o.z; return *this; }


		FUNCTION inline Vector3& toMax(const T val) { if (x < val)x = val; if (y < val)y = val; if (z < val)z = val; return *this; }
		template<typename T2>
		FUNCTION inline Vector3& toMax(const Vector3<T2>& o) { if (x < o.x)x = o.x; if (y < o.y)y = o.y; if (z < o.z)z = o.z; return *this; }

		//this function is to nullify values below a certain value.
		//FUUUUUUUUUUUUUUCK I called on of the class function ABS so I can't use the std abs function in this context............
		//FUNCTION inline Vector3& toEpsilonAbsToZero(const T val) { if (ABS_MACRO(x) < val)x = 0; if (ABS_MACRO(y) < val)y = 0; if (ABS_MACRO(z) < val)z = 0; return *this; }
		FUNCTION inline Vector3& toEpsilonAbsToZero(const T val) { Vector3 o = abs(); if (o.x < val)x = 0; if (o.y < val)y = 0; if (o.z < val)z = 0; return *this; }

		//host only functions
		inline std::string toString() { return std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z); }

	};

    using Vector3d = Vector3<RealCuda>;
    using Vector3i = Vector3<int>;
}


#endif
