#ifndef __Vector_h__
#define __Vector_h__

#include <cmath>
#include "SPlisHSPlasH\BasicTypes.h"

#ifdef __NVCC__
#define FUNCTION __host__ __device__
#else
#define FUNCTION 
#endif

template <typename T>
class Vector3 {
public:
	T x, y, z;

	FUNCTION Vector3(T a, T b, T c) { x = a; y = b; z = c; }
	FUNCTION Vector3(T val) { x = val; y = val; z = val; }
	//init from array without check
	template<typename T2>
	FUNCTION Vector3(T2 val[]) { x = val[0]; y = val[1]; z = val[2]; }

	template<typename T2>
	FUNCTION Vector3(const Vector3<T2>& v) { x = v.x; y = v.y; z = v.z; }

	FUNCTION Vector3() { setZero(); }

	FUNCTION ~Vector3() {}

	FUNCTION inline void setZero() { x = 0; y = 0; z = 0; }
	FUNCTION inline static Vector3 Zero() { return Vector3(0, 0, 0); }

	template<typename T2>
	FUNCTION inline Vector3& operator = (const Vector3<T2> &o) { x = o.x; y = o.y; z = o.z; return *this; }
	
	template<typename T2>
	FUNCTION inline Vector3& operator-= (const Vector3<T2> &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
	template<typename T2>
	FUNCTION inline friend Vector3 operator- (const Vector3& v1, const Vector3<T2> &v2) { return Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }
	FUNCTION inline Vector3& operator-= (const T val) { x -= val; y -= val; z -= val; return *this; }
	FUNCTION inline friend Vector3 operator- (const Vector3& v1, const T val) { return Vector3(v1.x - val, v1.y - val, v1.z - val); }
	FUNCTION inline friend Vector3 operator- (const T val, const Vector3& v1) { return Vector3(v1.x - val, v1.y - val, v1.z - val); }
	
	template<typename T2>
	FUNCTION inline Vector3& operator+= (const Vector3<T2> &o) { x += o.x; y += o.y; z += o.z; return *this; }
	template<typename T2>
	FUNCTION inline friend Vector3 operator+ (const Vector3& v1, const Vector3<T2> &v2) { return Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }
	FUNCTION inline Vector3& operator+= (const T val) { x += val; y += val; z += val; return *this; }
	FUNCTION inline friend Vector3 operator+ (const Vector3& v1, const T val) { return Vector3(v1.x + val, v1.y + val, v1.z + val); }
	FUNCTION inline friend Vector3 operator+ (const T val, const Vector3& v1) { return Vector3(v1.x + val, v1.y + val, v1.z + val); }

	template<typename T2>
	FUNCTION inline Vector3& operator*= (const Vector3<T2> &o) { x *= o.x; y *= o.y; z *= o.z; return *this; }
	template<typename T2>
	FUNCTION inline friend Vector3 operator* (const Vector3& v1, const Vector3<T2> &v2) { return Vector3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z); }
	FUNCTION inline Vector3& operator*= (const T val) { x *= val; y *= val; z *= val; return *this; }
	FUNCTION inline friend Vector3 operator* (const Vector3& v1, const T val) { return Vector3(v1.x * val, v1.y * val, v1.z * val); }
	FUNCTION inline friend Vector3 operator* (const T val, const Vector3& v1) { return Vector3(v1.x * val, v1.y * val, v1.z * val); }

	template<typename T2>
	FUNCTION inline Vector3& operator/= (const Vector3<T2> &o) { x /= o.x; y /= o.y; z /= o.z; return *this; }
	template<typename T2>
	FUNCTION inline friend Vector3 operator/ (const Vector3& v1, const Vector3<T2> &v2) { return Vector3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z); }
	FUNCTION inline Vector3& operator/= (const T val) { x /= val; y /= val; z /= val; return *this; }
	FUNCTION inline friend Vector3 operator/ (const Vector3& v1, const T val) { return Vector3(v1.x / val, v1.y / val, v1.z / val); }
	FUNCTION inline friend Vector3 operator/ (const T val, const Vector3& v1) { return Vector3(v1.x / val, v1.y / val, v1.z / val); }

	FUNCTION inline T squaredNorm() { return x*x + y*y + z*z; }
	FUNCTION inline T squaredNorm() const { return x*x + y*y + z*z; }
	FUNCTION inline T norm() { return std::sqrt(squaredNorm()); }
	FUNCTION inline T norm() const { return std::sqrt(squaredNorm()); }

	template<typename T2>
	FUNCTION inline T dot(const Vector3<T2> &o) const { return x * o.x + y * o.y + z * o.z; }
	template<typename T2>
	FUNCTION inline Vector3 cross(const Vector3<T2> &o) const { return Vector3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x); }

	FUNCTION inline T avg() { return (x + y + z) / 3.0; }
	FUNCTION inline Vector3& toAbs() { if (x < 0)x *= -1; if (y < 0)y *= -1; if (z < 0)z *= -1; return *this; }
	FUNCTION inline Vector3& clampTo(const T val) { if (x < val)x = val; if (y < val)y = val; if (z < val)z = val; return *this; }
	
	template<typename T2>
	FUNCTION inline Vector3& clampTo(const Vector3<T2> &o) { if (x < o.x)x = o.x; if (y < o.y)y = o.y; if (z < o.z)z = o.z; return *this; }
	

	FUNCTION inline Vector3& toAvg() { (*this) = Vector3(avg()); return *this; }

	FUNCTION inline Vector3& toFloor() { x = (int)x; y = (int)y; z = (int)z; return *this;}

};

using Vector3d = Vector3<RealCuda>;

#endif
