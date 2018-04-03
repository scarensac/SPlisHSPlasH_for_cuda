#ifndef __Vector_h__
#define __Vector_h__

#include <cmath>

#ifdef __NVCC__
#define FUNCTION __host__ __device__
#else
#define FUNCTION 
#endif

class Vector3d {
public:
	double x, y, z;

	FUNCTION Vector3d(double a, double b, double c) { x = a; y = b; z = c; }
	FUNCTION Vector3d(double val) { x = val; y = val; z = val; }

	FUNCTION Vector3d() { setZero(); }

	FUNCTION ~Vector3d() {}

	FUNCTION inline void setZero() { x = 0; y = 0; z = 0; }
	FUNCTION inline static Vector3d Zero() { return Vector3d(0, 0, 0); }

	FUNCTION inline Vector3d& operator = (const Vector3d &o) { x = o.x; y = o.y; z = o.z; return *this; }

	FUNCTION inline Vector3d& operator-= (const Vector3d &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
	FUNCTION inline friend Vector3d operator- (const Vector3d& v1, const Vector3d &v2) { return Vector3d(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }
	FUNCTION inline friend Vector3d operator- (const Vector3d& v1, const double val) { return Vector3d(v1.x - val, v1.y - val, v1.z - val); }
	FUNCTION inline friend Vector3d operator- (const double val, const Vector3d& v1) { return Vector3d(v1.x - val, v1.y - val, v1.z - val); }
	
	FUNCTION inline Vector3d& operator+= (const Vector3d &o) { x += o.x; y += o.y; z += o.z; return *this; }
	FUNCTION inline friend Vector3d operator+ (const Vector3d& v1, const Vector3d &v2) { return Vector3d(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }
	FUNCTION inline friend Vector3d operator+ (const Vector3d& v1, const double val) { return Vector3d(v1.x + val, v1.y + val, v1.z + val); }
	FUNCTION inline friend Vector3d operator+ (const double val, const Vector3d& v1) { return Vector3d(v1.x + val, v1.y + val, v1.z + val); }

	FUNCTION inline Vector3d& operator*= (const double val) { x *= val; y *= val; z *= val; return *this; }
	FUNCTION inline friend Vector3d operator* (const Vector3d& v1, const double val) { return Vector3d(v1.x * val, v1.y * val, v1.z * val); }
	FUNCTION inline friend Vector3d operator* (const double val, const Vector3d& v1) { return Vector3d(v1.x * val, v1.y * val, v1.z * val); }

	FUNCTION inline Vector3d& operator/= (const double val) { x /= val; y /= val; z /= val; return *this; }
	FUNCTION inline friend Vector3d operator/ (const Vector3d& v1, const double val) { return Vector3d(v1.x / val, v1.y / val, v1.z / val); }
	FUNCTION inline friend Vector3d operator/ (const double val, const Vector3d& v1) { return Vector3d(v1.x / val, v1.y / val, v1.z / val); }

	FUNCTION inline double squaredNorm() { return x*x + y*y + z*z; }
	FUNCTION inline double squaredNorm() const { return x*x + y*y + z*z; }
	FUNCTION inline double norm() { return std::sqrt(squaredNorm()); }
	FUNCTION inline double norm() const { return std::sqrt(squaredNorm()); }

	FUNCTION inline double dot(const Vector3d &o) { return x * o.x + y * o.y + z * o.z; }

	inline Vector3d& toAbs() { if (x < 0)x *= -1; if (y < 0)y *= -1; if (z < 0)z *= -1; return *this; }
	inline Vector3d& clampTo(const double val) { if (x < val)x = val; if (y < val)y = val; if (z < val)z = val; return *this; }
	inline Vector3d& clampTo(const Vector3d &o) { if (x < o.x)x = o.x; if (y < o.y)y = o.y; if (z < o.z)z = o.z; return *this; }
	inline double avg() { return (x + y + z) / 3.0; }
	inline Vector3d& toAvg() { (*this) = Vector3d(avg()); return *this; }

};

#endif
