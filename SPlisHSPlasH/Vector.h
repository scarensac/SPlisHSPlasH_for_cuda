#ifndef __Vector_h__
#define __Vector_h__

#include <cmath>

class Vector3d {
public:
	double x, y, z;

	Vector3d(double a, double b, double c) { x = a; y = b; z = c; }

	Vector3d() { setZero(); }

	~Vector3d() {}

	inline void setZero() { x = 0; y = 0; z = 0; }
	inline static Vector3d Zero() { return Vector3d(0,0,0); }

	inline Vector3d& operator = (const Vector3d &o) { x = o.x; y = o.y; z = o.z; return *this; }

	inline Vector3d& operator-= (const Vector3d &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
	inline friend Vector3d operator- (const Vector3d& v1, const Vector3d &v2) { return Vector3d(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }
	
	inline Vector3d& operator+= (const Vector3d &o) { x += o.x; y += o.y; z += o.z; return *this; }
	inline friend Vector3d operator+ (const Vector3d& v1, const Vector3d &v2) { return Vector3d(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }

	inline Vector3d& operator*= (const double val) { x *= val; y *= val; z *= val; return *this; }
	inline friend Vector3d operator* (const Vector3d& v1, const double val) { return Vector3d(v1.x * val, v1.y * val, v1.z * val); }
	inline friend Vector3d operator* (const double val, const Vector3d& v1) { return Vector3d(v1.x * val, v1.y * val, v1.z * val); }

	inline double squaredNorm() { return x*x + y*y + z*z; }
	inline double squaredNorm() const { return x*x + y*y + z*z; }
	inline double norm() { return std::sqrt(squaredNorm()); }
	inline double norm() const { return std::sqrt(squaredNorm()); }

	inline double dot(const Vector3d &o){ return x * o.x + y * o.y + z * o.z; }

};

#endif
