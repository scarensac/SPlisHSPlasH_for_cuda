#ifndef __Vector_h__
#define __Vector_h__


class Vector3d {
public:
	double x, y, z;

	Vector3d(double a, double b, double c) { x = a; y = b; z = c; }

	Vector3d() { setZero(); }

	~Vector3d() {}

	void setZero() { x = 0; y = 0; z = 0; }
};

#endif
