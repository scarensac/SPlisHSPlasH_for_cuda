#ifndef __Quaternion_h__
#define __Quaternion_h__

#include <cmath>
#include "SPlisHSPlasH\BasicTypes.h"
#include "SPlisHSPlasH\Vector.h"

#ifdef __NVCC__
#define FUNCTION __host__ __device__
#else
#define FUNCTION 
#endif


namespace SPH
{

	class Quaternion {
	public:
		RealCuda s;
		Vector3d v;

        FUNCTION Quaternion() { v = Vector3d(0, 0, 0); s = 1; }
		FUNCTION Quaternion(RealCuda x, RealCuda y, RealCuda z, RealCuda w) { v.x = x; v.y = y; v.z = z; s = w; }
		FUNCTION Quaternion(Vector3d v_i, RealCuda w) { v = v_i; s = w; }
		//init from array without check
		template<typename T2>
		FUNCTION Quaternion(T2 val[]) { v.x = val[0]; v.y = val[1]; v.z = val[2]; s = val[3]; }

		//this constructor create a queaternion from an euler matrix
		FUNCTION void fromEulerMatrix(RealCuda* m) {

			double tr = m[0] + m[4] + m[8];

			if (tr > 0) {
				double S = sqrt(tr + 1.0) * 2; // S=4*qw
				s = 0.25 * S;
				v.x = (m[7] - m[5]) / S;
				v.y = (m[2] - m[6]) / S;
				v.z = (m[3] - m[1]) / S;
			}
			else if ((m[0] > m[4])&(m[0] > m[8])) {
				double S = sqrt(1.0 + m[0] - m[4] - m[8]) * 2; // S=4*qx
				s = (m[7] - m[5]) / S;
				v.x = 0.25 * S;
				v.y = (m[1] + m[3]) / S;
				v.z = (m[2] + m[6]) / S;
			}
			else if (m[4] > m[8]) {
				double S = sqrt(1.0 + m[4] - m[0] - m[8]) * 2; // S=4*qy
				s = (m[2] - m[6]) / S;
				v.x = (m[1] + m[3]) / S;
				v.y = 0.25 * S;
				v.z = (m[5] + m[7]) / S;
			}
			else {
				double S = sqrt(1.0 + m[8] - m[0] - m[4]) * 2; // S=4*qz
				s = (m[3] - m[1]) / S;
				v.x = (m[2] + m[6]) / S;
				v.y = (m[5] + m[7]) / S;
				v.z = 0.25 * S;
			}
		}

		//*
		FUNCTION Vector3d rotate(const Vector3d& u) const {
			//uRot = q * (0, u) * q' = (s, v) * (0, u) * (s, -v)
			Vector3d t = u * s + v.cross(u);
			return v*u.dot(v) + t * s + v.cross(t);
		}
		//*/
	};

}

#endif
