#ifndef SPHKERNELSBASE_H
#define SPHKERNELSBASE_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>

namespace SPH
{
	/** \brief Cubic spline kernel.
	*/
	template <class Vector> class CubicKernelBase
	{
	protected:
		static Real m_radius;
		static Real m_k;
		static Real m_l;
		static Real m_W_zero;
	public:
		static Real getRadius() { return m_radius; }
		static void setRadius(Real val)
		{
			m_radius = val;
			static const Real pi = static_cast<Real>(M_PI);

			const Real h3 = m_radius*m_radius*m_radius;
			m_k = 8.0 / (pi*h3);
			m_l = 48.0 / (pi*h3);
			m_W_zero = W(Vector::Zero());
		}

	public:
		static Real W(const Real r)
		{
			Real res = 0.0;
			const Real q = r / m_radius;
			if (q <= 1.0)
			{
				if (q <= 0.5)
				{
					const Real q2 = q*q;
					const Real q3 = q2*q;
					res = m_k * (6.0*q3 - 6.0*q2 + 1.0);
				}
				else
				{
					res = m_k * (2.0*pow(1.0 - q, 3));
				}
			}
			return res;
		}

		static Real W(const Vector &r)
		{
			return W(r.norm());
		}

		static Vector gradW(const Vector &r)
		{
			Vector res;
			const Real rl = r.norm();
			const Real q = rl / m_radius;
			if (q <= 1.0)
			{
				if (rl > 1.0e-6)
				{
					const Vector gradq = r * ((Real) 1.0 / (rl*m_radius));
					if (q <= 0.5)
					{
						res = m_l*q*((Real) 3.0*q - (Real) 2.0)*gradq;
					}
					else
					{
						const Real factor = 1.0 - q;
						res = m_l*(-factor*factor)*gradq;
					}
				}
			}
			else
				res.setZero();

			return res;
		}

		static Real W_zero()
		{
			return m_W_zero;
		}
	};

}

#endif
