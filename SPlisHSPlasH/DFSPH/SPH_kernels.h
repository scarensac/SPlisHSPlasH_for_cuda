#ifndef __SPHkernels_h__
#define __SPHkernels_h__

#include "SPlisHSPlasH\BasicTypes.h"
#include <string>
#include <vector>



#include "SPlisHSPlasH\Vector.h"
#include "SPlisHSPlasH\Quaternion.h"

#include "DFSPH_define_c.h"


namespace SPH
{
//this is an eigen less cubic kernel
/** \brief Cubic spline kernel.
    */
class CubicKernelPerso
{
protected:
    RealCuda m_radius;
    RealCuda m_k;
    RealCuda m_l;
    RealCuda m_W_zero;
public:
	FUNCTION RealCuda getRadius() { return m_radius; }
	FUNCTION RealCuda getRadius() const { return m_radius; }
    FUNCTION void setRadius(RealCuda val)
    {
        m_radius = val;
        static const RealCuda pi = static_cast<RealCuda>(M_PI);

        const RealCuda h3 = m_radius*m_radius*m_radius;
        m_k = 8.0 / (pi*h3);
        m_l = 48.0 / (pi*h3);
        m_W_zero = W(Vector3d::Zero());
    }

public:
    FUNCTION RealCuda W(const RealCuda r) const
    {
        RealCuda res = 0.0;
        const RealCuda q = r / m_radius;
        if (q <= 1.0)
        {
            if (q <= 0.5)
            {
                const RealCuda q2 = q*q;
                const RealCuda q3 = q2*q;
                res = m_k * (6.0*q3 - 6.0*q2 + 1.0);
            }
            else
            {
                const RealCuda factor = 1.0 - q;
                //res = m_k * (2.0*pow(1.0 - q, 3));
                res = m_k * (2.0*factor*factor*factor);
            }
        }
        return res;
    }

    FUNCTION inline RealCuda W(const Vector3d &r) const
    {
        return W(r.norm());
    }

    FUNCTION Vector3d gradW(const Vector3d &r) const
    {
        Vector3d res;
        const RealCuda rl = r.norm();
        const RealCuda q = rl / m_radius;
        if (q <= 1.0)
        {
            if (rl > 1.0e-6)
            {
                const Vector3d gradq = r * ((RealCuda) 1.0 / (rl*m_radius));
                if (q <= 0.5)
                {
                    res = m_l*q*((RealCuda) 3.0*q - (RealCuda) 2.0)*gradq;
                }
                else
                {
                    const RealCuda factor = 1.0 - q;
                    res = m_l*(-factor*factor)*gradq;
                }
            }
        }
        else
            res.setZero();

        return res;
    }

    FUNCTION inline RealCuda W_zero() const
    {
        return m_W_zero;
    }
};

class PrecomputedCubicKernelPerso
{
public:
    RealCuda* m_W;
    RealCuda* m_gradW;
    RealCuda m_radius;
    RealCuda m_radius2;
    RealCuda m_invStepSize;
    RealCuda m_W_zero;
    unsigned int m_resolution;
public:

    PrecomputedCubicKernelPerso(){
        m_W=NULL;
        m_gradW=NULL;
    }

    FUNCTION RealCuda getRadius() { return m_radius; }
    void setRadius(RealCuda val);
    void freeMemory();

public:
    FUNCTION RealCuda W(const Vector3d &r) const
    {
        RealCuda res = 0.0;
        const RealCuda r2 = r.squaredNorm();
        if (r2 <= m_radius2)
        {
            const RealCuda r = sqrt(r2);
            const unsigned int pos = (unsigned int)(r * m_invStepSize);
            res = m_W[pos];
        }
        return res;
    }

    FUNCTION RealCuda W(const RealCuda r) const
    {
        RealCuda res = 0.0;
        if (r <= m_radius)
        {
            const unsigned int pos = (unsigned int)(r * m_invStepSize);
            res = m_W[pos];
        }
        return res;
    }

    FUNCTION Vector3d gradW(const Vector3d &r) const
    {
        Vector3d res;
        const RealCuda r2 = r.squaredNorm();
        if (r2 <= m_radius2)
        {
            const RealCuda rl = sqrt(r2);
            const unsigned int pos = (unsigned int)(rl * m_invStepSize);
            res = m_gradW[pos] * r;
        }
        else
            res.setZero();

        return res;
    }

    FUNCTION inline RealCuda W_zero() const
    {
        return m_W_zero;
    }
};


/** \brief Cohesion kernel used for the surface tension method of Akinci el al. \cite Akinci:2013.
        */
class CohesionKernelGPU
{
protected:
    RealCuda m_radius;
    RealCuda m_k;
    RealCuda m_c;
    RealCuda m_W_zero;
public:
    FUNCTION RealCuda getRadius() { return m_radius; }
    FUNCTION void setRadius(const RealCuda val)
    {
        m_radius = val;
        RealCuda pi = 3.14159265358979323846;
        m_k = 32. / (pi*pow(m_radius, 9));
        m_c = pow(m_radius, 6) / 64.0;
        m_W_zero = W(0);
    }

public:

    /**
            * W(r,h) = (32/(pi h^9))(h-r)^3*r^3					if h/2 < r <= h
            *          (32/(pi h^9))(2*(h-r)^3*r^3 - h^6/64		if 0 < r <= h/2
            */
    FUNCTION RealCuda W(const RealCuda r) const
    {
        RealCuda res = 0.0;
        const RealCuda r2 = r*r;
        const RealCuda radius2 = m_radius*m_radius;
        if (r2 <= radius2)
        {
            const RealCuda r1 = sqrt(r2);
            const RealCuda r3 = r2*r1;
            RealCuda factor=m_radius - r1;
            res=m_k*factor*factor*factor*r3;
            if (r1 <= 0.5*m_radius){
                res = 2.0*res - m_c;
            }

        }
        return res;
    }

    FUNCTION RealCuda W(const Vector3d &r) const
    {
        return W(r.norm());
    }

    FUNCTION RealCuda W_zero() const
    {
        return m_W_zero;
    }
};

/** \brief Adhesion kernel used for the surface tension method of Akinci el al. \cite Akinci:2013.
            */
class AdhesionKernelGPU
{
protected:
    RealCuda m_radius;
    RealCuda m_k;
    RealCuda m_W_zero;
public:
    FUNCTION RealCuda getRadius() { return m_radius; }
    FUNCTION void setRadius(RealCuda val)
    {
        m_radius = val;
        m_k = 0.007 / powf(m_radius, 3.25);
        m_W_zero = W(0);
    }

public:

    /**
                * W(r,h) = (0.007/h^3.25)(-4r^2/h + 6r -2h)^0.25					if h/2 < r <= h
                */
    FUNCTION RealCuda W(const RealCuda r) const
    {
        RealCuda res = 0.0;
        const RealCuda r2 = r*r;
        const RealCuda radius2 = m_radius*m_radius;
        if (r2 <= radius2)
        {
            const RealCuda r = sqrt(r2);
            if (r > 0.5*m_radius)
                res = m_k*powf(-4.0*r2 / m_radius + 6.0*r - 2.0*m_radius, 0.25);
        }
        return res;
    }

    FUNCTION RealCuda W(const Vector3d &r) const
    {
        return W(r.norm());
    }

    FUNCTION RealCuda W_zero() const
    {
        return m_W_zero;
    }
};

}

#endif //__SPHkernels_h__


