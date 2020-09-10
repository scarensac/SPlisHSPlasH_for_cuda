

//NOTE1:	seems that virtual function can't be used with managed allocation
//			so I'll use a template to have an equivalent solution
//			the  template parameter allow the représentation of various shapes
//NOTE2:	0 ==> plane: only use this for paralel planes (so max 2)
//			it's just a fast to compute solution if you need bands of fluid near the borders of the simulation
//			however since as soon as you get more than 2 planes the distance computation become pretty heavy
//			it's better to use another solution to represent the same surface (in particular in the case of boxes
//NOTE3:	1 ==> rectangular cuboid. 

class BufferFluidSurface
{
	int type;

	//this is for the planes
	Vector3d o;
	Vector3d n;

	//this is for the cuboid
	Vector3d center;
	Vector3d halfLengths;

	//this for radius based geometries
	RealCuda radius;
public:
	bool destructor_activated;

	inline BufferFluidSurface() {
		destructor_activated = false;
		type = -1;
	}


	inline ~BufferFluidSurface() {
		if (destructor_activated) {
			
		}
	}

	int getType() { return type; }

	inline int setPlane(Vector3d o_i, Vector3d n_i) {
		if (type < 0) {
			type = 0;
		}
		else if (type != 0) {
			return -1;
		}
		
		o = o_i;
		n = n_i;
		return 0;
	}

	inline int setCuboid(Vector3d c_i, Vector3d hl_i) {
		if (type < 0) {
			type = 1;
		}
		else if (type != 1) {
			return -1;
		}

		center = c_i;
		halfLengths = hl_i;
		return 0;
	}

	inline int setCylinder(Vector3d c_i, RealCuda half_h, RealCuda r) {
		if (type < 0) {
			type = 2;
		}
		else if (type != 2) {
			return -1;
		}

		center = c_i;
		halfLengths = Vector3d(0,half_h,0);
		radius = r;
		return 0;
	}

	inline void copy (const BufferFluidSurface& other) {
		type = -1;
		switch (other.type) {
		case 0: {
			setPlane(other.o, other.n);
			break;
		}
		case 1: {
			setCuboid(other.center, other.halfLengths);
			break;
		}
		case 2: {
			setCylinder(other.center, other.halfLengths.y, other.radius);
			break;
		}
		default: {; }//it should NEVER reach here
		};
	}

	inline void move(const Vector3d& d) {
		switch (type) {
		case 0: {
			o+=d;
			break;
		}
		case 1: {
			center += d;
			break;
		}
		case 2: {
			center += d;
			break;
		}
		default: {; }//it should NEVER reach here
		};
	}

	std::string toString() {
		std::ostringstream oss;
		//*
		switch (type) {
		case 0: {
			oss << "plane "  << " (o,n)  " << o.x << "   " << o.y << "   " << o.z << "  //  "
				"   " << n.x << "   " << n.y << "   " << n.z<< std::endl;
			break;
		}
		case 1: {
			oss << "Cuboid center: " << center.toString() << "   halfLengths: " << halfLengths.toString() << std::endl;
			break;
		}
		case 2: {
			oss << "Cylinder center: " << center.toString() << "  radius: "<<radius<<"   height: " << halfLengths.y << std::endl;
			break;
		}
		default: {; }//it should NEVER reach here
		};
		//*/
		return oss.str();
	}



	//to know if we are on the inside of each plane we can simply use the dot product*
	FUNCTION inline bool isinside(Vector3d p) {
		//*
		switch (type) {
		case 0: {
			Vector3d v = p - o;
			if (v.dot(n) < 0) {
				return false;
			}
			break;
		}
		case 1: {
			Vector3d v = p - center;
			v.toAbs();
			return (v.x < halfLengths.x) && (v.y < halfLengths.y) && (v.z < halfLengths.z);
			break;
		}
		case 2: {
			Vector3d v = p - center;
			if (abs(v.y) > halfLengths.y) { return false; }

			v.y = 0;
			return (v.norm()< radius);
			break;
		}
		default: {; }//it should NEVER reach here
		}
		//*/
		return true;
	}

	FUNCTION inline RealCuda distanceToSurface(Vector3d p) {
		RealCuda dist;
		//*
		switch (type) {
		case 0: {
			dist = abs((p - o).dot(n));
			break;
		}
		case 1: {
			Vector3d v = p - center;
			if (isinside(p)) {
				v.toAbs();

				dist = MIN_MACRO_CUDA(MIN_MACRO_CUDA((halfLengths.x - v.x), (halfLengths.y - v.y)), (halfLengths.z - v.z));

			}
			else {
				///TODO I'm nearly sure I can simply use the abs here but it would not realy impact the globals performances
				///		so I won't take the risk
				Vector3d v2 = v;
				if (v.x < -halfLengths.x) { v.x = -halfLengths.x; }
				else if (v.x > halfLengths.x) { v.x = halfLengths.x; }
				if (v.y < -halfLengths.y) { v.y = -halfLengths.y; }
				else if (v.y > halfLengths.y) { v.y = halfLengths.y; }
				if (v.z < -halfLengths.z) { v.z = -halfLengths.z; }
				else if (v.z > halfLengths.z) { v.z = halfLengths.z; }
				dist = (v - v2).norm();
			}

			break;
		}
		case 2: {
			Vector3d v = p - center;
			v.toAbs();
			if (isinside(p)) {

				//faster to compute that waty but you cound express it the same way as for the outside distance
				dist = halfLengths.y - v.y;
				v.y = 0;

				dist = MIN_MACRO_CUDA(dist,radius-v.norm());

			}
			else {
				Vector3d temp_v(v.x, 0, v.z);
				temp_v = Vector3d(MAX_MACRO_CUDA(temp_v.norm()-radius,0),MAX_MACRO_CUDA(v.y-halfLengths.y,0),0);
				dist = temp_v.norm();
			}

			break;
		}
		default: {; }//it should NEVER reach here
		}
		//*/
		return dist;
	}

	FUNCTION inline RealCuda distanceToSurfaceSigned(Vector3d p) {
		//you could most likely optimize the implementation for some of the types but it's not critical so why bother
		return  distanceToSurface(p) * ((isinside(p)) ? 1 : -1);;
	}

};