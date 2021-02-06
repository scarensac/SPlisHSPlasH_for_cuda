#include<fstream>
#include "OBJ_Loader.h"

namespace objl {
	SPH::Vector3d vector3ToVector3d(const Vector3& v) {
		return Vector3d(v.X, v.Y, v.Z);
	}
}

//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
//just modified to return the distance to the triangle and true as long as the line intersect the triangle at some point
template<bool compute_t, bool compute_p>
FUNCTION bool RayIntersectsTriangle(Vector3d rayOrigin,
	Vector3d rayVector,
	Vector3d vertex0, Vector3d vertex1, Vector3d vertex2,
	Vector3d& outIntersectionPoint, RealCuda& t)
{
	const float EPSILON = 0.0000001;
	Vector3d edge1, edge2, h, s, q;
	float a, f, u, v;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;
	h = rayVector.cross(edge2);
	a = edge1.dot(h);
	if (a > -EPSILON && a < EPSILON)
		return false;    // This ray is parallel to this triangle.

	f = 1.0 / a;
	s = rayOrigin - vertex0;
	u = f * s.dot(h);
	if (u < 0.0 || u > 1.0)
		return false; 
	q = s.cross(edge1);
	v = f * rayVector.dot(q);
	if (v < 0.0 || u + v > 1.0)
		return false;	//the ray has no intersection with the triangle, I guess

	// At this stage we can compute t to find out where the intersection point is on the line.
	if (compute_t) {
		t = f * edge2.dot(q);
		if (compute_p) {
			outIntersectionPoint = rayOrigin + rayVector * t;
		}
	}
	return true;
}

//the mesh class for the mesh based structure
//for now I'll only allow triangular meshs
class MeshPerso 
{
public:
	int nbVectex;
	int nbfaces;

	//vertexes
	Vector3d* v;
	
	//faces
	Vector3i* f;

	//normals btw I'll the the face normal not the vertex normal
	Vector3d* n;

	//an offset representing that the mesh was moved
	Vector3d offset;

	Vector3d Bmin;
	Vector3d Bmax;
	
	MeshPerso() { nbfaces = -1; nbVectex = -1; }

	//technically it's a delete but since I'm doing some not indeapth copy some times 
	//I want absolute control over the memory release
	void clear() {
		CUDA_FREE_PTR(v);
		CUDA_FREE_PTR(f);
		CUDA_FREE_PTR(n);
	}

	inline void cpy(MeshPerso* o) {
		clear();

		//set the nb of elems
		nbVectex = o->nbVectex;
		nbfaces = o->nbfaces;

		//alloc with managed since I'l load the data on cpu
		cudaMallocManaged(&(v), nbVectex * sizeof(Vector3d));
		cudaMallocManaged(&(f), nbfaces * sizeof(Vector3i));
		cudaMallocManaged(&(n), nbfaces * sizeof(Vector3d));

		//and copy the data
		gpuErrchk(cudaMemcpy(v, o->v, nbVectex * sizeof(Vector3d), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(f, o->f, nbfaces * sizeof(Vector3i), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(n, o->n, nbfaces * sizeof(Vector3d), cudaMemcpyDeviceToDevice));

		Bmin = o->Bmin;
		Bmax = o->Bmax;
		offset = o->offset;
	}

	inline bool isInitialized() { return (nbVectex > 0) && (nbfaces > 0); }

	//no check on the size os be carefull
	FUNCTION inline Vector3d getVertex(int i) { return v[i] + offset; }
	FUNCTION inline Vector3d getBmin() { return Bmin + offset; }
	FUNCTION inline Vector3d getBmax() { return Bmax + offset; }

	inline void applyOffset(Vector3d off) { offset += off; }

	//this will return 0 unless an error happens
	inline int loadFromFile(std::string fileName) {
		// Initialize Loader
		objl::Loader Loader;

		// Load .obj File
		std::ostringstream oss;
		//oss << "data/models/" << fileName;
		oss <<  fileName;
		bool loadout = Loader.LoadFile(oss.str());

		std::cout << "MeshPerso::loadFromFile trying to load the following file: " << fileName << std::endl;

		// If so continue
		if (loadout){
			if (Loader.LoadedMeshes.size() != 1) {
				std::cout << "multiple meshes were loaded for some reason. nb mesh loaded: " << Loader.LoadedMeshes.size() << std::endl;
				return -2;
			}
	
			// I only have one mesh so no need to loop
			objl::Mesh curMesh = Loader.LoadedMeshes[0];

			//set the nb of elems
			nbVectex = curMesh.Vertices.size();
			nbfaces = curMesh.Indices.size() / 3;

			if ((curMesh.Indices.size() % 3) != 0) {
				std::cout << "the number of indicies in not a multiple of 3. nb indices " << curMesh.Indices.size() << std::endl;
				return -3;
			}

			//alloc with managed since I'l load the data on cpu
			cudaMallocManaged(&(v), nbVectex * sizeof(Vector3d));
			cudaMallocManaged(&(f), nbfaces * sizeof(Vector3i));
			cudaMallocManaged(&(n), nbfaces * sizeof(Vector3d));

			//Ill load it on CPU since I can't be bothered to do it on gpu
			for (int i = 0; i < nbVectex; ++i) {
				Vector3d pos= Vector3d(objl::vector3ToVector3d(curMesh.Vertices[i].Position));
				v[i] = pos;

				//store the bounding box
				Bmin.toMin(pos);
				Bmax.toMax(pos);
			}

			//and load the faces data
			for (int i = 0; i < nbfaces; ++i) {
				Vector3i f_i(curMesh.Indices[3 * i + 0], curMesh.Indices[3 * i + 1], curMesh.Indices[3 * i + 2]);
				f[i] = f_i;

				Vector3d n_i(0);
				//the main problem is that the normal in the obj in on the vertices not the face
				//so here is a conversion (just an avg)
				if (false) {
					n_i += Vector3d(objl::vector3ToVector3d(curMesh.Vertices[f_i.x].Normal));
					n_i += Vector3d(objl::vector3ToVector3d(curMesh.Vertices[f_i.y].Normal));
					n_i += Vector3d(objl::vector3ToVector3d(curMesh.Vertices[f_i.z].Normal));
					n_i /= 3;
				}

				//actually those are the normal for the lighting probably so 
				//since only the geometry has an interest for me
				//I'm better off recomputing the geometric noraml for the face
				//let's however note that the sign of the normal will be pure bs by doing that
				//I cna solve that a bit by estimatin that the normal on the vertexes should still point
				//outward, so I can check the dot product between my computed normal and the avg(vertex_normal) and
				//if it is negative I'll multiply it by -1
				if (true){
					Vector3d v1 = getVertex(f_i.y) - getVertex(f_i.x);
					Vector3d v2 = getVertex(f_i.z) - getVertex(f_i.x);

					n_i = v1.cross(v2);

					Vector3d n_i_v(0);
					n_i_v += Vector3d(objl::vector3ToVector3d(curMesh.Vertices[f_i.x].Normal));
					n_i_v += Vector3d(objl::vector3ToVector3d(curMesh.Vertices[f_i.y].Normal));
					n_i_v += Vector3d(objl::vector3ToVector3d(curMesh.Vertices[f_i.z].Normal));
					
					if (n_i.dot(n_i_v)<0) {
						n_i *= -1;
					}
				}

				n_i.toUnit();
				n[i] = n_i;
			}
			
		}
		else {
			// Output Error
			std::cout << "Failed to Load File. May have failed to find it or it was not an .obj file."<<std::endl;
		
			return -1;
		}

		std::cout << "MeshPerso::loadFromFile successful" << std::endl;


		//reset the offset since we have a new mesh
		offset = Vector3d(0, 0, 0);
		return 0;
	}

	FUNCTION int isInside(Vector3d p, bool* result) {
		if (!isInitialized()) {
			printf("MeshPerso::isInside: trying to use an un-initialized mesh surface\n");
			return -1;
		}

		//do a check using the bounding box to accelerate the computations
		if ((p < getBmin()) || p > getBmax()) {
			*result = false;
			return 0;
		}


		//so I will use the simple throw a ray an count the number of triangle that I hit
		//no need to parralelize that since Il'll call it for multiples points at the same time (anyway I can't on old gpu)

		//the idea is to go through all faces and check collisions
		//some absurdly random values to decrease the probabilities of hitting the border of a face
		Vector3d u(2.575, 6.19763, 5.986);
		u.toUnit();

		int count_intersect = 0;
		for (int i = 0; i < nbfaces; ++i) {
			Vector3i f_i = f[i];

			Vector3d intersect_p(0);
			RealCuda intersect_dist=0;
			
			if (RayIntersectsTriangle<true,false>(p, u, getVertex(f_i.x), getVertex(f_i.y), getVertex(f_i.z), intersect_p, intersect_dist)) {
				if (intersect_dist > 0) {
					count_intersect++;
				}
				//printf("one intersection found with t= %f\n",intersect_dist);
			}
		}

		//printf("final count of intersections %i\n", count_intersect);

		//to see if the count is even or odd chaeck the first bit
		if ((count_intersect & 1) == 0) {
			*result=false;
			//printf("is outside\n");
		}
		else {
			*result=true;
		}
		return 0;
	}

	//the absolue distance
	FUNCTION int distance(Vector3d p, RealCuda* result) {
		if (!isInitialized()) {
			printf("MeshPerso::distance: trying to use an un-initialized mesh surface\n");
			return -1;
		}

		//ok so to find the minimum distance I need to:
		//	1: find the closest vertex
		//	2: find the distance between my point and all the faces/edges containing that vertex
		//  3: take the min of those distances

		//1: first let's find the closest vertex
		//I don't need to care about the fact of multiples points beeing at the same distance because it would still mean that the 
		//min dist is on one face adjascent to either one
		RealCuda dist_sq = (p - v[0]).squaredNorm();
		int id_closest = 0;

		for (int i = 1; i < nbVectex; ++i) {
			RealCuda dist_sq_i = (p - v[i]).squaredNorm();
			if (dist_sq > dist_sq_i) {
				dist_sq = dist_sq_i;
				id_closest = i;
			}
		}

		//2/3: let's find the min dist
		//for each face containng the vertex I need to check the projection on the face and on the edge to be sure
		//no ned to reset the sqdist since it actuall contains one of the possible cases that is the vertex is actually 
		//the projection of our point
		Vector3d p_proj = v[id_closest];
		for (int i = 0; i < nbfaces; ++i) {
			Vector3i f_i = f[i];

			if ((f_i.x == id_closest) || (f_i.y == id_closest) || (f_i.z == id_closest)) {
				//first let's try to project on the edges
				//note: only the 2 edges adjacent to the closest are usefull
				if ((f_i.x == id_closest) || (f_i.y == id_closest)) {
					Vector3d v1 = p - f_i.x;
					Vector3d v2 = f_i.y - f_i.x;

					//compue the projected length
					RealCuda dist_on_line = v1.dot(v2.unit());

					//check that the point is actualla on the edge...
					if (dist_on_line > 0 && dist_on_line < v2.norm()) {
						Vector3d p_proj_i = f_i.x + v2.unit() * dist_on_line;
						RealCuda dist_sq_i = (p - p_proj_i).squaredNorm();
						if (dist_sq > dist_sq_i) {
							dist_sq = dist_sq_i;
							p_proj = p_proj_i;
						}
					}
				}

				if ((f_i.x == id_closest) || (f_i.z == id_closest)) {
					Vector3d v1 = p - f_i.x;
					Vector3d v2 = f_i.z - f_i.x;

					//compue the projected length
					RealCuda dist_on_line = v1.dot(v2.unit());

					//check that the point is actualla on the edge...
					if (dist_on_line > 0 && dist_on_line < v2.norm()) {
						Vector3d p_proj_i = f_i.x + v2.unit() * dist_on_line;
						RealCuda dist_sq_i = (p - p_proj_i).squaredNorm();
						if (dist_sq > dist_sq_i) {
							dist_sq = dist_sq_i;
							p_proj = p_proj_i;
						}
					}
				}

				if ((f_i.y == id_closest) || (f_i.z == id_closest)) {
					Vector3d v1 = p - f_i.y;
					Vector3d v2 = f_i.z - f_i.y;

					//compue the projected length
					RealCuda dist_on_line = v1.dot(v2.unit());

					//check that the point is actualla on the edge...
					if (dist_on_line > 0 && dist_on_line < v2.norm()) {
						Vector3d p_proj_i = f_i.y + v2.unit() * dist_on_line;
						RealCuda dist_sq_i = (p - p_proj_i).squaredNorm();
						if (dist_sq > dist_sq_i) {
							dist_sq = dist_sq_i;
							p_proj = p_proj_i;
						}
					}
				}

				//now we have only one possibility remaining that is the closest point is on the face
				//for that to happen the only choise is for the point to project to be in from of the face
				//so I can just cast a ray with the normal to the fact as the direction (though I need to ignore the direction)
				//of the ray and handle the sign of the distance later since the intersection with the edges don't contain the information anyway

				Vector3d u = n[i];
				Vector3d intersect_p(0);
				RealCuda intersect_dist = 0;

				if (RayIntersectsTriangle<true, true>(p, u, getVertex(f_i.x), getVertex(f_i.y), getVertex(f_i.z), intersect_p, intersect_dist)) {
					if (dist_sq > (intersect_dist * intersect_dist)) {
						dist_sq = (intersect_dist * intersect_dist);
						p_proj = intersect_p;
					}
				}
			}
		}

		*result = SQRT_MACRO_CUDA(dist_sq);
		return 0;
	}


	//sadly I don't realy see a way to obtain the signed distance directly
	//so ther will most likaly be no choice other than checking if the point is inside
	//positive means inside
	FUNCTION int distanceSigned(Vector3d p, RealCuda* result) {
		if (!isInitialized()) {
			printf("MeshPerso::distanceSigned: trying to use an un-initialized mesh surface\n");
			return -1;
		}


		RealCuda dist;
		bool inside;
		distance(p, &dist);
		isInside(p, &inside);

		*result=	dist * ((inside) ? 1 : -1);
		return 0;
	}
	
};


//NOTE1:	seems that virtual function can't be used with managed allocation
//			so I'll use a template to have an equivalent solution
//			the  template parameter allow the représentation of various shapes
//NOTE2:	0 ==> plane: only use this for paralel planes (so max 2)
//			it's just a fast to compute solution if you need bands of fluid near the borders of the simulation
//			however since as soon as you get more than 2 planes the distance computation become pretty heavy
//			it's better to use another solution to represent the same surface (in particular in the case of boxes
//NOTE3:	1 ==> rectangular cuboid. 
//			2 ==> cylinder
//			3 ==> triangular mesh (the distance functions will be WAAAAAAYYY slower)
//					also to actually work the mesh MUST be fully enclosed

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

	//and this is the mesh for the mesh based geometries
	MeshPerso* mesh;

	//reverse the surface so that the exterior become the interior
	bool isReversedSurface;

	//to prevent the destruction when sending a copy to the kernels
	bool destructorActivated;
public:

	inline BufferFluidSurface() {
		type = -1;
		mesh = NULL;
		isReversedSurface = false;
		destructorActivated = false;
	}


	inline ~BufferFluidSurface() {
		if (destructorActivated) {
			clear(true);
		}
	}

	void clear(bool in_depth=false) {
		if (type == 3) {
			if (in_depth) {
				if (mesh != NULL) {
					mesh->clear();
				}
			}
			CUDA_FREE_PTR(mesh);
		}
	}

	void prepareForDestruction() { destructorActivated = true; }


	inline BufferFluidSurface& operator = (const BufferFluidSurface& other) {
		type= other.type;

		//this is for the planes
		o = other.o;
		n = other.n;

		//this is for the cuboid
		center = other.center;
		halfLengths = other.halfLengths;

		//this for radius based geometries
		radius = other.radius;

		//and this is the mesh for the mesh based geometries
		mesh = other.mesh;

		//reverse the surface so that the exterior become the interior
		isReversedSurface = other.isReversedSurface;


		return *this;
	}

	int getType() { return type; }

	void setReversedSurface(bool v) { isReversedSurface = v; }
	bool getReversedSurface() { return isReversedSurface ; }

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

	inline int setMesh(std::string fileName) {
		if (type < 0) {
			type = 3;
		}
		else if (type != 2) {
			return -1;
		}

		cudaMallocManaged(&(mesh), sizeof(MeshPerso));

		mesh->loadFromFile(fileName);

		return 0;
	}

	inline int setMesh(MeshPerso* mesh_i) {
		if (type < 0) {
			type = 3;
		}
		else if (type != 2) {
			return -1;
		}

		mesh=mesh_i;

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
		case 3: {
			cudaMallocManaged(&(mesh), sizeof(MeshPerso));
			mesh->cpy(other.mesh);
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
		case 3: {
			mesh->applyOffset(d);
			break;
		}
		default: {; }//it should NEVER reach here
		};
	}

	std::string toString() {
		std::ostringstream oss;
		//*
		if (isReversedSurface) {
			oss << "Reversed ";
		}
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
			oss << "Cylinder center: " << center.toString() << "  radius: " << radius << "   height: " << halfLengths.y << std::endl;
			break;
		}
		case 3: {
			if (mesh->isInitialized()) {
				oss << "Mesh with nbVertices: " << mesh->nbVectex << "  and nbFaces: " << mesh->nbfaces <<
					 "  and bounding box (bmin//bmax): "<< mesh->getBmin().toString() <<" // "<<mesh->getBmax().toString() <<std::endl;
			}
			else {
				oss << "Mesh not initialized (wtf)" << std::endl;
			}
			break;
		}
		default: {; }//it should NEVER reach here
		};
		//*/
		return oss.str();
	}



	//to know if we are on the inside of each plane we can simply use the dot product*
	FUNCTION inline bool isinside(Vector3d p) {
		bool result;
		//*
		switch (type) {
		case 0: {
			Vector3d v = p - o;
			if (v.dot(n) < 0) {
				result= false;
			}
			else {
				result = true;
			}
			break;
		}
		case 1: {
			Vector3d v = p - center;
			v.toAbs();
			result= (v.x < halfLengths.x) && (v.y < halfLengths.y) && (v.z < halfLengths.z);
			break;
		}
		case 2: {
			Vector3d v = p - center;
			if (abs(v.y) > halfLengths.y) { return false; }

			v.y = 0;
			result = (v.norm() < radius);
			break;
		}
		case 3: {
			bool inside;
			int res=mesh->isInside(p, &inside);
			if (res != 0) {
				printf("BufferFluidSurface::isInside: the is inside function of the mesh returned an error\n");
			}
			result = inside;
		
			break;
		}
		default: {; }//it should NEVER reach here
		}
		//*/

		if (getReversedSurface()) {
			result = !result;
		}


		return result;
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
		case 3: {
			int res = mesh->distance(p, &dist);
			if (res != 0) {
				printf("BufferFluidSurface::distanceToSurface: the is distance function of the mesh returned an error\n");
			}
		}
		default: {; }//it should NEVER reach here
		}
		//*/
		return dist;
	}

	FUNCTION inline RealCuda distanceToSurfaceSigned(Vector3d p) {
		//you could most likely optimize the implementation for some of the types but it's not critical so why bother
		//well actually for the mesh it may be critical if I actuall find a way so ... let's handle the special cases
		RealCuda dist = 0;
		switch (type) {
		case 3: {
			RealCuda dist;
			int res = mesh->distanceSigned(p, &dist);
			if (res != 0) {
				printf("BufferFluidSurface::distanceToSurfaceSigned: the distanceSigned function of the mesh returned an error\n");
			}
			break;
		}
		default: {
			dist= distanceToSurface(p) * ((isinside(p)) ? 1 : -1);
		}
		}

		

		return  dist;
	}

};



/**
This class is used to have a simple way to aggregate multiples iso surfaces
Technically it has to be made so that I can define an aggregate of surface as an intersection or an union of the surfaces
Curretnly I won't code the distance function because I don't have anyuse for them
just note that coding the distance function when the point is outside of even a singel surface might be straigth hell
//*/

class SurfaceAggregation {
	int numSurface;
	int numSurfaceMax;

	BufferFluidSurface* surfaces;

	//true if we are using the union of the surfaces,
	//false if we are using the intersection of the surfaces
	bool isUnion;

	//to prevent the destruction when sending a copy to the kernels
	bool destructorActivated;
public:

	SurfaceAggregation(bool isUnion_i = false) { 
		numSurface = 0; numSurfaceMax = 0; surfaces = NULL; 
		setIsUnion(isUnion_i);
		destructorActivated = false;
	}

	~SurfaceAggregation() {
		if (destructorActivated) {
			//actually I must not clear the surfaces since when i add a surface to an aggregation I do not do a a deep copy
			//this is most likely not good but it's life for now
			//anyway since I have to delete the array the destructor will be called so it's fucked anyway
			///TODO repair that by adding a deep copy when adding a surface to the aggeration
			///		see the clear function for more detail on the problem
			//*
			clear(true);
		}
	}

	void prepareForDestruction() { destructorActivated = true; }

	void setIsUnion(bool val) { isUnion = val; }

	void addSurface(const BufferFluidSurface& s) {
		//first make some space if needed
		if (numSurface + 1 > numSurfaceMax) {
			int numSurfaceMax_back = numSurfaceMax;
			BufferFluidSurface* surfaces_back = surfaces;

			numSurfaceMax = (numSurface + 1) * 2;

			cudaMallocManaged(&(surfaces), numSurfaceMax * sizeof(BufferFluidSurface));

			//copy the already existing ones
			//maybe doing it with memcpy is better...
			for (int i = 0; i < numSurface; ++i)
			{
				surfaces[i] = surfaces_back[i];
			}	

			//free the old memory space
			CUDA_FREE_PTR(surfaces_back);
		}
		//then add the new surface
		surfaces[numSurface] = s;

		numSurface++;
	}

	void clear(bool in_depth = false) {
		///TODO repair that by adding a deep copy when adding a surface to the aggeration
		///		this is the same problem as the destructor is if it is solved there it is fine here
		///		and btw the porblem comme from the fact that the mesh based surface
		///		has a pointer to a mesh which gets deleted when clear but since
		///		we do not have a pointer on the structure but a top level copy
		///		we explose the mesh memory allocation but the surface we copied do not know it has been destroyed...
		/*
		for (int i = 0; i < numSurface; ++i)
		{
			surfaces[i].clear(in_depth);
		}
		//*/
		CUDA_FREE_PTR(surfaces);
		numSurface = 0;
		numSurfaceMax = 0;
	}

	std::string toString() {
		std::ostringstream oss;
		oss << "Surface aggregation, aggreg type: "<<(isUnion?"union":"intersection")<<"   nbrSurface: " << numSurface << std::endl;
		for (int i = 0; i < numSurface; ++i)
		{
			oss<<surfaces[i].toString();
		}
		return oss.str();
	}

	//check if the point is inside the aggregatio of surface depending on the type of aggregation
	FUNCTION bool isinside(Vector3d p) {
		for (int i = 0; i < numSurface; ++i)
		{
			bool val = surfaces[i].isinside(p);

			if (isUnion) {
				//for a union I can return true as long as the point is inside at least one surface
				if (val) { return true; }
			}
			else {
				//for an intersection I can return false is the point is outside of even one surface
				if (!val) { return false; }
			}
		}

		//if we reach each we know the result depending on the union bool
		if (isUnion) {
			//not iside a single surface
			return false; 
		}
		else {
			//inside all surfaces
			return true;
		}
	}

	//WARNING curretnly this can only be used for intersection type of aggregation 
	//It will return a positive number inside the aggregation and a negative number outside (note I'm not sure the negative number works
	//the template parameters can be used to skip the inside/outide computations (and the system will return 0 in this case)
	///TODO see if there is any way to handle other cases
	template <bool compute_inside, bool compute_outside>
	FUNCTION RealCuda distanceSigned(Vector3d p) {
		RealCuda dist = 0;//if the point is outide the reported distance will be 0
		if (isinside(p)) {
			if (compute_inside) {
				dist = surfaces[0].distanceToSurface(p);
				for (int i = 1; i < numSurface; ++i)
				{
					RealCuda cur_dist = surfaces[i].distanceToSurface(p);
					if (dist > cur_dist) {
						dist = cur_dist;
					}
				}
			}
		}
		else {
			if (compute_outside) {
				//nah I have no idea on how to compute that ...
				//I would have to project the point on the survface wich would take a fucking long time
			}
		}
		return dist;
	}

	
};