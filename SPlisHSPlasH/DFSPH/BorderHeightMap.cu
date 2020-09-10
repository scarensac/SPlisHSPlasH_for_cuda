
class BorderHeightMap {
public:
	int samplingCount;
	Vector3d* samplingPositions;
	RealCuda* heights;

	BorderHeightMap() {}

	void init(SPH::DFSPHCData& data) {
		//I'll hardcode it for now
		//so the simu domain is -2;2  -0.5;0.5

		Vector3d points[4];
		points[0] = Vector3d(-2, 0, -0.5);
		points[1] = Vector3d(2, 0, -0.5);
		points[2] = Vector3d(2, 0, 0.5);
		points[3] = Vector3d(-2, 0, 0.5);

		//count the nbr of points and allocate
		//space between sampling will be the particle radius
		samplingCount = 0;
		RealCuda samplingSpacing = data.particleRadius;
		Vector3d startPt = points[3];
		for (int i = 0; i < 4; ++i) {
			Vector3d endPt = points[i];

			RealCuda length = (endPt - startPt).norm();
			samplingCount += floorf(length / samplingSpacing);

			startPt = endPt;
		}
		samplingCount += 4;//to add the 4 anchor points

		cudaMallocManaged(&(samplingPositions), sizeof(Vector3d)*samplingCount);
		cudaMallocManaged(&(heights), sizeof(RealCuda)*samplingCount);

		//and now we can set the values
		startPt = points[3];
		int count = 0;
		RealCuda base_height = 1.0;//0.47;
		for (int i = 0; i < 4; ++i) {
			Vector3d endPt = points[i];


			Vector3d delta = endPt - startPt;
			RealCuda length = delta.norm();
			delta *= samplingSpacing / length;
			int localCount = floorf(length / samplingSpacing);

			Vector3d pos = startPt;
			samplingPositions[count] = pos;
			heights[count] = base_height;
			count++;
			for (int j = 0; j < localCount; ++j) {
				pos += delta;

				samplingPositions[count] = pos;
				heights[count] = base_height;
				count++;
			}

			startPt = endPt;
		}
	}

	FUNCTION RealCuda getHeight(Vector3d pos) {
		//let's do a basic way: check all the position and keep the closest
		/// TODO CODE an acceleration structure so I don't have to explore every points, 
		///			you can use a structure similar to the one for the neightbors sice you won't have to rebuild it
		/// BTW I can use a similar structure to  handel the normal map for complexborders
		RealCuda distSq = (pos - samplingPositions[0]).squaredNorm();
		RealCuda height = heights[0];
		for (int i = 1; i < samplingCount; ++i) {
			RealCuda localDistSq = (pos - samplingPositions[i]).squaredNorm();
			if (localDistSq < distSq) {
				distSq = localDistSq;
				height = heights[i];
			}
		}

		return height;
	}
};