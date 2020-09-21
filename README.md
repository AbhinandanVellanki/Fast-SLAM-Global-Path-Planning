# Fast-SLAM-Global-Path-Planning

# Fast SLAM Implementation

FastSLAM is a Rao-Blackwellized particle filter for simultaneous localization and mapping. The pose of the robot in the environment is represented by a particle filter. Furthermore, each particle carries a map of the environment, which it uses for localization. In the case of landmark-based FastSLAM, the map is represented by a Kalman Filter, estimating the mean position and covariance of landmarks.

landmark-based FastSLAM algorithm:

1) data: This folder contains files representing the world definition and sensor readings used by the filter.

2) starter code: This folder contains the FastSLAM starter code.

3) doc This folder contains the detailed listing of the algorithm as a PDF file.

# Global Path Planning

Done using Dijkstra and A* algorithms