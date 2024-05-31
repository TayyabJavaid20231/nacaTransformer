#Interpolate
from ml_collections import ConfigDict
import numpy as np
import faiss
import cuspatial, cudf
from stl import mesh




def interpolate(config: ConfigDict, data, wing_data, mach):


	xmin, xmax, ymin, ymax = config.preprocess.dim
	nx, ny = config.vit.img_size
	k = config.preprocess.num_neighbors

	deltax = (xmax - xmin) / (nx-1)

	xb = data[:, 0:2]
	yb = data[:, 2:]
	xq = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)].reshape(2, -1).T





	dist, idx, weights, divisor, yq = get_KNNs(xq, xb, yb, deltax, xmin, ymin, k, config.preprocess.gpu_id)
	target_data = np.concatenate((xq, yq), axis=1)






	wing_points_idx = find_points_inside(xq, wing_data)
	sdf_dist, _ = find_knn(wing_data, target_data[:, 0:2], 1,
				           config.preprocess.gpu_id)

		    # set sdf values inside wing equal to minus 1
	sdf_dist[wing_points_idx, :] = -1

		    # set values for all fields inside wing geometry to zero
	target_data[wing_points_idx, 2:] = 0



	mach_data = np.copy(sdf_dist)
	mach_data = np.where(mach_data == -1, -1, mach)
	mach_data[wing_points_idx, :] = 0


	x = mach_data

		    # Delete point coordinates from decoder input
		#y = target_data[:, -9:]

	y = target_data[:, [-4,-2,-1]]

		    # # Define thermodynamic properties of air at ICAO standard atmosphere
	T0 = 288.15  # [K] Total temperature
	p0 = 101325  # [Pa] Total pressure
	gamma = 1.4  # [-] Ratio of specific heats
	R = 287.058  # [J/(kg*K)] Specific gas constant for dry air
		    #
		    # # M = config.preprocess.mach[1]
	M = mach
		    #
	T = T0 / (1 + 0.5 * (gamma - 1) * M ** 2)
	p_inf = p0 * (1 + 0.5 * (gamma - 1) * M ** 2) ** (-gamma / (gamma - 1))
	u_inf = M * np.sqrt(gamma * R * T)
		    #
		    # # Normalise pressure by freestream pressure
		#y[:, 0] /= p_inf
	y[:, 0] = (y[:, 0] - p_inf) / (gamma * 0.5 * p_inf * mach**2)
	y[wing_points_idx,0] = 0
		    #
		    # # Normalise velocities by freestream velocity
	y[:, 1] /= u_inf
	y[:, 2] /= u_inf
	
	return x, y




def sourceToTargetMapping(data: np.ndarray, deltaX: int, xmin: int, ymin: int):
	"""
	Map points of source mesh to target mesh
	Parameters
	----------
	data: numpy.ndarray
          raw average flow field
          
          
        Returns
    	-------
    	target_grid_mapping: numpy.ndarray
          x and y coordinates of target map and the index of array corresponds to data
          points of source grid
          It should give which source index should mapped to which target point. 
        distance: numpy.ndarray
          distance between target and source pair
	"""	
	target_grid_mapping = np.ndarray([np.shape(data)[0],2], dtype='float32')
	distance = np.ndarray([np.shape(data)[0],1], dtype='float32')



	target_grid_mapping[:,0] = data[:,0] - xmin
	target_grid_mapping[:,1] = data[:,1] - ymin

	
	#target_grid_mapping = (np.rint(data[:, 0:2]/deltaX))*deltaX
	target_grid_mapping = (np.rint(target_grid_mapping/deltaX))*deltaX



	target_grid_mapping[:,0] = target_grid_mapping[:,0] + xmin	
	target_grid_mapping[:,1] = target_grid_mapping[:,1] + ymin



	distance = np.power(np.power((data[:,0]-target_grid_mapping[:,0]),2) + np.power((data[:,1]-target_grid_mapping[:,1]),2),1)
	return distance, target_grid_mapping
	
def calculate_K(target_grid_mapping: np.ndarray, target_coordinates: np.ndarray):
	"""
	Finding all k nearest neighbours and their indexes
	Parameters
	----------
	target_grid_mapping: numpy.ndarray
          mapping which contains sources points mapped to target coordinates
        target_coordinates: numpy.ndarray
          Target coordinates for which knn's are required
          
        Returns
    	-------
    	k: numpy.ndarray
          contains all knn number in an array, contaning indexes 
          corresponding to source grid
          
        """
	target_grid_mapping = np.round(target_grid_mapping,3)
	target_coordinates = np.round(target_coordinates,3)
	a = ((1*(target_grid_mapping[:,0] ==  target_coordinates[0])))
	b = ((1*(target_grid_mapping[:,1] == target_coordinates[1])))
	k = np.where((a*b) == 1)
	return k
	
	

def find_points_inside(target_points, wing_points):
    """

    Parameters
    ----------
    target_points: numpy.ndarray
                   points of interpolation grid
    wing_points: numpy.ndarray
                 points of wing geometry. Must start and end at the same
                 point in clock- or counterclockwise direction

    Returns
    -------
    points_in_wing_idx: numpy.ndarray
                        indexes of target_points array which are inside wing
    """
    # Convert numpy arrays to GeoSeries for points_in_polygon(args)
    pts = cuspatial.GeoSeries.from_points_xy(
        cudf.Series(target_points.flatten()))
    plygon = cuspatial.GeoSeries.from_polygons_xy(
        cudf.Series(wing_points.flatten()).astype(float),
        cudf.Series([0, wing_points.shape[0]]),
        cudf.Series([0, 1]),
        cudf.Series([0, 1])
    )

    # find indexes of points within NACA airfoil shape
    df = cuspatial.point_in_polygon(pts, plygon)
    df.rename(columns={df.columns[0]: "inside"}, inplace=True)

    points_in_wing_idx = df.index[df['inside'] == True].to_numpy()

    return points_in_wing_idx


def find_knn(xb: np.ndarray, xq: np.ndarray, k: int, gpu_id: int):
    """
    Find k-nearest neighbours for a query vector based on the input coordinates
    using GPU-accelerated kNN algorithm. More information on
    https://github.com/facebookresearch/faiss/wiki

    Parameters
    ----------
    xb: numpy.ndarray
        coordinate points of raw data
    xq: numpy.ndarray
        query vector with interpolation points
    k: int
       number of nearest neighbours
    gpu_id: int
            ID of GPU which shall be used

    Returns
    -------
    (dist, indexes): (numpy.ndarray, numpy.ndarray)
                     distances of k nearest neighbours, the index for
                     the corresponding points in the xb array

    """

    _, d = xq.shape

    xb = np.ascontiguousarray(xb, dtype='float32')
    xq = np.ascontiguousarray(xq, dtype='float32')

    res = faiss.StandardGpuResources()

    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    gpu_index.add(xb)
    distances, neighbors = gpu_index.search(xq, k)

    return distances, neighbors


def get_KNNs(xq, xb, yb, deltax, xmin, ymin, k_min, gpu_id):
	

	index = [[0]]*np.shape(xq)[0]
	distance = [[0]]*np.shape(xq)[0]
	weights = [[0]]*np.shape(xq)[0]
	divisor = [[0]]*np.shape(xq)[0]
	#yq = [[0]]*np.shape(xq)[0]
	yq = np.ndarray([np.shape(xq)[0], np.shape(yb)[-1]])#.reshape(np.shape(xq)[0], -1)

	#Mapping to get Fixed KNN's 
	dis, idx = find_knn(xb, xq, k_min, gpu_id)

	#Mapping of all source points to nearest traget points
	dist, source2targetmapping = sourceToTargetMapping(xb, deltax, xmin, ymin)

	#Merging both mapping
	for i in range(np.shape(xq)[0]):
		k = calculate_K(source2targetmapping, xq[i])
		_, indexofIntersected, _ = np.intersect1d(k, idx[i], assume_unique=True, return_indices=True)
		k = np.delete(k, indexofIntersected)
		index[i] = np.append(idx[i], k)
		distance[i] = np.append(dis[i], np.take(dist, k))
		weights[i] = np.power(np.reciprocal(distance[i], out=np.zeros_like(distance[i]), where=distance[i] != 0), 2)
		divisor[i] = np.where(np.sum(weights[i]) == 0, 1e-23, np.sum(weights[i]))
		yq[i] = np.sum(weights[i] * (yb[index[i]].T), axis = 1) / divisor[i]


	return distance, index, weights, divisor, yq



def extractWingData(stl_dir):
#	stl_dir = '0010_0.stl'
	points = mesh.Mesh.from_file(stl_dir)[:, 0:3]

    # remove duplicate points of side surface at x=-1
	points = points[points[:, 2] > 0][:, 0:2]

    # find index where points of side surface end
	idx = np.where(points == np.max(points[:, 0]))[0][-1]

    # remove redundant points from .stl-file type
	points = points[: idx + 1, :]
	return points






















