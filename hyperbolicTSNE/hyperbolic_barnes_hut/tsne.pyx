# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
#
# This file implements hyperbolic t-SNE components efficiently using Cython.
# The implementation is based on the tSNE code from Christopher Moody and 
# and Nick Travers available at https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_barnes_hut_tsne.pyx
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libc.math cimport sqrt, log, acosh, cosh, cos, sin, M_PI, atan2, tanh, atanh, isnan, fabs, fmin, fmax
from libc.stdlib cimport malloc, free, realloc
from cython.parallel cimport prange, parallel
from libc.string cimport memcpy
from libcpp.vector cimport vector
from libc.stdint cimport SIZE_MAX

np.import_array()


cdef char* EMPTY_STRING = ""

# Smallest strictly positive value that can be represented by floating
# point numbers for different precision levels. This is useful to avoid
# taking the log of zero when computing the KL divergence.
cdef float FLOAT32_TINY = np.finfo(np.float32).tiny

# Useful to void division by zero or divergence to +inf.
cdef float FLOAT64_EPS = np.finfo(np.float64).eps

cdef double EPSILON = 1e-5
cdef double MAX_TANH = 15.0
cdef double BOUNDARY = 1 - EPSILON
cdef int RANGE = 0
cdef int ANGLE = 1
cdef double MACHINE_EPSILON = np.finfo(np.double).eps
cdef int TAKE_TIMING = 1
cdef int AREA_SPLIT = 0
cdef int GRAD_FIX = 1

cdef double clamp(double n, double lower, double upper) nogil:
    cdef double t = lower if n < lower else n
    return upper if t > upper else t

##################################################
# QuadTree
##################################################
ctypedef np.npy_float64 DTYPE_t          # Type of X
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef struct Cell:
    # Base storage structure for cells in a QuadTree object

    # Tree structure
    SIZE_t parent              # Parent cell of this cell
    SIZE_t[4] children         # Array pointing to children of this cell

    # Cell description
    SIZE_t cell_id             # Id of the cell in the cells array in the Tree
    SIZE_t point_index         # Index of the point at this cell (only defined
                               # in non empty leaf)
    bint is_leaf               # Does this cell have children?
    DTYPE_t squared_max_width  # Squared value of the maximum width w
    SIZE_t depth               # Depth of the cell in the tree
    SIZE_t cumulative_size     # Number of points included in the subtree with
                               # this cell as a root.

    # Internal constants
    DTYPE_t[2] center          # Store the center for quick split of cells
    DTYPE_t[2] barycenter      # Keep track of the center of mass of the cell
    DTYPE_t lorentz_factor_sum      # TODO

    # Cell boundaries
    DTYPE_t[2] min_bounds      # Inferior boundaries of this cell (inclusive)
    DTYPE_t[2] max_bounds      # Superior boundaries of this cell (exclusive)

# Build the corresponding numpy dtype for Cell.
# This works by casting `dummy` to an array of Cell of length 1, which numpy
# can construct a `dtype`-object for. See https://stackoverflow.com/q/62448946
# for a more detailed explanation.
cdef Cell dummy;
CELL_DTYPE = np.asarray(<Cell[:1]>(&dummy)).dtype

assert CELL_DTYPE.itemsize == sizeof(Cell)

ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (Cell*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes"
                              % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience

# This is effectively an ifdef statement in Cython
# It allows us to write printf debugging lines
# and remove them at compile time
cdef enum:
    DEBUGFLAG = 0

cdef extern from "time.h":
    # Declare only what is necessary from `tm` structure.
    ctypedef long clock_t
    clock_t clock() nogil
    double CLOCKS_PER_SEC


from cpython cimport Py_INCREF, PyObject, PyTypeObject

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(np.ndarray arr, PyObject* obj)

cdef DTYPE_t er_to_hr(DTYPE_t er) nogil:
    return acosh(1 + 2 * er * er / (1 - er * er + MACHINE_EPSILON))

cdef DTYPE_t hr_to_er(DTYPE_t hr) nogil:
    cdef double ch = cosh(hr)

    return sqrt((ch - 1) / (ch + 1))

cdef DTYPE_t sq_norm(DTYPE_t x, DTYPE_t y) nogil:
    return x ** 2 + y ** 2

cdef DTYPE_t poincare_to_klein(DTYPE_t c, DTYPE_t sq_n) nogil:
    return 2 * c / (1 + sq_n)

cdef DTYPE_t klein_to_poincare(DTYPE_t c, DTYPE_t sq_n) nogil:
    return c / (1 + sqrt(1 - sq_n))

cdef DTYPE_t lorentz_factor(DTYPE_t sq_n) nogil:
    return 1 / sqrt(1 - sq_n)

cdef DTYPE_t distance_polar(DTYPE_t r, DTYPE_t phi, DTYPE_t r2, DTYPE_t phi2) nogil:
    cdef:
        double u0 = r * cos(phi)
        double u1 = r * sin(phi)

        double v0 = r2 * cos(phi2)
        double v1 = r2 * sin(phi2)

    return distance(u0, u1, v0, v1)

cdef DTYPE_t mid_range(DTYPE_t min_r, DTYPE_t max_r) nogil:
    cdef:
        # TODO: check alpha
        double alpha = 1.
        double res = cosh(alpha * er_to_hr(max_r)) + cosh(alpha * er_to_hr(min_r))

    if AREA_SPLIT:
        return hr_to_er(acosh(res / 2) / alpha)
    else:
        return (min_r + max_r) / 2

cdef DTYPE_t get_max_for_polar_rect(DTYPE_t[2] min_bounds, DTYPE_t[2] max_bounds) nogil:
    return max(distance_polar(max_bounds[RANGE], min_bounds[ANGLE], max_bounds[RANGE], max_bounds[ANGLE]),
               distance_polar(max_bounds[RANGE], min_bounds[ANGLE], min_bounds[RANGE], max_bounds[ANGLE]),
               distance_polar(max_bounds[RANGE], min_bounds[ANGLE], min_bounds[RANGE], min_bounds[ANGLE]))

cdef class _QuadTree:
    """Array-based representation of a QuadTree.

    This class is currently working for indexing 2D data (regular QuadTree) and
    for indexing 3D data (OcTree). It is planned to split the 2 implementations
    using `Cython.Tempita` to save some memory for QuadTree.

    Note that this code is currently internally used only by the Barnes-Hut
    method in `sklearn.manifold.TSNE`. It is planned to be refactored and
    generalized in the future to be compatible with nearest neighbors API of
    `sklearn.neighbors` with 2D and 3D data.
    """

    # Parameters of the tree
    cdef public int n_dimensions         # Number of dimensions in X
    cdef public int verbose              # Verbosity of the output
    cdef SIZE_t n_cells_per_cell         # Number of children per node. (2 ** n_dimension)

    # Tree inner structure
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t cell_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef public SIZE_t n_points          # Total number of points
    cdef Cell* cells                     # Array of nodes

    def __cinit__(self, int n_dimensions, int verbose):
        """Constructor."""
        # Parameters of the tree
        self.n_dimensions = n_dimensions
        self.verbose = verbose
        self.n_cells_per_cell = 2 ** self.n_dimensions

        # Inner structures
        self.max_depth = 0
        self.cell_count = 0
        self.capacity = 0
        self.n_points = 0
        self.cells = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.cells)

    property cumulative_size:
        def __get__(self):
            return self._get_cell_ndarray()['cumulative_size'][:self.cell_count]

    property leafs:
        def __get__(self):
            return self._get_cell_ndarray()['is_leaf'][:self.cell_count]

    def init_py(self, X):
        """Initialize a tree from an array of points X python"""
        cdef:
            int i
            DTYPE_t r
            DTYPE_t[2] min_bounds, max_bounds

        n_samples = X.shape[0]

        capacity = 100
        self._resize(capacity)

        min_bounds[RANGE] = 1.
        max_bounds[RANGE] = 0.

        for i in range(n_samples):
            r = sqrt(X[i, 0] ** 2 + X[i, 1] ** 2)

            if r < min_bounds[RANGE]:
                min_bounds[RANGE] = r

            if r > max_bounds[RANGE]:
                max_bounds[RANGE] = r

        min_bounds[ANGLE] = 0.
        max_bounds[ANGLE] = 2. * M_PI

        if self.verbose > 10:
            for i in range(self.n_dimensions):
                printf("[QuadTree] bounding box axis %i : [%f, %f]\n",
                       i, min_bounds[i], max_bounds[i])

        # Create the initial node with boundaries from the dataset
        self._init_root(min_bounds, max_bounds)

    def insert_py(self, x, i):
        """Insert point python"""
        cdef:
            DTYPE_t[2] pt
        pt[RANGE] = sqrt(x[0] ** 2 + x[1] ** 2)
        pt[ANGLE] = atan2(x[1], x[0])

        pt[ANGLE] = pt[ANGLE] if pt[ANGLE] > 0 else pt[ANGLE] + 2 * M_PI

        self.insert_point(pt, i)

    def build_tree(self, X):
        """Build a tree from an array of points X."""
        cdef:
            int i
            DTYPE_t[2] pt
            DTYPE_t[2] min_bounds, max_bounds

        # validate X and prepare for query
        # X = check_array(X, dtype=DTYPE_t, order='C')
        n_samples = X.shape[0]

        capacity = 100
        self._resize(capacity)

        min_bounds[RANGE] = 1.
        max_bounds[RANGE] = 0.

        for i in range(n_samples):
            r = sqrt(X[i, 0] ** 2 + X[i, 1] ** 2)

            if r < min_bounds[RANGE]:
                min_bounds[RANGE] = r

            if r > max_bounds[RANGE]:
                max_bounds[RANGE] = r

        min_bounds[ANGLE] = 0.
        max_bounds[ANGLE] = 2. * M_PI

        if self.verbose > 10:
            for i in range(self.n_dimensions):
                printf("[QuadTree] bounding box axis %i : [%f, %f]\n",
                       i, min_bounds[i], max_bounds[i])

        # Create the initial node with boundaries from the dataset
        self._init_root(min_bounds, max_bounds)

        for i in range(n_samples):
            pt[RANGE] = sqrt(X[i, 0] ** 2 + X[i, 1] ** 2)
            pt[ANGLE] = atan2(X[i, 1], X[i, 0])

            pt[ANGLE] = pt[ANGLE] if pt[ANGLE] > 0 else pt[ANGLE] + 2 * M_PI

            # if self.verbose > 10:
            #     printf("[QuadTree] Inserting point: [%f, %f]\n", pt[RANGE], pt[ANGLE])
            self.insert_point(pt, i)

        # Shrink the cells array to reduce memory usage
        self._resize(capacity=self.cell_count)

    cdef int insert_point(self, DTYPE_t[2] point, SIZE_t point_index,
                          SIZE_t cell_id=0) nogil except -1:
        """Insert a point in the QuadTree."""
        cdef int ax
        cdef DTYPE_t n_frac
        cdef SIZE_t selected_child
        cdef Cell* cell = &self.cells[cell_id]
        cdef SIZE_t n_point = cell.cumulative_size
        cdef DTYPE_t temp_norm
        cdef DTYPE_t temp_lorentz
        cdef DTYPE_t[2] poincare_point
        cdef DTYPE_t[2] klein_point

        if self.verbose > 10:
            printf("[QuadTree] Inserting depth %li\n", cell.depth)

        # If the cell is an empty leaf, insert the point in it
        if cell.cumulative_size == 0:
            cell.cumulative_size = 1
            self.n_points += 1

            poincare_point[0] = point[RANGE] * cos(point[ANGLE])
            poincare_point[1] = point[RANGE] * sin(point[ANGLE])

            cell.barycenter[0] = poincare_point[0]
            cell.barycenter[1] = poincare_point[1]

            temp_norm = sq_norm(poincare_point[0], poincare_point[1])
            temp_norm = sq_norm(poincare_to_klein(poincare_point[0], temp_norm),
                                poincare_to_klein(poincare_point[1], temp_norm))
            temp_lorentz = lorentz_factor(temp_norm)
            cell.lorentz_factor_sum = temp_lorentz

            cell.point_index = point_index
            if self.verbose > 10:
                printf("[QuadTree] inserted point %li in cell %li\n",
                       point_index, cell_id)
            return cell_id

        # If the cell is not a leaf, update cell internals and
        # recurse in selected child
        if not cell.is_leaf:
            poincare_point[0] = point[RANGE] * cos(point[ANGLE])
            poincare_point[1] = point[RANGE] * sin(point[ANGLE])

            temp_norm = sq_norm(poincare_point[0], poincare_point[1])

            klein_point[0] = poincare_to_klein(poincare_point[0], temp_norm)
            klein_point[1] = poincare_to_klein(poincare_point[1], temp_norm)

            temp_norm = sq_norm(klein_point[0],
                                klein_point[1])
            temp_lorentz = lorentz_factor(temp_norm)

            temp_norm = sq_norm(cell.barycenter[0], cell.barycenter[1])
            cell.barycenter[0] = ((poincare_to_klein(cell.barycenter[0], temp_norm) * cell.lorentz_factor_sum) + temp_lorentz * klein_point[0]) / (cell.lorentz_factor_sum + temp_lorentz)
            cell.barycenter[1] = ((poincare_to_klein(cell.barycenter[1], temp_norm) * cell.lorentz_factor_sum) + temp_lorentz * klein_point[1]) / (cell.lorentz_factor_sum + temp_lorentz)

            temp_norm = sq_norm(cell.barycenter[0], cell.barycenter[1])
            cell.barycenter[0] = klein_to_poincare(cell.barycenter[0], temp_norm)
            cell.barycenter[1] = klein_to_poincare(cell.barycenter[1], temp_norm)

            cell.lorentz_factor_sum = cell.lorentz_factor_sum + temp_lorentz

            # Euclidean barycenter
            # cell.barycenter[1] = (n_point * cell.barycenter[1] + (point[RANGE] * sin(point[ANGLE]))) / (n_point + 1)

            # Increase the size of the subtree starting from this cell
            cell.cumulative_size += 1

            # Insert child in the correct subtree
            selected_child = self._select_child(point, cell)
            if self.verbose > 49:
                printf("[QuadTree] selected child %li\n", selected_child)
            if selected_child == -1:
                self.n_points += 1
                return self._insert_point_in_new_child(point, cell, point_index)
            return self.insert_point(point, point_index, selected_child)

        # Finally, if the cell is a leaf with a point already inserted,
        # split the cell in n_cells_per_cell if the point is not a duplicate.
        # If it is a duplicate, increase the size of the leaf and return.
        if self._is_duplicate(point, cell.barycenter):
            if self.verbose > 10:
                printf("[QuadTree] found a duplicate!\n")
            cell.cumulative_size += 1
            self.n_points += 1
            return cell_id

        # In a leaf, the barycenter correspond to the only point included
        # in it.
        poincare_point[RANGE] = sqrt(cell.barycenter[0] ** 2 + cell.barycenter[1] ** 2)
        poincare_point[ANGLE] = atan2(cell.barycenter[1], cell.barycenter[0])

        poincare_point[ANGLE] = poincare_point[ANGLE] if poincare_point[ANGLE] > 0 else poincare_point[ANGLE] + 2 * M_PI
        self._insert_point_in_new_child(poincare_point, cell, cell.point_index,
                                        cell.cumulative_size)
        return self.insert_point(point, point_index, cell_id)

    # XXX: This operation is not Thread safe
    cdef SIZE_t _insert_point_in_new_child(self, DTYPE_t[2] point, Cell* cell,
                                          SIZE_t point_index, SIZE_t size=1
                                          ) nogil:
        """Create a child of cell which will contain point."""

        # Local variable definition
        cdef:
            SIZE_t cell_id, cell_child_id, parent_id
            DTYPE_t[2] save_point
            DTYPE_t width
            DTYPE_t temp_norm
            Cell* child
            int i

        # If the maximal capacity of the Tree have been reached, double the capacity
        # We need to save the current cell id and the current point to retrieve them
        # in case the reallocation
        if self.cell_count + 1 > self.capacity:
            parent_id = cell.cell_id
            for i in range(self.n_dimensions):
                save_point[i] = point[i]
            self._resize(SIZE_MAX)
            cell = &self.cells[parent_id]
            point = save_point

        # Get an empty cell and initialize it
        cell_id = self.cell_count
        self.cell_count += 1
        child  = &self.cells[cell_id]

        self._init_cell(child, cell.cell_id, cell.depth + 1)
        child.cell_id = cell_id

        # Set the cell as an inner cell of the Tree
        cell.is_leaf = False
        cell.point_index = -1

        # Set the correct boundary for the cell, store the point in the cell
        # and compute its index in the children array.
        cell_child_id = 0
        for i in range(self.n_dimensions):
            cell_child_id *= 2
            if point[i] >= cell.center[i]:
                cell_child_id += 1
                child.min_bounds[i] = cell.center[i]
                child.max_bounds[i] = cell.max_bounds[i]
            else:
                child.min_bounds[i] = cell.min_bounds[i]
                child.max_bounds[i] = cell.center[i]

        child.center[ANGLE] = (child.min_bounds[ANGLE] + child.max_bounds[ANGLE]) / 2.
        child.center[RANGE] = mid_range(child.min_bounds[RANGE], child.max_bounds[RANGE])

        child.barycenter[0] = point[RANGE] * cos(point[ANGLE])
        child.barycenter[1] = point[RANGE] * sin(point[ANGLE])

        temp_norm = sq_norm(child.barycenter[0], child.barycenter[1])
        temp_norm = sq_norm(poincare_to_klein(child.barycenter[0], temp_norm),
                            poincare_to_klein(child.barycenter[1], temp_norm))
        child.lorentz_factor_sum = lorentz_factor(temp_norm)

        width = get_max_for_polar_rect(child.min_bounds, child.max_bounds)
        child.squared_max_width = width * width

        # Store the point info and the size to account for duplicated points
        child.point_index = point_index
        child.cumulative_size = size

        # Store the child cell in the correct place in children
        cell.children[cell_child_id] = child.cell_id

        if self.verbose > 10:
            printf("[QuadTree] inserted point %li in new child %li\n",
                   point_index, cell_id)

        return cell_id


    cdef bint _is_duplicate(self, DTYPE_t[2] point1, DTYPE_t[2] point2) nogil:
        """Check if the two given points are equals."""
        cdef int i
        cdef bint res = True
        # for i in range(self.n_dimensions):
        #     # Use EPSILON to avoid numerical error that would overgrow the tree
        #     res &= fabs(point1[i] - point2[i]) <= EPSILON
        res &= fabs((point1[0] * cos(point1[1])) - point2[0]) <= EPSILON
        res &= fabs((point1[0] * sin(point1[1])) - point2[1]) <= EPSILON
        return res


    cdef SIZE_t _select_child(self, DTYPE_t[2] point, Cell* cell) nogil:
        """Select the child of cell which contains the given query point."""
        cdef:
            int i
            SIZE_t selected_child = 0

        for i in range(self.n_dimensions):
            # Select the correct child cell to insert the point by comparing
            # it to the borders of the cells using precomputed center.
            selected_child *= 2
            if point[i] >= cell.center[i]:
                selected_child += 1
        return cell.children[selected_child]

    cdef void _init_cell(self, Cell* cell, SIZE_t parent, SIZE_t depth) nogil:
        """Initialize a cell structure with some constants."""
        cell.parent = parent
        cell.is_leaf = True
        cell.depth = depth
        cell.squared_max_width = 0
        cell.cumulative_size = 0
        for i in range(self.n_cells_per_cell):
            cell.children[i] = SIZE_MAX

    cdef void _init_root(self, DTYPE_t[2] min_bounds, DTYPE_t[2] max_bounds) nogil:
        """Initialize the root node with the given space boundaries"""
        cdef:
            int i
            DTYPE_t width
            Cell* root = &self.cells[0]

        self._init_cell(root, -1, 0)
        for i in range(self.n_dimensions):
            root.min_bounds[i] = min_bounds[i]
            root.max_bounds[i] = max_bounds[i]

        root.center[ANGLE] = (max_bounds[ANGLE] + min_bounds[ANGLE]) / 2.
        root.center[RANGE] = mid_range(min_bounds[RANGE], max_bounds[RANGE])

        width = get_max_for_polar_rect(min_bounds, max_bounds)
        root.squared_max_width =  width * width
        root.cell_id = 0

        self.cell_count += 1

    cdef long summarize(self, DTYPE_t[2] point, DTYPE_t* results,
                        float squared_theta=.5, SIZE_t cell_id=0, long idx=0
                        ) nogil:
        """Summarize the tree compared to a query point.

        Input arguments
        ---------------
        point : array (n_dimensions)
             query point to construct the summary.
        cell_id : integer, optional (default: 0)
            current cell of the tree summarized. This should be set to 0 for
            external calls.
        idx : integer, optional (default: 0)
            current index in the result array. This should be set to 0 for
            external calls
        squared_theta: float, optional (default: .5)
            threshold to decide whether the node is sufficiently far
            from the query point to be a good summary. The formula is such that
            the node is a summary if
                node_width^2 / dist_node_point^2 < squared_theta.
            Note that the argument should be passed as theta^2 to avoid
            computing square roots of the distances.

        Output arguments
        ----------------
        results : array (n_samples * (n_dimensions+2))
            result will contain a summary of the tree information compared to
            the query point:
            - results[idx:idx+n_dimensions] contains the coordinate-wise
                difference between the query point and the summary cell idx.
                This is useful in t-SNE to compute the negative forces.
            - result[idx+n_dimensions+1] contains the squared euclidean
                distance to the summary cell idx.
            - result[idx+n_dimensions+2] contains the number of point of the
                tree contained in the summary cell idx.

        Return
        ------
        idx : integer
            number of elements in the results array.
        """
        cdef:
            int i, idx_d = idx + self.n_dimensions
            bint duplicate = True
            double dist
            Cell* cell = &self.cells[cell_id]

        results[idx_d] = 0.
        for i in range(self.n_dimensions):
            results[idx + i] = distance_grad_q(point, cell.barycenter, i)
            duplicate &= fabs(results[idx + i]) <= EPSILON

        dist = distance_q(point, cell.barycenter)
        results[idx_d] = dist * dist

        # Do not compute self interactions
        if duplicate and cell.is_leaf:
            return idx

        # Check whether we can use this node as a summary
        # It's a summary node if the angular size as measured from the point
        # is relatively small (w.r.t. to theta) or if it is a leaf node.
        # If it can be summarized, we use the cell center of mass
        # Otherwise, we go a higher level of resolution and into the leaves.
        if cell.is_leaf or ((cell.squared_max_width / results[idx_d]) < squared_theta):
            # printf("[QuadTree] Theta check: %g, %g, %f\n", cell.squared_max_width, results[idx_d], squared_theta)
            # For debugging
            # results[idx_d + 1] = cell_id
            results[idx_d + 1] = <DTYPE_t> cell.cumulative_size
            return idx + self.n_dimensions + 2

        else:
            # Recursively compute the summary in nodes
            for c in range(self.n_cells_per_cell):
                child_id = cell.children[c]
                if child_id != -1:
                    idx = self.summarize(point, results, squared_theta, child_id, idx)

        return idx

    def get_cell(self, point):
        """return the id of the cell containing the query point or raise
        ValueError if the point is not in the tree
        """
        cdef DTYPE_t[2] query_pt
        cdef int i

        assert len(point) == self.n_dimensions, (
            "Query point should be a point in dimension {}."
            .format(self.n_dimensions))

        for i in range(self.n_dimensions):
            query_pt[i] = point[i]

        return self._get_cell(query_pt, 0)

    cdef int _get_cell(self, DTYPE_t[2] point, SIZE_t cell_id=0
                       ) nogil except -1:
        """guts of get_cell.

        Return the id of the cell containing the query point or raise ValueError
        if the point is not in the tree"""
        cdef:
            SIZE_t selected_child
            Cell* cell = &self.cells[cell_id]

        if cell.is_leaf:
            if self._is_duplicate(cell.barycenter, point):
                if self.verbose > 99:
                    printf("[QuadTree] Found point in cell: %li\n",
                           cell.cell_id)
                return cell_id
            with gil:
                raise ValueError("Query point not in the Tree.")

        selected_child = self._select_child(point, cell)
        if selected_child > 0:
            if self.verbose > 99:
                printf("[QuadTree] Selected_child: %li\n", selected_child)
            return self._get_cell(point, selected_child)
        with gil:
            raise ValueError("Query point not in the Tree.")

    # Pickling primitives

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (_QuadTree, (self.n_dimensions, self.verbose),
                           self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["cell_count"] = self.cell_count
        d["capacity"] = self.capacity
        d["n_points"] = self.n_points
        d["cells"] = self._get_cell_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.cell_count = d["cell_count"]
        self.capacity = d["capacity"]
        self.n_points = d["n_points"]

        if 'cells' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        cell_ndarray = d['cells']

        if (cell_ndarray.ndim != 1 or
                cell_ndarray.dtype != CELL_DTYPE or
                not cell_ndarray.flags.c_contiguous):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = cell_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)

        cells = memcpy(self.cells, (<np.ndarray> cell_ndarray).data,
                       self.capacity * sizeof(Cell))


    # Array manipulation methods, to convert it to numpy or to resize
    # self.cells array

    cdef np.ndarray _get_cell_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.cell_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Cell)
        cdef np.ndarray arr
        Py_INCREF(CELL_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   CELL_DTYPE, 1, shape,
                                   strides, <void*> self.cells,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array!")
        return arr

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.cells != NULL:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 9  # default initial value to min
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.cells, capacity)

        # if capacity smaller than cell_count, adjust the counter
        if capacity < self.cell_count:
            self.cell_count = capacity

        self.capacity = capacity
        return 0

    def _py_summarize(self, DTYPE_t[:] query_pt, DTYPE_t[:, :] X, float angle):
        # Used for testing summarize
        cdef:
            DTYPE_t[:] summary
            int n_samples, n_dimensions

        n_samples = X.shape[0]
        n_dimensions = X.shape[1]
        summary = np.empty(4 * n_samples, dtype=np.float64)

        idx = self.summarize(&query_pt[0], &summary[0], angle * angle)
        return idx, summary


#################################################
# Dist and Dist Grad functions
#################################################
cdef double distance_q(DTYPE_t* u, DTYPE_t* v) nogil:
    return distance(u[0], u[1], v[0], v[1])

cdef double dot_q(DTYPE_t* u, DTYPE_t* v) nogil:
    return u[0] * v[0] + u[1] * v[1]

cdef double distance_grad_q(DTYPE_t* u, DTYPE_t* v, int ax) nogil:
    return distance_grad(u[0], u[1], v[0], v[1], ax)

cpdef double distance(double u0, double u1, double v0, double v1) nogil:
    if fabs(u0 - v0) <= EPSILON and fabs(u1 - v1) <= EPSILON:
        return 0.

    cdef:
        double uv2 = ((u0 - v0) * (u0 - v0)) + ((u1 - v1) * (u1 - v1))
        double u_sq = clamp(u0 * u0 + u1 * u1, 0, BOUNDARY)
        double v_sq = clamp(v0 * v0 + v1 * v1, 0, BOUNDARY)
        double alpha = 1. - u_sq
        double beta = 1. - v_sq
        double result = acosh( 1. + 2. * uv2 / ( alpha * beta ) )

    return result

cdef double distance_grad(double u0, double u1, double v0, double v1, int ax) nogil:
    if fabs(u0 - v0) <= EPSILON and fabs(u1 - v1) <= EPSILON:
        return 0.

    cdef:
        double a = u0 - v0
        double b = u1 - v1
        double uv2 = a * a + b * b

        double u_sq = clamp(u0 * u0 + u1 * u1, 0, BOUNDARY)
        double v_sq = clamp(v0 * v0 + v1 * v1, 0, BOUNDARY)
        double alpha = 1. - u_sq
        double beta = 1. - v_sq

        double gamma = 1. + (2. / (alpha * beta)) * uv2
        double shared_scalar = 4. / fmax(beta * sqrt((gamma * gamma) - 1.), MACHINE_EPSILON)

        double u_scalar = (v_sq - 2. * (u0 * v0 + u1 * v1) + 1.) / (alpha * alpha)
        double v_scalar = 1. / alpha

    if ax == 0:
        return shared_scalar * (u_scalar * u0 - v_scalar * v0)
    else:
        return shared_scalar * (u_scalar * u1 - v_scalar * v1)

cdef void exp_map_single(double* x, double* v, double* res) nogil:
    cdef double x_norm_sq, metric, v_norm, v_scalar
    cdef double* y = <double*> malloc(sizeof(double) * 2)

    x_norm_sq = clamp(x[0] ** 2 + x[1] ** 2, 0, BOUNDARY)

    metric = 2. / (1. - x_norm_sq)
    v_norm = sqrt(v[0] ** 2 + v[1] ** 2)

    v_scalar = tanh(clamp((metric * v_norm) / 2., -MAX_TANH, MAX_TANH))

    for j in range(2):
        y[j] = (v[j] / v_norm) * v_scalar

    mobius_addition(x, y, res)
    free(y)

cpdef void exp_map(double[:, :] x, double[:, :] v, double[:, :] out, int num_threads) nogil:
    cdef double* exp_map_res = <double*> malloc(sizeof(double) * 2)

    for i in range(x.shape[0]):
        exp_map_single(&x[i, 0], &v[i, 0], exp_map_res)

        for j in range(2):
            out[i, j] = exp_map_res[j]

    free(exp_map_res)

cdef void mobius_addition(double* x, double* y, double* res) nogil:
    cdef double y_norm_sq, x_norm_sq, x_scalar, y_scalar, r_term, denominator

    x_norm_sq = clamp(x[0] ** 2 + x[1] ** 2, 0, BOUNDARY)
    y_norm_sq = y[0] ** 2 + y[1] ** 2

    r_term = 1. + 2. * (x[0] * y[0] + x[1] * y[1])

    x_scalar = (r_term + y_norm_sq)
    y_scalar = (1. - x_norm_sq)

    denominator = r_term + x_norm_sq * y_norm_sq

    for i in range(2):
        res[i] = (x_scalar * x[i] + y_scalar * y[i]) / denominator

cdef void log_map_single(double* x, double* y, double* res) nogil:
    cdef double x_norm_sq, metric, y_scalar

    x_norm_sq = clamp(x[0] ** 2 + x[1] ** 2, 0, BOUNDARY)

    metric = 2. / (1. - x_norm_sq)

    cdef double* u = <double*> malloc(sizeof(double) * 2)
    for j in range(2):
        u[j] = -x[j]

    cdef double* mobius_res = <double *> malloc(sizeof(double) * 2)
    mobius_addition(u, y, mobius_res)

    free(u)

    mob_add_norm = sqrt(mobius_res[0] ** 2 + mobius_res[1] ** 2)
    y_scalar = atanh(fmin(mob_add_norm, 1. - EPSILON))

    for j in range(2):
        res[j] = (2. / metric) * y_scalar * (mobius_res[j] / mob_add_norm)

    free(mobius_res)

cpdef void log_map(double[:, :] x, double[:, :] y, double[:, :] out, int num_threads) nogil:
    cdef double* log_map_res = <double*> malloc(sizeof(double) * 2)

    for i in range(x.shape[0]):
        log_map_single(&x[i, 0], &y[i, 0], log_map_res)

        for j in range(2):
            out[i, j] = log_map_res[j]

    free(log_map_res)

cpdef void constrain(double[:, :] y, double[:, :] out, int num_threads) nogil:
    for i in range(y.shape[0]):
        point_norm = sqrt(y[i, 0] ** 2 + y[i, 1] ** 2)

        for j in range(2):
            if point_norm >= BOUNDARY:
                out[i, j] = (y[i, j] / point_norm) - EPSILON
            else:
                out[i, j] = y[i, j]

cpdef void poincare_dists(double[:, :] y, double[:, :] out) nogil:
    cdef:
        long i, j

    with nogil, parallel(num_threads=12):
        for i in prange(0, y.shape[0], schedule='static'):
            for j in range(0, y.shape[0]):
                if i == j:
                    continue
                out[i, j] = distance(y[i, 0], y[i, 1], y[j, 0], y[j, 1])

def distance_grad_py(double[:] u, double[:] v, int ax):
    return distance_grad(u[0], u[1], v[0], v[1], ax)

def distance_py(double[:] u, double[:] v):
    return distance(u[0], u[1], v[0], v[1],)


#######################################
# Exact
#######################################
cdef double exact_compute_gradient(float[:] timings,
                            double[:] val_P,
                            double[:, :] pos_reference,
                            np.int64_t[:] neighbors,
                            np.int64_t[:] indptr,
                            double[:, :] tot_force,
                            InfinityQuadTree iqt,
                            float theta,
                            int dof,
                            long start,
                            long stop,
                            bint compute_error,
                            int num_threads) nogil:
    # Having created the tree, calculate the gradient
    # in two components, the positive and negative forces
    cdef:
        long i, coord
        int ax
        long n_samples = pos_reference.shape[0]
        int n_dimensions = 2
        double sQ
        double error
        clock_t t1 = 0, t2 = 0

    cdef double* neg_f = <double*> malloc(sizeof(double) * n_samples * n_dimensions)
    cdef double* pos_f = <double*> malloc(sizeof(double) * n_samples * n_dimensions)

    if TAKE_TIMING:
        t1 = clock()
    sQ = exact_compute_gradient_negative(pos_reference, neighbors, indptr, neg_f, iqt, dof, theta, start,
                                   stop, num_threads)
    
    if TAKE_TIMING:
        t2 = clock()
        timings[2] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    if TAKE_TIMING:
        t1 = clock()

    error = compute_gradient_positive(val_P, pos_reference, neighbors, indptr,
                                      pos_f, n_dimensions, dof, sQ, start,
                                      10, compute_error, num_threads)

    if TAKE_TIMING:
        t2 = clock()
        timings[3] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    for i in prange(start, n_samples, nogil=True, num_threads=num_threads, schedule='static'):
        for ax in range(n_dimensions):
            coord = i * n_dimensions + ax
            tot_force[i, ax] = pos_f[coord] - (neg_f[coord] / sQ)

    free(neg_f)
    free(pos_f)
    return error

#######################################
# Exact Negative
#######################################
cdef double exact_compute_gradient_negative(double[:, :] pos_reference,
                                      np.int64_t[:] neighbors,
                                      np.int64_t[:] indptr,
                                      double* neg_f,
                                      InfinityQuadTree iqt,
                                      int dof,
                                      float theta,
                                      long start,
                                      long stop,
                                      int num_threads) nogil:
    cdef:
        int ax
        int n_dimensions = 2
        int offset = n_dimensions + 2
        long i, j, k, idx
        long n = stop - start
        long dta = 0
        long dtb = 0
        double size, dist2s, mult
        double qijZ, sum_Q = 0.0
        long n_samples = indptr.shape[0] - 1
        double dij, qij, dij_sq

    with nogil, parallel(num_threads=num_threads):
        for i in prange(start, n_samples, schedule='static'):
            # Init the gradient vector
            for ax in range(n_dimensions):
                neg_f[i * n_dimensions + ax] = 0.0

            for j in range(start, n_samples):
                if i == j:
                    continue
                dij = distance(pos_reference[i, 0], pos_reference[i, 1], pos_reference[j, 0], pos_reference[j, 1])
                dij_sq = dij * dij

                qij = 1. / (1. + dij_sq)

                if GRAD_FIX:
                    # New Fix
                    mult = qij * qij * dij
                else:
                    # Old Solution
                    mult = qij * qij

                sum_Q += qij
                for ax in range(n_dimensions):
                    neg_f[i * n_dimensions + ax] += mult * distance_grad(pos_reference[i, 0], pos_reference[i, 1], pos_reference[j, 0], pos_reference[j, 1], ax)

    # Put sum_Q to machine EPSILON to avoid divisions by 0
    sum_Q = max(sum_Q, FLOAT64_EPS)
    return sum_Q

#####################################################
# Grad
#####################################################
cdef double compute_gradient(float[:] timings,
                            double[:] val_P,
                            double[:, :] pos_reference,
                            np.int64_t[:] neighbors,
                            np.int64_t[:] indptr,
                            double[:, :] tot_force,
                            InfinityQuadTree iqt,
                            float theta,
                            int dof,
                            long start,
                            long stop,
                            bint compute_error,
                            int num_threads) nogil:
    # Having created the tree, calculate the gradient
    # in two components, the positive and negative forces
    cdef:
        long i, coord
        int ax
        long n_samples = pos_reference.shape[0]
        int n_dimensions = 2
        double sQ
        double error
        clock_t t1 = 0, t2 = 0

    cdef double* neg_f = <double*> malloc(sizeof(double) * n_samples * n_dimensions)
    cdef double* pos_f = <double*> malloc(sizeof(double) * n_samples * n_dimensions)

    if TAKE_TIMING:
        t1 = clock()
    # sQ = compute_gradient_negative(pos_reference, neighbors, indptr, neg_f, qt, dof, theta, start,
    #                                stop, num_threads)
    sQ = compute_gradient_negative(pos_reference, neg_f, iqt, dof, theta, start,
                                   stop, num_threads)
    if TAKE_TIMING:
        t2 = clock()
        timings[2] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    if TAKE_TIMING:
        t1 = clock()

    error = compute_gradient_positive(val_P, pos_reference, neighbors, indptr,
                                      pos_f, n_dimensions, dof, sQ, start,
                                      10, compute_error, num_threads)

    if TAKE_TIMING:
        t2 = clock()
        timings[3] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    for i in prange(start, n_samples, nogil=True, num_threads=num_threads, schedule='static'):
        for ax in range(n_dimensions):
            coord = i * n_dimensions + ax
            tot_force[i, ax] = pos_f[coord] - (neg_f[coord] / sQ)

    free(neg_f)
    free(pos_f)
    return error

cdef double compute_gradient_positive(double[:] val_P,
                                     double[:, :] pos_reference,
                                     np.int64_t[:] neighbors,
                                     np.int64_t[:] indptr,
                                     double* pos_f,
                                     int n_dimensions,
                                     int dof,
                                     double sum_Q,
                                     np.int64_t start,
                                     int verbose,
                                     bint compute_error,
                                     int num_threads) nogil:
    # Sum over the following expression for i not equal to j
    # grad_i = p_ij (1 + ||y_i - y_j||^2)^-1 (y_i - y_j)
    # This is equivalent to compute_edge_forces in the authors' code
    # It just goes over the nearest neighbors instead of all the data points
    # (unlike the non-nearest neighbors version of `compute_gradient_positive')
    cdef:
        int ax
        long i, j, k
        long n_samples = indptr.shape[0] - 1
        double C = 0.0
        double dij, qij, pij, mult, dij_sq

    with nogil, parallel(num_threads=num_threads):
        for i in prange(start, n_samples, schedule='static'):
            # Init the gradient vector
            for ax in range(n_dimensions):
                pos_f[i * n_dimensions + ax] = 0.0
            # Compute the positive interaction for the nearest neighbors
            for k in range(indptr[i], indptr[i+1]):
                j = neighbors[k]
                pij = val_P[k]

                dij = distance(pos_reference[i, 0], pos_reference[i, 1], pos_reference[j, 0], pos_reference[j, 1])
                dij_sq = dij * dij

                qij = 1. / (1. + dij_sq)

                if GRAD_FIX:
                    # New Fix
                    mult = pij * qij * dij
                else:
                    # Old solution
                    mult = pij * qij

                # only compute the error when needed
                if compute_error:
                    qij = qij / sum_Q
                    C += pij * log(max(pij, FLOAT32_TINY) / max(qij, FLOAT32_TINY))
                for ax in range(n_dimensions):
                    pos_f[i * n_dimensions + ax] += mult * distance_grad(pos_reference[i, 0], pos_reference[i, 1], pos_reference[j, 0], pos_reference[j, 1], ax)
    return C

cdef double compute_gradient_negative(double[:, :] pos_reference,
                                      double* neg_f,
                                      InfinityQuadTree iqt,
                                      int dof,
                                      float theta,
                                      long start,
                                      long stop,
                                      int num_threads) nogil:
    if stop == -1:
        stop = pos_reference.shape[0]
    cdef:
        int ax
        int n_dimensions = 2
        int offset = n_dimensions + 2
        long i, j, idx
        long n = stop - start
        long dta = 0
        long dtb = 0
        double size
        double dist2s, mult
        double qijZ, sum_Q = 0.0
        double* force
        double* summary
        double* pos
        double* neg_force
        int n_centers 
        int center_of_mass_idx
        CenterOfMass center_of_mass
        clock_t t1 = 0, t2 = 0, t3 = 0
        vector[CenterOfMass] results
    with nogil, parallel(num_threads=num_threads):
        # Define thread-local buffers
        summary = <double *> malloc(sizeof(double) * n * offset)
        force = <double *> malloc(sizeof(double) * n_dimensions)
        pos = <double *> malloc(sizeof(double) * n_dimensions)
        neg_force = <double *> malloc(sizeof(double) * n_dimensions)

        for i in prange(start, stop, schedule='static'):
            # Clear the arrays
            for ax in range(n_dimensions):
                force[ax] = 0.0
                neg_force[ax] = 0.0
                pos[ax] = pos_reference[i, ax]

            # Find which nodes are summarizing and collect their centers of mass
            # deltas, and sizes, into vectorized arrays
            t1 = clock()
            #idx = qt.summarize(pos, summary, theta*theta)

            idx = iqt.approximate_centers_of_mass(pos_reference[i, 0], pos_reference[i, 1], theta*theta, summary)

            t2 = clock()

            # Compute the t-SNE negative force
            # for the digits dataset, walking the tree
            # is about 10-15x more expensive than the
            # following for loop
            for j in range(idx // 4):
                dist2s = summary[j * offset + n_dimensions]
                size = summary[j * offset + n_dimensions + 1]
                qijZ = 1. / (1. + dist2s)  # 1/(1+dist)

                # if size > 1:
                #     printf("[QuadTree] Size: %g, %g\n", dist2s, dist2s * size / dist2s)

                sum_Q += size * qijZ   # size of the node * q

                if GRAD_FIX:
                    # New Fix
                    mult = size * qijZ * qijZ * sqrt(dist2s)
                else:
                    # Old Solution
                    mult = size * qijZ * qijZ

                for ax in range(n_dimensions):
                    neg_force[ax] += mult * summary[j * offset + ax]

                neg_force[0] += mult * distance_grad(pos_reference[i, 0], pos_reference[i, 1], summary[j * 4 + 0], summary[j * 4 + 1], 0)
                neg_force[1] += mult * distance_grad(pos_reference[i, 0], pos_reference[i, 1], summary[j * 4 + 0], summary[j * 4 + 1], 1)

            for ax in range(2):
                neg_f[i * n_dimensions + ax] = neg_force[ax]
            
            # t3 = clock()

            # dta += t2 - t1
            # dtb += t3 - t2

        free(force)
        free(pos)
        free(neg_force)
        free(summary)

        #printf("[t-SNE] Tree: %li clock ticks | ", dta)
        #printf("Force computation: %li clock ticks\n", dtb)
        printf("neg_f[0] %f\n", neg_f[0])
        printf("neg_f[1] %f\n", neg_f[1])

    # Put sum_Q to machine EPSILON to avoid divisions by 0
    sum_Q = max(sum_Q, FLOAT64_EPS)
    printf("END\n")
    return sum_Q

def gradient(float[:] timings,
             double[:] val_P,
             double[:, :] pos_output,
             np.int64_t[:] neighbors,
             np.int64_t[:] indptr,
             double[:, :] forces,
             float theta,
             int n_dimensions,
             int verbose,
             int dof=1,
             long skip_num_points=0,
             bint compute_error=1,
             int num_threads=1,
             bint exact=1,
             bint area_split=0,
             bint grad_fix=0):
    cdef double C
    cdef int n
    cdef vector[Point] iqt_initialization
    cdef InfinityQuadTree infinity_qt
    cdef clock_t t1 = 0, t2 = 0
    cdef n_samples = pos_output.shape[0]
    cdef a
    cdef b

    global AREA_SPLIT
    AREA_SPLIT = area_split

    global GRAD_FIX
    GRAD_FIX = grad_fix

    if not exact:
        if TAKE_TIMING:
            t1 = clock()

        #qt.build_tree(pos_output)
        #########################
        # Port points into a vector
        iqt_initialization = vector[Point](0)
        for i in range(n_samples):
            a = sqrt(pos_output[i, 0] * pos_output[i, 0] + pos_output[i, 1] * pos_output[i, 1])
            b = atan2(pos_output[i, 1], pos_output[i, 0])
            b = b if b > 0 else b + 2 * M_PI

            a = clamp(a, 0.0, 1 - BOUNDARY)
            iqt_initialization.push_back(Point(a * cos(b), a*sin(b)))

        printf("STARTED BUILDING\n")
        infinity_qt = InfinityQuadTree(iqt_initialization)
        printf("FINISHED BUILDING\n")

        if TAKE_TIMING:
            t2 = clock()
            timings[0] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    if TAKE_TIMING:
        t1 = clock()
    if exact:
        C = exact_compute_gradient(timings, val_P, pos_output, neighbors, indptr, forces,
                             infinity_qt, theta, dof, skip_num_points, -1, compute_error,
                             num_threads)
    else:
        C = compute_gradient(timings, val_P, pos_output, neighbors, indptr, forces,
                              infinity_qt, theta, dof, skip_num_points, -1, compute_error,
                             num_threads)
    if TAKE_TIMING:
        t2 = clock()
        timings[1] = ((float) (t2 - t1)) / CLOCKS_PER_SEC

    if not compute_error:
        C = np.nan
    return C

###################################
###################################

cdef extern from "point.hpp":
    cdef struct Point:
        DTYPE_t x
        DTYPE_t y

cdef extern from "centre_of_mass.hpp":
    cdef struct CenterOfMass:
        Point position
        SIZE_t number_of_accumulated_points
        double distance_to_target

cdef extern from "cell.hpp":
    cdef struct Cell:
        SIZE_t parent_idx

        vector[SIZE_t] children_idx
        bint is_leaf

        Point barycenter
        Point center

        Point min_bounds
        Point max_bounds

        SIZE_t cumulative_size
        SIZE_t depth
        DTYPE_t lorentz_factor

cdef extern from "infinity_quad_tree.hpp":
    cdef cppclass InfinityQuadTree:
        InfinityQuadTree() except +
        InfinityQuadTree(vector[Point] points) except +
        vector[Cell] get_nodes() nogil
        SIZE_t approximate_centers_of_mass(DTYPE_t x, DTYPE_t y, double theta_sq, DTYPE_t* combined_results) nogil except +
