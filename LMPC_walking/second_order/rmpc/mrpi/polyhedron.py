from numpy import array, hstack, vstack, reshape, ones, divide, dot, cos, sin
from cdd import Polyhedron as cddPolyhedron, Matrix as cddMatrix, RepType
from matplotlib.collections import PatchCollection
from cvxopt import matrix as cvxoptMatrix, solvers
from numpy import ndarray, all, empty, zeros
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from warnings import warn

class polyhedron:
    def __init__(self, _A=[], _b=[], _Aeq=[], _beq=[], _vertices=[], _rays=[]):
        if len(_A)==0 and len(_Aeq)==0:
            self.hasHrep = False
        else:
            self.hasHrep = True
        if len(_vertices)==0 and len(_rays)==0:
            self.hasVrep = False
        else:
            self.hasVrep = True
        if len(_A) == 0:
            self.A = []
        else:
            self.A =_A
            self.dim = _A.shape[1]
        if len(_b) == 0:
            self.b = []
        else:
            self.b =_b.reshape(_b.shape[0],1)
        if len(_Aeq) == 0:
            self.Aeq = []
        else:
            self.Aeq =_Aeq
            self.dim = _Aeq.shape[1]
        if len(_beq) == 0:
            self.beq = []
        else:
            self.beq =_beq.reshape(_beq.shape[0],1)
        if len(_vertices) == 0:
            self.vertices = []
        else:
            self.vertices = _vertices
            self.dim = _vertices.shape[1]
        if len(_rays) == 0:
            self.rays = []
        else:
            self.rays = _rays
            self.dim = _rays.shape[1]

    def construct_cdd_Hpolyhedron(self):
        if self.Aeq==[] and self.beq==[] :
            mat = cddMatrix(hstack([self.b, -self.A]), number_type='float')
        else:
            mat = cddMatrix(hstack([vstack([self.b, self.beq]),
                            vstack([-self.A, -self.Aeq])]), number_type='float')
        mat.rep_type = RepType.INEQUALITY
        cdd_Hpolyhedron = cddPolyhedron(mat)
        return cdd_Hpolyhedron

    def compute_Vrep(self):
        P = self.construct_cdd_Hpolyhedron()
        generators = P.get_generators()
        V = array(generators)
        vertices = []
        rays = []
        free_coordinates = []
        for i in range(V.shape[0]):
            if generators.lin_set and i in generators.lin_set:
                free_coordinates.append(list(V[i, 1:]).index(1.))
            elif V[i, 0] != 1:     # 1 = vertex, 0 = ray (check cdd python API)
                warn('Polyhedron is not a polytope !')
                rays.append(V[i, 1:])
            else:
                vertices.append(V[i, 1:])
        self.vertices = array(vertices)
        self.rays     = array(rays)
        self.hasVrep = True
        return

    def construct_cdd_Vpolyhedron(self):
        V = vstack(self.vertices)
        t = ones((V.shape[0], 1)) #1 for vertices (check cdd python API)
        tV = hstack([t, V])
        mat = cddMatrix(tV, number_type='float')
        mat.rep_type = RepType.GENERATOR
        cdd_Vpolyhedron = cddPolyhedron(mat)
        return cdd_Vpolyhedron

    def compute_Hrep(self):
        P = self.construct_cdd_Vpolyhedron()
        bA = array(P.get_inequalities())# the polyhedron is given by b+Ax>= 0
                                        # bA is [b A] (check cdd python API)
        if bA.shape == (0,):  # empty polyhedron bA == []
            return bA
        # transform to inequalities and equalities to canonical form Ax<=b
        b, A = array(bA[:, 0]), array(-bA[:, 1:])
        # get inequalities
        i = 0
        for b_i in b:
            if b_i == 0:
                break
            i = i + 1
        self.b, self.A   = array(bA[0:i, 0]), array(-bA[0:i, 1:])
        #for j in range(self.A.shape[0]):
        #    if self.A[j] < 0.0:
        #        self.b[j] = self.b[j]/-self.A[j]
        #        self.A[j] = -1.0
        #    else:
        #        self.b[j] = self.b[j]/self.A[j]
        #        self.A[j] = 1.0
        self.b = self.b.reshape(self.b.shape[0],1)
        # get equalities
        self.beq, self.Aeq = array(bA[i::, 0]), array(-bA[i::, 1:])
        self.hasHrep = True
        return

    def minVrep(self):
        if not self.hasVrep:
            self.compute_Vrep()
            self.hasVrep == True
        # in case of 1D vertices or you have only two points (edge)
        if self.vertices.shape[1]==1 or self.vertices.shape[0]==2:
            print('you need three 2D vertices to compute the convex hull')
            return
        else: #compute the convex hull
            hull = ConvexHull(self.vertices)
        self.vertices = self.vertices[hull.vertices, :]
        return

    ## TODO: unit test
    def projection(self, dim):
        if dim > self.dim:
            print('the dim of projection has to be less than the polyedron dim')
        if self.dim == dim:
            return self
        projected_vertices = []
        projected_rays = []
        if not self.hasVrep:
            self.compute_Vrep()
        for i in range(self.vertices.shape[0]):
            projected_vertices.append(self.vertices[i,0:dim])
        if len(self.rays) > 0:
            for i in range(self.rays.shape[0]):
                projected_rays.append(self.rays[i,0:dim])
        projected_polyhedron = polyhedron([],[],[],[],array(projected_vertices),
                                          array(projected_rays))
        return projected_polyhedron

    def affineMap(self, H):
        if not self.hasVrep:
            self.compute_Vrep()
        vertices, rays = [], []
        for i in range(self.vertices.shape[0]):
            vertices.append(dot(self.vertices[i], H.T))
        if len(self.rays)>0:
            for i in range(self.rays.shape[0]):
                rays.append(dot(self.rays[i], H.T))
        mapped_polyhedron = polyhedron([],[],[],[],array(vertices), array(rays))
        return mapped_polyhedron

    def scale(self, scale):
        scaled_vertices = []
        scaled_rays = []
        if not self.hasVrep:
            self.compute_Vrep()
        for i in range(self.vertices.shape[0]):
            scaled_vertices.append(scale*self.vertices[i])
        if len(self.rays) == 0:
            scaled_rays = []
        else:
            for i in range(self.rays.shape[0]):
                scaled_rays.append(scale*self.rays[i])
        scaled_polyhedron = polyhedron([],[],[],[], array(scaled_vertices),
                            array(scaled_rays))
        scaled_polyhedron.hasVrep = True
        return scaled_polyhedron

    def minkowskiSum(P1, P2):
        if not P1.hasVrep:
            P1.compute_Vrep()
        if not P2.hasVrep:
            P2.compute_Vrep()
        if len(P1.vertices) == 0 and len(P2.vertices) == 0:
            return polyhedron()
        # check if any of the sets are empty
        if len(P1.vertices) == 0:
            summed_polyhedron = P2
            return summed_polyhedron
        if len(P2.vertices) == 0:
            summed_polyhedron = P1
            return summed_polyhedron
        sums = []
        for a in P1.vertices:
            for b in P2.vertices:
                sums.append(tuple([sum(x) for x in zip(a,b)]))
        summed_vertices = list(dict.fromkeys(sums)) # list of tuples of
                                                    # length <=|A||B| points
        summed_polyhedron = polyhedron([],[],[],[],array(summed_vertices),[])
        summed_polyhedron.hasVrep = True
        return summed_polyhedron

    def contains(self, x, abs_tol=1e-8):
        if not isinstance(x, ndarray):
            x = array(x)
        Ax = self.A.dot(x)
        Ax = Ax.reshape(Ax.shape[0],1)
        test = Ax - self.b < abs_tol
        return all(test)

    def minkowskiDiff(P1, P2):
        diffs = []
        if not P1.hasVrep:
            P1.compute_Vrep()
        if not P2.hasVrep:
            P2.compute_Vrep()
        vertices_A = P1.vertices
        vertices_B = P2.vertices
        if len(vertices_A) == 0 and len(vertices_B) == 0:
            return polyhedron()
        if not P1.hasHrep:
            P1.compute_Hrep()
        for a in vertices_A:
            for b in vertices_B:
                temp_vertex = array(tuple([sum(x) for x in zip(a,b)]))
                print(temp_vertex)
                if P1.contains(temp_vertex):
                    diffs.append(temp_vertex)
        differenced_polyhedron = polyhedron([],[],[],[],array(diffs),[])
        differenced_polyhedron.hasVrep = True
        return differenced_polyhedron

    def compute_extreme(self,c, solver='glpk'):
        if not self.hasHrep:
            self.compute_Hrep()
        m   = self.dim
        c   = cvxoptMatrix(c.reshape((c.shape[0]),1))
        A   = cvxoptMatrix(self.A)
        b   = cvxoptMatrix(self.b)
        solvers.options['show_progress'] = False
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
        if self.Aeq==[] and self.beq==[]:
            sol = solvers.lp(-c, A, b, solver='glpk')
        else:
            Aeq = cvxoptMatrix(self.Aeq)
            beq = cvxoptMatrix(self.beq)
            sol = solvers.lp(-c, A, b, Aeq, beq, solver='glpk')
        supp = dot(c.T, sol['x'])
        return sol['x'], supp

    def plot_polygon(self, alpha=.4, color='g', linestyle='solid', fill=True,
            linewidth=None, title="outer approximation of mRPI set"):
        plt.rc('text', usetex = True)
        plt.rc('font', family ='serif')
        if not self.hasVrep:
            self.compute_Vrep()
        points = self.vertices
        ax = plt.gca()
        # 1D case -> append a zero to the projected dimension to plot
        if points.shape[1] == 1:
            t = zeros((points.shape[0], 1))  # first column is 1 for vertices
            points = hstack([points, t])
            plt.plot(points[:,0], points[:,1])
            ax.autoscale_view()
            ax.relim()
            ax.grid(color=(0,0,0), linestyle='--', linewidth=0.3)
            ax.set_title(title)
            plt.show()
            return
        # edge case (only two points)
        if points.shape[0] == 2:
            plt.plot(points[:,0], points[:,1])
            ax.autoscale_view()
            ax.relim()
            ax.grid(color=(0,0,0), linestyle='--', linewidth=0.3)
            ax.set_title(title)
            plt.show()
            return
        else:
            hull = ConvexHull(points)
            points = points[hull.vertices, :]
            patch = plt.Polygon(points, color=color, linestyle=linestyle,
                        fill=fill,linewidth=linewidth)
            ax.add_patch(patch)
            ax.autoscale_view()
            ax.relim()
            ax.grid(color=(0,0,0), linestyle='--', linewidth=0.3)
            ax.set_title(title)
            plt.xlabel('$c\,(m)$', fontsize='19')
            plt.ylabel('$\dot{c}\,(\frac{m}{s})$', fontsize='19')
            plt.show()
        return

def plot_polygon_list(list_of_polygons, color='green', linestyle='solid',
                    fill=True, linewidth=None,
                    title="outer-epsilon approximation of mRPI set"):
    ana_color=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,0,1),(0,0,1)]
    p_list = []
    x_all  = empty((0,2))
    figure, ax = plt.subplots()
    for polygon in list_of_polygons:
        x_all = vstack((x_all, polygon.vertices))
        patch = plt.Polygon(polygon.vertices, color=color, fill=fill,
                            linestyle=linestyle,linewidth=linewidth, zorder=-1)
        ax.add_patch(patch)
        ax.autoscale_view()
        ax.relim()
        ax.grid(color=(0,0,0), linestyle='--', linewidth=0.2)
        #ax.set_title(title)
        figure.canvas.draw()
        figure.show()
        figure.canvas.flush_events()
    plt.xlabel('c (m)', usetex=True, fontsize='19')
    plt.ylabel('$\dot{c}$ (m/s)', usetex=True, fontsize='19')
    #plt.legend()

    return
