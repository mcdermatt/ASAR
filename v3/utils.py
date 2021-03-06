import numpy as np
import tensorflow as tf
from vedo import *
import vtk

def get_cluster(rads, thresh = 0.3, mnp = 25): #mnp = 50, thresh = 0.2
    """ Identifies radial bounds which contain the first cluster in a spike 
            that is closest to the ego-vehicle 
        
        rads = tensor containing radii of points in each spike
        thresh = must be this close to nearest neighbor to be considered part of a cluster
        mnp = minimum number of points a cluster must contain to be considered
            """

    #TODO: try dymacally lowering <max_buffer> value as algorithm progresses

    max_buffer = 0.2 #0.5
    # max_buffer = 0.1 #0.5


    #notes for spherical ICET MC 
    #  0.1 -> mountain
    #  0.2 -> shadows (need slightly longer basin of attraction to grab on to wall)

    #fix dimensions
    if len(tf.shape(rads)) < 2:
        rads = rads[:,None]

    #replace all zeros in rads (result of converting ragged -> standard tensor) with some arbitrarily large value
    mask = tf.cast(tf.math.equal(rads, 0), tf.float32)*1000
    rads = rads + mask
    # print(rads)

    #sort in ascending order for each column in tensor
    top_k = tf.math.top_k(tf.transpose(rads), k = tf.shape(rads)[0])
    # print("\n top_k \n", top_k[1])
    rads = tf.transpose(tf.gather(tf.transpose(rads), top_k[1], batch_dims = 1))
    rads = tf.reverse(rads, axis = tf.constant([0]))
    # print("rads \n", rads)


    # calculate the forward difference between neighboring points
    z = tf.zeros([1, tf.shape(rads)[1].numpy()])
    shifted = tf.concat((rads[1:], z), axis = 0)
    diff = shifted - rads
    # diff = tf.math.abs(rads - shifted) #debug 6/9/22
    # print("\n diff \n", diff)

    # #find where difference jumps
    jumps = tf.where(diff > thresh)
    # print("\n jumps \n", jumps) #[idx of jump, which spike is jumping]

    #find where the first large cluster occurs in each spike
    #   using numpy here because we're not working with the full dataset and 
    #   it's easier if we use in place operations
    bounds = np.zeros([tf.shape(rads)[1].numpy(), 2])
    for i in range(tf.shape(rads)[1].numpy()):

        #get the indices of jumps for the ith spike
        jumps_i = tf.gather(jumps, tf.where(jumps[:,1] == i))[:,0].numpy()
        jumps_i = np.append(np.zeros([1,2], dtype = np.int32), jumps_i, axis = 0)#need to add zeros to the beginning
    
        # print("jumps_i", i, " \n", jumps_i)  

        last = 0
        count = 1
        while True:

            #degbug
            # print(tf.shape(jumps_i))
            if tf.shape(jumps_i)[0] < 2:
                bounds[i,:] = 0
                break

            #check and see if this jump contains a sufficient number of points
            if jumps_i[count ,0] - last > mnp:
                # print(last, count)

                #set bounds at edges of cluster
                # bounds[i, 0] = rads[jumps_i[count - 1, 0] + 1, i]
                # bounds[i, 1] = rads[jumps_i[count, 0], i] 

                #extend cluster bounds halfway to next point
                buffer_dist1 = (rads[jumps_i[count - 1, 0] , i] - rads[jumps_i[count - 1, 0] - 1, i]) / 2
                # print("b1", buffer_dist1)
                if abs(buffer_dist1) > max_buffer:
                    buffer_dist1 = max_buffer
                bounds[i, 0] =  rads[jumps_i[count - 1, 0] + 1, i] - buffer_dist1

                buffer_dist2 = (rads[jumps_i[count, 0] + 1, i] - rads[jumps_i[count, 0], i]) / 2
                # print("b2", buffer_dist2)
                if buffer_dist2 > max_buffer:
                    buffer_dist2 = max_buffer
                bounds[i, 1] =  rads[jumps_i[count, 0], i] + buffer_dist2
                
                break 

            last = jumps_i[count, 0]
            count += 1

            #if no useful clusters appear
            if count == tf.shape(jumps_i)[0]:
                bounds[i, :] = 0
                break

    bounds = tf.convert_to_tensor(bounds)

    return(bounds)


def R2Euler(mat):
    """determines euler angles from euler rotation matrix"""

    if len( tf.shape(mat) ) == 2:
        mat = mat[None, :, :]

    R_sum = np.sqrt(( mat[:,0,0]**2 + mat[:,0,1]**2 + mat[:,1,2]**2 + mat[:,2,2]**2 ) / 2)

    phi = np.arctan2(-mat[:,1,2],mat[:,2,2])
    theta = np.arctan2(mat[:,0,2], R_sum)
    psi = np.arctan2(-mat[:,0,1], mat[:,0,0])

    angs = np.array([phi, theta, psi])
    return angs

def R_tf(angs):
    """generates rotation matrix using euler angles
    angs = tf.constant(phi, theta, psi) (aka rot about (x,y,z))
            can be single set of angles or batch for multiple cells
    """

    if len(tf.shape(angs)) == 1:
        angs = angs[None,:]

    phi = angs[:,0]
    theta = angs[:,1]
    psi = angs[:,2]

    mat = tf.Variable([[cos(theta)*cos(psi), sin(psi)*cos(phi) + sin(phi)*sin(theta)*cos(psi), sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi)],
                       [-sin(psi)*cos(theta), cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(psi) + sin(theta)*sin(psi)*cos(phi)],
                       [sin(theta), -sin(phi)*cos(theta), cos(phi)*cos(theta)]
                        ])

    mat = tf.transpose(mat, [2, 0, 1])
    mat = tf.squeeze(mat)
    return mat

def jacobian_tf(p_point, angs):
    """calculates jacobian for point using TensorFlow
        angs = tf.constant[phi, theta, psi] aka (x,y,z)"""

    phi = angs[0]
    theta = angs[1]
    psi = angs[2]

    #correct method using tf.tile
    eyes = tf.tile(-tf.eye(3), [tf.shape(p_point)[1] , 1])

    # (deriv of R() wrt phi).dot(p_point)
    #   NOTE: any time sin/cos operator is used, output will be 1x1 instead of constant (not good)
    Jx = tf.tensordot(tf.Variable([[tf.constant(0.), (-sin(psi)*sin(phi) + cos(phi)*sin(theta)*cos(psi)), (cos(phi)*sin(psi) + sin(theta)*sin(phi)*cos(psi))],
                                   [tf.constant(0.), (-sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)), (cos(phi)*cos(psi) - sin(theta)*sin(psi)*sin(phi))], 
                                   [tf.constant(0.), (-cos(phi)*cos(theta)), (-sin(phi)*cos(theta))] ]), p_point, axes = 1)

    # (deriv of R() wrt theta).dot(p_point)
    Jy = tf.tensordot(tf.Variable([[(-sin(theta)*cos(psi)), (cos(theta)*sin(phi)*cos(psi)), (-cos(theta)*cos(phi)*cos(psi))],
                                   [(sin(psi)*sin(theta)), (-cos(theta)*sin(phi)*sin(psi)), (cos(theta)*sin(psi)*cos(phi))],
                                   [(cos(theta)), (sin(phi)*sin(theta)), (-sin(theta)*cos(phi))] ]), p_point, axes = 1)

    Jz = tf.tensordot(tf.Variable([[(-cos(theta)*sin(psi)), (cos(psi)*cos(phi) - sin(phi)*sin(theta)*sin(psi)), (cos(psi)*sin(phi) + sin(theta)*cos(phi)*sin(psi)) ],
                                       [(-cos(psi)*cos(theta)), (-sin(psi)*cos(phi) - sin(phi)*sin(theta)*cos(psi)), (-sin(phi)*sin(psi) + sin(theta)*cos(psi)*cos(phi))],
                                       [tf.constant(0.),tf.constant(0.),tf.constant(0.)]]), p_point, axes = 1)

    Jx_reshape = tf.reshape(tf.transpose(Jx), shape = (tf.shape(Jx)[0]*tf.shape(Jx)[1],1))
    Jy_reshape = tf.reshape(tf.transpose(Jy), shape = (tf.shape(Jy)[0]*tf.shape(Jy)[1],1))
    Jz_reshape = tf.reshape(tf.transpose(Jz), shape = (tf.shape(Jz)[0]*tf.shape(Jz)[1],1))

    J = tf.concat([eyes, Jx_reshape, Jy_reshape, Jz_reshape], axis = 1) #was this

    return J

class Ell(Mesh):
    """
    Build a 3D ellipsoid centered at position `pos`.

    |projectsphere|

    |pca| |pca.py|_
    """
    def __init__(self, pos=(0, 0, 0), axis1= 1, axis2 = 2, axis3 = 3, angs = np.array([0,0,0]),
                 c="cyan4", alpha=1, res=24):

        self.center = pos
        self.va_error = 0
        self.vb_error = 0
        self.vc_error = 0
        self.axis1 = axis1
        self.axis2 = axis2
        self.axis3 = axis3
        self.nr_of_points = 1 # used by pcaEllipsoid

        if utils.isSequence(res):
            res_t, res_phi = res
        else:
            res_t, res_phi = 2*res, res

        elliSource = vtk.vtkSphereSource()
        elliSource.SetThetaResolution(res_t)
        elliSource.SetPhiResolution(res_phi)
        elliSource.Update()
        l1 = axis1
        l2 = axis2
        l3 = axis3
        self.va = l1
        self.vb = l2
        self.vc = l3
        axis1 = 1
        axis2 = 1
        axis3 = 1
        angle = angs[0] #np.arcsin(np.dot(axis1, axis2))
        theta = angs[1] #np.arccos(axis3[2])
        phi =  angs[2] #np.arctan2(axis3[1], axis3[0])

        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Scale(l1, l2, l3)

        #needed theta and angle to be negative before messing with E_xz, E_yz...
        t.RotateZ(np.rad2deg(phi))
        t.RotateY(-np.rad2deg(theta)) #flipped sign here 5/19
        t.RotateX(-np.rad2deg(angle)) #flipped sign here 5/19
        
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(elliSource.GetOutput())
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()
        self.transformation = t

        Mesh.__init__(self, pd, c, alpha)
        self.phong()
        self.GetProperty().BackfaceCullingOn()
        self.SetPosition(pos)
        self.Length = -np.array(axis1) / 2 + pos
        self.top = np.array(axis1) / 2 + pos
        self.name = "Ell"