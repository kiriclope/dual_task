from . import constants as gv

def unit_vector(vector): 
    """ Returns the unit vector of the vector.  """ 
    u = vector / ( np.linalg.norm(vector) + gv.eps ) 
    return u 

def angle_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """ 
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.arccos( np.clip( np.dot(v1_u, v2_u), -1.0, 1.0) ) 

def cos_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """

    idx = np.unique(np.concatenate( ( np.where(v1!=0)[0], np.where(v2!=0)[0]) ) )
    
    v1 = v1[idx] 
    v2 = v2[idx] 
    
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.clip( np.dot(v1_u, v2_u), -1.0, 1.0)

def get_angle(coefs, v0):
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """    
    alpha=np.empty(coefs.shape[0]) 
    for i in np.arange(0, coefs.shape[0]): 
        alpha[i] = angle_between(v0, coefs[i])
    return alpha 

def get_cos(coefs, v0): 
    """ Returns the cosine of the angle alpha between vector coefs[0] (early delay) and coefs[1] (late delay) """ 
    cos_alp=[] 
    for j in np.arange(0, coefs.shape[0]):  
        cos_alp.append( cos_between(v0, coefs[j]) ) 
    return cos_alp 
