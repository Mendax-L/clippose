import numpy as np

# Reference translation vector
t_x2 = np.array([-111.3412, -6.856851, 682.35516])
# t_x2 = np.array([-16.667253, 8.227341, 737.03766])

t_12 = np.array([-0.45437132, 0.06880743, 0.88815102])
# t_12 = np.array([ 0.07372925, -0.17202289, -0.98232995])

# Predicted rotation matrix
R_12 = np.array([
    [ 0.99000109, -0.12566024, -0.06408852],
    [ 0.12246689,  0.99113300, -0.05154835],
    [ 0.06999782,  0.04318420,  0.99661198]
])
# R_12 = np.array([[ 0.90844819, -0.37421362, -0.18624193],
#  [ 0.3924849,   0.91692956,  0.07208186],
#  [ 0.14379671, -0.13857978,  0.97985619]])
# Ground truth translation vector
t_x1_gt = np.array([-35.76937, 46.32161, 791.82837])
# t_x1_gt = np.array([-38.68748, 34.07101, 771.578])
# Step 1: Transform t_x2 using R_12 and t_12 to get predicted t_x1
t_x1_pred = t_x2 - R_12 @ t_12*10

print("Predicted t_x1:", t_x1_pred)

# Step 2: Compute the error between predicted t_x1 and ground truth t_x1_gt
error_vector = t_x1_pred - t_x1_gt
error_norm = np.linalg.norm(error_vector)

print("Error Vector:", error_vector)
print("Error Norm:", error_norm)

# Optional: Compute individual component errors
error_x = error_vector[0]
error_y = error_vector[1]
error_z = error_vector[2]

print(f"Error in x: {error_x}")
print(f"Error in y: {error_y}")
print(f"Error in z: {error_z}")
