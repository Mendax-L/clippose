import numpy as np

# Quaternion class for easier manipulation
class Quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def to_numpy(self):
        return np.array([self.x, self.y, self.z, self.w])

    @staticmethod
    def from_numpy(arr):
        return Quaternion(arr[0], arr[1], arr[2], arr[3])

# Average Quaternion function
def average_quaternion(cumulative, new_rotation, first_rotation, add_amount):
    """
    Averages multiple quaternions, taking care of quaternion inversion when necessary.
    
    :param cumulative: A numpy array storing cumulative quaternion values.
    :param new_rotation: The new quaternion to add to the average.
    :param first_rotation: The first quaternion of the array to be averaged.
    :param add_amount: The current count of quaternions being averaged.
    :return: The normalized average quaternion.
    """
    # Before we add the new rotation to the average, we check if the quaternion needs to be inverted.
    # This is because q and -q represent the same rotation, but we need to ensure all quaternions have consistent signs.
    if not are_quaternions_close(new_rotation, first_rotation):
        new_rotation = inverse_sign_quaternion(new_rotation)
    
    # Average the values
    add_det = 1.0 / float(add_amount)
    
    cumulative[0] += new_rotation[0]  # w
    cumulative[1] += new_rotation[1]  # x
    cumulative[2] += new_rotation[2]  # y
    cumulative[3] += new_rotation[3]  # z
    
    w = cumulative[0] * add_det
    x = cumulative[1] * add_det
    y = cumulative[2] * add_det
    z = cumulative[3] * add_det
    
    return normalize_quaternion(x, y, z, w)

# Normalize quaternion to unit length
def normalize_quaternion(x, y, z, w):
    """
    Normalize the quaternion (x, y, z, w) to unit length.
    """
    length_d = 1.0 / (w**2 + x**2 + y**2 + z**2)
    w *= length_d
    x *= length_d
    y *= length_d
    z *= length_d
    
    return Quaternion(x, y, z, w)

# Inverse the sign of quaternion components
def inverse_sign_quaternion(q):
    """
    Change the sign of all quaternion components.
    This is not the same as the inverse quaternion.
    """
    return Quaternion(-q.x, -q.y, -q.z, -q.w)

# Check if two quaternions are close to each other (e.g., one quaternion is the inverse of the other)
def are_quaternions_close(q1, q2):
    """
    Check if two quaternions are close to each other. This can be used to check if q1 and q2
    represent the same rotation but with the signs flipped.
    """
    dot = quaternion_dot(q1, q2)
    
    return dot >= 0.0

# Dot product of two quaternions
def quaternion_dot(q1, q2):
    """
    Compute the dot product of two quaternions.
    """
    return q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w

# Example usage:
if __name__ == "__main__":
    # Example quaternions (in numpy array form)
    first_rotation = Quaternion(1, 0, 0, 0)  # Example quaternion
    new_rotation = Quaternion(0.999, 0, 0, 0.044)  # Another example quaternion
    
    # Cumulative values (we'll store w, x, y, z in a numpy array)
    cumulative = np.zeros(4)  # [w, x, y, z]
    
    add_amount = 1  # This would be updated as more quaternions are added
    
    # Adding the first quaternion
    average = average_quaternion(cumulative, new_rotation.to_numpy(), first_rotation.to_numpy(), add_amount)
    print(f"Average Quaternion: {average.x}, {average.y}, {average.z}, {average.w}")
