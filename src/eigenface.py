import numpy as np
import os
import cv2

# Implementasi langkah 1: menyiapkan data — ambil semua path gambar dari folder dataset
def get_image_paths(dataset_path):
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Implementasi langkah 1: mengubah 1 gambar jadi vektor
def image_to_vector(img):
    image = cv2.imread(img)
    if image is None:
        raise ValueError(f"Gagal membaca gambar: {img}")
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = gray_image.flatten()
    return result

# Implementasi langkah 1: mengubah semua gambar jadi matriks vektor
def vector_to_matrix(image_paths):
    matrix = []
    print(f"Processing {len(image_paths)} images...")
    for i, path in enumerate(image_paths):
        try:
            vec = image_to_vector(path)
            matrix.append(vec)
            if (i + 1) % 10 == 0:  # Progress indicator
                print(f"Processed {i + 1}/{len(image_paths)} images")
        except Exception as e:
            print(f"Skip {path}: {e}")
    return np.array(matrix)

# Implementasi langkah 2: menghitung nilai tengah (mean)
def mean(matrix_image_vector):
    mean_vector = np.mean(matrix_image_vector, axis=0)
    return mean_vector.reshape(1, -1)

# Implementasi langkah 3: menghitung selisih antara tiap image vector dengan nilai mean
def selisih(mean_value, matrix_image_vector):
    matrix_selisih = matrix_image_vector - mean_value[0]
    return matrix_selisih

def covariance(matrix_selisih):
# Implementasi langkah 4: menghitung matriks kovarian
    n = matrix_selisih.shape[0]
    # Menggunakan dot product numpy untuk efisiensi
    matrix_covarian = np.dot(matrix_selisih, matrix_selisih.T) / n
    return matrix_covarian

# Implementasi QR Decomposition
def qr_decomposition_simple(A):
    # QR Decomposition sederhana menggunakan Gram-Schmidt
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        
        for i in range(j):
            R[i, j] = np.sum(Q[:, i] * A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        R[j, j] = np.sqrt(np.sum(v * v))
        
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = v
    
    return Q, R

# Implementasi eigenvalue dan eigenvector
def eig_qr_simple(matrix, max_iter=20):
    n = matrix.shape[0]
    A = matrix.copy()
    V = np.eye(n)
    
    for _ in range(max_iter):
        Q, R = qr_decomposition_simple(A)
        A = np.dot(R, Q)
        V = np.dot(V, Q)
    
    eigenvalues = np.diag(A)
    return eigenvalues, V

# Implementasi langkah 5 & 6: eigenspace
def eig(matrix_covarian):
    print("Computing eigenvalues and eigenvectors...")
    
    # Jika matrix terlalu besar, ambil sampel atau kurangi dimensi
    if matrix_covarian.shape[0] > 50:
        print("Matrix too large, using simplified approach...")
        # SVD sederhana
        eigenvalues, eigenvectors = eig_qr_simple(matrix_covarian[:20, :20], max_iter=10)
        full_eigenvectors = np.zeros((matrix_covarian.shape[0], 20))
        full_eigenvectors[:20, :] = eigenvectors
        return full_eigenvectors.T
    else:
        eigenvalues, eigenvectors = eig_qr_simple(matrix_covarian)
        return eigenvectors.T

# Implementasi proyeksi
def projection(matrix_image_vector, reduced_eig_vectors):

    print("Computing projections...")
    # Ambil hanya beberapa eigenvector pertama untuk efisiensi
    num_components = min(10, reduced_eig_vectors.shape[0])  # Maksimal 10 komponen
    selected_eigenvectors = reduced_eig_vectors[:num_components]
    
    projection_result = np.dot(matrix_image_vector.T, selected_eigenvectors.T)
    return projection_result.T

# Implementasi weight dataset
def weight_dataset(dataset_projection, matrix_selisih):
    print("Computing weights...")
    weight_data = np.dot(matrix_selisih, dataset_projection.T)
    return weight_data

# Euclidean Distance
def euclidean_distance_manual(vec1, vec2):
    diff = vec1 - vec2
    squared_sum = 0
    for val in diff:
        squared_sum += val * val
    return np.sqrt(squared_sum)

# Implementasi pengenalan wajah
def recognise_unknown_face(dataset_path, test_face_path, mean_dataset, projection_vectors, weight_data,
                           threshold_percentage=20, threshold_euclid=15000):
    
    print("Processing test image...")
    test_face = cv2.imread(test_face_path)
    test_face = cv2.resize(test_face, (64, 64), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(test_face, cv2.COLOR_BGR2GRAY)
    test_face_vector = gray_image.flatten()

    # Hitung selisih dengan mean
    vector_selisih_test_face = test_face_vector - mean_dataset

    # Hitung weight test face
    weight_test_face = np.dot(vector_selisih_test_face, projection_vectors.T)

    print("Computing distances...")
    # Hitung jarak euclidean dengan setiap wajah dalam dataset
    euclid_distances = []
    for i, weight_sample in enumerate(weight_data):
        distance = euclidean_distance_manual(weight_sample, weight_test_face)
        euclid_distances.append(distance)
    
    euclid_distances = np.array(euclid_distances)
    
    # Cari jarak minimum
    min_index = np.argmin(euclid_distances)
    min_distance = euclid_distances[min_index]
    max_distance = np.max(euclid_distances)
    
    # Hitung similarity
    if max_distance > 0:
        similarity = ((max_distance - min_distance) / max_distance) * 100
    else:
        similarity = 100
    
    # Alternatif perhitungan similarity berdasarkan threshold
    alt_similarity = max(0, 100 - (min_distance / 100))
    similarity = max(similarity, alt_similarity)
    
    image_files = get_image_paths(dataset_path)
    
    print(f"Min distance: {min_distance:.2f}")
    print(f"Max distance: {max_distance:.2f}")
    print(f"Similarity: {similarity:.2f}%")
    
    # Threshold logic
    if similarity >= threshold_percentage or min_distance <= threshold_euclid:
        return (image_files[min_index], similarity)
    else:
        return (None, similarity)