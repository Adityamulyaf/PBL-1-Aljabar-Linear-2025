import eigenface

def run(image_path, sample_path, threshold_percent=40, threshold_euclid=5000):
    # Ambil semua path gambar dari folder dataset
    image_paths = eigenface.get_image_paths(image_path)

    # Ubah semua gambar jadi matriks vektor
    dataset_mat = eigenface.vector_to_matrix(image_paths)

    # Proses eigenface
    dataset_mean = eigenface.mean(dataset_mat)
    normalised_dataset = eigenface.selisih(dataset_mean, dataset_mat)
    cov_dataset = eigenface.covariance(normalised_dataset)
    mat_eig_vec = eigenface.eig(cov_dataset)
    dataset_projection_mat = eigenface.projection(dataset_mat, mat_eig_vec)
    weight_dataset = eigenface.weight_dataset(dataset_projection_mat, normalised_dataset)

    # Pengenalan wajah test image
    result_path, match_percentage = eigenface.recognise_unknown_face(
        image_path,
        sample_path,
        dataset_mean[0],
        dataset_projection_mat,
        weight_dataset,
        threshold_percentage=threshold_percent,
        threshold_euclid=threshold_euclid
    )

    return result_path, match_percentage
