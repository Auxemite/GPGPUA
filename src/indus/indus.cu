#ifndef FIX_GPU_CUH
#define FIX_GPU_CUH

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include "../image.hh"

// Fonction pour corriger une image sur le GPU
void fix_image_gpu(Image& image) {
    const int image_size = image.width * image.height;

    // Allocation de la mémoire GPU pour les données de l'image
    thrust::device_vector<int> d_buffer(image.buffer, image.buffer + image_size);

    // Étape 1 : Compactage des données pour enlever les valeurs -27
    constexpr int garbage_val = -27;
    thrust::device_vector<int> d_predicate(image_size);

    // Remplir le vecteur de prédicat (1 si différent de -27, 0 sinon)
    thrust::transform(
        d_buffer.begin(), 
        d_buffer.end(), 
        d_predicate.begin(), 
        [=] __device__ (int val) { return val != garbage_val ? 1 : 0; }
    );

    // Utiliser CUB pour calculer la somme exclusive sur le vecteur de prédicat
    thrust::device_vector<int> d_exclusive_scan(image_size);
    size_t temp_storage_bytes = 0;
    void* d_temp_storage = nullptr;

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, 
        temp_storage_bytes, 
        thrust::raw_pointer_cast(d_predicate.data()), 
        thrust::raw_pointer_cast(d_exclusive_scan.data()), 
        image_size
    );

    // Allouer l'espace de stockage temporaire
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Lancer à nouveau le scan exclusif avec le stockage temporaire
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, 
        temp_storage_bytes, 
        thrust::raw_pointer_cast(d_predicate.data()), 
        thrust::raw_pointer_cast(d_exclusive_scan.data()), 
        image_size
    );

    // Utiliser Thrust pour compacter les données en utilisant le prédicat
    thrust::device_vector<int> d_compacted_buffer(image_size);
    auto end_iter = thrust::copy_if(
        d_buffer.begin(), 
        d_buffer.end(), 
        d_predicate.begin(), 
        d_compacted_buffer.begin(), 
        thrust::identity<int>()
    );

    int new_size = end_iter - d_compacted_buffer.begin();

    // Libérer la mémoire temporaire
    cudaFree(d_temp_storage);

    // Étape 2 : Application de la transformation sur les pixels
    thrust::transform(
        d_compacted_buffer.begin(),
        d_compacted_buffer.begin() + new_size,
        thrust::make_counting_iterator(0),
        d_compacted_buffer.begin(),
        [=] __device__ (int pixel, int idx) {
            switch (idx % 4) {
                case 0: return pixel + 1;
                case 1: return pixel - 5;
                case 2: return pixel + 3;
                case 3: return pixel - 8;
            }
            return pixel; // Par défaut
        }
    );

    // Étape 3 : Histogramme et égalisation
    thrust::device_vector<int> d_histogram(256, 0);
    thrust::for_each(
        d_compacted_buffer.begin(),
        d_compacted_buffer.begin() + new_size,
        [=] __device__ (int pixel) {
            atomicAdd(thrust::raw_pointer_cast(d_histogram.data()) + pixel, 1);
        }
    );

    // Calculer le scan inclusif de l'histogramme pour l'égalisation
    thrust::inclusive_scan(
        d_histogram.begin(),
        d_histogram.end(),
        d_histogram.begin()
    );

    // Trouver le premier élément non nul dans le CDF
    int cdf_min = *thrust::find_if(
        d_histogram.begin(),
        d_histogram.end(),
        [] __device__ (int val) { return val > 0; }
    );

    // Appliquer l'égalisation
    thrust::transform(
        d_compacted_buffer.begin(),
        d_compacted_buffer.begin() + new_size,
        d_compacted_buffer.begin(),
        [=] __device__ (int pixel) {
            return roundf(((thrust::raw_pointer_cast(d_histogram.data())[pixel] - cdf_min) /
                           static_cast<float>(new_size - cdf_min)) * 255.0f);
        }
    );

    // Copier le buffer compacté corrigé vers l'image CPU
    thrust::copy(
        d_compacted_buffer.begin(),
        d_compacted_buffer.begin() + new_size,
        image.buffer
    );
}

#endif // FIX_GPU_CUH
