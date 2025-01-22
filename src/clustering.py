import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def dbscan(points, eps=0.5, min_samples=200):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return labels

def monte_carlo_kde(points, bandwidth: float, sample_size: int = 500, num_samples: int = 10):
    densities = []
    for _ in range(num_samples):
        sample_indices = np.random.choice(len(points), sample_size, replace=False)
        sample_points = points[sample_indices]

        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(sample_points)
        
        log_density = kde.score_samples(points)
        densities.append(np.exp(log_density))

    # Aggregate results (e.g., average density estimates)
    density = np.mean(densities, axis=0)
    return density


def get_peaks(density, min_peak_points, sigma, plot=False):
    density = np.sort(density)
    density = density[int(0.1 * len(density)):]
    density_values, bin_edges = np.histogram(density, bins=100) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    smoothed_density = gaussian_filter1d(density_values, sigma=sigma)

    peaks, _ = find_peaks(smoothed_density, height=min_peak_points)  # Adjust height to filter out small peaks

    peak_boundaries = []
    for peak in peaks:
        start, end = peak, peak

        while start > 0 and smoothed_density[start - 1] < smoothed_density[start]:
            start -= 1
        
        while end < len(smoothed_density) - 1 and smoothed_density[end + 1] < smoothed_density[end]:
            end += 1

        peak_boundaries.append((start, end))
    return peak_boundaries, bin_centers


def get_densest_cluster(points, min_peak_points, kde_samples=1000, sigma=2.5, colors=None, plot=False):
    # density = monte_carlo_kde(points, bandwidth=1, sample_size=max(len(points) // 100, 3000))  
    density = monte_carlo_kde(points, bandwidth=1, sample_size=kde_samples)  
    peak_boundaries, bin_centers = get_peaks(density, min_peak_points, sigma=sigma, plot=True)
    first_peak_end_index = peak_boundaries[0][0]
    first_peak_end = bin_centers[first_peak_end_index]
    print(f"First peak ends at density {first_peak_end}")
    points = points[density > first_peak_end]
    density = density[density > first_peak_end]
    return points, density

def remove_outliers(points, eps=None, min_samples=50):
    if eps is None:
        density = monte_carlo_kde(points, bandwidth=1.0)
        mean_density = np.mean(density)
        median_density = np.median(density)
        eps = (1 / median_density) ** (1 / 3)
        print(f"Mean density: {mean_density:.2f}")
        print(f"Median density: {median_density:.2f}")
        print(f"Estimated epsilon: {eps:.2f}")
        print(f"Estimated epsilon using mean density: {(1 / mean_density) ** (1 / 3):.2f}")

    dbscan_labels = dbscan(points, eps=eps, min_samples=min_samples)
    unique_labels = np.unique(dbscan_labels)
    print(f"Number of clusters: {len(unique_labels)}")
    # return cluster with the most points. dbscan_labels has -1 for outliers
    cluster_sizes = np.bincount(dbscan_labels[dbscan_labels != -1])
    largest_cluster_label = np.argmax(cluster_sizes)
    largest_cluster_points = points[dbscan_labels == largest_cluster_label]
    return largest_cluster_points
