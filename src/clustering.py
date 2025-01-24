import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import time
import plotly.graph_objs as go



def dbscan(points, eps=0.5, min_samples=200):
    print(f"Running DBSCAN with epsilon {eps} and min_samples {min_samples}.")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return labels

def monte_carlo_kde(points, bandwidth: float, sample_size: int = 500, num_samples: int = 10):
    print(f"Running Monte Carlo KDE with bandwidth {bandwidth}, sample size {sample_size}, and {num_samples} samples.")
    start_time = time.perf_counter()
    def process_sample():
        sample_indices = np.random.choice(len(points), sample_size, replace=False)
        sample_points = points[sample_indices]

        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(sample_points)
        
        log_density = kde.score_samples(points)
        return np.exp(log_density)

    densities = Parallel(n_jobs=-1)(delayed(process_sample)() for _ in range(num_samples))
    print(f"Monte Carlo KDE took {time.perf_counter() - start_time:.2f} seconds.")
    return np.mean(densities, axis=0)


def get_peaks(density, min_peak_points, sigma, out_path):
    print(f"Finding peaks with minimum {min_peak_points} points.")
    density = np.sort(density)
    density = density[int(0.3 * len(density)):]
    density_values, bin_edges = np.histogram(density, bins=100) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    smoothed_density = gaussian_filter1d(density_values, sigma=sigma)

    peaks, _ = find_peaks(smoothed_density, height=min_peak_points, distance=10)  # Adjust height to filter out small peaks

    peak_boundaries = []
    for peak in peaks:
        start, end = peak, peak

        while start > 0 and smoothed_density[start - 1] < smoothed_density[start]:
            start -= 1
        
        while end < len(smoothed_density) - 1 and smoothed_density[end + 1] < smoothed_density[end]:
            end += 1

        peak_boundaries.append((start, end))

    fig = go.Figure()

    # Original density trace
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=density_values,
        mode='lines',
        name="Original Density",
        line=dict(color='#1f77b4')  # Consistent blue
    ))

    # Smoothed density trace
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=smoothed_density,
        mode='lines',
        line=dict(color='#ff7f0e'),  # Consistent orange
        name="Smoothed Density"
    ))

    # Detected peaks
    fig.add_trace(go.Scatter(
        x=bin_centers[peaks],
        y=smoothed_density[peaks],
        mode='markers',
        marker=dict(color='#d62728', size=10, symbol='x'),  # Consistent red
        name="Detected Peaks"
    ))

    # Peak boundaries
    for idx, (start, end) in enumerate(peak_boundaries):
        fig.add_trace(go.Scatter(
            x=[bin_centers[start], bin_centers[start]],
            y=[0, max(density_values)],
            mode='lines',
            line=dict(color='green', dash='dash'),
            showlegend=idx == 0,
            name="Peak Separation" if idx == 0 else None
        ))

    # Layout for consistent style
    fig.update_layout(
        title="Density Histogram with Peak Detection",
        xaxis_title="Density",
        yaxis_title="Number of Points",
        plot_bgcolor='rgba(240, 240, 240, 1)',  # Light gray background
        paper_bgcolor='rgba(240, 240, 240, 1)',  # Match plot background
        legend=dict(
            x=1.02,  # Slightly outside the graph on the right
            y=1.0,  # Aligns with the top of the graph
            xanchor='left',  # Anchors the legend's left side at x=1.02
            yanchor='top',  # Anchors the legend's top side at y=1.0
            bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent legend background
            bordercolor='black',  # Border color
            borderwidth=1  # Border width
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12,
        )
    )

    fig.write_html(f"{out_path}/plots/density.html")
    return peak_boundaries, bin_centers


def get_densest_cluster(points, out_path, min_peak_points, kde_samples=1000, sigma=3):
    print(f"Calculating density for {len(points)} points...")
    density = monte_carlo_kde(points, bandwidth=1, sample_size=kde_samples)  
    peak_boundaries, bin_centers = get_peaks(density, min_peak_points, sigma, out_path)
    first_peak_end_index = peak_boundaries[1][1]
    first_peak_end = bin_centers[first_peak_end_index]
    print(f"First peak ends at density {first_peak_end}")
    points = points[density > first_peak_end]
    density = density[density > first_peak_end]
    return points, density

def remove_outliers(points, eps=None, min_samples=50):
    print("Removing outliers...")
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
