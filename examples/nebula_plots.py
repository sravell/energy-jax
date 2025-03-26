"""
    Visualization function + helpers for a nebula plot.
    This is a 2D projection of the data with the score field overlaid.
"""
import io
import math
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.decomposition import PCA
import jax
from jax import numpy as jnp
import equinox as eqx
from PIL import Image


def get_nebula_score_field_plot_as_fig(X, y, score_fn, label_subset, label_names):
    """_summary_

    Args:
        X (np.array): input data.  must be flat bc PCA.  takes shape (num_examples, data_dim)
        y (np.array): labels.  shape (num_examples,).  Not one-hot.
        score_fn (Callable): a function that takes an array of shape (K, data_dim)
            and returns an array of shape (K, data_dim) representing the score field.  K is usually the mesh size.
        label_subset (List[str]): the labels that are actually represented.  I like subsetting to debug.
        label_names (List[str]): all label names.  This corresponds to the indices in y.

    Returns:
        a matplotlib figure object
    """
    plt.style.use("dark_background")
    fig, lattice_in_projection_coords, projection_matrix = _get_nebula_base_elements(
        X, y, label_subset, label_names, mesh_size=20, in_color=True
    )
    ax = fig.axes[0]

    add_score_field_to_nebula(
        fig, ax, lattice_in_projection_coords, projection_matrix, score_fn
    )
    fig.suptitle(
        "PCA projection of data and modeled score field", fontsize=18, fontweight="bold"
    )
    return fig


def get_nebula_energy_landscape_plot_as_fig(
    X, y, negative_phase_samples, ebm, label_subset, label_names
):
    """Makes a plot with:
    - the data scatter
    - the energy landscape
    - the negative phase samples

    Args:
        X (np.array): input data.  must be flat bc PCA.  takes shape (num_examples, data_dim)
        y (np.array): labels.  shape (num_examples,).  Not one-hot.
        negative_phase_samples (np.array): samples from model.  shape (num_samples, data_dim)
        ebm (Callable): a function that takes an array of shape (K, data_dim)
            and returns an array of shape (K, energy) representing the energies.  K is usually the mesh size.
        label_subset (List[str]): the labels that are actually represented.  I like subsetting to debug.
        label_names (List[str]): all label names.  This corresponds to the indices in y.

    Returns:
        a matplotlib figure
    """
    plt.style.use("dark_background")
    fig, lattice_in_projection_coords, projection_matrix = _get_nebula_base_elements(
        X, y, label_subset, label_names, mesh_size=500, in_color=False
    )
    ax = fig.axes[0]

    add_energy_heatmap_to_nebula(
        fig, ax, lattice_in_projection_coords, projection_matrix, ebm
    )
    if negative_phase_samples is not None:
        negative_phase_samples_in_projection_coords = (
            negative_phase_samples @ projection_matrix.T
        )
        ax.scatter(
            negative_phase_samples_in_projection_coords[:, 0],
            negative_phase_samples_in_projection_coords[:, 1],
            marker="o",
            facecolors="none",
            edgecolors="black",
            alpha=1.0,
        )
    fig.suptitle(
        "PCA projection of data and modeled energy landscape",
        fontsize=18,
        fontweight="bold",
    )
    return fig


def add_energy_heatmap_to_nebula(
    fig, ax, lattice_in_projection_coords, projection_matrix, ebm
):
    inverse_projection_matrix = jnp.linalg.pinv(projection_matrix)
    lattice_in_data_coords = inverse_projection_matrix @ lattice_in_projection_coords.T
    energies = ebm(lattice_in_data_coords.T)

    # Plot
    lattice_x = lattice_in_projection_coords[:, 0]
    lattice_y = lattice_in_projection_coords[:, 1]

    canyon_cp = sns.color_palette("Spectral", as_cmap=True)

    ax.imshow(
        energies.reshape(500, 500),
        extent=[lattice_x.min(), lattice_x.max(), lattice_y.min(), lattice_y.max()],
        origin="lower",
        aspect="auto",
        cmap=canyon_cp,
    )
    fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=canyon_cp,
            norm=plt.Normalize(vmin=energies.min(), vmax=energies.max()),
        ),
        label="Magnitude",
        orientation="vertical",
        pad=0.01,
    )


def _get_nebula_base_elements(
    X, y, label_subset, label_names, mesh_size, in_color=True
):
    """Makes a plot with the data scatter only

    Args:
        X (np.array): input data.  must be flat bc PCA.  takes shape (num_examples, data_dim)
        y (np.array): labels.  shape (num_examples,).  Not one-hot.
        label_subset (List[str]): the labels that are actually represented.  I like subsetting to debug.
        label_names (List[str]): all label names.  This corresponds to the indices in y.

    Returns:
        a matplotlib figure object
    """
    # generate noise and black images
    plt.style.use("dark_background")
    noise_images = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(len(X) // len(label_subset), *X.shape[1:])
    )
    # black_images = jnp.ones_like(noise_images)
    # augmented_label_names = label_names + ["Noise", "All-black"]
    augmented_label_names = label_names + ["Noise"]

    # get the PCA projection
    projection_matrix = get_projection_matrix(
        # X, y, [noise_images, black_images], label_subset, label_names
        X,
        y,
        [noise_images],
        label_subset,
        label_names,
    )

    # Plot the 2D projection scatter plot
    data_in_projection_coords = X @ projection_matrix.T

    # convert synthetic images
    # black_images_in_projection_coords = black_images @ projection_matrix.T
    noise_images_in_projection_coords = noise_images @ projection_matrix.T

    # make mesh
    (
        max_x,
        min_x,
        max_y,
        min_y,
        x_margin,
        y_margin,
        lattice_in_projection_coords,
    ) = get_mesh_properties(
        [
            data_in_projection_coords,
            noise_images_in_projection_coords,
            # black_images_in_projection_coords,
        ],
        mesh_size=mesh_size,
    )

    # set up plot
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("black")
    scatter_cp = sns.husl_palette(len(augmented_label_names), l=0.5)

    def get_scatter_properties_of_label(label):
        if label in label_subset:
            marker = "o"
        elif label == "Noise":
            marker = "+"
        else:
            marker = "x"

        if in_color:
            return {
                "color": scatter_cp[augmented_label_names.index(label)],
                "alpha": 2 * get_alpha(len(X)),
                "marker": marker,
            }
        else:
            return {
                "color": "white",
                "alpha": get_alpha(len(X)),
                "facecolors": "none",
                "edgecolors": "white",
                "marker": marker,
            }

    # plot the scatters
    for class_name in label_subset:
        cluster_projection = data_in_projection_coords[
            y == label_names.index(class_name)
        ]
        add_single_cluster_to_nebula(
            ax,
            cluster_projection,
            class_name,
            **get_scatter_properties_of_label(class_name)
        )
    add_single_cluster_to_nebula(
        ax,
        noise_images_in_projection_coords,
        "Noise",
        **get_scatter_properties_of_label("Noise")
    )
    # add_single_cluster_to_nebula(
    #     ax,
    #     black_images_in_projection_coords,
    #     "All-black",
    #     **get_scatter_properties_of_label("All-black")
    # )
    ax.axis(
        xmin=min_x - x_margin,
        xmax=max_x + x_margin,
        ymin=min_y - y_margin,
        ymax=max_y + y_margin,
    )
    # fig.tight_layout()
    return fig, lattice_in_projection_coords, projection_matrix


def fig_to_buf(fig):
    buf = io.BytesIO()  # Create a buffer
    fig.savefig(buf, format="png")  # Save the figure to the buffer in PNG format
    buf.seek(0)  # Rewind the buffer to the start
    plt.close(fig)  # Close the figure to free memory
    return buf


def img_arr_to_buf(arr):
    fig, ax = plt.subplots()  # Create a new figure
    ax.imshow(arr, cmap="gray")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)  # Close the figure
    return buf


def get_mesh_properties(list_of_data_arrays, mesh_size):
    all_data_in_projection_coords = jnp.vstack(list_of_data_arrays)
    max_x = np.max(all_data_in_projection_coords[:, 0])
    min_x = np.min(all_data_in_projection_coords[:, 0])
    max_y = np.max(all_data_in_projection_coords[:, 1])
    min_y = np.min(all_data_in_projection_coords[:, 1])
    x_margin = (max_x - min_x) * 0.05
    y_margin = (max_y - min_y) * 0.05
    grid_x, grid_y = jnp.meshgrid(
        jnp.linspace(min_x - x_margin, max_x + x_margin, mesh_size),
        jnp.linspace(min_y - y_margin, max_y + y_margin, mesh_size),
    )
    lattice_in_projection_coords = jnp.vstack([grid_x.ravel(), grid_y.ravel()]).T
    return max_x, min_x, max_y, min_y, x_margin, y_margin, lattice_in_projection_coords


def add_single_cluster_to_nebula(ax, data, cluster_name, color, **kwargs):
    # data scatter
    ax.scatter(data[:, 0], data[:, 1], color=color, **kwargs)
    projected_centroid = jnp.mean(data, axis=0)
    ax.text(
        projected_centroid[0],
        projected_centroid[1],
        cluster_name,
        color=color,
        fontsize=16,
        fontweight="bold",
    )


def add_score_field_to_nebula(
    fig, ax, lattice_in_projection_coords, projection_matrix, score_fn
):
    inverse_projection_matrix = jnp.linalg.pinv(projection_matrix)
    lattice_in_data_coords = inverse_projection_matrix @ lattice_in_projection_coords.T
    scores = score_fn(lattice_in_data_coords.T)
    scores_in_projection_coords = scores @ projection_matrix.T

    normalized_scores, magnitudes = normalize(scores_in_projection_coords)

    # quiver_cp = get_starry_redblue_palette()
    quiver_cp = sns.color_palette("Spectral", as_cmap=True)

    ax.quiver(
        lattice_in_projection_coords[:, 0],
        lattice_in_projection_coords[:, 1],
        normalized_scores[:, 0],
        normalized_scores[:, 1],
        color=quiver_cp(magnitudes),  # Map the magnitudes to a color
        scale=None,
        scale_units="xy",
        angles="xy",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=quiver_cp,
            norm=plt.Normalize(vmin=magnitudes.min(), vmax=magnitudes.max()),
        ),
        label="Magnitude",
        orientation="vertical",
        pad=0.01,
    )


def darken(c):
    """darken a mpl color"""
    return sns.dark_palette(c, n_colors=3)[1]


def lighten(c):
    """lighten a mpl color"""
    return sns.light_palette(c, n_colors=3)[1]


def get_projection_matrix(X, y, synthetic_Xs, label_subset, label_names):
    """get the two biggest principal components of PCA as projection dimensions"""
    input_matrix_to_PCA = X[
        jnp.isin(y, jnp.array([label_names.index(l) for l in label_subset]))
    ]
    input_matrix_to_PCA = jnp.vstack([input_matrix_to_PCA] + synthetic_Xs)
    pca = PCA(n_components=2)
    pca.fit(input_matrix_to_PCA)
    return pca.components_


def get_starry_redblue_palette():
    """color palette for score field.  redshift=small, blueshift=large"""
    mid_lum_red = colorsys.hls_to_rgb(0, 0.7, 1)
    high_lum_blue = colorsys.hls_to_rgb(2 / 3, 0.9, 1)
    return mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", [mid_lum_red, high_lum_blue]
    )


def normalize(vectors):
    """returns the normalized vectors and their norms"""
    norms = np.linalg.norm(vectors, axis=1)
    return (vectors / norms[:, np.newaxis], np.linalg.norm(vectors, axis=1))


def buf_to_image(buf):
    """converts the io output to a displayable image for the ipynb"""
    image = Image.open(buf)

    if image.mode == "RGBA":
        image = image.convert("RGB")

    return jnp.array(image)


def one_hot_to_int(one_hot):
    """converts one-hot labels to integers"""
    return jnp.argmax(one_hot, axis=-1)


def get_score_fn(ebm):
    """gets score from the ebm and makes it parallelizable over an array axis"""
    return eqx.filter_vmap(eqx.filter_grad(lambda x: -ebm(x)))


def get_energy_fn(ebm):
    return eqx.filter_vmap(lambda x: ebm(x))


def get_score_fn_in_2d(ebm):
    """takes flattened data, converts to 2D, calculates score, and re-flattens"""
    score_fn = get_score_fn(ebm)

    def score_fn_in_2d(x):
        return score_fn(x.reshape(x.shape[0], 28, 28)).reshape(x.shape[0], -1)

    return score_fn_in_2d


def get_energy_fn_in_2d(ebm):
    """takes flattened data, converts to 2D, calculates score, and re-flattens"""
    energy_fn = eqx.filter_vmap(lambda x: jnp.sum(ebm(x)))

    def energy_fn_in_2d(x):
        return energy_fn(x.reshape(x.shape[0], 28, 28)).reshape(x.shape[0], -1)

    return energy_fn_in_2d


def flatten_data(data):
    """flattens images"""
    return data.reshape(data.shape[0], -1)


def unflatten_data(data, shape):
    """unflattens images"""
    return data.reshape(data.shape[0], *shape)


def get_alpha(num_points):
    """solving for f(x) = a*e^(bx)
    where f(30000) = 0.1, f(3000) = 0.3
    we want points clouds with fewer points to still be visible across runs

    Args:
        num_points (_type_): _description_
    """
    a = 0.34
    b = -4e-5

    return a * math.exp(b * num_points)
