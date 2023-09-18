from nilearn import datasets
from nilearn.plotting import find_parcellation_cut_coords


def get_coordinates(labels_in):
    # Load the AAL atlas
    atlas = datasets.fetch_atlas_aal(version="SPM12")

    # Get the labels of the regions
    labels = atlas["labels"]

    region_indices = [labels.index(region.replace("\r", "")) for region in labels_in]

    coords = find_parcellation_cut_coords(labels_img=atlas.maps)
    region_coords = [coords[i] for i in region_indices]

    return region_coords
