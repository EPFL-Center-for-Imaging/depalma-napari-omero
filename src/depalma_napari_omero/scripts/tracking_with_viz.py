import tifffile
import sys
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from depalma_napari_omero.tumor_tracking import run_tracking, recompute_labels

if __name__ == "__main__":
    timeseries = tifffile.imread(
        "/home/wittwer/Desktop/depalma_test/512/Animal-3/Animal-3_labels_2_scans.tif"
    )
    print(f"{timeseries.shape=}, {timeseries.dtype=}")

    linkage_df, grouped_df = run_tracking(timeseries)
    print(linkage_df)
    timeseries_corrected = recompute_labels(timeseries, linkage_df)

    # grouped_df.to_csv('/home/wittwer/Desktop/depalma_test/512/Animal-3/summary.csv')

    # # Save a figure
    # fig, ax = plt.subplots(dpi=120)
    # # ax.set_title(f'{specimen}')
    # sns.lineplot(data=grouped_df, x="scan", y="volume", hue="tumor", ax=ax)
    # fig_path = '/home/wittwer/Desktop/depalma_test/512/Animal-3/summary.png'
    # # plt.savefig(fig_path, transparent=False, facecolor='w')
    # # plt.close()
    # plt.show()

    import napari

    viewer = napari.view_labels(timeseries, ndisplay=3)
    viewer.add_labels(timeseries_corrected)
    tracks = linkage_df[["tumor", "scan", "z", "y", "x"]].values.astype(float)
    viewer.add_tracks(
        tracks,
        name="Tracks (Trackpy)",
        tail_length=len(timeseries),
        head_length=len(timeseries),
    )
    napari.run()
