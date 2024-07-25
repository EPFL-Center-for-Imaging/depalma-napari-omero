import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
import trackpy as tp

def run_tracking(timeseries: np.ndarray) -> pd.DataFrame:
    """Simple linkage of objects between time frames based on object location and volume."""
    df = pd.DataFrame(columns=['z', 'y', 'x', 'frame'])

    for t, frame in enumerate(timeseries):
        # Skip if there are no objects in the frame
        if frame.sum() == 0:
            continue

        df_frame = pd.DataFrame(regionprops_table(frame, properties=['centroid', 'area', 'label']))
        df_frame['frame'] = t
        df_frame = df_frame.rename(columns={'centroid-0': 'z', 'centroid-1': 'y', 'centroid-2': 'x', 'area': 'volume'})
        df = pd.merge(df_frame, df, how='outer')

    # Trackpy linking
    linkage_df = tp.link(df, search_range=50, memory=0)
    linkage_df = linkage_df.rename(columns={'particle': 'tumor', 'frame': 'scan', 'label': 'label'})
    # By default, the first particle index is zero, but that makes it render as background, so we add 1:
    linkage_df['tumor'] = linkage_df['tumor'] + 1
    grouped_df = linkage_df.groupby(['tumor', 'scan']).mean()

    return linkage_df, grouped_df

def recompute_labels(timeseries, linkage_df):
    print(f"{len(np.unique(timeseries))=}")
    corrected_timeseries = np.zeros_like(timeseries)

    for _, row in linkage_df.iterrows():
        t, z, y, x, new_label, old_label = row['scan'], row['z'], row['y'], row['x'], row['tumor'], row['label']
        t = int(t)
        z = int(z)
        y = int(y)
        x = int(x)
        new_label = int(new_label)
        old_label = int(old_label)

        corrected_timeseries[t][timeseries[t] == old_label] = new_label

    print(f"{len(np.unique(corrected_timeseries))=}")

    return corrected_timeseries

if __name__=='__main__':
    import tifffile
    image = tifffile.imread('/home/wittwer/Desktop/depalma_test/sandra/registered_C41555_labels_timeseries.tif')
    linkage_df, grouped_df = run_tracking(image)
    import pdb; pdb.set_trace()
