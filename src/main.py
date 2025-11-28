import os
from pathlib import Path
import bioformats
import javabridge
import objectset_creation, Object
import FeatureSpace_Clustering
from Segmentation import run_segmentation


def main():
    # Start Java VM
    javabridge.start_vm(class_path=bioformats.JARS)

    # Check if interested_focus data exists
    my_file = Path(os.getcwd() + "\\Workspace_data/interesting_objects.dat")
    if my_file.exists():
        objects = objectset_creation.load_data("Workspace_data/interesting_objects.dat")
    else:
        # Check if data data exists
        my_file = Path(os.getcwd() + "\\Workspace_data/data.dat")
        if my_file.exists():
            list_of_objects = objectset_creation.load_data("Workspace_data/data.dat")
        else:
            # If neither interested_focus nor data data exists, read raw files
            list_of_objects = objectset_creation.read_raw_files()
            objectset_creation.save_data(list_of_objects, "Workspace_data/data.dat")

        # Get indices of interesting focus objects and save them if you don't have any then add all
        interesting_index = objectset_creation.interesting_objects()
        objects = [list_of_objects[d] for d in interesting_index]
        objectset_creation.save_data(objects, "Workspace_data/interesting_objects.dat")
    javabridge.kill_vm()



    objects, n_clusters = FeatureSpace_Clustering.kmeans_clustering(objects, isPlot=0, certainty=0.5, toSave=0)

    if not Path(os.getcwd() + "\\Workspace_data/clustered_objects.dat").exists():
        objectset_creation.save_data(objects, "Workspace_data/clustered_objects.dat")
        objectset_creation.save_data(n_clusters, "Workspace_data/best_n_clusters.dat")

    objects , settings = run_segmentation(objects, "Workspace_data/adjusted_objects.dat" , n_clusters = n_clusters, toSave = 1)


# Run the script if it's executed directly
if __name__ == '__main__':
    main()

