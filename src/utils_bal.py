from utils import *
from utils_augm import *




def compute_image_copy_allocation(class_object_df, R, first_upper, target_total_objects, verbose=True):
    """
    Compute image copy allocation for data balancing augmentation.
    
    Args:
        class_object_df: DataFrame with class object counts per image
        R: Number of bins
        first_upper: Upper limit of first bin
        target_total_objects: Target number of objects per class
        verbose: Whether to print allocation details
    
    Returns:
        tuple: (image_copy_allocation DataFrame, image_copy_allocation_by_class_and_bin dict)
    """
    # Collect bin_stats for all classes
    class_bin_stats_dict = {}
    for class_id in class_object_df.columns:
        counts = class_object_df[class_id].tolist()
        image_names = class_object_df.index.tolist()
        bins = make_bins_for_class(counts, R, first_upper)
        if not bins:
            continue
        bin_stats = bin_stats_for_class(counts, bins, image_names)
        allocated, _, _, _ = allocate_copies_to_target(bin_stats, target_total_objects, method='proportional_objects')
        class_bin_stats_dict[class_id] = allocated

    bin_image_allocation_df = get_bin_image_allocation_df(class_bin_stats_dict)
    
    if verbose:
        print("\nBin image allocation important informations for augmentation:")
        for class_id in class_bin_stats_dict.keys():
            print(f"\n⚡ CLASS {class_id} START ⚡")
            class_rows = bin_image_allocation_df[bin_image_allocation_df['class_id'] == class_id]
            for _, row in class_rows.iterrows():
                print(f"Bin {row['range']}📊: copies_to_add={row['copies_to_add']}✅, images={row['image_names']}")

    # Distribute copies_to_add uniformly among image_names
    allocation_records = []
    image_copy_allocation_by_class_and_bin = {}

    for class_id in class_bin_stats_dict.keys():
        class_rows = bin_image_allocation_df[bin_image_allocation_df['class_id'] == class_id]
        allocation_by_bin = {}
        for _, row in class_rows.iterrows():
            bin_range = row['range']
            bin_df = pd.DataFrame([row])
            current_bin_allocation = distribute_uniform_from_df(bin_df)
            
            allocation_by_bin[bin_range] = current_bin_allocation
            
            for img, copies in current_bin_allocation.items():
                allocation_records.append({
                    'image_name': img,
                    'class_name': class_id,
                    'num_copies': copies
                })

        image_copy_allocation_by_class_and_bin[class_id] = allocation_by_bin

    image_copy_allocation = pd.DataFrame(allocation_records)

    if verbose:
        print("\nFinal image copy allocation for augmentation (separated by class AND bin):")
        for class_id, bins in image_copy_allocation_by_class_and_bin.items():
            print(f"\nClass {class_id}📊:")
            for bin_range, allocation in bins.items():
                print(f"  Bin {bin_range}:")
                for img, copies in allocation.items():
                    print(f"    - {img}: copies_to_add={copies}✅")

    return image_copy_allocation, image_copy_allocation_by_class_and_bin