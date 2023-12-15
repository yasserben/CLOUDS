from scipy.ndimage import label, center_of_mass
from scipy.ndimage import binary_erosion
from scipy.ndimage import label, sum as ndi_sum
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# from detectron2.structures import ImageList


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def mask_to_random_coordinates(binary_mask_tensor, num_samples=10):
    # Convert the binary mask tensor to a NumPy array
    binary_mask_np = binary_mask_tensor.cpu().detach().numpy()

    # Get the indices of the white pixels
    white_pixels = np.argwhere(binary_mask_np == 1)

    # Randomly select num_samples coordinates
    if white_pixels.size > 0:  # Check to avoid error if there are no white pixels
        selected_indices = np.random.choice(
            white_pixels.shape[0], num_samples, replace=False
        )
        white_pixels = white_pixels[selected_indices]

    # Create a NumPy array of ones with length num_samples
    ones_array = np.ones(num_samples)

    return white_pixels, ones_array


def mask_to_center_coordinates(binary_mask_tensor, num_samples=10):
    # Convert the binary mask tensor to a NumPy array
    binary_mask_np = binary_mask_tensor.cpu().detach().numpy()

    # Label connected components
    labeled, num_features = label(binary_mask_np)

    # If no features are found, return empty arrays
    if num_features == 0:
        return np.array([]), np.array([])

    # Calculate the area of each component
    areas = np.bincount(labeled.ravel())[1:]  # Ignore background

    # Get the label of the largest component
    largest_component_label = (
        np.argmax(areas) + 1
    )  # Add 1 as bincount starts counting from 0

    # Extract the coordinates of the largest component
    coordinates = np.argwhere(labeled == largest_component_label)

    # Calculate the center of the largest component
    center = center_of_mass(
        binary_mask_np, labels=labeled, index=largest_component_label
    )

    # Sort the coordinates by their distance to the center and select the closest ones
    sorted_coordinates = sorted(coordinates, key=lambda x: np.linalg.norm(x - center))[
        :num_samples
    ]

    # Create a NumPy array of ones with the number of selected coordinates
    ones_array = np.ones(len(sorted_coordinates))

    return np.array(sorted_coordinates), ones_array


def select_points_around_center(mask, num_points, radius=10):
    labeled, num_features = label(mask)
    if num_features == 0:
        return np.array([])

    areas = np.bincount(labeled.ravel())[1:]
    largest_component_label = np.argmax(areas) + 1
    coordinates = np.argwhere(labeled == largest_component_label)
    center = center_of_mass(mask, labels=labeled, index=largest_component_label)

    # Get coordinates within a certain radius around the center
    distances = np.linalg.norm(coordinates - center, axis=1)
    nearby_coordinates = coordinates[distances < radius]

    # If there are more nearby coordinates than required, select a subset
    if nearby_coordinates.shape[0] > num_points:
        indices = np.random.choice(
            nearby_coordinates.shape[0], num_points, replace=False
        )
        nearby_coordinates = nearby_coordinates[indices]
    else:
        nearby_coordinates = np.array([])

    return nearby_coordinates


def select_points_largest(mask, num_points):
    labeled, _ = label(mask)
    coordinates = np.argwhere(labeled == labeled.max())
    if coordinates.shape[0] > num_points:
        indices = np.random.choice(coordinates.shape[0], num_points, replace=False)
        coordinates = coordinates[indices]
    else:
        coordinates = np.array([])

    return coordinates


def separate_shapes_list(input_dicts, size_threshold):
    # Create a list to hold the output dictionaries
    separated_shapes_list = []

    # Process each input dictionary
    for input_dict in input_dicts:
        # Extract masks and labels from the current input dictionary
        multi_mask_np = input_dict["masks"].cpu().detach().numpy()
        labels_np = input_dict["labels"].cpu().detach().numpy()

        # Create a dictionary to hold separated binary masks for each original mask
        separated_shapes_dict = {}

        # Process each mask in the multi_mask_np array
        for i, (binary_mask_np, lbl) in enumerate(zip(multi_mask_np, labels_np)):
            # Label connected components
            labeled, num_features = label(binary_mask_np)

            # Calculate the area of each component
            areas = ndi_sum(binary_mask_np, labeled, range(num_features + 1))

            # Create a list to hold separate binary masks for each shape
            separate_masks = []

            for j in range(1, num_features + 1):
                # Check if the area of the current shape exceeds the size threshold
                if areas[j] > size_threshold:
                    # Create a binary mask for the current shape
                    shape_mask = (labeled == j).astype(int)
                    separate_masks.append(shape_mask)

            # Add the separate masks to the dictionary using the class ID as the key
            separated_shapes_dict[lbl] = np.array(separate_masks)

        # Add the current separated_shapes_dict to the output list
        separated_shapes_list.append(separated_shapes_dict)

    return separated_shapes_list


from scipy.ndimage import label, center_of_mass


def select_points_near_center_simple(mask, num_points):
    labeled, num_features = label(mask)
    if num_features == 0:
        return np.array([])

    areas = np.bincount(labeled.ravel())[1:]
    largest_component_label = np.argmax(areas) + 1
    coordinates = np.argwhere(labeled == largest_component_label)
    center = center_of_mass(mask, labels=labeled, index=largest_component_label)

    # Calculate distances from the center
    distances = np.linalg.norm(coordinates - center, axis=1)

    # Sort coordinates by distance
    sorted_indices = np.argsort(distances)

    # Select a certain number of points closest to the center
    selected_coordinates = coordinates[sorted_indices[:num_points]]

    return selected_coordinates


def get_fixed_points(
    separated_masks_list,
    num_points=20,
    erosion_size=2,
    apply_erosion=True,
    selection_mode="random",
):
    eroded_points_list = []
    structure_element = np.ones((erosion_size, erosion_size))

    for separated_masks_dict in separated_masks_list:
        eroded_points_dict = {}
        for label, masks in separated_masks_dict.items():
            eroded_points_masks_list = []
            for mask in masks:
                processed_mask = (
                    binary_erosion(mask, structure=structure_element)
                    if apply_erosion
                    else mask
                )
                coordinates = np.column_stack(np.where(processed_mask == 1))[:, ::-1]

                if num_points == 0:
                    raise ValueError("The number of points must be greater than zero.")

                if selection_mode == "random":
                    if coordinates.shape[0] >= num_points:
                        indices = np.random.choice(
                            coordinates.shape[0], num_points, replace=False
                        )
                        coordinates = coordinates[indices]
                    else:
                        coordinates = np.array([])
                elif selection_mode == "center":
                    if coordinates.shape[0] >= num_points:
                        coordinates = select_points_around_center(
                            processed_mask, num_points
                        )
                        if coordinates.shape[0] == num_points:
                            coordinates = coordinates[:, ::-1]
                    else:
                        coordinates = np.array([])
                elif selection_mode == "largest_component":
                    if coordinates.shape[0] >= num_points:
                        coordinates = select_points_largest(processed_mask, num_points)
                        if coordinates.shape[0] == num_points:
                            coordinates = coordinates[:, ::-1]
                    else:
                        coordinates = np.array([])
                else:
                    raise ValueError(
                        "Invalid selection_mode. Choose from 'random', 'center', or 'largest_component'."
                    )

                eroded_points_masks_list.append(
                    torch.tensor(coordinates.copy(), dtype=torch.float32)
                )

            eroded_points_dict[label] = eroded_points_masks_list

        eroded_points_list.append(eroded_points_dict)

    return eroded_points_list


def get_points_percentage(
    separated_masks_list,
    percentage=20,
    erosion_size=2,
    apply_erosion=True,
    selection_mode="random",
):
    eroded_points_list = []
    structure_element = np.ones((erosion_size, erosion_size))

    for separated_masks_dict in separated_masks_list:
        eroded_points_dict = {}
        for label, masks in separated_masks_dict.items():
            eroded_points_masks_list = []
            for mask in masks:
                processed_mask = (
                    binary_erosion(mask, structure=structure_element)
                    if apply_erosion
                    else mask
                )
                coordinates = np.column_stack(np.where(processed_mask == 1))[:, ::-1]
                num_points = int((percentage / 100) * coordinates.shape[0])

                # Threshold the number of points to a maximum of 100
                num_points = min(num_points, 100)

                if num_points == 0:
                    raise ValueError("The percentage is too low to select any points.")

                if selection_mode == "random":
                    if coordinates.shape[0] >= num_points:
                        indices = np.random.choice(
                            coordinates.shape[0], num_points, replace=False
                        )
                        coordinates = coordinates[indices]
                    else:
                        raise ValueError(
                            "Not enough points to select the specified percentage of coordinates."
                        )
                elif selection_mode == "center":
                    coordinates = select_points_around_center(
                        processed_mask, num_points
                    )
                elif selection_mode == "largest_component":
                    coordinates = select_points_largest(processed_mask, num_points)
                else:
                    raise ValueError(
                        "Invalid selection_mode. Choose from 'random', 'center', or 'largest_component'."
                    )

                eroded_points_masks_list.append(
                    torch.tensor(coordinates, dtype=torch.float32)
                )

            eroded_points_dict[label] = eroded_points_masks_list

        eroded_points_list.append(eroded_points_dict)

    return eroded_points_list


def visualize_points_on_mask(points, mask_shape, point_radius=3):
    """
    Visualize points on the mask.

    Parameters:
    - points: tensor of size (N, 20, 2) containing the points coordinates.
    - mask_shape: tuple containing the height and width of the masks.
    - point_radius: the radius of the points to be visualized on the mask.
    """
    # Convert points to numpy if it's a torch tensor
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    # Create a blank mask with zeros
    mask = np.zeros(mask_shape, dtype=np.uint8)

    # Iterate over all sets of points
    for point_set in points:
        for (x, y) in point_set:
            # Convert coordinates to integers
            x, y = int(x), int(y)

            # Draw circles around the points to make them visible
            for dx in range(-point_radius, point_radius + 1):
                for dy in range(-point_radius, point_radius + 1):
                    if dx * dx + dy * dy <= point_radius * point_radius:
                        new_x, new_y = x + dx, y + dy
                        if (
                            0 <= new_x < mask_shape[1] and 0 <= new_y < mask_shape[0]
                        ):  # Check boundaries
                            mask[new_y, new_x] = 1  # Set the pixel value to 1 (white)

    # Plot the mask with points
    plt.imshow(mask, cmap="gray")
    plt.show()


def save_points_on_mask(points, mask_shape, point_radius=3):
    """
    Visualize points on the mask.

    Parameters:
    - points: tensor of size (N, 20, 2) containing the points coordinates.
    - mask_shape: tuple containing the height and width of the masks.
    - point_radius: the radius of the points to be visualized on the mask.
    """
    # Convert points to numpy if it's a torch tensor
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    # Create a blank mask with zeros
    mask = np.zeros(mask_shape, dtype=np.uint8)

    # Iterate over all sets of points
    for point_set in points:
        for (x, y) in point_set:
            # Convert coordinates to integers
            x, y = int(x), int(y)

            # Draw circles around the points to make them visible
            for dx in range(-point_radius, point_radius + 1):
                for dy in range(-point_radius, point_radius + 1):
                    if dx * dx + dy * dy <= point_radius * point_radius:
                        new_x, new_y = x + dx, y + dy
                        if (
                            0 <= new_x < mask_shape[1] and 0 <= new_y < mask_shape[0]
                        ):  # Check boundaries
                            mask[new_y, new_x] = 1  # Set the pixel value to 1 (white)

    # Plot the mask with points
    image = Image.fromarray(mask)
    image.save("./points_on_mask.jpg")
    # plt.imshow(mask, cmap="gray")
    # plt.show()


def create_ones_tensor(input_tensor):
    # Get the size of the input tensor
    P, N, _ = input_tensor.shape

    # Create a new tensor of ones with size (P, N)
    ones_tensor = torch.ones((P, N), dtype=torch.float32)

    return ones_tensor


def dict_to_tensor(input_dict):
    non_empty_tensors = []
    counts_per_key = {}

    for key, values in input_dict.items():
        counts_per_key[key] = len(values)
        non_empty_values = [value for value in values if value.nelement() > 0]
        if non_empty_values:
            non_empty_tensors.extend(non_empty_values)

    if non_empty_tensors:
        non_empty_tensors = torch.stack(non_empty_tensors)
    else:
        non_empty_tensors = torch.empty((0, 20, 2))

    return non_empty_tensors, counts_per_key


def reconstruct_dict(tensor, counts_per_key):
    # Create a dictionary to hold the separated tensors for each original key
    reconstructed_dict = {}

    # Index to keep track of where to slice the tensor
    index = 0

    for key, count in counts_per_key.items():
        if count > 0:
            # Slice the tensor to get the separated tensors for this key
            separated_tensors = tensor[index : index + count]
            index += count  # Update the index for the next slice
        else:
            # If count is 0, create an empty tensor with the same other dimensions as the input tensor
            empty_shape = (0,) + tensor.shape[1:]
            separated_tensors = torch.empty(empty_shape, dtype=tensor.dtype)

        # Add the separated tensors to the dictionary
        reconstructed_dict[key] = separated_tensors

    return reconstructed_dict


def union_of_masks(masks_dict):
    # Initialize final_mask as None
    final_mask = None

    for key, masks in masks_dict.items():
        if masks.nelement() == 0:  # Skip empty tensors
            continue

        # If final_mask is still None, initialize it with 19s using the shape of the current mask
        if final_mask is None:
            device = masks.device  # Get the device of the masks
            final_mask = torch.full(masks[0].shape, 19, dtype=torch.int, device=device)

        # Iterate over individual masks of the current class
        for mask in masks:
            # If key is 0, then directly update the final_mask where the original mask is 1
            if key == 0:
                final_mask = torch.where(
                    mask == 1,
                    torch.tensor(key, dtype=torch.int, device=mask.device),
                    final_mask,
                )
            else:
                # Create a mask of class indices where the original mask is 1
                class_mask = torch.where(
                    mask == 1,
                    torch.tensor(key, dtype=torch.int, device=mask.device),
                    torch.tensor(0, dtype=torch.int, device=mask.device),
                )

                # Update the final mask with the class indices
                final_mask = torch.where(class_mask != 0, class_mask, final_mask)

    return final_mask


def separate_outputs_by_filename(input_data, model_output):
    # Initialize the output dictionaries
    output_gta = {
        "pred_logits": [],
        "pred_masks": [],
        "aux_outputs": [{} for _ in range(9)],
    }
    output_sd = {
        "pred_logits": [],
        "pred_masks": [],
        "aux_outputs": [{} for _ in range(9)],
    }

    for i, item in enumerate(input_data):
        filename = item.get("file_name", "")

        # Separate the model outputs based on the filename
        if "gta" in filename or "cityscapes" in filename or "synthia" in filename:
            output_gta["pred_logits"].append(model_output["pred_logits"][i])
            output_gta["pred_masks"].append(model_output["pred_masks"][i])
            for j in range(9):
                output_gta["aux_outputs"][j].setdefault("pred_logits", []).append(
                    model_output["aux_outputs"][j]["pred_logits"][i]
                )
                output_gta["aux_outputs"][j].setdefault("pred_masks", []).append(
                    model_output["aux_outputs"][j]["pred_masks"][i]
                )
        elif "sd" in filename:
            output_sd["pred_logits"].append(model_output["pred_logits"][i])
            output_sd["pred_masks"].append(model_output["pred_masks"][i])
            for j in range(9):
                output_sd["aux_outputs"][j].setdefault("pred_logits", []).append(
                    model_output["aux_outputs"][j]["pred_logits"][i]
                )
                output_sd["aux_outputs"][j].setdefault("pred_masks", []).append(
                    model_output["aux_outputs"][j]["pred_masks"][i]
                )

    # Convert lists to tensors
    output_gta["pred_logits"] = (
        torch.stack(output_gta["pred_logits"]) if output_gta["pred_logits"] else None
    )
    output_gta["pred_masks"] = (
        torch.stack(output_gta["pred_masks"]) if output_gta["pred_masks"] else None
    )
    output_sd["pred_logits"] = (
        torch.stack(output_sd["pred_logits"]) if output_sd["pred_logits"] else None
    )
    output_sd["pred_masks"] = (
        torch.stack(output_sd["pred_masks"]) if output_sd["pred_masks"] else None
    )

    for j in range(9):
        if output_gta["aux_outputs"][j].get("pred_logits"):
            output_gta["aux_outputs"][j]["pred_logits"] = torch.stack(
                output_gta["aux_outputs"][j]["pred_logits"]
            )
            output_gta["aux_outputs"][j]["pred_masks"] = torch.stack(
                output_gta["aux_outputs"][j]["pred_masks"]
            )
        else:
            output_gta["aux_outputs"][j] = None

        if output_sd["aux_outputs"][j].get("pred_logits"):
            output_sd["aux_outputs"][j]["pred_logits"] = torch.stack(
                output_sd["aux_outputs"][j]["pred_logits"]
            )
            output_sd["aux_outputs"][j]["pred_masks"] = torch.stack(
                output_sd["aux_outputs"][j]["pred_masks"]
            )
        else:
            output_sd["aux_outputs"][j] = None

    return output_gta, output_sd


def extract_outputs_by_index(model_output, indices):
    if isinstance(indices, int):
        indices = [indices]

    extracted_output = {
        "pred_logits": torch.stack([model_output["pred_logits"][i] for i in indices]),
        "pred_masks": torch.stack([model_output["pred_masks"][i] for i in indices]),
        "aux_outputs": [
            {
                "pred_logits": torch.stack(
                    [model_output["aux_outputs"][j]["pred_logits"][i] for i in indices]
                ),
                "pred_masks": torch.stack(
                    [model_output["aux_outputs"][j]["pred_masks"][i] for i in indices]
                ),
            }
            for j in range(9)
        ],
    }

    return extracted_output


def separate_dicts_by_filename(dict_list):
    sd_list = [d for d in dict_list if "generated" in str(d["image_id"]).lower()]
    order_list = [
        i for i, d in enumerate(dict_list) if "generated" in str(d["image_id"]).lower()
    ]
    return sd_list, order_list


def get_order_target(dict_list):
    order_list = [i for i, d in enumerate(dict_list) if "sd" in d["file_name"].lower()]
    return order_list


def transform_masks(input_dict):
    labels = []
    masks = []

    for label, mask_tensor in input_dict.items():
        if mask_tensor.size(0) > 0:  # Check if there are masks to merge
            merged_mask = torch.max(
                mask_tensor, dim=0
            ).values  # Merge masks belonging to the same class
            labels.append(label)
            masks.append(merged_mask)

    # Check if there are labels and masks to stack
    if labels and masks:
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        masks_tensor = torch.stack(masks, dim=0)
    else:
        labels_tensor = torch.empty((0,), dtype=torch.int64)
        masks_tensor = torch.empty((0, mask_tensor.size(2), mask_tensor.size(3)))

    output_dict = {"labels": labels_tensor.to("cuda"), "masks": masks_tensor.squeeze(1)}

    return output_dict


def select_best_masks(masks, mask_qualities):
    # Get the indices of the masks with the highest quality for each batch
    _, max_quality_indices = mask_qualities.max(dim=1)

    # Select the masks corresponding to the highest quality indices
    best_masks = masks[torch.arange(masks.size(0)), max_quality_indices]
    return best_masks


def include_negative_points(points):
    B, N, _ = points.shape
    negative_points = torch.zeros((B, B * N, 2))
    negative = torch.zeros((B * N))
    positive = torch.ones((B * N))
    point_labels = torch.ones((B, B * N))
    for i in range(B):
        negative_points[i] = torch.cat(
            (points[0:i], points[i : i + 1], points[i + 1 :])
        ).flatten(start_dim=0, end_dim=1)
        point_labels[i] = torch.cat(
            (negative[0:i], positive[i : i + 1], negative[i + 1 :])
        )
    return negative_points, point_labels


def remove_intersecting_pixels(masks):
    # Check where the sum along the first dimension (across all masks) is greater than 1
    intersections = masks.sum(dim=0) > 1

    # Set the intersecting pixels to 0 in all masks
    masks[:, intersections] = 0

    return masks


def merge_and_sum_dicts(dict1, dict2):
    if dict1 is None:
        return dict2 if dict2 is not None else {}
    elif dict2 is None:
        return dict1

    if set(dict1.keys()) != set(dict2.keys()):
        raise ValueError("Both dictionaries should have the same keys")

    summed_dict = {key: dict1[key] + dict2[key] for key in dict1}
    return summed_dict


def extract_rows_from_tensors(
    input_dict,
    indices,
    copy_keys=["text_classifier", "num_templates"],
):
    """
    Extract specific rows from each tensor in a dictionary based on provided indices.

    Parameters:
    - input_dict: dictionary containing tensors of shape (N,...)
    - indices: tensor containing the indices of the rows to be extracted

    Returns:
    - output_dict: dictionary with tensors of shape (M,...) where M is the length of indices
    """
    output_dict = {}
    for key, tensor in input_dict.items():
        if key in copy_keys:
            output_dict[key] = tensor
        else:
            output_dict[key] = tensor[indices]
    return output_dict


def get_elements_by_index(
    images_norm_list,
    order_target,
    size_divisibility,
):
    from detectron2.structures import ImageList

    images_norm_list_target = [images_norm_list[i] for i in order_target]
    images_norm_target = ImageList.from_tensors(
        images_norm_list_target, size_divisibility
    )
    # clip_vis_dense_target = clip_vis_dense[order_target]
    # outputs_target = extract_outputs_by_index(outputs, order_target)
    # return outputs_target, clip_vis_dense_target, images_norm_target
    return images_norm_target


def process_segmentation_maps(segmentation_maps):
    """
    Process a list of dictionaries each containing a segmentation map to extract labels and
    corresponding binary masks, excluding specified label values.

    Parameters:
    - segmentation_maps (list of dict): A list of dictionaries each containing a key 'sem_seg'
      with a value of a 2D tensor of shape (H, W) representing a segmentation map.
      H is the height, and W is the width of the map.

    Returns:
    - List of dictionaries: Each dictionary contains:
        - 'labels' (torch.Tensor): A tensor of unique label indices present in the segmentation
          map, excluding 19 and 255.
        - 'masks' (torch.Tensor): A tensor containing binary masks associated with each unique
          label index, excluding the masks for labels 19 and 255. Each mask is of shape (H, W).
    """
    results = []

    for item in segmentation_maps:
        seg_map = item["sem_seg"]  # Directly use the 2D tensor
        labels = torch.unique(seg_map, sorted=True)
        labels = labels[labels != 19]
        labels = labels[labels != 255]

        masks = torch.stack([(seg_map == label).float() for label in labels])

        results.append({"labels": labels, "masks": masks})

    return results
