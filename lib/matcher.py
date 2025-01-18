import sys
sys.path.append('/home/mendax/project/CLIPPose')
import matplotlib.pyplot as plt
import torch
from lightglue import LightGlue, SuperPoint, viz2d, SIFT
from lightglue.utils import load_image, match_pair, extract_features
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from PIL import Image

# Function to extract and match keypoints between two images
def extract_and_match(image_1, image_2, max_keypoints=2048, device=None):
    """
    Extracts and matches keypoints between two images using SuperPoint and LightGlue.

    Args:
    - image_1 (str): First image.
    - image_2 (str): Second image.
    - max_keypoints (int): Maximum number of keypoints to detect (default 2048).
    - device (torch.device): The device to run the computation on, either 'cuda' or 'cpu'.

    Returns:
    - matches: The matched keypoints between the two images.
    """
    torch.set_grad_enabled(False)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    # Load SuperPoint and LightGlue models
    extractor = SIFT(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="sift").eval().to(device)

    print(f"Using device: {device}")

    # Load images
    image0 = load_image(image_1)
    image1 = load_image(image_2)

    # Extract features + correspondences
    feats0, feats1, matches01 = match_pair(
        extractor, matcher, image0.to(device), image1.to(device), non_blocking=True
    )
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Visualize results
    viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
    plt.savefig("xxxx.png")

    return m_kpts0, m_kpts1, matches

def extract_and_match_parallel(test_image, refer_images, max_keypoints=2048, device=None, trust_threshold=0):
    """
    Extracts and matches keypoints between one image and a list of images using SuperPoint and LightGlue.
    Filters keypoints in `image_0` based on a trust threshold.

    Args:
    - test_image (str): The first image.
    - refer_images (list of str): List of second images to match with the first image.
    - max_keypoints (int): Maximum number of keypoints to detect (default 2048).
    - device (torch.device): The device to run the computation on, either 'cuda' or 'cpu'.
    - trust_threshold (float): Minimum proportion of matches required for a keypoint to be considered trusted.

    Returns:
    - results: A list of tuples containing filtered matched keypoints for each image in `refer_images`.
    - trusted_keypoints: Keypoints in `image_0` that are considered trusted.
    """
    torch.set_grad_enabled(False)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    # Load SuperPoint and LightGlue models
    extractor = SIFT(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="sift").eval().to(device)

    print(f"Using device: {device}")

    # Load the first image
    test_image = load_image(test_image).to(device)
    f0 = extract_features(extractor, test_image.to(device), non_blocking=True)

    # Initialize a counter for keypoint matches in image_0
    keypoint_counter = Counter()

    # Function to process a single image pair
    def process_single_pair(refer_image):
        refer_image = load_image(refer_image).to(device)
        f1 = extract_features(extractor, refer_image.to(device), non_blocking=True)
        feats0, feats1, matches01 = match_pair(matcher, f0, f1)
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        # Update the keypoint match counter
        keypoint_counter.update(matches[..., 0].tolist())

        # Save visualizations (optional, remove if not needed)
        # viz2d.plot_images([test_image, refer_image])
        # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
        # plt.savefig(f"x0x.png")

        return m_kpts0, m_kpts1, matches

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor() as executor:
        raw_results = list(executor.map(process_single_pair, refer_images))

    # Determine trusted keypoints
    total_refers = len(refer_images)
    # print(total_refers)
    trusted_indices = {k for k, v in keypoint_counter.items() if v / total_refers >= trust_threshold }
    # print(trusted_indices)
    # Filter matches to retain only trusted keypoints
    filtered_results = []
    for m_kpts0, m_kpts1, matches in raw_results:
        print(f"过滤前：{matches.shape}")   
        trusted_mask = [i for i, idx in enumerate(matches[..., 0]) if idx.item() in trusted_indices]
        # print(trusted_mask)

        filtered_m_kpts0 = m_kpts0[trusted_mask]
        filtered_m_kpts1 = m_kpts1[trusted_mask]
        filtered_matches = matches[trusted_mask]
        print(f"过滤后：{filtered_matches.shape}")
        filtered_results.append((filtered_m_kpts0, filtered_m_kpts1, filtered_matches))
    return filtered_results,raw_results

if __name__ == "__main__":
    test_image = Image.open("/home/mendax/project/CLIPPose/refers/obj_1/001_000200_000000.png")
    refer_images = [test_image, test_image, test_image]

    results = extract_and_match_parallel(test_image, refer_images)

    for idx, (m_kpts0, m_kpts1, matches) in enumerate(results):
        print(f"Matches for refer_{idx}: {len(matches)}")