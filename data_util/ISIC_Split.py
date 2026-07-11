import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

# CSV file
df = pd.read_csv('/data_util/isic/class_id.csv')

# Image root directory
base_dir = '/data/ISIC/ISIC/ISIC2018_Task1-2_Training_Input/'

# Class mapping
class_dict = {
    'seborrheic_keratosis': '1',
    'nevus': '2',
    'melanoma': '3'
}

# Create subdirectories if they do not exist
for d in ['1', '2', '3']:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)

# Use a progress bar and skip existing images
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    src_path = os.path.join(base_dir, row['ID'] + '.jpg')
    dest_path = os.path.join(base_dir, class_dict[row['Class']], row['ID'] + '.jpg')

    if os.path.exists(dest_path):
        continue  # Already processed, skip

    try:
        img = Image.open(src_path)
        img_resized = img.resize((512, 512))
        img_resized.save(dest_path)
    except Exception as e:
        # print(f"Skip {row['ID']}: {e}")
        continue