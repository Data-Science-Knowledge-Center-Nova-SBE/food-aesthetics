# import dependencies
from food_aesthetics.model import FoodAesthetics
import argparse, os, time
import pandas as pd
from pathlib import Path

# 1. create a path function 
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# 2. parametrize script 
parser = argparse.ArgumentParser()
parser.add_argument('path',
    help='Path to images folder',
    type=dir_path
)
args = parser.parse_args()
path_dir = Path(args.path)

# 3. init model 
fa = FoodAesthetics()
print('Scoring the pictures..')

# 4. score images
images = os.listdir(path_dir) # load images
images = [image for image in images if '.jpeg' in image] # exlude non-jpeg

out = []

for image in images:
    image_path = path_dir / image
    aes = fa.aesthetic_score(image_path)
    out.append(aes)

print('Scored all pictures.')

# 5. export
df = pd.DataFrame({
    "image":images,
    "aesthetic_score":out
})
timestr = time.strftime("%Y%m%d-%H%M%S")
df.to_csv(f'./output/aesthetic_scores-{timestr}.csv', index = False)
print('Exported the csv file correctly in the output/ folder.')
