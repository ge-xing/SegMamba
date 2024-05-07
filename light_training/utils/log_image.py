

import os 
from PIL import Image 


def log_image(save_dir, split, images,
                global_step, current_epoch):
    root = os.path.join(save_dir, "images", split)
    for k in images:

        filename = "{}_gs-{:06}_e-{:06}.png".format(
            k,
            global_step,
            current_epoch,
            )
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        
        Image.fromarray(images[k]).save(path)