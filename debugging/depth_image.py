import database
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

db = database.RTABSQliteDatabase("databases/241211-32334 PM.db")

db_depth_imgs = db.extract_depth_images()
db_depth_img = db_depth_imgs[0]

pil_depth_img = np.asarray(Image.open("/home/dts/Desktop/rtabmap-splatting/1.png"))
pil_depth_img = np.array(pil_depth_img) * 1.0 / 1000

bounds_real = (pil_depth_img.max(), pil_depth_img.min())
bounds_fake = (db_depth_img.max(), db_depth_img.min())

out = (db_depth_img - pil_depth_img).mean()

plt.imshow(db_depth_img - pil_depth_img)
plt.colorbar()
plt.show()