import database
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

db = database.RTABSQliteDatabase("databases/241211-32334 PM.db")

db.camera.K