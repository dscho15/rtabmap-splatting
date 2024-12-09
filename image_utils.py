from PIL import Image
import io
import numpy as np

def __decode_depth_image(byte_sequence: str):
        
    image = Image.open(io.BytesIO(byte_sequence))
    
    depth_image = np.array(image)
    
    assert len(depth_image.shape) == 3, "Invalid image shape"
    assert depth_image.shape[-1] == 4, "Invalid image channels"
    
    h, w, c = depth_image.shape
    
    depth_image = depth_image.reshape((h * w, c))
    
    # decoding pattern (BGRA -> RGBA)
    depth_image = depth_image[:, [2, 1, 0, 3]]
    
    depth_image = depth_image.reshape(h, w, c)
    
    depth_bytes = depth_image.tobytes()
    
    image = np.frombuffer(depth_bytes, dtype=np.float32)
    
    return image.reshape(h, w)

def extract_depth_images(data: list) -> list:
    
    depth_images = [__decode_depth_image(d.depth) for d in data]
    
    depth_images = np.array(depth_images)
    
    return depth_images

def extract_images(data: list) -> list:
    
    images = [io.BytesIO(d.image) for d in data]
    
    images = [Image.open(image) for image in images]
    
    images = np.asarray(images)
    
    return images