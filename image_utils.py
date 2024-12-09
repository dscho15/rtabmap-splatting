from PIL import Image
import io
import numpy as np

def load_image(image_path):
    
    return Image.open(image_path)


def decode_depth_image(byte_sequence: str):
        
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