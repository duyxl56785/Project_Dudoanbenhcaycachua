import numpy as np

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def preprocess_image(image_array, target_size=(224, 224)):
    resized = np.resize(image_array, (target_size[0], target_size[1], 3))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def test_class_labels():
    assert len(CLASSES) == 10
    assert "Tomato___healthy" in CLASSES
    assert "Tomato___Late_blight" in CLASSES

def test_image_preprocessing():
    dummy_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    processed = preprocess_image(dummy_image)
    
    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0
