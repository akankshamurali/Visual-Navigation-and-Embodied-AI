import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.preprocessing import normalize
from joblib import dump
from pathlib import Path
import time
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class AdvancedFeatureExtractor:
    def __init__(self, model_type='efficientnet', input_shape=(224, 224), use_gpu=True):
        """
        Advanced feature extractor with multiple model support.
        
        Args:
            model_type (str): Feature extraction model type
            input_shape (tuple): Input image dimensions
            use_gpu (bool): Enable GPU acceleration
        """
        # Enable mixed precision for improved performance
        set_global_policy('mixed_float16')
        
        self.input_shape = input_shape
        self.model_type = model_type
        self.use_gpu = use_gpu
        
        # Model and preprocessing selection
        self._setup_model()
    
    def _setup_model(self):
        """Initialize feature extraction model based on selected type."""
        try:
            if self.model_type == 'efficientnet':
                base_model = EfficientNetB0(
                    weights='imagenet', 
                    include_top=False, 
                    pooling='avg',
                    input_shape=self.input_shape + (3,)
                )
                self.preprocess_fn = efficientnet_preprocess
            elif self.model_type == 'resnet':
                base_model = ResNet50V2(
                    weights='imagenet', 
                    include_top=False, 
                    pooling='avg',
                    input_shape=self.input_shape + (3,)
                )
                self.preprocess_fn = resnet_preprocess
            elif self.model_type == 'vgg':
                base_model = VGG16(
                    weights='imagenet', 
                    include_top=False, 
                    pooling='avg',
                    input_shape=self.input_shape + (3,)
                )
                self.preprocess_fn = vgg_preprocess
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Freeze base model layers to use as feature extractor
            for layer in base_model.layers:
                layer.trainable = False
            
            self.feature_extractor = Model(
                inputs=base_model.input, 
                outputs=base_model.output
            )
            
            # GPU detection and logging
            if self.use_gpu and tf.config.list_physical_devices('GPU'):
                logging.info(f"Using GPU with {self.model_type.upper()} model")
        except Exception as e:
            logging.error(f"Model initialization error: {e}")
            raise
    
    def _robust_image_loader(self, path):
        """
        Robust image loading with multiple preprocessing steps.
        
        Args:
            path (str): Image file path
        
        Returns:
            Preprocessed image tensor or None
        """
        try:
            # Read image with error handling
            image = tf.io.read_file(path)
            
            # Attempt multiple decoding strategies
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            
            # Resize and preprocess
            image = tf.image.resize(image, self.input_shape)
            image = self.preprocess_fn(image)
            
            return image
        except Exception as e:
            logging.warning(f"Could not process image {path}: {e}")
            return None
    
    def extract_features(self, image_paths, batch_size=32):
        """
        Extract deep features with advanced processing and tracking.
        
        Args:
            image_paths (list): Paths to images
            batch_size (int): Processing batch size
        
        Returns:
            Normalized feature matrix
        """
        # Create TensorFlow dataset for efficient processing
        path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        image_ds = path_ds.map(
            self._robust_image_loader, 
            num_parallel_calls=tf.data.AUTOTUNE
        ).filter(lambda x: x is not None)
        
        # Batch and prefetch for performance
        image_ds = image_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Feature extraction
        features = []
        start_time = time.time()
        total_batches = len(image_paths) // batch_size + bool(len(image_paths) % batch_size)
        
        for batch_idx, batch_images in enumerate(image_ds):
            try:
                batch_features = self.feature_extractor.predict(batch_images, verbose=0)
                features.append(batch_features)
                
                # Progress tracking
                elapsed_time = time.time() - start_time
                avg_batch_time = elapsed_time / (batch_idx + 1)
                remaining_time = (total_batches - batch_idx - 1) * avg_batch_time
                
                logging.info(
                    f"Batch {batch_idx+1}/{total_batches} "
                    f"({(batch_idx+1)/total_batches*100:.1f}%) "
                    f"- Est. remaining: {remaining_time/60:.1f} minutes"
                )
            
            except Exception as e:
                logging.error(f"Batch processing error: {e}")
        
        # Combine and normalize features
        features = np.vstack(features)
        features = normalize(features, axis=1, norm='l2')
        
        return features

def build_feature_database(
    directory_path, 
    output_dir, 
    batch_size=32, 
    model_type='efficientnet', 
    input_shape=(224, 224),
    use_gpu=True
):
    """
    Comprehensive feature database builder.
    
    Args:
        directory_path (str): Source image directory
        output_dir (str): Output directory for features
        batch_size (int): Processing batch size
        model_type (str): Feature extraction model
        input_shape (tuple): Input image dimensions
        use_gpu (bool): Enable GPU acceleration
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find image paths with expanded format support
    image_paths = [
        str(p) for p in Path(directory_path).rglob("*")
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    ]
    
    if not image_paths:
        raise ValueError(f"No images found in {directory_path}")
    
    logging.info(f"Found {len(image_paths)} images to process")
    
    # Initialize feature extractor
    extractor = AdvancedFeatureExtractor(
        model_type=model_type, 
        input_shape=input_shape,
        use_gpu=use_gpu
    )
    
    # Extract features
    features = extractor.extract_features(image_paths, batch_size)
    
    # Create shortened image names (image name with number)
    # short_image_names = [
    #     f"image_{idx+1}{Path(path).suffix}" 
    #     for idx, path in enumerate(image_paths)
    # ]
    
    # Save features and paths
    feature_file = os.path.join(output_dir, 'image_features.joblib')
    paths_file = os.path.join(output_dir, 'image_paths.joblib')
    
    logging.info(f"Saving features to {feature_file}")
    dump(features, feature_file, compress=3)
    
    logging.info(f"Saving image names to {paths_file}")
    dump([s.split("\\")[-1] for s in image_paths], paths_file, compress=3)
    
    logging.info("Feature extraction complete!")
    logging.info(f"Total images processed: {len(image_paths)}")
    logging.info(f"Feature shape: {features.shape}")

if __name__ == "__main__":
    # Configuration
    # IMAGE_DIR = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\final\\Finalfinal\\features\\clean_images_final_maze2"
    # OUTPUT_DIR = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\final\\Finalfinal\\features\\maze2features_joblib"
    IMAGE_DIR = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\final\\Finalfinal\\features\\clean_images_final_maze1"
    OUTPUT_DIR = "C:\\Desktop\\NYU\\3rd_sem\\Perception\\final\\Finalfinal\\features\\maze1features_joblib"

    build_feature_database(
        directory_path=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=32,
        model_type='efficientnet',  # Options: 'efficientnet', 'resnet', 'vgg'
        input_shape=(224, 224),
        use_gpu=True
    )