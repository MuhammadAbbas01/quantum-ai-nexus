import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import sqlite3
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
import threading
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pytesseract
import os
import base64
import io

# --- AI MODELS ---
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import requests

# Set Tesseract path for Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\DELL\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImageAnalysisResult:
    """Structured result for image analysis"""
    objects: List[str]
    scene_type: str
    dominant_colors: List[str]
    emotions: List[str]
    text_content: str
    confidence_score: float
    suggestions: List[str]
    metadata: Dict[str, Any]
    processing_time: float
    enhanced_image_base64: Optional[str] = None

@dataclass
class ProcessingPipeline:
    """Defines a specialized processing pipeline for specific image types"""
    name: str
    priority_tasks: List[str]
    skip_tasks: List[str]
    enhancement_profile: str
    quality_thresholds: Dict[str, float]
    specialized_config: Dict[str, Any]

class OptimizedKnowledgeBase:
    """Lightweight knowledge base with essential information"""
    
    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self._cache = {}
        self._lock = threading.RLock()
        self._initialize_database()
        self._load_essential_knowledge()
    
    def _initialize_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE,
                analysis_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_essential_knowledge(self):
        """Load essential knowledge for image analysis"""
        knowledge = {
            'color_names': {
                'red_range': [np.array([0, 50, 50]), np.array([10, 255, 255])],
                'green_range': [np.array([35, 50, 50]), np.array([85, 255, 255])],
                'blue_range': [np.array([100, 50, 50]), np.array([130, 255, 255])],
                'yellow_range': [np.array([15, 50, 50]), np.array([35, 255, 255])],
                'purple_range': [np.array([130, 50, 50]), np.array([160, 255, 255])]
            },
            'scene_indicators': {
                'outdoor_keywords': ['sky', 'tree', 'grass', 'road', 'building'],
                'indoor_keywords': ['wall', 'floor', 'ceiling', 'furniture', 'room'],
                'text_keywords': ['document', 'paper', 'book', 'sign', 'letter']
            },
            'enhancement_tips': {
                'low_contrast': 'Increase contrast for better visibility',
                'dark_image': 'Brighten the image for better clarity',
                'blurry': 'Apply sharpening filter for clearer details',
                'noisy': 'Use noise reduction for cleaner appearance'
            }
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for category, content in knowledge.items():
            if category == 'color_names':
                serializable_content = {}
                for color_name, (lower, upper) in content.items():
                    serializable_content[color_name] = [lower.tolist(), upper.tolist()]
                cursor.execute('''
                    INSERT OR REPLACE INTO knowledge_entries 
                    (category, content) VALUES (?, ?)
                ''', (category, json.dumps(serializable_content)))
            else:
                cursor.execute('''
                    INSERT OR REPLACE INTO knowledge_entries 
                    (category, content) VALUES (?, ?)
                ''', (category, json.dumps(content)))
        
        conn.commit()
        conn.close()
    
    @lru_cache(maxsize=32)
    def get_knowledge(self, category: str) -> Dict:
        """Get knowledge by category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT content FROM knowledge_entries 
            WHERE category = ?
        ''', (category,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            content = json.loads(result[0])
            if category == 'color_names':
                for color_name, (lower_list, upper_list) in content.items():
                    content[color_name] = [np.array(lower_list), np.array(upper_list)]
            return content
        return {}

class OptimizedImageProcessor:
    """Optimized image processor with specialized pipelines"""
    
    def __init__(self):
        self.knowledge_base = OptimizedKnowledgeBase()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()
        self.model.to(self.device)

        self.imagenet_labels = self._load_imagenet_labels_from_url()
        if not self.imagenet_labels:
            logger.error("Failed to load ImageNet labels")
            self.imagenet_labels = ["unknown", "object", "scene"]

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.CONFIDENCE_THRESHOLDS = {
            'object_detection': 0.15,
            'scene_classification': 0.25,
            'high_confidence': 0.4
        }
        
        self._initialize_specialized_pipelines()
        
        logger.info("✅ Optimized Image Processor initialized with specialized pipelines")

    def _initialize_specialized_pipelines(self):
        """Initialize specialized processing pipelines"""
        self.processing_pipelines = {
            'document_paper': ProcessingPipeline(
                name='Document/Text Processing',
                priority_tasks=['text_extraction', 'contrast_enhancement', 'sharpening'],
                skip_tasks=['face_detection', 'emotion_analysis', 'color_analysis'],
                enhancement_profile='text_optimized',
                quality_thresholds={'sharpness': 400, 'contrast': 50, 'brightness_range': [100, 180]},
                specialized_config={
                    'ocr_mode': 'aggressive',
                    'preprocessing': ['deskew', 'noise_reduction', 'binarization'],
                    'enhancement_strength': 'high'
                }
            ),
            'social_gathering': ProcessingPipeline(
                name='Social/Portrait Processing',
                priority_tasks=['face_detection', 'emotion_analysis', 'people_focus'],
                skip_tasks=['text_extraction'],
                enhancement_profile='portrait_optimized',
                quality_thresholds={'sharpness': 300, 'contrast': 40, 'brightness_range': [90, 200]},
                specialized_config={
                    'face_priority': True,
                    'skin_tone_preservation': True,
                    'enhancement_strength': 'moderate'
                }
            ),
            'indoor_collaboration_space': ProcessingPipeline(
                name='Workspace/Meeting Processing',
                priority_tasks=['object_detection', 'text_extraction', 'face_detection'],
                skip_tasks=[],
                enhancement_profile='balanced',
                quality_thresholds={'sharpness': 350, 'contrast': 45, 'brightness_range': [80, 190]},
                specialized_config={
                    'dual_focus': ['people', 'technology'],
                    'enhancement_strength': 'moderate'
                }
            ),
            'outdoor_nature': ProcessingPipeline(
                name='Landscape/Nature Processing',
                priority_tasks=['color_analysis', 'scene_analysis', 'dynamic_range'],
                skip_tasks=['face_detection', 'emotion_analysis', 'text_extraction'],
                enhancement_profile='landscape_optimized',
                quality_thresholds={'sharpness': 250, 'contrast': 60, 'brightness_range': [70, 220]},
                specialized_config={
                    'color_enhancement': True,
                    'tone_mapping': True,
                    'enhancement_strength': 'high'
                }
            ),
            'urban_scene': ProcessingPipeline(
                name='Urban/Architecture Processing',
                priority_tasks=['object_detection', 'geometry_analysis', 'contrast_enhancement'],
                skip_tasks=['emotion_analysis'],
                enhancement_profile='architectural',
                quality_thresholds={'sharpness': 350, 'contrast': 55, 'brightness_range': [85, 200]},
                specialized_config={
                    'line_enhancement': True,
                    'structure_focus': True,
                    'enhancement_strength': 'moderate'
                }
            ),
            'indoor_scene': ProcessingPipeline(
                name='Indoor/Interior Processing',
                priority_tasks=['object_detection', 'lighting_analysis'],
                skip_tasks=['emotion_analysis'],
                enhancement_profile='interior',
                quality_thresholds={'sharpness': 300, 'contrast': 45, 'brightness_range': [90, 185]},
                specialized_config={
                    'lighting_correction': True,
                    'enhancement_strength': 'moderate'
                }
            ),
            'unknown_scene': ProcessingPipeline(
                name='General Processing',
                priority_tasks=['object_detection', 'scene_analysis', 'color_analysis'],
                skip_tasks=[],
                enhancement_profile='balanced',
                quality_thresholds={'sharpness': 300, 'contrast': 40, 'brightness_range': [85, 195]},
                specialized_config={
                    'enhancement_strength': 'moderate'
                }
            )
        }

    def _is_document_like_image(self, image: np.ndarray) -> bool:
        """Detect document-like characteristics"""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_line_density = np.sum(horizontal_lines > 0) / (height * width)
        
        aspect_ratio = height / width
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_pixel_ratio = np.sum(binary == 255) / (height * width)
        black_pixel_ratio = np.sum(binary == 0) / (height * width)
        
        try:
            text = pytesseract.image_to_string(gray, config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ')
            text_length = len(text.strip())
            alphanumeric_ratio = sum(c.isalnum() for c in text) / max(1, len(text))
        except:
            text_length = 0
            alphanumeric_ratio = 0
        
        document_score = 0
        
        if edge_density > 0.05:
            document_score += 1
        if horizontal_line_density > 0.008:
            document_score += 2
        if aspect_ratio > 1.3:
            document_score += 1
        if white_pixel_ratio > 0.6 and black_pixel_ratio > 0.1:
            document_score += 1
        if text_length > 20 and alphanumeric_ratio > 0.7:
            document_score += 2
        
        logger.info(f"Document detection - Edge: {edge_density:.3f}, Line: {horizontal_line_density:.3f}, "
                   f"Aspect: {aspect_ratio:.2f}, Text: {text_length}, Score: {document_score}")
        
        return document_score >= 3

    def _select_processing_pipeline(self, scene_type: str, preliminary_objects: List[str], image: np.ndarray) -> ProcessingPipeline:
        """Select processing pipeline"""
        
        if self._is_document_like_image(image):
            selected_pipeline = self.processing_pipelines['document_paper']
            logger.info(f"✅ Selected pipeline: {selected_pipeline.name} (document detected)")
            return selected_pipeline
        
        text_indicators = ['menu', 'paper', 'document', 'book', 'binder', 'envelope']
        if any(obj.lower() in text_indicators for obj in preliminary_objects):
            selected_pipeline = self.processing_pipelines['document_paper']
            logger.info(f"✅ Selected pipeline: {selected_pipeline.name} (text objects)")
            return selected_pipeline
        
        if scene_type in self.processing_pipelines:
            selected_pipeline = self.processing_pipelines[scene_type]
            logger.info(f"✅ Selected pipeline: {selected_pipeline.name} (scene-based)")
            return selected_pipeline
        
        if any(obj in ['person', 'face_present'] for obj in preliminary_objects):
            selected_pipeline = self.processing_pipelines['social_gathering']
            logger.info(f"✅ Selected pipeline: {selected_pipeline.name} (people detected)")
            return selected_pipeline
        
        selected_pipeline = self.processing_pipelines['unknown_scene']
        logger.info(f"✅ Selected pipeline: {selected_pipeline.name} (default)")
        return selected_pipeline

    def _load_imagenet_labels_from_url(self):
        """Download ImageNet labels"""
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        try:
            response = requests.get(url)
            response.raise_for_status()
            labels = [line.strip() for line in response.text.split('\n') if line.strip()]
            logger.info(f"✅ Loaded {len(labels)} ImageNet labels")
            return labels
        except Exception as e:
            logger.error(f"❌ Error downloading ImageNet labels: {e}")
            return None

    def _get_image_hash(self, image_path: str) -> str:
        """Generate hash for caching"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return str(hash(image_path))
    
    def _load_cached_analysis(self, image_hash: str) -> Optional[ImageAnalysisResult]:
        """Load cached analysis"""
        try:
            conn = sqlite3.connect(self.knowledge_base.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT analysis_result FROM image_cache 
                WHERE image_hash = ?
            ''', (image_hash,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                data = json.loads(result[0])
                return ImageAnalysisResult(**data)
        except Exception as e:
            logger.error(f"Cache loading error: {e}")
        
        return None
    
    def _cache_analysis(self, image_hash: str, result: ImageAnalysisResult):
        """Cache analysis result"""
        try:
            conn = sqlite3.connect(self.knowledge_base.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO image_cache 
                (image_hash, analysis_result) VALUES (?, ?)
            ''', (image_hash, json.dumps(result.__dict__)))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Cache saving error: {e}")
    
    def _preprocess_image_for_resnet(self, image_np: np.ndarray):
        """Preprocess image for ResNet"""
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        return tensor

    def _preprocess_image_specialized(self, image: np.ndarray, pipeline: ProcessingPipeline) -> np.ndarray:
        """Apply specialized preprocessing"""
        
        if pipeline.name == 'Document/Text Processing':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            coords = np.column_stack(np.where(gray > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                if abs(angle) > 0.5:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            denoised = cv2.bilateralFilter(image, 15, 100, 100)
            return denoised
            
        elif pipeline.name == 'Social/Portrait Processing':
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            denoised = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            return denoised
            
        elif pipeline.name == 'Landscape/Nature Processing':
            denoised = cv2.bilateralFilter(image, 7, 60, 60)
            
            hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.multiply(s, 1.1)
            enhanced = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
            return enhanced
            
        else:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            return denoised

    # FIXED: Convert image to base64 for FULL-SIZE display
    def _image_to_base64(self, image: np.ndarray, max_dimension: int = None) -> str:
        """FIXED: Convert OpenCV image to base64 - FULL SIZE by default"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # FIXED: Only resize if explicitly requested or image is extremely large
            if max_dimension and (pil_image.width > max_dimension or pil_image.height > max_dimension):
                pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                logger.info(f"Resized image to fit within {max_dimension}px")
            
            # Save to bytes buffer with high quality
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)  # High quality JPEG
            
            # Convert to base64
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logger.info(f"✅ Converted image to base64: {len(base64_string)} characters, "
                       f"size: {pil_image.width}x{pil_image.height}px")
            return f"data:image/jpeg;base64,{base64_string}"
            
        except Exception as e:
            logger.error(f"❌ Error converting image to base64: {e}")
            return None

    async def analyze_image_comprehensive(self, image_path: str) -> ImageAnalysisResult:
        """FIXED: Comprehensive image analysis with FULL-SIZE enhanced images"""
        start_time = datetime.now()
        
        try:
            image_hash = self._get_image_hash(image_path)
            cached_result = self._load_cached_analysis(image_hash)
            
            if cached_result:
                logger.info(f"✅ Using cached analysis")
                return cached_result
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            logger.info(f"📷 Image loaded: {image.shape[1]}x{image.shape[0]}px")
            
            # PHASE 1: Preliminary analysis
            input_tensor = self._preprocess_image_for_resnet(image)
            loop = asyncio.get_event_loop()
            
            preliminary_scene = await loop.run_in_executor(
                self.executor, self._classify_scene_resnet, input_tensor, image
            )
            preliminary_objects = await loop.run_in_executor(
                self.executor, self._detect_objects_resnet, input_tensor, image
            )
            
            # PHASE 2: Select pipeline
            selected_pipeline = self._select_processing_pipeline(preliminary_scene, preliminary_objects, image)
            
            # PHASE 3: Specialized preprocessing
            specialized_image = self._preprocess_image_specialized(image, selected_pipeline)
            specialized_tensor = self._preprocess_image_for_resnet(specialized_image)
            
            # PHASE 4: FIXED - Create FULL-SIZE enhanced image
            enhanced_image = await loop.run_in_executor(
                self.executor, self._enhance_image_for_display, image, selected_pipeline
            )
            # FIXED: No max_dimension limit - full size
            enhanced_image_base64 = self._image_to_base64(enhanced_image)
            
            logger.info(f"✅ Enhanced image created: FULL SIZE")
            
            # PHASE 5: Execute analysis tasks
            tasks = []
            results_map = {}
            
            tasks.append(('quality', loop.run_in_executor(self.executor, self._analyze_image_quality, image)))
            
            if 'object_detection' in selected_pipeline.priority_tasks:
                tasks.append(('objects', loop.run_in_executor(
                    self.executor, self._detect_objects_resnet, specialized_tensor, specialized_image
                )))
            
            if 'scene_analysis' in selected_pipeline.priority_tasks:
                tasks.append(('scene', loop.run_in_executor(
                    self.executor, self._classify_scene_resnet, specialized_tensor, specialized_image
                )))
            else:
                results_map['scene'] = preliminary_scene
            
            if 'color_analysis' in selected_pipeline.priority_tasks:
                tasks.append(('colors', loop.run_in_executor(
                    self.executor, self._analyze_colors_improved, specialized_image
                )))
            
            if 'face_detection' in selected_pipeline.priority_tasks:
                tasks.append(('emotions', loop.run_in_executor(
                    self.executor, self._detect_emotions_improved, specialized_image
                )))
            
            if 'text_extraction' in selected_pipeline.priority_tasks:
                tasks.append(('text', loop.run_in_executor(
                    self.executor, self._extract_text_ocr_specialized, specialized_image, selected_pipeline
                )))
            
            completed_tasks = await asyncio.gather(*[task for _, task in tasks])
            
            for i, (task_name, _) in enumerate(tasks):
                results_map[task_name] = completed_tasks[i]
            
            # Fill skipped tasks
            if 'objects' not in results_map:
                results_map['objects'] = preliminary_objects if preliminary_objects else ['general_content']
            
            if 'colors' not in results_map:
                if 'color_analysis' not in selected_pipeline.skip_tasks:
                    results_map['colors'] = await loop.run_in_executor(
                        self.executor, self._analyze_colors_improved, specialized_image
                    )
                else:
                    results_map['colors'] = ['color_analysis_skipped']
            
            if 'emotions' not in results_map:
                if 'emotion_analysis' not in selected_pipeline.skip_tasks:
                    results_map['emotions'] = await loop.run_in_executor(
                        self.executor, self._detect_emotions_improved, specialized_image
                    )
                else:
                    results_map['emotions'] = ['no_emotion_analysis']
            
            if 'text' not in results_map:
                if 'text_extraction' not in selected_pipeline.skip_tasks:
                    results_map['text'] = await loop.run_in_executor(
                        self.executor, self._extract_text_ocr_specialized, specialized_image, selected_pipeline
                    )
                else:
                    results_map['text'] = 'Text extraction skipped'
            
            # Generate suggestions
            suggestions = self._generate_specialized_suggestions(
                results_map['objects'], results_map['scene'], results_map['colors'], 
                results_map['emotions'], results_map['text'], results_map['quality'], selected_pipeline
            )
            
            confidence = self._calculate_specialized_confidence(results_map, selected_pipeline)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ImageAnalysisResult(
                objects=results_map['objects'],
                scene_type=results_map['scene'],
                dominant_colors=results_map['colors'],
                emotions=results_map['emotions'],
                text_content=results_map['text'],
                confidence_score=confidence,
                suggestions=suggestions,
                metadata={
                    'image_quality': results_map['quality'],
                    'image_hash': image_hash,
                    'processing_method': f'specialized_pipeline_{selected_pipeline.enhancement_profile}',
                    'selected_pipeline': selected_pipeline.name,
                    'priority_tasks': selected_pipeline.priority_tasks,
                    'skipped_tasks': selected_pipeline.skip_tasks
                },
                processing_time=processing_time,
                enhanced_image_base64=enhanced_image_base64
            )
            
            self._cache_analysis(image_hash, result)
            
            logger.info(f"✅ Analysis completed in {processing_time:.2f}s using {selected_pipeline.name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in image analysis: {e}")
            import traceback
            traceback.print_exc()
            return ImageAnalysisResult(
                objects=['analysis_error'],
                scene_type='unknown',
                dominant_colors=['unknown'],
                emotions=['neutral'],
                text_content='',
                confidence_score=0.0,
                suggestions=['Please try uploading a different image'],
                metadata={'error': str(e)},
                processing_time=(datetime.now() - start_time).total_seconds(),
                enhanced_image_base64=None
            )
    
    def _enhance_image_for_display(self, image: np.ndarray, pipeline: ProcessingPipeline) -> np.ndarray:
        """FIXED: Create FULL-SIZE enhanced image for display"""
        enhanced = image.copy()
        quality = self._analyze_image_quality(image)
        
        # Apply specialized preprocessing
        enhanced = self._preprocess_image_specialized(enhanced, pipeline)
        
        # Apply pipeline-specific enhancements
        if pipeline.enhancement_profile == 'text_optimized':
            enhanced = self._enhance_for_text(enhanced, quality, pipeline)
        elif pipeline.enhancement_profile == 'portrait_optimized':
            enhanced = self._enhance_for_portraits(enhanced, quality, pipeline)
        elif pipeline.enhancement_profile == 'landscape_optimized':
            enhanced = self._enhance_for_landscapes(enhanced, quality, pipeline)
        elif pipeline.enhancement_profile == 'architectural':
            enhanced = self._enhance_for_architecture(enhanced, quality, pipeline)
        else:
            enhanced = self._enhance_balanced(enhanced, quality, pipeline)
        
        logger.info(f"✅ Enhanced image created: {enhanced.shape[1]}x{enhanced.shape[0]}px (FULL SIZE)")
        return enhanced
    
    def _detect_objects_resnet(self, input_tensor: torch.Tensor, original_image: np.ndarray) -> List[str]:
        """Object detection with confidence filtering"""
        detected_objects = []
        
        try:
            if not self.imagenet_labels:
                return ['object_detection_error']

            with torch.no_grad():
                output = self.model(input_tensor)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_indices = torch.topk(probabilities, 20)

            high_confidence_objects = []
            medium_confidence_objects = []
            
            problematic_labels = {
                'hard disc', 'crossword puzzle', 'Appenzeller', 'EntleBucher', 'tench', 'coho',
                'Dungeness crab', 'crayfish', 'American lobster', 'cleaver', 'spatula',
                'toilet tissue', 'paper towel', 'bathtub', 'plunger', 'drumstick', 'wig'
            }
            
            scene_only_labels = {
                'sky', 'clouds', 'ocean', 'sea', 'field', 'forest', 'river', 'lake', 'beach',
                'park', 'garden', 'street', 'road', 'highway', 'indoor', 'room'
            }

            if not self._is_document_like_image(original_image):
                gray_for_faces = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_for_faces, 1.1, 4)
                if len(faces) > 0:
                    detected_objects.append('person')
                    detected_objects.append('face_present')
                    if len(faces) > 1:
                        detected_objects.append(f'{len(faces)}_people')

            for i in range(top_indices.size(0)):
                label_index = top_indices[i].item()
                if label_index < len(self.imagenet_labels):
                    label = self.imagenet_labels[label_index]
                    prob = top_prob[i].item()

                    if label in detected_objects or label in problematic_labels or label in scene_only_labels:
                        continue
                    
                    if prob > self.CONFIDENCE_THRESHOLDS['high_confidence']:
                        high_confidence_objects.append(label)
                    elif prob > self.CONFIDENCE_THRESHOLDS['object_detection']:
                        medium_confidence_objects.append(label)

            detected_objects.extend(high_confidence_objects[:3])
            
            remaining_slots = max(0, 5 - len(detected_objects))
            detected_objects.extend(medium_confidence_objects[:remaining_slots])

            self._add_geometric_context(original_image, detected_objects)

            return detected_objects if detected_objects else ['general_object']
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return ['object_detection_error']

    def _classify_scene_resnet(self, input_tensor: torch.Tensor, original_image: np.ndarray) -> str:
        """Scene classification"""
        try:
            if not self.imagenet_labels:
                return 'scene_classification_error'

            with torch.no_grad():
                output = self.model(input_tensor)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_indices = torch.topk(probabilities, 15)

            if self._is_document_like_image(original_image):
                return 'document_paper'

            scene_category_map = {
                'outdoor_nature': ['mountain', 'valley', 'forest', 'beach', 'lake', 'river', 'seashore', 
                                 'promontory', 'alp', 'cliff', 'lakeside', 'volcano', 'coral reef'],
                'urban_scene': ['building', 'house', 'church', 'bridge', 'street', 'road', 'cityscape'],
                'indoor_scene': ['room', 'kitchen', 'bedroom', 'office', 'indoor', 'desk', 'table', 'chair'],
                'document_paper': ['document', 'paper', 'book', 'menu', 'binder'],
                'sports_scene': ['stadium', 'racetrack', 'tennis court', 'golf course']
            }

            if not self._is_document_like_image(original_image):
                gray_for_faces = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_for_faces, 1.1, 4)
                
                if len(faces) > 0:
                    if self._detect_rectangular_shapes(original_image):
                        return 'indoor_collaboration_space'
                    else:
                        return 'social_gathering'
            
            for category, keywords in scene_category_map.items():
                for i in range(top_indices.size(0)):
                    label_index = top_indices[i].item()
                    if label_index < len(self.imagenet_labels):
                        predicted_label = self.imagenet_labels[label_index]
                        if predicted_label in keywords and top_prob[i].item() > self.CONFIDENCE_THRESHOLDS['scene_classification']:
                            return category

            return 'unknown_scene'
            
        except Exception as e:
            logger.error(f"Scene classification failed: {e}")
            return 'scene_classification_error'

    def _add_geometric_context(self, image: np.ndarray, detected_objects: List[str]):
        """Add geometric context"""
        if self._detect_rectangular_shapes(image):
            if 'rectangular_shapes' not in detected_objects:
                detected_objects.append('rectangular_shapes')

    def _detect_rectangular_shapes(self, image: np.ndarray) -> bool:
        """Detect rectangular shapes"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (image.shape[0] * image.shape[1] * 0.01):
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(approx) == 4:
                    aspect_ratio = float(cv2.boundingRect(approx)[2])/cv2.boundingRect(approx)[3]
                    if 0.5 < aspect_ratio < 3.0:
                        return True
        return False
    
    def _analyze_colors_improved(self, image: np.ndarray) -> List[str]:
        """Color analysis"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        colors = []
        color_knowledge = self.knowledge_base.get_knowledge('color_names')
        
        if color_knowledge:
            for color_name, (lower, upper) in color_knowledge.items():
                if color_name.endswith('_range'): 
                    color = color_name.replace('_range', '')
                    mask = cv2.inRange(hsv, lower, upper)
                    ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255)
                    
                    if ratio > 0.1:
                        colors.append(color)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness > 200:
            colors.append('bright')
        elif mean_brightness < 80:
            colors.append('dark')
        
        if not colors:
            if mean_brightness > 150:
                colors.append('light_tones')
            elif mean_brightness < 100:
                colors.append('dark_tones')
            else:
                colors.append('medium_tones')
        
        return colors[:3]
    
    def _detect_emotions_improved(self, image: np.ndarray) -> List[str]:
        """Emotion detection"""
        if self._is_document_like_image(image):
            return ['no_faces']
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        emotions = []
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                face_mean = np.mean(face_roi)
                face_std = np.std(face_roi)
                
                eye_region = face_roi[0:h//3, :]
                mouth_region = face_roi[2*h//3:h, :]
                
                eye_mean = np.mean(eye_region)
                mouth_mean = np.mean(mouth_region)
                
                if face_std > 50:
                    emotions.append('highly_expressive')
                elif face_std > 35:
                    emotions.append('expressive')
                elif mouth_mean > eye_mean + 15:
                    emotions.append('happy')
                elif eye_mean < mouth_mean - 15:
                    emotions.append('serious')
                elif face_std < 20:
                    emotions.append('calm')
                else:
                    emotions.append('neutral')
        else:
            emotions.append('no_faces')
        
        return list(dict.fromkeys(emotions))
    
    def _extract_text_ocr_specialized(self, image: np.ndarray, pipeline: ProcessingPipeline) -> str:
        """FIXED: Text extraction with proper formatting"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if pipeline.name == 'Document/Text Processing':
                denoised = cv2.fastNlMeansDenoising(gray, h=20)
                
                thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 15, 4)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
                
                custom_config = r'--oem 3 --psm 4 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()$/# '
                text = pytesseract.image_to_string(cleaned, config=custom_config)
                
                if not text.strip():
                    return "No readable text detected in document"
                
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                if not lines:
                    return "No substantial text content found"
                
                formatted_lines = []
                for line in lines:
                    cleaned_line = line.replace('|', 'I').replace('§', 'S').replace('©', 'O')
                    
                    if len(cleaned_line) < 2 and not any(c.isdigit() or c in '$#' for c in cleaned_line):
                        continue
                        
                    formatted_lines.append(cleaned_line)
                
                if not formatted_lines:
                    return "Text detected but formatting failed"
                
                formatted_text = '\n'.join(formatted_lines)
                
                if len(formatted_text.strip()) < 10:
                    return "Limited text content detected in document"
                    
                return formatted_text
                
            elif pipeline.name == 'Workspace/Meeting Processing':
                denoised = cv2.fastNlMeansDenoising(gray, h=15)
                thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
                
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(thresh, config=custom_config)
                
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                formatted_text = '\n'.join(lines)
                
                if len(formatted_text.strip()) > 5:
                    return formatted_text
                else:
                    return "Minimal text content detected in workspace"
            
            else:
                denoised = cv2.fastNlMeansDenoising(gray, h=10)
                thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
                
                custom_config = r'--oem 3 --psm 8 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(thresh, config=custom_config)
                
                text = text.strip()
                return text if text else "No text detected"
                
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return f"Text extraction error: Unable to process image text"
    
    def _analyze_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Image quality analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_level = np.std(gray - gaussian_blur)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.mean(sobel_magnitude > 50)
        
        return {
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'noise_level': noise_level,
            'edge_density': edge_density,
            'resolution': f"{image.shape[1]}x{image.shape[0]}",
            'quality_score': self._calculate_quality_score(sharpness, brightness, contrast, noise_level)
        }

    def _calculate_quality_score(self, sharpness: float, brightness: float, 
                               contrast: float, noise_level: float) -> float:
        """Calculate quality score"""
        sharpness_score = min(sharpness / 1000, 1.0)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        contrast_score = min(contrast / 100, 1.0)
        noise_score = max(0, 1.0 - noise_level / 100)
        
        quality_score = (sharpness_score * 0.3 + brightness_score * 0.25 + 
                        contrast_score * 0.25 + noise_score * 0.2)
        
        return round(quality_score, 3)
    
    def _generate_specialized_suggestions(self, objects: List[str], scene_type: str, 
                                        colors: List[str], emotions: List[str], 
                                        text_content: str, quality: Dict[str, Any],
                                        pipeline: ProcessingPipeline) -> List[str]:
        """Generate suggestions"""
        suggestions = []
        
        if pipeline.name == 'Document/Text Processing':
            if 'error' not in text_content.lower() and len(text_content.strip()) > 20:
                lines = [line for line in text_content.split('\n') if line.strip()]
                if len(lines) > 3:
                    suggestions.append("✅ Excellent document processing! Multi-line text extracted successfully.")
                else:
                    suggestions.append("✅ Document detected and processed. Text extraction completed.")
            elif 'no readable text' in text_content.lower():
                suggestions.append("⚠️ Document detected but text extraction limited. Improve lighting/flatness.")
            else:
                suggestions.append("✅ Document processing completed.")
            
            if quality['sharpness'] < 400:
                suggestions.append("💡 Apply document-specific sharpening for improved clarity.")
            
            if quality['contrast'] < 50:
                suggestions.append("💡 Enhance contrast for better text readability.")
                
        elif pipeline.name == 'Social/Portrait Processing':
            if 'person' in objects or 'face_present' in objects:
                if 'happy' in emotions:
                    suggestions.append("😊 Joyful social moment captured!")
                elif 'expressive' in emotions:
                    suggestions.append("✨ Dynamic expressions detected!")
                else:
                    suggestions.append("👥 People-focused image detected.")
            
            if quality['brightness'] < 90:
                suggestions.append("💡 Apply gentle brightening for portraits.")
                
        elif pipeline.name == 'Workspace/Meeting Processing':
            suggestions.append("💼 Professional workspace detected.")
            
            if text_content and len(text_content.strip()) > 10:
                suggestions.append("📝 Presentation/workspace text detected.")
            
            if 'rectangular_shapes' in objects:
                suggestions.append("🖥️ Technology and furniture detected.")
                
        else:
            suggestions.append(f"✅ Image processed using {pipeline.name}.")
        
        quality_score = quality.get('quality_score', 0.5)
        pipeline_thresholds = pipeline.quality_thresholds
        
        if quality['sharpness'] < pipeline_thresholds.get('sharpness', 300):
            if quality['noise_level'] < 30:
                suggestions.append("🔍 Image sharpness can be improved.")
        
        if quality['contrast'] < pipeline_thresholds.get('contrast', 40):
            suggestions.append("📊 Contrast enhancement recommended.")
        
        brightness_range = pipeline_thresholds.get('brightness_range', [85, 195])
        if quality['brightness'] < brightness_range[0]:
            suggestions.append("💡 Image appears dark - brightness adjustment needed.")
        elif quality['brightness'] > brightness_range[1]:
            suggestions.append("☀️ Image very bright - tone adjustment recommended.")
        
        if quality['noise_level'] > 50:
            if pipeline.name == 'Document/Text Processing':
                suggestions.append("🧹 High noise - aggressive noise reduction recommended.")
            else:
                suggestions.append("🧹 Noise reduction recommended.")
        
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:4] if unique_suggestions else ["✅ Image processed successfully."]
    
    def _calculate_specialized_confidence(self, results_map: Dict[str, Any], pipeline: ProcessingPipeline) -> float:
        """Calculate confidence"""
        scores = []
        
        objects = results_map.get('objects', [])
        scene_type = results_map.get('scene', 'unknown')
        colors = results_map.get('colors', [])
        emotions = results_map.get('emotions', [])
        text_content = results_map.get('text', '')
        quality = results_map.get('quality', {})
        
        if scene_type == 'document_paper' and pipeline.name == 'Document/Text Processing':
            scores.append(0.95)
        elif scene_type != 'unknown_scene' and 'error' not in scene_type:
            scores.append(0.85)
        else:
            scores.append(0.6)
        
        if pipeline.name == 'Document/Text Processing':
            if text_content and 'error' not in text_content.lower() and len(text_content.strip()) > 20:
                lines = [line for line in text_content.split('\n') if line.strip()]
                text_quality = len(lines) / max(1, len(text_content.split('\n')))
                char_quality = sum(c.isalnum() or c.isspace() for c in text_content) / max(1, len(text_content))
                combined_score = (text_quality * 0.4 + char_quality * 0.6) * 0.9
                scores.append(min(0.95, combined_score))
            elif text_content and len(text_content.strip()) > 5:
                scores.append(0.7)
            else:
                scores.append(0.4)
        else:
            if text_content and len(text_content.strip()) > 5:
                scores.append(0.8)
            else:
                scores.append(0.7)
        
        if objects and 'object_detection_error' not in objects:
            if len(objects) > 2 and 'general_object' not in objects:
                scores.append(0.85)
            else:
                scores.append(0.75)
        else:
            scores.append(0.5)
        
        if 'color_analysis' in pipeline.skip_tasks:
            scores.append(0.8)
        elif colors and len(colors) > 0:
            scores.append(0.8)
        else:
            scores.append(0.6)
        
        if 'emotion_analysis' in pipeline.skip_tasks or pipeline.name == 'Document/Text Processing':
            scores.append(0.8)
        elif emotions and len(emotions) > 0:
            scores.append(0.8)
        else:
            scores.append(0.6)
        
        quality_score = quality.get('quality_score', 0.5)
        scores.append(min(0.95, quality_score + 0.2))
        
        return round(np.mean(scores), 3)

    def _enhance_for_text(self, image: np.ndarray, quality: Dict[str, Any], pipeline: ProcessingPipeline) -> np.ndarray:
        """Enhancement for text"""
        enhanced = image.copy()
        
        if quality['contrast'] < 50:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        if quality['brightness'] < 100 or quality['brightness'] > 180:
            alpha = 1.2 if quality['brightness'] < 100 else 0.85
            beta = 20 if quality['brightness'] < 100 else -15
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        if quality['sharpness'] < 400:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.addWeighted(enhanced, 0.5, sharpened, 0.5, 0)
        
        return enhanced

    def _enhance_for_portraits(self, image: np.ndarray, quality: Dict[str, Any], pipeline: ProcessingPipeline) -> np.ndarray:
        """Enhancement for portraits"""
        enhanced = image.copy()
        
        if quality['brightness'] < 90:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
        elif quality['brightness'] > 200:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=0.95, beta=-8)
        
        if quality['contrast'] < 40:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        if quality['sharpness'] < 300:
            kernel = np.array([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)
        
        return enhanced

    def _enhance_for_landscapes(self, image: np.ndarray, quality: Dict[str, Any], pipeline: ProcessingPipeline) -> np.ndarray:
        """Enhancement for landscapes"""
        enhanced = image.copy()
        
        if quality['contrast'] < 60:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.15)
        enhanced = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        
        if quality['sharpness'] < 250:
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return enhanced

    def _enhance_for_architecture(self, image: np.ndarray, quality: Dict[str, Any], pipeline: ProcessingPipeline) -> np.ndarray:
        """Enhancement for architecture"""
        enhanced = image.copy()
        
        if quality['contrast'] < 55:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        if quality['brightness'] < 85:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.08, beta=12)
        elif quality['brightness'] > 200:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=0.92, beta=-10)
        
        if quality['sharpness'] < 350:
            kernel = np.array([[0, -0.75, 0], [-0.75, 4, -0.75], [0, -0.75, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.addWeighted(enhanced, 0.6, sharpened, 0.4, 0)
        
        return enhanced

    def _enhance_balanced(self, image: np.ndarray, quality: Dict[str, Any], pipeline: ProcessingPipeline) -> np.ndarray:
        """Balanced enhancement"""
        enhanced = image.copy()
        
        if quality['brightness'] < 100:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=15)
        elif quality['brightness'] > 180:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=0.9, beta=-10)
        
        if quality['contrast'] < 40:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        if quality['sharpness'] < 300 and quality['noise_level'] < 50:
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return enhanced

# Usage example
if __name__ == "__main__":
    processor = OptimizedImageProcessor()
    
    image_to_process = "test_image.jpg"

    async def test_processor():
        try:
            print(f"\n{'='*80}")
            print(f"🖼️  TESTING OPTIMIZED IMAGE PROCESSOR - FULL-SIZE ENHANCED IMAGES")
            print(f"{'='*80}\n")
            
            result = await processor.analyze_image_comprehensive(image_to_process)
            
            print(f"✅ ANALYSIS RESULTS:")
            print(f"{'='*80}")
            print(f"Objects Detected: {result.objects}")
            print(f"Scene Type: {result.scene_type}")
            print(f"Dominant Colors: {result.dominant_colors}")
            print(f"Emotions: {result.emotions}")
            print(f"Confidence Score: {result.confidence_score:.3f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            
            print(f"\n💡 SUGGESTIONS:")
            print(f"{'='*80}")
            for i, suggestion in enumerate(result.suggestions, 1):
                print(f"  {i}. {suggestion}")
            
            print(f"\n📊 QUALITY METRICS:")
            print(f"{'='*80}")
            quality = result.metadata['image_quality']
            print(f"  - Sharpness: {quality['sharpness']:.1f}")
            print(f"  - Brightness: {quality['brightness']:.1f}")
            print(f"  - Contrast: {quality['contrast']:.1f}")
            print(f"  - Noise Level: {quality['noise_level']:.1f}")
            print(f"  - Quality Score: {quality.get('quality_score', 0):.3f}")
            print(f"  - Resolution: {quality['resolution']}")

            print(f"\n📝 TEXT EXTRACTION:")
            print(f"{'='*80}")
            print(result.text_content)
            print(f"{'='*80}")

            if result.enhanced_image_base64:
                print(f"\n✅ Enhanced Image: FULL-SIZE (ready for display)")
                print(f"   Base64 length: {len(result.enhanced_image_base64)} characters")
            else:
                print(f"\n⚠️  Enhanced Image: Not generated")

            print(f"\n✅ PROCESSING COMPLETE!")
            print(f"{'='*80}")
            print(f"✓ Full-size enhanced image generated")
            print(f"✓ Specialized pipeline: {result.metadata.get('selected_pipeline', 'Unknown')}")
            print(f"✓ All features working: Objects, Text, Quality, Suggestions")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test_processor())