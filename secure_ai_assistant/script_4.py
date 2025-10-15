# Create the image scaling attack detector
image_scaling_detector_code = '''"""
Image Scaling Attack Detector
Implements the multi-interpolation scaling attack detection from the Trail of Bits research
"""

import cv2
import numpy as np
import pytesseract
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from PIL import Image
import io


@dataclass
class ScalingAttackResult:
    """Result of image scaling attack detection"""
    is_attack: bool
    severity_score: float  # 0.0 - 1.0
    evidence: List[Dict]
    detected_text: str
    explanation: str
    method_used: str


class ImageScalingDetector:
    """Detects image scaling attacks using multi-interpolation analysis"""
    
    def __init__(self, config: Dict):
        self.config = config.get("security", {}).get("image_scaling", {})
        self.scale_factors = self.config.get("scale_factors", [0.25, 0.33, 0.5])
        self.ssim_threshold = self.config.get("ssim_threshold", 0.85)
        self.ocr_enabled = self.config.get("ocr_enabled", True)
        
        # Interpolation methods to test
        self.interpolation_methods = {
            "INTER_AREA": cv2.INTER_AREA,
            "INTER_LINEAR": cv2.INTER_LINEAR, 
            "INTER_CUBIC": cv2.INTER_CUBIC,
            "INTER_NEAREST": cv2.INTER_NEAREST
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _load_image(self, image_path_or_bytes) -> np.ndarray:
        """Load image from file path or bytes"""
        if isinstance(image_path_or_bytes, str):
            return cv2.imread(image_path_or_bytes)
        elif isinstance(image_path_or_bytes, bytes):
            nparr = np.frombuffer(image_path_or_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise ValueError("Input must be file path or bytes")
    
    def _resize_with_method(self, image: np.ndarray, scale_factor: float, 
                          interpolation: int) -> np.ndarray:
        """Resize image with specific interpolation method"""
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        
        # Downscale
        downscaled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        # Upscale back to original size
        upscaled = cv2.resize(downscaled, (width, height), interpolation=cv2.INTER_LINEAR)
        
        return upscaled
    
    def _extract_text_ocr(self, image: np.ndarray) -> str:
        """Extract text using OCR"""
        if not self.ocr_enabled:
            return ""
        
        try:
            # Convert to grayscale for better OCR
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply some preprocessing for better OCR
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            text = pytesseract.image_to_string(gray, config='--psm 6')
            return text.strip()
            
        except Exception as e:
            self.logger.warning(f"OCR failed: {e}")
            return ""
    
    def _analyze_scaling_pair(self, original: np.ndarray, processed: np.ndarray, 
                            method: str, scale_factor: float) -> Dict:
        """Analyze a single scaling operation"""
        
        # Convert to grayscale for analysis
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            proc_gray = processed
        
        # Calculate structural similarity
        ssim_score = ssim(orig_gray, proc_gray)
        
        # Calculate PSNR
        psnr_score = psnr(orig_gray, proc_gray)
        
        # Extract text from both versions
        orig_text = self._extract_text_ocr(original)
        proc_text = self._extract_text_ocr(processed)
        
        # Check for text differences (potential hidden content revelation)
        text_diff = proc_text != orig_text
        new_text_revealed = len(proc_text.strip()) > len(orig_text.strip())
        
        return {
            "method": method,
            "scale_factor": scale_factor,
            "ssim": float(ssim_score),
            "psnr": float(psnr_score),
            "original_text": orig_text,
            "processed_text": proc_text,
            "text_changed": text_diff,
            "new_text_revealed": new_text_revealed,
            "suspicious": ssim_score < self.ssim_threshold or text_diff or new_text_revealed
        }
    
    def detect_attack(self, image_input) -> ScalingAttackResult:
        """
        Main detection method
        
        Args:
            image_input: Image file path, bytes, or numpy array
            
        Returns:
            ScalingAttackResult with detection results
        """
        try:
            # Load image
            if isinstance(image_input, np.ndarray):
                original = image_input
            else:
                original = self._load_image(image_input)
            
            if original is None:
                raise ValueError("Could not load image")
            
            evidence = []
            all_revealed_text = []
            max_severity = 0.0
            
            # Test different interpolation methods and scale factors
            for method_name, method_cv in self.interpolation_methods.items():
                for scale_factor in self.scale_factors:
                    try:
                        # Process image with this method and scale
                        processed = self._resize_with_method(original, scale_factor, method_cv)
                        
                        # Analyze the result
                        analysis = self._analyze_scaling_pair(
                            original, processed, method_name, scale_factor
                        )
                        
                        if analysis["suspicious"]:
                            evidence.append(analysis)
                            
                            # Calculate severity based on multiple factors
                            severity = 0.0
                            if analysis["ssim"] < 0.7:
                                severity += 0.3
                            if analysis["new_text_revealed"]:
                                severity += 0.4
                            if analysis["text_changed"]:
                                severity += 0.3
                            
                            max_severity = max(max_severity, severity)
                            
                            # Collect revealed text
                            if analysis["processed_text"]:
                                all_revealed_text.append(analysis["processed_text"])
                    
                    except Exception as e:
                        self.logger.warning(f"Analysis failed for {method_name} @ {scale_factor}: {e}")
                        continue
            
            # Determine if this is an attack
            is_attack = len(evidence) > 0 and max_severity > 0.2
            
            # Generate explanation
            explanation = self._generate_explanation(evidence, is_attack, max_severity)
            
            # Find the method that revealed the most suspicious content
            method_used = "multi-interpolation"
            if evidence:
                best_evidence = max(evidence, key=lambda x: len(x.get("processed_text", "")))
                method_used = f"{best_evidence['method']} @ {best_evidence['scale_factor']}x"
            
            return ScalingAttackResult(
                is_attack=is_attack,
                severity_score=max_severity,
                evidence=evidence[:3],  # Limit evidence to top 3 for readability
                detected_text=" ".join(set(all_revealed_text)),
                explanation=explanation,
                method_used=method_used
            )
            
        except Exception as e:
            self.logger.error(f"Image scaling detection failed: {e}")
            return ScalingAttackResult(
                is_attack=False,
                severity_score=0.0,
                evidence=[],
                detected_text="",
                explanation=f"Detection failed: {str(e)}",
                method_used="error"
            )
    
    def _generate_explanation(self, evidence: List[Dict], is_attack: bool, 
                            severity: float) -> str:
        """Generate human-readable explanation"""
        if not is_attack:
            return "No scaling attack detected. Image appears benign across all tested interpolation methods."
        
        explanations = []
        
        for ev in evidence[:2]:  # Top 2 pieces of evidence
            method = ev["method"]
            scale = ev["scale_factor"]
            
            if ev["new_text_revealed"]:
                explanations.append(
                    f"Hidden text revealed using {method} interpolation at {scale}x scale"
                )
            elif ev["text_changed"]:
                explanations.append(
                    f"Text content changed after {method} processing at {scale}x scale"
                )
            elif ev["ssim"] < 0.7:
                explanations.append(
                    f"Significant structural changes detected with {method} at {scale}x scale (SSIM: {ev['ssim']:.2f})"
                )
        
        severity_desc = "low"
        if severity > 0.6:
            severity_desc = "high"
        elif severity > 0.3:
            severity_desc = "medium"
        
        base_explanation = f"Scaling attack detected with {severity_desc} severity ({severity:.2f}). "
        return base_explanation + ". ".join(explanations) + "."
    
    def quick_check(self, image_input) -> bool:
        """Quick check for scaling attacks (faster, less thorough)"""
        result = self.detect_attack(image_input)
        return result.is_attack
'''

with open('image_scaling_detector.py', 'w') as f:
    f.write(image_scaling_detector_code)

print("✓ image_scaling_detector.py created")