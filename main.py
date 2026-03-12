"""
Facial Expression Detection Program
Detects emotions: Happy, Sad, Angry, Surprised, Neutral, Fear, Disgust
"""

import cv2
import numpy as np
from deepface import DeepFace
import time
from collections import deque
import threading

class ExpressionDetector:
    def __init__(self, ip_address=None):
        self.ip_address = ip_address
        self.cap = None
        self.expression_buffer = deque(maxlen=10)  # Store last 10 predictions for smoothing
        self.frame_count = 0
        self.process_every_n_frames = 3  # Process every 3rd frame for performance
        self.current_expression = "Analyzing..."
        self.current_confidence = 0.0
        self.lock = threading.Lock()
        
        # Color scheme for expressions
        self.expression_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'surprise': (0, 255, 255), # Yellow
            'neutral': (200, 200, 200),# Gray
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 128, 128)   # Olive
        }
        
        # Emoji representations
        self.expression_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😮',
            'neutral': '😐',
            'fear': '😨',
            'disgust': '🤢'
        }
    
    def setup_capture(self):
        """Initialize video capture"""
        if self.ip_address:
            url = f"https://{self.ip_address}/video"
            self.cap = cv2.VideoCapture(url)
        else:
            self.cap = cv2.VideoCapture(0)  # Default webcam
        
        if not self.cap.isOpened():
            raise Exception("Failed to open video capture")
        
        # Optimize capture settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def detect_expression(self, frame):
        """Detect facial expression using DeepFace"""
        try:
            # Analyze the frame
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            # Handle both list and dict returns
            if isinstance(result, list):
                result = result[0]
            
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            confidence = emotions[dominant_emotion]
            
            return dominant_emotion, confidence, emotions
            
        except Exception as e:
            return None, 0, None
    
    def smooth_prediction(self, expression):
        """Smooth predictions using a buffer"""
        self.expression_buffer.append(expression)
        
        # Get most common expression in buffer
        if len(self.expression_buffer) > 0:
            return max(set(self.expression_buffer), key=self.expression_buffer.count)
        return expression
    
    def draw_ui(self, frame, expression, confidence, all_emotions=None):
        """Draw enhanced UI on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Top bar with expression
        color = self.expression_colors.get(expression, (200, 200, 200))
        cv2.rectangle(overlay, (0, 0), (width, 80), color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Main expression text
        emoji = self.expression_emojis.get(expression, '')
        text = f"{emoji} {expression.upper()}"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Confidence bar
        bar_width = int((width - 40) * (confidence / 100))
        cv2.rectangle(frame, (20, height - 60), (20 + bar_width, height - 40), color, -1)
        cv2.rectangle(frame, (20, height - 60), (width - 20, height - 40), (255, 255, 255), 2)
        
        # Confidence text
        conf_text = f"Confidence: {confidence:.1f}%"
        cv2.putText(frame, conf_text, (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw all emotion scores on the right side
        if all_emotions:
            y_offset = 100
            cv2.putText(frame, "All Emotions:", (width - 200, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            # Sort emotions by confidence
            sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
            
            for emotion, score in sorted_emotions:
                emoji = self.expression_emojis.get(emotion, '')
                text = f"{emoji} {emotion}: {score:.1f}%"
                cv2.putText(frame, text, (width - 200, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit | 'S' to save screenshot", 
                   (20, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("🎭 Facial Expression Detector")
        print("=" * 50)
        print("Starting camera...")
        
        self.setup_capture()
        
        cv2.namedWindow("Expression Detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Expression Detector", 800, 600)
        
        print("✓ Camera ready!")
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Save screenshot")
        print("\nDetecting expressions...\n")
        
        screenshot_count = 0
        
        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process expression detection every N frames
            if self.frame_count % self.process_every_n_frames == 0:
                expression, confidence, all_emotions = self.detect_expression(frame)
                
                if expression:
                    smoothed_expression = self.smooth_prediction(expression)
                    
                    with self.lock:
                        self.current_expression = smoothed_expression
                        self.current_confidence = confidence
                        self.all_emotions = all_emotions
            
            # Draw UI with current expression
            with self.lock:
                frame = self.draw_ui(
                    frame, 
                    self.current_expression, 
                    self.current_confidence,
                    getattr(self, 'all_emotions', None)
                )
            
            cv2.imshow("Expression Detector", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Goodbye!")
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            
            self.frame_count += 1
            
            # Maintain frame rate
            elapsed_time = time.time() - start_time
            remaining_time = max(0, (1 / 30) - elapsed_time)
            time.sleep(remaining_time)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    print("\n" + "="*50)
    print("   FACIAL EXPRESSION DETECTION PROGRAM")
    print("="*50 + "\n")
    
    choice = input("Choose input source:\n  1. Webcam\n  2. IP Camera\n\nEnter choice (1 or 2): ").strip()
    
    ip_address = None
    if choice == '2':
        ip_address = input("Enter IP Address: ").strip()
    
    try:
        detector = ExpressionDetector(ip_address)
        detector.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure your camera is connected")
        print("  - For IP camera, check the IP address and connection")
        print("  - Install required packages: pip install deepface opencv-python tf-keras")


if __name__ == "__main__":
    main()
