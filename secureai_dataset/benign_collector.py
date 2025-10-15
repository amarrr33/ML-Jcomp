"""High-quality benign image collection from real sources."""
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import random
import time


def download_real_image(path: Path, category="portraits", index=0):
    """Download real images from reliable sources."""
    # Use Lorem Picsum for reliable, high-quality placeholder images
    sources = [
        f"https://picsum.photos/512/512?random={index}",  # General images
        f"https://source.unsplash.com/512x512/?{category}",  # Category-specific
    ]
    
    for url in sources:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(response.content)
                return {"path": str(path), "label": "benign", "source": url, "category": category}
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    
    # If downloads fail, create a high-quality fallback
    return create_fallback_image(path, category, index)


def create_fallback_image(path: Path, category: str, index: int, size=(512, 512)):
    """Create high-quality fallback images when downloads fail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if category == "portraits":
        img = create_realistic_portrait(size, index)
    elif category == "landscapes":
        img = create_realistic_landscape(size, index)
    elif category == "documents":
        img = create_realistic_document(size, index)
    elif category == "screenshots":
        img = create_realistic_screenshot(size, index)
    else:
        img = create_realistic_portrait(size, index)
    
    img.save(path, quality=95)
    return {"path": str(path), "label": "benign", "category": category, "type": "fallback"}


def create_realistic_portrait(size=(512, 512), seed=0):
    """Create a realistic-looking portrait."""
    # Create gradient background
    img = Image.new('RGB', size, (245, 245, 220))  # Beige background
    draw = ImageDraw.Draw(img)
    
    # Add subtle gradient
    for y in range(size[1]):
        alpha = int(255 * (y / size[1]))
        color = (245 - alpha//10, 245 - alpha//10, 220 - alpha//10)
        draw.line([(0, y), (size[0], y)], fill=color)
    
    # Draw face oval
    center_x, center_y = size[0] // 2, size[1] // 2
    face_width = size[0] // 3
    face_height = int(size[1] * 0.4)
    
    # Skin tones
    skin_tones = [(245, 222, 196), (210, 180, 140), (160, 120, 90), (120, 90, 70)]
    skin_color = skin_tones[seed % len(skin_tones)]
    
    # Face
    draw.ellipse([center_x - face_width//2, center_y - face_height//2,
                  center_x + face_width//2, center_y + face_height//2], 
                 fill=skin_color, outline=(100, 100, 100))
    
    # Eyes
    eye_y = center_y - face_height//4
    eye_offset = face_width//4
    for eye_x in [center_x - eye_offset, center_x + eye_offset]:
        # Eye white
        draw.ellipse([eye_x-12, eye_y-8, eye_x+12, eye_y+8], fill=(255, 255, 255))
        # Iris
        iris_colors = [(70, 130, 180), (139, 69, 19), (0, 128, 0), (105, 105, 105)]
        iris_color = iris_colors[seed % len(iris_colors)]
        draw.ellipse([eye_x-7, eye_y-5, eye_x+7, eye_y+5], fill=iris_color)
        # Pupil
        draw.ellipse([eye_x-3, eye_y-3, eye_x+3, eye_y+3], fill=(0, 0, 0))
    
    # Nose
    nose_y = center_y
    draw.polygon([(center_x, nose_y-8), (center_x-4, nose_y+8), (center_x+4, nose_y+8)], 
                 outline=(100, 100, 100))
    
    # Mouth
    mouth_y = center_y + face_height//4
    draw.arc([center_x-15, mouth_y-8, center_x+15, mouth_y+8], 0, 180, fill=(139, 69, 19), width=3)
    
    # Hair
    hair_colors = [(139, 69, 19), (0, 0, 0), (255, 215, 0), (165, 42, 42)]
    hair_color = hair_colors[seed % len(hair_colors)]
    draw.ellipse([center_x - face_width//2 - 20, center_y - face_height//2 - 40,
                  center_x + face_width//2 + 20, center_y], fill=hair_color)
    
    return img


def create_realistic_landscape(size=(512, 512), seed=0):
    """Create a realistic landscape."""
    img = Image.new('RGB', size, (135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)
    
    # Sky gradient
    for y in range(size[1]//2):
        blue_val = 235 - int(y * 0.3)
        draw.line([(0, y), (size[0], y)], fill=(135, 206, max(blue_val, 180)))
    
    # Mountains
    mountain_points = []
    for x in range(0, size[0], 50):
        height = size[1]//2 + random.randint(-80, -20) + (seed * 13) % 60
        mountain_points.append((x, height))
    mountain_points.append((size[0], size[1]))
    mountain_points.append((0, size[1]))
    
    draw.polygon(mountain_points, fill=(34, 139, 34))
    
    # Trees
    for i in range(10):
        x = (i * size[0]//10) + random.randint(-20, 20)
        tree_height = random.randint(40, 80)
        tree_y = size[1] - tree_height
        # Trunk
        draw.rectangle([x-5, tree_y+tree_height-20, x+5, size[1]], fill=(139, 69, 19))
        # Leaves
        draw.ellipse([x-15, tree_y, x+15, tree_y+tree_height-20], fill=(0, 100, 0))
    
    # Sun
    sun_x, sun_y = size[0] - 80, 60
    draw.ellipse([sun_x-25, sun_y-25, sun_x+25, sun_y+25], fill=(255, 255, 0))
    
    return img


def create_realistic_document(size=(512, 512), seed=0):
    """Create a realistic document."""
    img = Image.new('RGB', size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Document header
    draw.rectangle([20, 20, size[0]-20, 60], fill=(240, 248, 255))
    draw.text((30, 30), "BUSINESS REPORT - CONFIDENTIAL", fill=(0, 0, 139), font=font)
    
    # Content lines
    content = [
        "Executive Summary",
        "",
        "Q3 Performance Analysis:",
        "• Revenue: $2.3M (+12% YoY)",
        "• Customer Growth: 15,000 new users",
        "• Market Share: Increased to 8.2%",
        "",
        "Key Initiatives:",
        "1. Product Development Roadmap",
        "2. International Expansion Plan",
        "3. Customer Retention Strategy",
        "",
        "Risk Assessment:",
        "- Competitive pressure increasing",
        "- Supply chain considerations",
        "- Regulatory compliance updates",
        "",
        "Recommendations:",
        "Continue current growth trajectory",
        "Invest in R&D for next quarter",
        "",
        "Classification: INTERNAL USE ONLY"
    ]
    
    y = 80
    for line in content:
        if line.startswith("Executive") or line.startswith("Q3") or line.startswith("Key") or line.startswith("Risk") or line.startswith("Recommendations"):
            draw.text((25, y), line, fill=(139, 0, 0), font=font)  # Red headers
        elif line.startswith("•") or line.startswith("-"):
            draw.text((40, y), line, fill=(0, 0, 0), font=font)
        elif line.startswith(("1.", "2.", "3.")):
            draw.text((40, y), line, fill=(0, 0, 139), font=font)
        elif line.startswith("Classification"):
            draw.rectangle([20, y-2, size[0]-20, y+18], fill=(255, 0, 0))
            draw.text((25, y), line, fill=(255, 255, 255), font=font)
        else:
            draw.text((25, y), line, fill=(0, 0, 0), font=font)
        y += 20
        if y > size[1] - 40:
            break
    
    # Border
    draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=(0, 0, 0), width=2)
    
    return img


def create_realistic_screenshot(size=(512, 512), seed=0):
    """Create a realistic application screenshot."""
    img = Image.new('RGB', size, (248, 248, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Title bar
    draw.rectangle([0, 0, size[0], 30], fill=(70, 130, 180))
    draw.text((10, 8), "SecureAI Personal Assistant v2.1", fill=(255, 255, 255), font=font)
    
    # Navigation bar
    draw.rectangle([0, 30, size[0], 60], fill=(230, 230, 250))
    nav_items = ["Dashboard", "Calendar", "Messages", "Settings"]
    x = 20
    for item in nav_items:
        draw.text((x, 38), item, fill=(0, 0, 0), font=font)
        x += len(item) * 8 + 30
    
    # Main content area
    draw.rectangle([10, 70, size[0]-10, size[1]-10], fill=(255, 255, 255), outline=(200, 200, 200))
    
    # Dashboard widgets
    widgets = [
        ("Today's Schedule", 20, 90),
        ("• 9:00 AM - Team Meeting", 30, 110),
        ("• 11:30 AM - Client Call", 30, 130),
        ("• 2:00 PM - Project Review", 30, 150),
        ("", 20, 170),
        ("Recent Messages", 20, 190),
        ("📧 New email from sarah@company.com", 30, 210),
        ("💬 Slack: 3 unread messages", 30, 230),
        ("📱 SMS: Appointment reminder", 30, 250),
        ("", 20, 270),
        ("Quick Actions", 20, 290),
    ]
    
    for text, x, y in widgets:
        if text.startswith(("Today's", "Recent", "Quick")):
            draw.text((x, y), text, fill=(70, 130, 180), font=font)
        else:
            draw.text((x, y), text, fill=(0, 0, 0), font=font)
    
    # Action buttons
    buttons = ["Schedule Meeting", "Check Email", "Set Reminder"]
    y = 310
    for button in buttons:
        btn_width = len(button) * 8 + 20
        draw.rectangle([30, y, 30 + btn_width, y + 25], fill=(100, 149, 237), outline=(70, 130, 180))
        draw.text((40, y + 5), button, fill=(255, 255, 255), font=font)
        y += 35
    
    return img


def generate_benign_image(path: Path, size=(512, 512), seed=0):
    """Main function to generate high-quality benign images."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine category from path
    if "portraits" in str(path):
        # Try real download first, fallback to synthetic
        result = download_real_image(path, "portraits", seed)
        if result and result.get("source"):
            return result
        return create_fallback_image(path, "portraits", seed, size)
    elif "landscapes" in str(path):
        result = download_real_image(path, "landscapes", seed)
        if result and result.get("source"):
            return result
        return create_fallback_image(path, "landscapes", seed, size)
    elif "documents" in str(path):
        return create_fallback_image(path, "documents", seed, size)
    elif "screenshots" in str(path):
        return create_fallback_image(path, "screenshots", seed, size)
    else:
        # Default to portraits with real download attempt
        result = download_real_image(path, "portraits", seed)
        if result and result.get("source"):
            return result
        return create_fallback_image(path, "portraits", seed, size)