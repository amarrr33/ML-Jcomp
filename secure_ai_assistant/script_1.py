# Create requirements.txt file
requirements = """# Core dependencies
ollama
google-api-python-client
google-auth
google-auth-oauthlib
google-auth-httplib2
opencv-python
pillow
pytesseract
shap
lime
watchdog
pydantic
fastapi
uvicorn
sqlite-utils

# Additional dependencies for enhanced functionality
requests
numpy
scikit-image
scikit-learn
matplotlib
pandas
python-dotenv
click
colorama
tqdm
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements.strip())

print("✓ requirements.txt created")