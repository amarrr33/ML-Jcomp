# Create the main configuration file
config_yaml = """# SecureAI Personal Assistant Configuration

# Local LLM Settings
llm:
  provider: "ollama"
  model: "llama3.2:3b-instruct-q4_0"
  endpoint: "http://localhost:11434"
  max_tokens: 2048
  temperature: 0.7
  system_prompt: |
    You are a helpful personal assistant that prioritizes privacy and security.
    Always ask for explicit user confirmation before performing sensitive actions.
    Never execute commands that could harm the system or compromise security.

# Gemini Fallback Settings
gemini:
  enabled: true
  model: "gemini-pro"
  rate_limit: 60  # requests per minute
  fallback_threshold: 0.8  # confidence threshold to trigger fallback

# Security Settings
security:
  image_scaling:
    enabled: true
    interpolation_methods: ["INTER_AREA", "INTER_LINEAR", "INTER_CUBIC"]
    scale_factors: [0.25, 0.33, 0.5]
    ssim_threshold: 0.85
    ocr_enabled: true
  
  text_injection:
    enabled: true
    severity_thresholds:
      low: 0.3
      medium: 0.6
      high: 0.8
    patterns:
      - "ignore previous"
      - "override instructions"
      - "exfiltrate"
      - "send to"
      - "disable safety"
      - "you must follow"
      - "system prompt:"
      - "base64 decode and run"
  
  xai:
    enabled: true
    max_features: 6
    cache_explanations: true

# Integration Settings
integrations:
  calendar:
    enabled: true
    provider: "google"
    scopes: ["https://www.googleapis.com/auth/calendar"]
  
  files:
    enabled: true
    allowed_extensions: [".txt", ".md", ".pdf", ".docx", ".csv"]
    sandbox_root: "./data/sandbox"
    max_file_size: "10MB"
  
  voice:
    enabled: false
    provider: "system"

# Logging and Storage
logging:
  level: "INFO"
  file: "./logs/assistant.log"
  max_size: "10MB"
  backup_count: 5

database:
  path: "./data/assistant.db"
  backup_enabled: true
  backup_interval: "24h"

# UI Settings
ui:
  mode: "cli"  # cli, web, desktop
  theme: "dark"
  notifications: true
"""

with open('app_config.yaml', 'w') as f:
    f.write(config_yaml.strip())

print("✓ app_config.yaml created")