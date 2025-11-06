# test_gemini_models.py
import google.generativeai as genai
import os

genai.configure(api_key=os.environ['GEMINI_KEY'])

# Test different model name formats
model_names = [
    "gemini-1.5-pro",
    "models/gemini-1.5-pro", 
    "gemini-1.5-pro-latest",
    "models/gemini-1.5-pro-latest",
    "gemini-pro"
]

print("Testing Gemini model names...")
print("=" * 60)

for name in model_names:
    try:
        model = genai.GenerativeModel(name)
        response = model.generate_content("Say 'hello'")
        print(f"✅ {name} - WORKS!")
    except Exception as e:
        error_msg = str(e)[:80]
        print(f"❌ {name} - {error_msg}")

print("=" * 60)
print("Use whichever model name shows ✅ in your config.py")
