import google.generativeai as genai

genai.configure(api_key='AIzaSyB3HUscGjTeeUpttYa8y-bYM357joxp96Q')

print("Available Gemini models:")
print("=" * 60)

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")
            print(f"  Display name: {model.display_name}")
            print()
except Exception as e:
    print(f"Error listing models: {e}")

print("=" * 60)
