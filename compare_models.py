import ollama
import time

def test_model(model_name, prompt):
    """Test a model and return response + timing"""
    start = time.time()
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        elapsed = time.time() - start
        
        return {
            'model': model_name,
            'response': response['message']['content'],
            'time': elapsed,
            'tokens': response.get('eval_count', 0)
        }
    except Exception as e:
        return {
            'model': model_name,
            'error': str(e),
            'time': time.time() - start
        }

# Test prompt
prompt = """
Explain what an API is in exactly 2 sentences.
Make it understandable for someone with no technical background.
"""

# Models to compare (make sure these are installed)
models = ['llama3.2', 'mistral', 'phi']

print("PROMPT:")
print(prompt)
print("\n" + "="*60 + "\n")

for model in models:
    result = test_model(model, prompt)
    
    print(f"Model: {result['model']}")
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Response: {result['response']}")
        print(f"Time: {result['time']:.2f}s")
        print(f"Tokens: {result['tokens']}")
    print("\n" + "-"*60 + "\n")