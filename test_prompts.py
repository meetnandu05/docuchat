import requests
import json

API_URL = "http://localhost:8000/chat"

def test_prompt(description, messages, temperature=0.7):
    """Test a prompt and display results"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(
            API_URL,
            json={
                "messages": messages,
                "temperature": temperature
            }
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        result = response.json()
        
        # Debug: print the actual response structure
        # print(f"DEBUG - Full response: {result}")
        
        # Handle different response formats
        if 'response' in result:
            response_text = result['response']
        elif 'detail' in result:
            response_text = f"Error: {result['detail']}"
        else:
            response_text = str(result)
        
        print(f"\nResponse:\n{response_text}")
        print(f"\nTokens - Prompt: {result.get('prompt_tokens')}, Completion: {result.get('completion_tokens')}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")
        return None
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return None

# Test 1: Zero-shot vs Few-shot
print("\n" + "="*60)
print("COMPARISON: Zero-shot vs Few-shot")
print("="*60)

# Zero-shot
test_prompt(
    "Zero-shot: Extract email",
    [
        {"role": "user", "content": "Extract the email from: Contact John at john.doe@example.com for details"}
    ]
)

# Few-shot
test_prompt(
    "Few-shot: Extract email",
    [
        {"role": "user", "content": """
Extract email addresses from text.

Text: "Reach out to Sarah at sarah@company.com"
Email: sarah@company.com

Text: "Contact John at john.doe@example.com for details"
Email:
        """}
    ]
)

# Test 2: Temperature comparison
print("\n" + "="*60)
print("COMPARISON: Temperature Effects")
print("="*60)

prompt = [{"role": "user", "content": "Write a creative opening line for a sci-fi story."}]

test_prompt("Low temperature (0.2)", prompt, temperature=0.2)
test_prompt("Medium temperature (0.7)", prompt, temperature=0.7)
test_prompt("High temperature (1.5)", prompt, temperature=1.5)

# Test 3: Role-based prompting
print("\n" + "="*60)
print("COMPARISON: With and without role")
print("="*60)

code = """
def process_data(data):
    result = []
    for i in data:
        if i > 0:
            result.append(i * 2)
    return result
"""

test_prompt(
    "Without role",
    [{"role": "user", "content": f"Review this code:\n{code}"}]
)

test_prompt(
    "With role",
    [
        {"role": "system", "content": "You are a senior Python developer. Review code focusing on performance, readability, and best practices."},
        {"role": "user", "content": f"Review this code:\n{code}"}
    ]
)

# Test 4: Chain-of-thought
print("\n" + "="*60)
print("COMPARISON: Direct vs Chain-of-thought")
print("="*60)

test_prompt(
    "Direct question",
    [{"role": "user", "content": "Is 127 a prime number?"}]
)

test_prompt(
    "Chain-of-thought",
    [{"role": "user", "content": """
Is 127 a prime number? Let's think step by step:
1) First, recall what makes a number prime
2) Check if 127 is divisible by small primes
3) Conclude whether it's prime
    """}]
)