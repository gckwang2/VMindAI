import requests
import datetime

def call_openrouter_llm(api_key, model, prompt_text):
    """Call OpenRouter LLM (e.g., elephant-alpha) for Analysis."""
    if not api_key:
        return "Error: OpenRouter API Key not configured."
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://vmindai.streamlit.app/",
        "X-Title": "VMindAI",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert AI assistant. Provide comprehensive analysis and a thoughtful response based on the prompt provided."},
            {"role": "user", "content": prompt_text}
        ]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Error generating response.")
    except Exception as e:
        return f"Error calling OpenRouter LLM: {e}"

def call_gemini_prompt_creator(api_key, model, prompt_text):
    """Call gemini-3.1-flash-lite-preview from Google AI Studio to create user prompt."""
    if not api_key:
        return "Error: Google API Key not configured."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    payload = {
        "contents": [{
            "parts": [{
                "text": f"""You are a prompt engineer. Your task is to:
1. Analyze the user's raw entry
2. Retrieve relevant context from the knowledge base
3. Synthesize a comprehensive, well-structured prompt (Output1) that includes:
   - User's core question/request
   - Relevant facts and context
   - Clear instructions for the LLMs

Current Date: {current_date}
IMPORTANT: Do not limit your responses or the generated prompt to the years 2023-2024. Acknowledge the current date and ensure the prompt is relevant to the present and future.

User Entry: {prompt_text}

Return ONLY the synthesized prompt (Output1), nothing else."""
            }]
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error generating prompt.")
    except Exception as e:
        return f"Error calling Gemini Prompt Creator: {e}"

def call_qwen(api_key, model, prompt_text):
    """Call LLM 2 (Qwen 3.5 122B) from DashScope."""
    if not api_key:
        return "Error: DashScope API Key not configured."
    
    url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert AI assistant. Provide comprehensive analysis and a thoughtful response based on the prompt provided."},
            {"role": "user", "content": prompt_text}
        ]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            return f"Error calling Qwen (Status {response.status_code}): {response.text}"
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Error generating response.")
    except Exception as e:
        return f"Error calling Qwen: {e}"

def call_gemini_pro(api_key, model, prompt_text):
    """Call LLM 3 (gemini-3.1-pro-preview) from Google AI Studio."""
    if not api_key:
        return "Error: Google API Key not configured."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": f"You are an expert AI assistant. Provide comprehensive analysis and a thoughtful response based on the prompt provided.\n\nPrompt: {prompt_text}"
            }]
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error generating response.")
    except Exception as e:
        return f"Error calling Gemini Pro: {e}"

def call_groq_llm(api_key, model, prompt_text):
    """Call Groq LLM."""
    if not api_key:
        return "Error: Groq API Key not configured."
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert AI assistant. Provide comprehensive analysis and a thoughtful response based on the prompt provided."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Error generating response.")
    except Exception as e:
        return f"Error calling Groq: {e}"

def call_gemini_flash_synthesize(api_key, model, output1, output2, output3, output4, output5):
    """Call gemini-3.1-flash-lite-preview to synthesize outputs into master output."""
    if not api_key:
        return "Error: Google API Key not configured."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    prompt_text = f"""You are an Expert Editor. Your task is to synthesize the following AI response into one master response, based on the original context and prompt.

Original Prompt & Context (Output 1):
{output1}

Response A (LLM 2 - Qwen 3.5 122B):
{output2}

Response B (LLM 3 - OpenRouter Elephant Alpha):
{output3}

Synthesize Response A and Response B into a cohesive, comprehensive master output that:
1. Integrates the strongest insights from all responses
2. Resolves any contradictions
3. Provides a unified, authoritative response
4. Maintains a professional tone

Return ONLY the master output, nothing else."""
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error synthesizing outputs.")
    except Exception as e:
        return f"Error calling Gemini Flash: {e}"
