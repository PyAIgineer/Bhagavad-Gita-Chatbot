import gradio as gr
import requests
import os
import threading
import time
from typing import Dict, Any, List

# Configuration
API_URL = "http://localhost:8000"  # FastAPI backend URL

def query_chatbot(message):
    """Send a query to the chatbot API and return the response."""
    try:
        response = requests.post(f"{API_URL}/query", json={"query": message})
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        
        if data["type"] == "verse_lookup":
            result = data["result"]
            if "error" in result:
                return f"Error: {result['error']}"
            
            # Format the verse information
            chapter = result.get("chapter", "N/A")
            verse = result.get("verse", "N/A")
            
            formatted_response = f"📜 **Chapter {chapter}, Verse {verse}**\n\n"
            
            if "sanskrit" in result and result["sanskrit"]:
                formatted_response += f"**Sanskrit:**\n{result['sanskrit']}\n\n"
            
            if "transliteration" in result and result["transliteration"]:
                formatted_response += f"**Transliteration:**\n{result['transliteration']}\n\n"
            
            if "translation" in result and result["translation"]:
                formatted_response += f"**Translation:**\n{result['translation']}\n\n"
            
            if "purport" in result and result["purport"]:
                purport = result["purport"]
                
                # Check if this is a purport-specific query
                if "purport" in message.lower():
                    # Show full purport when specifically requested
                    formatted_response += f"**Purport:**\n{purport}\n\n"
                elif len(purport) > 1000:
                    # Otherwise truncate if very long
                    formatted_response += f"**Purport (excerpt):**\n{purport[:1000]}...\n\n(Full purport available on request)"
                else:
                    formatted_response += f"**Purport:**\n{purport}\n\n"
            
            return formatted_response
            
        elif data["type"] == "keyword_search":
            results = data["results"]
            if not results:
                return "No results found for your search query."
                
            formatted_response = f"🔍 **Found {len(results)} matches:**\n\n"
            
            for i, result in enumerate(results, 1):
                chapter = result.get("chapter", "N/A")
                verse = result.get("verse", "N/A")
                translation = result.get("translation", "")
                
                # Truncate translation if it's long
                if len(translation) > 150:
                    translation = translation[:150] + "..."
                
                formatted_response += f"{i}. Chapter {chapter}, Verse {verse}: {translation}\n\n"
            
            formatted_response += "Ask for a specific verse to see details."
            return formatted_response
            
        elif data["type"] == "error":
            return f"❌ Error: {data['message']}"
        
        else:
            return f"Unexpected response format: {data}"
            
    except Exception as e:
        return f"❌ Error communicating with the chatbot API: {str(e)}"

def check_backend_ready():
    """Check if the backend is ready before starting the UI."""
    max_retries = 30
    retry_interval = 2  # seconds
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/status/ready")
            if response.status_code == 200 and response.json().get("ready", False):
                print("Backend is ready!")
                return True
            else:
                print(f"Backend not ready yet. Retrying in {retry_interval} seconds...")
        except Exception as e:
            print(f"Error checking backend status: {e}")
        
        time.sleep(retry_interval)
    
    print("Timed out waiting for backend to be ready. Starting UI anyway...")
    return False

def create_gradio_app():
    """Create a Gradio interface that operates independently."""
    with gr.Blocks() as app:
        gr.Markdown("# 📚 Religious Book Chatbot")
        gr.Markdown("Ask questions about verses or search for keywords in religious texts")
        
        # Create a simple chat interface
        with gr.Row():
            with gr.Column():
                chatbox = gr.Chatbot(
                    value=[], 
                    elem_id="chatbox",
                    height=500,
                    label="Conversation with 🧘 Bhagavad Gita Bot",
                    avatar_images=("👤", "🧘"),
                    type="messages"  # Use messages format instead of tuples
                )
        
        with gr.Row():
            message = gr.Textbox(
                label="Your Question",
                placeholder="E.g., 'What does chapter 2 verse 41 say?' or 'Search for karma'",
                lines=2
            )
            
        with gr.Row():
            submit = gr.Button("Ask 🔍")
            clear = gr.Button("Clear Chat 🗑️")
        
        # Simple examples
        gr.Examples(
            examples=[
                "What does Chapter 1 Verse 32 say?",
                "Show me the translation of verse 1.35",
                "Tell me the purport for chapter 1 verse 36",
                "Search for wisdom",
            ],
            inputs=message
        )
        
        # Define update function without using any shared state
        def update_chat(user_message, history):
            if not user_message.strip():
                return "", history
            
            # Get bot response directly
            bot_message = query_chatbot(user_message)
            
            # Update history
            history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": bot_message}]
            return "", history
        
        # Connect buttons with no queue
        submit.click(
            fn=update_chat,
            inputs=[message, chatbox],
            outputs=[message, chatbox],
            queue=False
        )
        
        message.submit(
            fn=update_chat,
            inputs=[message, chatbox],
            outputs=[message, chatbox],
            queue=False
        )
        
        clear.click(lambda: [], None, chatbox, queue=False)
        
    return app

def run_gradio(port=7860):
    """Run the Gradio app on the specified port."""
    # Wait for the backend to be ready before starting
    check_backend_ready()
    
    print("Starting Gradio UI...")
    app = create_gradio_app()
    app.launch(
        server_port=port, 
        share=False, 
        server_name="0.0.0.0",
        quiet=True
    )

# For direct execution
if __name__ == "__main__":
    run_gradio()