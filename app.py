"""
Gradio Chat Application with Multiple AI Models (Streaming Version)

This app lets you chat with either:
- Local Ollama models (llama3.2:3b or llama3.3:70b)
- Claude API (requires API key)

Features:
- Streaming responses (text appears word-by-word)
- Save/load conversations to JSON files

Run with: python app.py
Access at: http://localhost:7860 (or your machine's IP:7860 from other devices)
"""

import os
import json
from datetime import datetime
from pathlib import Path
import gradio as gr

# Load environment variables from .env file (if it exists)
# This lets you store your API key in a .env file instead of exporting it
from dotenv import load_dotenv
load_dotenv()

# Import the AI libraries
import ollama
import anthropic


# ============================================================================
# CONFIGURATION
# ============================================================================

# Available models - the keys are what users see, values are the actual model names
MODELS = {
    "Llama 3.2 3B - Meta's small, quick model (128K context)": "llama3.2:3b",
    "Llama 3.3 70B - Meta's powerful reasoning model (128K context)": "llama3.3:70b",
    "Claude Sonnet - Anthropic's balanced model (200K context, API key required)": "claude-sonnet-4-20250514",
}

# Folder where conversations are saved
CONVERSATIONS_DIR = Path("conversations")


# ============================================================================
# CONVERSATION SAVE/LOAD FUNCTIONS
# ============================================================================

def ensure_conversations_dir():
    """Create the conversations folder if it doesn't exist."""
    CONVERSATIONS_DIR.mkdir(exist_ok=True)


def generate_title(history: list, max_length: int = 40) -> str:
    """
    Generate a short capitalized title from the conversation's first user message.

    Args:
        history: The chat history
        max_length: Maximum characters for the title

    Returns:
        A short descriptive title with proper capitalization
    """
    # Find the first user message
    for msg in history:
        if msg.get("role") == "user":
            content = extract_text_content(msg.get("content", ""))
            # Clean up: remove newlines, extra spaces
            title = " ".join(content.split())
            # Truncate if too long, add ellipsis
            if len(title) > max_length:
                title = title[:max_length - 3].rsplit(" ", 1)[0] + "..."
            # Capitalize each word for proper title case
            return title.title()
    return "Untitled Conversation"


def save_conversation(history: list) -> str:
    """
    Save the current conversation to a JSON file.

    Args:
        history: The chat history (list of message dicts)

    Returns:
        A status message indicating success or failure
    """
    # Don't save empty conversations
    if not history:
        return "Nothing to save - conversation is empty"

    ensure_conversations_dir()

    # Generate a title from the first user message
    title = generate_title(history)

    # Create filename with timestamp: conversation_2026-01-17_14-30-45.json
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"conversation_{timestamp}.json"
    filepath = CONVERSATIONS_DIR / filename

    # Build the data to save
    # We save both the history and some metadata
    data = {
        "saved_at": datetime.now().isoformat(),
        "title": title,
        "message_count": len(history),
        "history": history,
    }

    # Write to file
    # indent=2 makes the JSON human-readable
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return f"Saved: {title}"


def get_saved_conversations() -> list:
    """
    Get a list of all saved conversation files with their titles.

    Returns:
        List of tuples (display_name, filename) for Gradio dropdown.
        Display name shows title + date, newest first.
    """
    ensure_conversations_dir()

    # Find all JSON files in the conversations folder
    # sorted() with reverse=True puts newest first (because of timestamp in name)
    files = sorted(CONVERSATIONS_DIR.glob("*.json"), reverse=True)

    choices = []
    for f in files:
        # Try to read the title from the JSON file
        try:
            with open(f, "r") as file:
                data = json.load(file)
                title = data.get("title", "Untitled")
                # Extract date from filename (conversation_2026-01-17_14-30-45.json)
                # Show as "Title - Jan 17, 14:30"
                date_part = f.stem.replace("conversation_", "")  # 2026-01-17_14-30-45
                date_obj = datetime.strptime(date_part, "%Y-%m-%d_%H-%M-%S")
                date_display = date_obj.strftime("%b %d, %H:%M")
                display_name = f"{title} - {date_display}"
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback to just filename if we can't parse
            display_name = f.name

        # Gradio dropdown with tuples: (display_name, actual_value)
        choices.append((display_name, f.name))

    return choices


def load_conversation(filename: str) -> list:
    """
    Load a conversation from a JSON file.

    Args:
        filename: Name of the file to load

    Returns:
        The chat history from that file, or empty list if error
    """
    if not filename:
        return []

    filepath = CONVERSATIONS_DIR / filename

    if not filepath.exists():
        return []

    with open(filepath, "r") as f:
        data = json.load(f)

    # Return the history portion of the saved data
    return data.get("history", [])


# ============================================================================
# AI CHAT FUNCTIONS
# ============================================================================

def check_claude_api_key() -> bool:
    """Check if the ANTHROPIC_API_KEY environment variable is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    return bool(api_key)


def extract_text_content(content) -> str:
    """
    Extract plain text from message content.

    Gradio's Chatbot can send content in different formats:
    - String: "hello" (older format)
    - List: [{"type": "text", "text": "hello"}] (newer format)

    This helper handles both cases.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list) and len(content) > 0:
        # Extract text from the first text block
        first_item = content[0]
        if isinstance(first_item, dict) and "text" in first_item:
            return first_item["text"]
    return str(content)  # Fallback: convert to string


def chat_with_ollama_stream(message: str, history: list, model_name: str):
    """
    Stream a response from a local Ollama model.

    Uses 'yield' to return partial responses as they're generated,
    so users see text appearing word-by-word.
    """
    messages = []
    for msg in history:
        # Use helper to handle different content formats
        text = extract_text_content(msg["content"])
        messages.append({"role": msg["role"], "content": text})
    messages.append({"role": "user", "content": message})

    try:
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True,
        )

        partial_response = ""
        for chunk in stream:
            chunk_text = chunk["message"]["content"]
            partial_response += chunk_text
            yield partial_response

    except ollama.ResponseError as e:
        yield f"Error: {e.error}\n\nMake sure Ollama is running and the model is downloaded:\n`ollama pull {model_name}`"

    except Exception as e:
        yield f"Error connecting to Ollama: {str(e)}\n\nMake sure Ollama is running: `ollama serve`"


# Store the last Claude API token usage for display
# This is updated after each Claude API call
last_token_usage = {
    "input_tokens": 0,
    "output_tokens": 0,
    "model": None,
}

# Claude Sonnet pricing (per million tokens) as of 2025
# See: https://www.anthropic.com/pricing
CLAUDE_PRICING = {
    "input_per_million": 3.00,   # $3 per million input tokens
    "output_per_million": 15.00,  # $15 per million output tokens
}


def chat_with_claude_stream(message: str, history: list):
    """
    Stream a response from Claude API.
    Captures token usage for cost estimation.
    """
    global last_token_usage

    if not check_claude_api_key():
        yield (
            "Claude requires an API key.\n\n"
            "To use Claude:\n"
            "1. Get an API key from https://console.anthropic.com\n"
            "2. Set it as an environment variable:\n"
            "   `export ANTHROPIC_API_KEY=your_key_here`\n"
            "3. Restart this app"
        )
        return

    # Explicitly pass the API key from environment variable
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    messages = []
    for msg in history:
        # Use helper to handle different content formats
        text = extract_text_content(msg["content"])
        messages.append({"role": msg["role"], "content": text})
    messages.append({"role": "user", "content": message})

    try:
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=messages,
        ) as stream:
            partial_response = ""
            for text in stream.text_stream:
                partial_response += text
                yield partial_response

            # After streaming completes, capture token usage from the final message
            final_message = stream.get_final_message()
            if final_message and final_message.usage:
                last_token_usage = {
                    "input_tokens": final_message.usage.input_tokens,
                    "output_tokens": final_message.usage.output_tokens,
                    "model": "Claude Sonnet",
                }

    except anthropic.AuthenticationError:
        yield "Authentication failed. Please check your API key."

    except Exception as e:
        yield f"Error calling Claude API: {str(e)}"


def chat(message: str, history: list, model_choice: str):
    """Main chat function that routes to the appropriate AI model."""
    model_name = MODELS[model_choice]

    if "claude" in model_choice.lower():
        yield from chat_with_claude_stream(message, history)
    else:
        yield from chat_with_ollama_stream(message, history, model_name)


def stream_response(message: str, history: list, model_choice: str):
    """
    Generator that yields tuples to update multiple outputs.

    Pattern from llm-council: yield partial response to Markdown component,
    then append complete response to Chatbot history when done.

    During streaming: streaming box visible & large, chatbot small
    When complete: streaming box hidden, chatbot full size
    """
    if not message.strip():
        # No message, return unchanged
        yield (
            gr.update(value=history, height=400),  # chatbot full size
            gr.update(visible=False),  # streaming_group hidden
            "",  # streaming_display
            "",  # status_display
            gr.Button(visible=False),  # stop_btn
        )
        return

    # Add user message to history immediately
    history = history + [{"role": "user", "content": message}]

    # Show status and stop button
    model_short = model_choice.split(" - ")[0]
    status = f"*Generating response with {model_short}...*"

    # Stream the response - streaming box prominent, chatbot shrunk
    partial_response = ""
    for chunk in chat(message, history[:-1], model_choice):  # history without current user msg
        partial_response = chunk
        yield (
            gr.update(value=history, height=150),  # chatbot small during streaming
            gr.update(visible=True),  # streaming_group visible
            partial_response,  # streaming_display shows response
            status,  # status_display
            gr.Button(visible=True),  # stop_btn
        )

    # Streaming complete - append assistant response to history
    history = history + [{"role": "assistant", "content": partial_response}]

    # Hide streaming box, expand chatbot to full size
    yield (
        gr.update(value=history, height=400),  # chatbot full size
        gr.update(visible=False),  # streaming_group hidden
        "",  # streaming_display cleared
        "",  # status_display cleared
        gr.Button(visible=False),  # stop_btn hidden
    )


# ============================================================================
# CUSTOM THEME (Professional Blue-Grey)
# ============================================================================

# Color palette - Clinical blue-grey theme for a professional medical aesthetic
COLORS = {
    # Backgrounds - light slate/blue-grey tones
    "bg_app": "#F8FAFC",        # Very light slate background
    "bg_primary": "#FFFFFF",    # White cards
    "bg_secondary": "#F1F5F9",  # Light slate surface
    "bg_tertiary": "#E2E8F0",   # Slightly darker slate
    # Text - dark slate for readability
    "text_primary": "#0F172A",  # Dark slate (near black)
    "text_secondary": "#334155", # Medium slate
    "text_muted": "#64748B",    # Muted slate
    "text_disabled": "#94A3B8", # Light slate
    # Borders
    "border": "#CBD5E1",        # Slate border
    "border_subtle": "#E2E8F0", # Subtle slate
    # Accents - professional blue
    "accent_primary": "#2563EB",  # Blue 600
    "accent_secondary": "#3B82F6", # Blue 500
    "accent_data": "#0EA5E9",    # Sky blue
    "accent_warning": "#D97706", # Amber
    # Semantic
    "success": "#059669",       # Emerald
    "warning": "#D97706",       # Amber
    "error": "#DC2626",         # Red
    "info": "#0EA5E9",          # Sky blue
}

# Custom CSS applying the Clinical Blue-Grey color palette
CUSTOM_CSS = """
/* Global app background - light slate */
.gradio-container {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

/* Main content area */
.main, .contain {
    background-color: #F8FAFC !important;
}

/* Headers and titles */
h1, h2, h3, h4, .markdown h1, .markdown h2 {
    color: #0F172A !important;
}

.markdown p, .markdown {
    color: #334155 !important;
}

/* Panels and containers - white cards */
.panel, .form, .block {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
}

/* Chatbot container */
.chatbot {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
}

/* User messages - Professional blue accent */
.message.user {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
}

/* Assistant messages - light slate surface */
.message.bot {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
}

/* Input textbox */
textarea, input[type="text"], .textbox {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
    outline: none !important;
}

textarea::placeholder, input::placeholder {
    color: #94A3B8 !important;
}

/* Dropdown */
.dropdown, select, .wrap {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

/* Dropdown options */
.dropdown-item, option {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}

.dropdown-item:hover, option:hover {
    background-color: #F1F5F9 !important;
}

/* Dropdown list container */
ul[role="listbox"], .options {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
}

/* Primary button - Professional blue */
.primary, button.primary {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

.primary:hover, button.primary:hover {
    background-color: #1D4ED8 !important;
}

/* Secondary button - white with slate border */
.secondary, button.secondary {
    background-color: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

.secondary:hover, button.secondary:hover {
    background-color: #F1F5F9 !important;
}

/* Default buttons */
button {
    background-color: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

button:hover {
    background-color: #F1F5F9 !important;
}

/* Labels */
label, .label-wrap {
    color: #334155 !important;
}

label span {
    color: #334155 !important;
}

.info {
    color: #64748B !important;
}

/* Status text */
.status {
    color: #0EA5E9 !important;
}

/* Dividers / horizontal rules */
hr {
    border-color: #E2E8F0 !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #F8FAFC;
}

::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #2563EB;
}

/* Streaming output area - prominent box for AI response during generation */
#streaming-output {
    min-height: 200px;
    max-height: 350px;
    overflow-y: auto;
    padding: 16px;
    background-color: #FFFFFF !important;
    border: 2px solid #2563EB !important;
    border-radius: 8px;
    margin: 8px 0;
    font-size: 1rem;
    line-height: 1.6;
}

#status {
    color: #2563EB !important;
    font-style: italic;
    font-weight: 500;
    padding: 4px 0;
}

/* Code blocks in chat */
pre, code {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
}

/* Selected dropdown item */
.selected {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
}

/* Links */
a {
    color: #2563EB !important;
}

a:hover {
    color: #1D4ED8 !important;
}
"""

# ============================================================================
# GRADIO THEME (using the color palette)
# ============================================================================

# Build a custom Gradio theme using Clinical Blue-Grey color palette
# Professional medical aesthetic with blues and greys
custom_theme = gr.themes.Base(
    # Primary colors - Professional blue for main interactive elements
    primary_hue=gr.themes.Color(
        c50="#EFF6FF",   # Lightest blue
        c100="#DBEAFE",
        c200="#BFDBFE",
        c300="#93C5FD",
        c400="#60A5FA",
        c500="#3B82F6",  # Blue 500
        c600="#2563EB",  # Primary action (Blue 600)
        c700="#1D4ED8",  # Hover state
        c800="#1E40AF",
        c900="#1E3A8A",
        c950="#172554",
    ),
    # Secondary colors - slate tones
    secondary_hue=gr.themes.Color(
        c50="#F8FAFC",   # App background
        c100="#F1F5F9",  # Secondary surface
        c200="#E2E8F0",  # Tertiary surface
        c300="#CBD5E1",  # Borders
        c400="#94A3B8",
        c500="#64748B",  # Muted text
        c600="#475569",  # Labels
        c700="#334155",  # Secondary text
        c800="#1E293B",  # Primary text
        c900="#0F172A",  # Darkest text
        c950="#020617",
    ),
    # Neutral colors - slate for backgrounds and text
    neutral_hue=gr.themes.Color(
        c50="#F8FAFC",   # App background
        c100="#F1F5F9",  # Secondary surface
        c200="#E2E8F0",  # Subtle dividers
        c300="#CBD5E1",  # Borders
        c400="#94A3B8",
        c500="#64748B",  # Muted/placeholder
        c600="#475569",  # Labels
        c700="#334155",  # Secondary text
        c800="#1E293B",
        c900="#0F172A",  # Primary text
        c950="#020617",
    ),
).set(
    # Body and backgrounds - light slate
    body_background_fill="#F8FAFC",
    body_background_fill_dark="#F8FAFC",  # Keep light even in dark mode
    body_text_color="#0F172A",
    body_text_color_subdued="#334155",

    # Blocks and panels (white cards)
    block_background_fill="#FFFFFF",
    block_border_color="#CBD5E1",
    block_border_width="1px",
    block_label_text_color="#334155",
    block_title_text_color="#0F172A",

    # Inputs
    input_background_fill="#FFFFFF",
    input_border_color="#CBD5E1",
    input_border_width="1px",
    input_placeholder_color="#94A3B8",

    # Buttons - Professional blue
    button_primary_background_fill="#2563EB",
    button_primary_background_fill_hover="#1D4ED8",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#2563EB",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F1F5F9",
    button_secondary_text_color="#334155",
    button_secondary_border_color="#CBD5E1",

    # Borders and shadows
    border_color_primary="#CBD5E1",
    border_color_accent="#2563EB",
    shadow_drop="none",  # Use borders instead of shadows
    shadow_spread="0px",
)

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Create the Gradio app
with gr.Blocks(title="AI Chat") as app:

    gr.Markdown("# Hello", elem_classes=["title"])
    gr.Markdown("Chat with local and cloud AI models", elem_classes=["subtitle"])

    # Organize interface into tabs
    with gr.Tabs():

        # ====================================================================
        # TAB 1: Chat - main chat interface
        # ====================================================================
        with gr.TabItem("Chat"):
            # Model selection dropdown
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Llama 3.2 3B - Meta's small, quick model (128K context)",
                label="Select Model",
                info="Local models require Ollama running. Claude requires API key.",
            )

            # Streaming display - shows response as it's generated (ABOVE conversation)
            # Starts hidden, becomes prominent during streaming
            with gr.Group(visible=False) as streaming_group:
                status_display = gr.Markdown("", elem_id="status")
                streaming_display = gr.Markdown(
                    value="",
                    elem_id="streaming-output",
                )

            # Chatbot displays completed conversation history
            # Gradio 6.x uses messages format {"role": ..., "content": ...} by default
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
            )

            # User input
            msg_input = gr.Textbox(
                label="Your message",
                placeholder="Type your message here...",
                lines=2,
                interactive=True,
            )

            # Example prompts
            gr.Examples(
                examples=[
                    "What are the key considerations for anticoagulation management perioperatively?",
                    "Compare laparoscopic vs robotic approaches for hernia repair",
                    "Help me draft a patient education handout for gallbladder surgery",
                    "Compare the pros and cons of local AI models vs cloud AI",
                ],
                inputs=msg_input,
                label="Try an example",
            )

            # Action buttons
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop", visible=False)
                clear_btn = gr.Button("Clear")

            # Token usage display (only shown for Claude API calls)
            token_usage_display = gr.Markdown("")

            # Save conversation controls
            gr.Markdown("---")
            with gr.Row():
                save_btn = gr.Button("Save Conversation", variant="secondary")
                save_status = gr.Textbox(
                    label="Status",
                    interactive=False,  # User can't type here, it's just for display
                    scale=2,
                )

        # ====================================================================
        # TAB 2: History - load saved conversations
        # ====================================================================
        with gr.TabItem("History"):
            gr.Markdown("### Load Saved Conversations")
            gr.Markdown("Select a previously saved conversation to continue chatting.")

            with gr.Row():
                # Dropdown to select a saved conversation
                load_dropdown = gr.Dropdown(
                    choices=get_saved_conversations(),
                    label="Load Saved Conversation",
                    info="Select a conversation to load",
                    scale=2,
                )
                refresh_btn = gr.Button("Refresh List")
                load_btn = gr.Button("Load", variant="secondary")

    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================
    # These connect buttons/inputs to functions

    def update_token_usage(model_choice: str):
        """
        Display token usage and estimated cost after Claude API calls.
        Returns empty string for non-Claude models.
        """
        # Only show token usage for Claude models
        if "claude" not in model_choice.lower():
            return ""

        # Check if we have token usage data
        if last_token_usage["model"] is None:
            return ""

        input_tokens = last_token_usage["input_tokens"]
        output_tokens = last_token_usage["output_tokens"]
        total_tokens = input_tokens + output_tokens

        # Calculate estimated cost
        input_cost = (input_tokens / 1_000_000) * CLAUDE_PRICING["input_per_million"]
        output_cost = (output_tokens / 1_000_000) * CLAUDE_PRICING["output_per_million"]
        total_cost = input_cost + output_cost

        # Format the display string
        return (
            f"**Token Usage:** {total_tokens:,} tokens "
            f"({input_tokens:,} input, {output_tokens:,} output) · "
            f"**Est. Cost:** ${total_cost:.4f}"
        )

    def save_chat(history: list):
        """Save current conversation."""
        return save_conversation(history)

    def refresh_conversations():
        """Refresh the dropdown with latest saved conversations."""
        return gr.Dropdown(choices=get_saved_conversations())

    def load_chat(filename: str):
        """Load a saved conversation."""
        if not filename:
            return []
        return load_conversation(filename)

    # Submit on button click or Enter key
    # Chain: stream_response -> clear input -> update tokens
    submit_event = send_btn.click(
        fn=stream_response,
        inputs=[msg_input, chatbot, model_dropdown],
        outputs=[chatbot, streaming_group, streaming_display, status_display, stop_btn],
    ).then(
        fn=lambda: "",  # Clear input after submit
        outputs=[msg_input],
    ).then(
        fn=update_token_usage,
        inputs=[model_dropdown],
        outputs=[token_usage_display],
    )

    # Also trigger on Enter key
    enter_event = msg_input.submit(
        fn=stream_response,
        inputs=[msg_input, chatbot, model_dropdown],
        outputs=[chatbot, streaming_group, streaming_display, status_display, stop_btn],
    ).then(
        fn=lambda: "",
        outputs=[msg_input],
    ).then(
        fn=update_token_usage,
        inputs=[model_dropdown],
        outputs=[token_usage_display],
    )

    # Stop button cancels streaming and resets UI
    stop_btn.click(
        fn=lambda: (gr.update(height=400), gr.update(visible=False), "", "", gr.Button(visible=False)),
        outputs=[chatbot, streaming_group, streaming_display, status_display, stop_btn],
        cancels=[submit_event, enter_event],
    )

    # Clear button
    clear_btn.click(
        fn=lambda: ([], gr.update(visible=False), "", ""),
        outputs=[chatbot, streaming_group, streaming_display, status_display],
    )

    # Save button
    save_btn.click(save_chat, inputs=[chatbot], outputs=[save_status])

    # Refresh button updates the dropdown choices
    refresh_btn.click(refresh_conversations, outputs=[load_dropdown])

    # Load button - load into chatbot
    load_btn.click(load_chat, inputs=[load_dropdown], outputs=[chatbot])


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    # Ensure conversations directory exists on startup
    ensure_conversations_dir()

    print("\nStarting AI Chat app (with streaming)...")
    print("Access locally at: http://localhost:7860")
    print("Access from other devices at: http://<your-ip>:7860\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=custom_theme,
        css=CUSTOM_CSS,
        # Simple password protection - shows login screen before chat
        auth=("digital", "surgeon"),
    )
