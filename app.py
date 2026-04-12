import gradio as gr
from transformers import pipeline
import os

MODEL_PATH = "./saved_model"

# Professional department routing mapping
DEPARTMENT_ROUTING = {
    "electricity": "Department of Energy & Electricity ⚡",
    "water": "Municipal Water Authority 💧",
    "internet": "Telecom & IT Infrastructure 🌐",
    "road": "Public Works Department (PWD) 🚧",
    "garbage": "Waste Management & Sanitation 🗑️"
}

def classify_complaint(text):
    """Classify the input text and route appropriately."""
    if not text or not text.strip():
        return "⚠️ Error: Please enter a valid complaint."
        
    try:
        # Load the pipeline locally.
        # It's loaded inside the function to avoid breaking the UI launch if model isn't trained yet.
        classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)
        
        # Predict the label and score
        result = classifier(text)[0]
        label = result['label']
        confidence = result['score'] * 100
        
        # Determine Routing
        department = DEPARTMENT_ROUTING.get(label, "General Queries Department 🏢")
        
        # Format a beautifully styled Markdown output
        formatted_output = f"""
### 📊 AI Classification Result

- **Category Detected:** `{label.capitalize()}`
- **Confidence Score:** `{confidence:.2f}%`

🎯 **Action:** Routing ticket directly to **{department}**
        """
        
        return formatted_output
        
    except Exception as e:
        return f"⚠️ **Error: Model not found.**\nPlease ensure you run `python train.py` first to generate the `{MODEL_PATH}` directory.\n\n*Technical Details: {str(e)}*"

# -----------------
# Gradio UI Design
# -----------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🏛️ Smart City AI: Complaint Router")
    gr.Markdown("""
    **Transforming Civic Engagement powered by Machine Learning.**
    
    This interface reads citizen complaints (including casual text & Hinglish) and automatically classifies and routes them to the correct municipal department. 
    It operates completely locally on a fine-tuned `DistilBERT` model.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ✍️ Input Complaint")
            complaint_input = gr.Textbox(
                lines=5, 
                placeholder="Enter a complaint... e.g., 'light nahi aa rahi sector 4 mein, please fix immediately!'",
                label="Citizen Message"
            )
            submit_btn = gr.Button("Classify & Route", variant="primary")
            
            gr.Examples(
                examples=[
                    "There are massive potholes on the main highway causing terrible traffic.",
                    "Panī ki tankī leaking since yesterday night.",
                    "Wifi bahut slow chal raha hai yaar. WFH not possible today.",
                    "Transformer blew up near the crossing, sparks flying everywhere.",
                    "Garbage truck didn't come for 3 weeks, it's smelling very bad."
                ],
                inputs=complaint_input
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 AI Decision")
            output_display = gr.Markdown(label="Routing Information")
            
    # Connect UI components
    submit_btn.click(fn=classify_complaint, inputs=complaint_input, outputs=output_display)

if __name__ == "__main__":
    print(f"Starting Gradio Server... (Ensure '{MODEL_PATH}' exists for predictions)")
    demo.launch(share=False)
