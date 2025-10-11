import streamlit as st # type: ignore
from st_copy_to_clipboard import st_copy_to_clipboard # type: ignore
import os
import json
from PIL import Image # type: ignore
from dotenv import load_dotenv # type: ignore
from langchain_core.messages import HumanMessage # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
import base64
import warnings
import io
import zipfile


warnings.filterwarnings("ignore")


load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API key not found. Please set it in a .env file.")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
image_enhancer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-image-preview", temperature=0.5)
# --- Helper Functions ---

def safe_json_parse(json_string):
    """Safely parses a JSON string, returning None on failure."""
    try:
        # The model sometimes wraps the JSON in markdown backticks
        if json_string.startswith("```json"):
            json_string = json_string.strip("```json\n").strip()
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None


ADVANCED_IMAGE_PROMPT = """
You are an AI cataloguing assistant for B2B products.

Input:
Product Name: {product_name}
Reference product image uploaded by the user (main product image)
List of key specification attributes:
{specifications_string}

Task: Generate 2 professional B2B catalog images:

1) Spec Highlight Image (A+ Content Style):
Create a high-resolution catalog image of the full product.
AI should select 1–2 most visually significant key specifications and highlight them using zoom-in, callouts, or subtle visual emphasis.
White/clean background, realistic lighting, product fully visible and centered.
Maintain professional B2B catalog aesthetics.
No logos, unrelated text, humans, or body parts.

2) Second Image (AI-Selected Display Logic):
AI should choose the most suitable presentation style from the following list, based on the product and its specs:
- Close-Up / Macro Feature (highlight a key part, texture, or material)
- Exploded / Component View (show internal parts or modular design)
- Lifestyle/Contextual Setting (product in a realistic commercial/industrial environment)
- 360° / Multi-Angle View
- Infographic / Spec-Focused Layout (highlight 1–2 key specs visually)
Ensure the chosen logic maximizes clarity, professionalism, and visual appeal.
Product must be fully visible and clearly understood.
Maintain B2B styling, realistic lighting, clean backgrounds (where applicable).
No humans or body parts visible.

Output:
Provide 2 unique image files.
Both images must be consistent with the uploaded reference image, product name, and key specifications.
"""

SKU_QUESTION_GENERATION_PROMPT = """
You are an expert product specialist for the brand "{brand_name}".
Your goal is to identify the exact product SKU shown in the provided image.

1.  **Analyze and Infer:** Analyze the image of the {product_name}. Infer all specifications you can determine visually (e.g., color, basic shape, visible features).
2.  **Identify Ambiguities:** Based on your knowledge of {brand_name}'s product line, identify 0 to 5 critical, ambiguous specifications that are needed to differentiate this product from other similar models.
3.  **Generate Questions with Options:** For each ambiguous spec, formulate a question and provide 4 plausible multiple-choice options. These options should be realistic for {brand_name} products. The 5th question can optionally be for the Model Number if it's a key differentiator.

Return a JSON object with a single key "questions". The value should be a list of question objects.
- If no questions are needed, return an empty list: {{"questions": []}}
- Each question object must have three keys: "spec_name" (a short attribute name), "description" (the full question), and "options" (a list of 4 strings).

**Example Output:**
{{
  "questions": [
    {{
      "spec_name": "Capacity",
      "description": "Spec 1 (What is the battery's Ampere-hour capacity?)",
      "options": ["45Ah", "55Ah", "65Ah", "75Ah"]
    }},
    {{
      "spec_name": "Terminal Type",
      "description": "Spec 2 (What is the terminal layout?)",
      "options": ["Left Hand", "Right Hand", "Center Post", "Side Post"]
    }}
  ]
}}

Provide only the JSON response.
"""

### Part 2: The Code Implementation

#### **Step 1: Add a New Helper Function**

def generate_b2b_catalog_images(product_name, specifications_list):
    """
    Generates two advanced B2B catalog images based on a product's details.

    Returns:
        A list of two image bytes, or None if generation fails.
    """
    # Format the list of specifications into a clean string for the prompt
    spec_string = "\n".join([f"- {spec.get('attribute')}: {spec.get('value')}" for spec in specifications_list])

    # Construct the final, detailed prompt
    final_prompt = ADVANCED_IMAGE_PROMPT.format(
        product_name=product_name,
        specifications_string=spec_string
    )


    message = HumanMessage(content=[{"type": "text", "text": final_prompt},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(st.session_state.image_bytes).decode('utf-8')}"}},])
    
    with st.spinner("🤖 Generating advanced A+ catalog images... (This may take a moment)"):
        generated_images = invoke_image_model_with_tracking(image_enhancer_llm, message)

    # Process the response, expecting two image files
    if generated_images and len(generated_images) == 2:
        st.success("Advanced A+ images generated successfully!")
        return generated_images
    else:
        # This is the original, robust error handling for when generation fails.
        st.warning("Advanced image generation failed to return two images. Proceeding with the main image only.")
        return None


def render_product_listing(product_id, listing_data, image_bytes_list, image_mime_type):
    """
    Renders a single product listing, now with an interactive image rotation feature.
    This version correctly handles both single and multiple image lists.
    """

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        # --- NEW: Interactive Image Selector and Rotation ---
        if image_bytes_list:
            # Determine the currently selected image index
            # We use a unique key for each product's selector
            selector_key = f"image_selector_{product_id}"
            
            # If there's more than one image, show the selector.
            if len(image_bytes_list) > 1:
                selected_index = st.radio(
                    "Select Image View",
                    options=range(len(image_bytes_list)),
                    format_func=lambda i: f"Image {i + 1}",
                    key=selector_key,
                    horizontal=True,
                    label_visibility="collapsed"
                )
            else:
                # If there's only one image, the index is always 0.
                selected_index = 0

            # Display the currently selected image
            st.image(image_bytes_list[selected_index], use_container_width=True)

            # Add the Rotate button, which acts on the selected image
            if st.button("🔄 Rotate Current Image 90°", key=f"rotate_{product_id}", use_container_width=True):
                try:
                    current_image_bytes = image_bytes_list[selected_index]
                    image = Image.open(io.BytesIO(current_image_bytes))
                    rotated_image = image.rotate(-90, expand=True)
                    
                    buffer = io.BytesIO()
                    rotated_image.save(buffer, format="PNG")
                    
                    # Overwrite the old image data in the list with the new rotated data
                    image_bytes_list[selected_index] = buffer.getvalue()
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not rotate image: {e}")

            # --- Download Logic (Handles both single and multiple images) ---
            if len(image_bytes_list) > 1:
                # Download All as ZIP
                try:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        product_name_for_file = listing_data.get('product_name', 'product').strip().replace(' ', '_')
                        for i, img_bytes in enumerate(image_bytes_list):
                            zf.writestr(f"{product_name_for_file}_{i+1}.png", img_bytes)
                    st.download_button(
                        label="📥 Download All Images (.zip)",
                        data=zip_buffer.getvalue(),
                        file_name=f"{product_name_for_file}_images.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"Could not prepare ZIP file: {e}")
            else:
                # Download Single Image
                try:
                    product_name_for_file = listing_data.get('product_name', 'product').strip().replace(' ', '_')
                    st.download_button(
                        label="📥 Download Image",
                        data=image_bytes_list[0],
                        file_name=f"{product_name_for_file}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"Could not prepare image for download: {e}")
        else:
            st.warning("No images available to display.")

    with col2:
        # The entire display logic for text content is unchanged and remains correct.
        st.header(listing_data.get('product_name', 'Product Name Not Found'))
        st_copy_to_clipboard(listing_data.get('product_name', ''), f"📋 Copy Name")
        st.markdown("---")

        spec_title_col, spec_button_col = st.columns([4, 1])
        with spec_title_col:
            st.markdown("#### Specification")
        
        specs = listing_data.get('specifications', [])
        if specs and isinstance(specs, list):
            spec_string_to_copy = "\n".join([f"{spec.get('attribute', 'N/A')}: {spec.get('value', 'N/A')}" for spec in specs])
            with spec_button_col:
                st_copy_to_clipboard(spec_string_to_copy, f"📋 Copy Specs")

            with st.container(border=True):
                for spec in specs:
                    spec_col1, spec_col2 = st.columns(2)
                    spec_col1.markdown(f"**{spec.get('attribute', 'N/A')}**")
                    spec_col2.write(f"{spec.get('value', 'N/A')}")
        else:
            st.write("No specifications were generated.")
        
        st.write("")

        desc_title_col, desc_button_col = st.columns([4, 1])
        with desc_title_col:
            st.markdown("#### Description") # Using H4 for consistency
        with desc_button_col:
            st_copy_to_clipboard(listing_data.get('description', ''), "📋 Copy Desc.")

        st.write(listing_data.get('description', 'No description available.'))
        st.markdown("---")

        st.markdown("**Primary Keyword:**")
        st.code(listing_data.get('primary_keyword', 'N/A'))

def reset_session_state():
    """Resets the session state to start a new cataloging process."""
    usage = st.session_state.get("usage_stats", {}) 
    st.session_state.clear()
    st.session_state.usage_stats = usage
    st.session_state.step = "initial"

def invoke_text_model_with_tracking(llm, message):
    """Invokes a text model, tracks token usage, and returns the response content."""
    result = llm.invoke([message])
    print(result)
    usage = result.usage_metadata
    
    st.session_state.usage_stats["text_input_tokens"] += usage.get("input_tokens", 0)
    st.session_state.usage_stats["text_output_tokens"] += usage.get("output_tokens", 0)

    return result.content


def invoke_image_model_with_tracking(llm, message):
    """Invokes an image model, tracks usage, and returns a list of image bytes."""
    result = llm.invoke([message])
    usage = result.usage_metadata

    st.session_state.usage_stats["image_input_tokens"] += usage.get("input_tokens", 0)
    st.session_state.usage_stats["image_output_tokens"] += usage.get("output_tokens", 0)

    response_content = result.content
    if isinstance(response_content, list):
        image_bytes_list = []
        for part in response_content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                b64_data = part["image_url"]["url"].split(",")[1]
                image_bytes_list.append(base64.b64decode(b64_data))
        
        # Add the number of successfully generated images to the tracker
        st.session_state.usage_stats["images_generated"] += len(image_bytes_list)
        return image_bytes_list
        
    return None

# --- Main Application Logic ---

st.set_page_config(page_title="AI Cataloguing Assistant", layout="wide")
st.title("🤖 AI Cataloguing Assistant Prototype")

banner_text = "For the best experience, use Chrome. Accessible on iPhone, Android, and desktop/laptop."

# We use the <marquee> HTML tag to create the scrolling effect.
# Inline CSS is used to style it like a professional banner.
banner_html = f"""
    <marquee style="background-color: #D9534F; 
                    padding: 10px; 
                    border-radius: 5px; 
                    color: white;
                    font-family: sans-serif; 
                    font-size: 14px; 
                    font-weight: bold; 
                    text-transform: uppercase;" 
             scrollamount="4" 
             behavior="scroll" 
             direction="left">
        <b>Note:</b> {banner_text}
    </marquee>
"""

# The st.markdown function renders the HTML. unsafe_allow_html must be True.
st.markdown(banner_html, unsafe_allow_html=True)

st.write("")

# Initialize session state variables
if "step" not in st.session_state:
    st.session_state.step = "initial"
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "image_mime_type" not in st.session_state:
    st.session_state.image_mime_type = None 
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None
if "identified_products" not in st.session_state:
    st.session_state.identified_products = []
if "critical_questions" not in st.session_state:
    st.session_state.critical_questions = []
if "critical_attribute" not in st.session_state:
    st.session_state.critical_attribute = None
if "quality_issues_list" not in st.session_state:
    st.session_state.quality_issues_list = []
if "quality_issues" not in st.session_state:
    st.session_state.quality_issues = ""
if "enhanced_image_bytes" not in st.session_state:
    st.session_state.enhanced_image_bytes = None
if "final_listing" not in st.session_state:
    st.session_state.final_listing = None
if "create_all_flow" not in st.session_state:
    st.session_state.create_all_flow = False
if "processing_index" not in st.session_state:
    st.session_state.processing_index = 0
if "all_final_listings" not in st.session_state:
    st.session_state.all_final_listings = []
if "products_to_process" not in st.session_state:
    st.session_state.products_to_process = []
if "pre_extracted_images" not in st.session_state:
    st.session_state.pre_extracted_images = {}
if "usage_stats" not in st.session_state:
    st.session_state.usage_stats = {
        "text_input_tokens": 0,
        "text_output_tokens": 0,
        "image_input_tokens": 0,
        "image_output_tokens": 0,
        "images_generated": 0,
    }
if "customization_details" not in st.session_state:
    st.session_state.customization_details = None
if "is_branded_flow" not in st.session_state:
    st.session_state.is_branded_flow = False
if "brand_name" not in st.session_state:
    st.session_state.brand_name = None
if "sku_questions" not in st.session_state:
    st.session_state.sku_questions = []


# --- Step 0: Image Upload ---
if st.session_state.step == "initial":
    st.info("To begin, please provide a product image using one of the methods below.")

    # --- NEW: Create tabs for the two input methods ---
    tab1, tab2 = st.tabs(["📁 Upload an Image", "📸 Take a Photo"])

    with tab1:
        # This is the existing file uploader
        uploaded_file = st.file_uploader(
            "Choose an image file from your device...", 
            type=["jpg", "jpeg", "png","webp","bmp","tiff"]
        )

    with tab2:
        # --- NEW: Camera input widget ---
        # This will activate the user's camera and show a "Take photo" button.
        clicked_photo = st.camera_input(
            "Point your camera at the product and take a photo."
        )

    # --- UNIFIED LOGIC ---
    # This variable will hold the file data from whichever method the user chose.
    image_file = uploaded_file or clicked_photo

    if image_file is not None:
        # Process the image file, regardless of its source (upload or camera)
        image_data = image_file.getvalue()
        
        # We need to use io.BytesIO to handle the in-memory file for PIL
        st.session_state.uploaded_image = Image.open(io.BytesIO(image_data))
        
        # Store the image data in both the working and original state variables
        st.session_state.image_bytes = image_data
        st.session_state.original_image_bytes = image_data
        
        # Robustly get the MIME type. Camera input doesn't have a 'type' attribute,
        # so we default to 'image/png', which is a safe choice.
        mime_type = image_file.type if hasattr(image_file, 'type') else "image/png"
        st.session_state.image_mime_type = mime_type
        st.session_state.original_image_mime_type = mime_type
        
        # Proceed to the first step of the analysis workflow
        st.session_state.step = "identify_products"
        st.rerun()

# Display the uploaded image throughout the process
if st.session_state.uploaded_image:
    with st.sidebar:
        st.header("Uploaded Product")
        st.image(st.session_state.uploaded_image, use_container_width=True)
        if st.button("Start Over"):
            reset_session_state()
            st.rerun()
        st.markdown("---")
        with st.expander("📊 API Usage & Cost Estimate", expanded=True):
            stats = st.session_state.usage_stats
            
            st.write(f"**Text Input Tokens:** `{stats['text_input_tokens']}`")
            st.write(f"**Text Output Tokens:** `{stats['text_output_tokens']}`")
            st.write(f"**Image Input Tokens:** `{stats['image_input_tokens']}`")
            st.write(f"**Image Output Tokens:** `{stats['image_output_tokens']}`")
            st.write(f"**Images Generated:** `{stats['images_generated']}`")

            # Example pricing - replace with actuals if needed
            # Prices are per 1,000 tokens or per image
            text_input_cost = (stats['text_input_tokens'] *0.3)/1000000 
            text_output_cost = (stats['text_output_tokens'] * 2.5) /1000000
            image_prompt_cost = (stats['image_input_tokens']*0.30)/1000000 # Example price
            image_gen_cost = (stats['image_output_tokens'] * 30)/1000000 # Example price
            
            total_cost = text_input_cost + text_output_cost + image_prompt_cost + image_gen_cost
            
            st.markdown("---")
            st.metric(label="Estimated Total Cost", value=f"${total_cost:.4f} USD")
            
            if st.button("Reset Cost Tracker"):
                st.session_state.usage_stats = {
                    "text_input_tokens": 0, "text_output_tokens": 0,
                    "image_prompt_tokens": 0, "images_generated": 0,
                }
                st.rerun()

        st.caption("Costs are estimates based on sample pricing and may not be exact.")


if st.session_state.step == "identify_products":
    with st.spinner("Step 1: Identifying products in the image..."):
        prompt = """
            Analyze the provided image carefully and identify all distinct, primary products clearly visible. There shouldn't be any duplicates or variations of the same product.
            Don't leave anything out, only if the product is completely visible and present in the image.
            If same Multiple products are present treat them as single product and return one entry.
            For each product, you must determine if a recognizable brand is clearly visible.
            

            **CRITICAL RULES FOR BRAND IDENTIFICATION:**
            1.  **High Confidence Only:** Only identify a brand if the name or logo belongs to a **well-known, publicly recognized commercial brand**.
            2.  **No Ambiguity:** Do not identify generic text (e.g., "Made in China", "Heavy Duty", "12V") or unclear logos as a brand.
            3.  **Default to Non-Branded:** If you are not highly confident, you MUST classify the product as non-branded.

            Return the result as a JSON array of objects. Each object must have three keys:
            1. "product_name": A generic name for the product (e.g., "Car Battery", "Floor Lamp").
            2. "is_branded": A boolean (true/false), based on the critical rules above.
            3. "brand_name": The identified brand name as a string, or null if non-branded.

            Example for a clear, well-known brand:
            [{"product_name": "Automotive Battery", "is_branded": true, "brand_name": "Exide"}]

            Example for a generic or unbranded product:
            [{"product_name": "Tripod Floor Lamp", "is_branded": false, "brand_name": null}]

            Example for ambiguous text that should NOT be a brand:
            [{"product_name": "Power Inverter", "is_branded": false, "brand_name": null}]

            If only one product is clearly the main subject, return an array with a single item.
            If no clear product is visible, return an empty array.
            Provide only the JSON response.
            """

        message = HumanMessage(content=[{"type": "text", "text": prompt}, 
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(st.session_state.image_bytes).decode('utf-8')}"}},])
        
        response_content = invoke_text_model_with_tracking(llm, message)
        products_data = safe_json_parse(response_content)

        if products_data and isinstance(products_data, list) and len(products_data) > 0:
            if len(products_data) == 1:
                product = products_data[0]
                st.session_state.selected_product = product["product_name"]
                st.session_state.is_branded_flow = product["is_branded"]
                st.session_state.brand_name = product["brand_name"]
                
                # --- NEW: Workflow Routing ---
                if st.session_state.is_branded_flow:
                    st.success(f"Branded product identified: **{st.session_state.brand_name} {st.session_state.selected_product}**")
                    st.session_state.step = "quality_check" # Skip to quality check for branded items
                else:
                    st.success(f"Product identified: **{st.session_state.selected_product}**")
                    st.session_state.step = "quality_check" # The original non-branded flow
                st.rerun()
            elif len(products_data) > 1:
                # If multiple products, we still go to the confirmation step
                st.session_state.identified_products = products_data # Store the full data
                st.session_state.step = "confirm_product"
                st.rerun()
            else:
                st.session_state.step = "product_not_found_fail"
                st.rerun()
        else:
            st.error("Failed to identify products. The model's response was not as expected.")
            st.session_state.step = "product_not_found_fail"
            st.rerun()

if st.session_state.step == "product_not_found_fail":
    # Display a specific and clear error message for this failure case.
    st.error("ERROR: Product Identification Failed")
    st.warning("The AI could not identify a clear product in the uploaded image.")
    st.info("Please try again with a different image that clearly shows one or more products.")

    # Provide the button to restart the entire process.
    if st.button("Upload a New Image", use_container_width=True):
        # Call the reset function to clear all old data.
        reset_session_state()
        # Rerun the app to go back to the initial upload screen.
        st.rerun()

if st.session_state.step == "confirm_product":
    st.subheader("Visible Products:")
    st.write("Multiple items were detected. Please select the products you wish to process.")
    
    if "identified_products" in st.session_state:
        selections = {}

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Deselect All", use_container_width=True):
                # Set the value for each checkbox key in the session state to False
                for product_data in st.session_state.identified_products:
                    product_name = product_data.get("product_name", "")
                    st.session_state[f"check_{product_name}"] = False
                st.rerun()

        with st.container(border=True):
            st.write("**Select Products:**")
            # --- THE FIX: The 'product' variable is now a dictionary ---
            for product_data in st.session_state.identified_products:
                product_name = product_data.get("product_name", "Unknown Product")
                
                selections[product_name] = st.checkbox(
                    product_name.title(), 
                    value=True, 
                    key=f"check_{product_name}"
                )

        st.write("") # Add some space

        if st.button("🚀 Process Selected Products", use_container_width=True, type="primary"):
            # Find the full data for each product where the checkbox is ticked.
            products_to_create = [
                prod_data for prod_data in st.session_state.identified_products 
                if selections.get(prod_data.get("product_name"))
            ]

            if not products_to_create:
                st.warning("Please select at least one product to process.")
            else:
                if len(products_to_create) == 1:
                    # Single product flow
                    selected_data = products_to_create[0]
                    st.session_state.create_all_flow = False
                    st.session_state.selected_product = selected_data.get("product_name")
                    st.session_state.is_branded_flow = selected_data.get("is_branded")
                    st.session_state.brand_name = selected_data.get("brand_name")
                    st.session_state.step = "extract_selected_product" # Always extract
                    st.rerun()
                    
                else:
                    # Batch processing flow
                    st.session_state.products_to_process = products_to_create
                    st.session_state.create_all_flow = True
                    st.session_state.processing_index = 0
                    st.session_state.all_final_listings = []
                    st.session_state.step = "extract_selected_product"
                    st.rerun()

        if st.button("🔍 Products Not in This List", use_container_width=True):
            st.session_state.step = "product_not_listed_fail"
            st.rerun()

if st.session_state.step == "product_not_listed_fail":
    
    # Display the user's requested instructions
    st.subheader("Instruction: Please upload a clear image of the product.")
    
    # Use st.warning or st.info for the guideline to make it stand out
    st.warning(
        """
        **Guideline:** The uploaded image should clearly show the application or use of the product 
        so the module can contextually identify it.
        """
    )
    
    st.markdown("---")

    # Provide a clear button to go back to the start and try again.
    if st.button("🔄 Upload a New, Clear Image", use_container_width=True):
        # Call the reset function to clear all old data.
        reset_session_state()
        # Rerun the app to go back to the initial upload screen.
        st.rerun()

if st.session_state.step == "extract_selected_product":

    if st.session_state.create_all_flow:
        # If in a batch, reset the source image to the original multi-product one.
        st.session_state.image_bytes = st.session_state.original_image_bytes
        st.session_state.image_mime_type = st.session_state.original_image_mime_type
        
        # Get the data for the current item from our list.
        current_item_data = st.session_state.products_to_process[st.session_state.processing_index]
        
        # Set the state (name, brand, etc.) for this specific item.
        st.session_state.selected_product = current_item_data.get("product_name")
        st.session_state.is_branded_flow = current_item_data.get("is_branded")
        st.session_state.brand_name = current_item_data.get("brand_name")

    # The rest of the function proceeds as normal, now using the correct context.
    product_name = st.session_state.selected_product
    
    with st.spinner(f"Isolating '{st.session_state.selected_product}' from the image... This may take a moment."):
        
        extraction_prompt = f"""
        You are an expert digital imaging specialist tasked with isolating a single product from a composite image for a high-end B2B catalog.

        The user has provided an image containing multiple items and has selected the following product to be the main subject: "{product_name}".

        Your instructions are as follows:
        1.  **Identify and Isolate:** Accurately identify the "{product_name}" within the provided image.
        2.  **Regenerate a New Image:** Create a new image that contains ONLY the selected product.
        3.  **Create a B2B Standard Background:** Place the isolated product on a clean, non-distracting, solid light gray (#f0f0f0) or pure white (#ffffff) background.
        4.  **Maintain Product Integrity:** The product's appearance, color, lighting, texture, and orientation must be perfectly preserved. Do not alter the product itself in any way.
        5.  **Remove All Distractions:** All other products, text, logos, or background clutter from the original image must be completely removed.
        6.  **Ensure Photorealism:** The final output must be a high-resolution, photorealistic image.

        The final output should be only the regenerated image file.
        """


        extraction_message = HumanMessage(content=[{"type": "text", "text": extraction_prompt}, 
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(st.session_state.image_bytes).decode('utf-8')}"}},])
        
        # Use the powerful image generation model for this task
        extracted_images = invoke_image_model_with_tracking(image_enhancer_llm, extraction_message)

        # Process the response to get the new image data
        if extracted_images and len(extracted_images) >= 1:
            new_image_bytes = extracted_images[0]

            st.session_state.image_bytes = new_image_bytes
            st.session_state.image_mime_type = "image/png" # Generated images are typically PNG
            st.session_state.uploaded_image = Image.open(io.BytesIO(new_image_bytes))

            st.success(f"Successfully isolated the {product_name}.")
            st.session_state.step = "quality_check"
            st.rerun()

        else:
            st.error("AI image extraction failed. The model did not return a valid image.")
            st.warning("You can proceed with the original multi-product image or start over.")
            
            col1, col2 = st.columns(2)
            if col1.button("Proceed with Original Image", use_container_width=True):
                st.session_state.step = "quality_check"
                st.rerun()
            if col2.button("Start Over", use_container_width=True):
                reset_session_state()
                st.rerun()


# --- Step 1: Image Quality Check ---
if st.session_state.step == "quality_check":

    local_issues = []
    min_dimension = 450
    min_dimension = 450
    try:
        image = Image.open(io.BytesIO(st.session_state.image_bytes))
        width, height = image.size
        
        if width < min_dimension or height < min_dimension:
            st.warning(f"Image resolution is low ({width}x{height}). Minimum recommended is {min_dimension}x{min_dimension}.")
            st.session_state.quality_issues_list = ["low_resolution"]
            st.session_state.step = "offer_enhancement"
            st.rerun() 
            
    except Exception as e:
        st.error(f"Could not read image dimensions: {e}")
        st.session_state.quality_issues = "Could not read image file for quality check."
        st.session_state.step = "quality_fail"
        st.rerun()

    with st.spinner("Step 1: Performing Image Quality Check..."):
        prompt = """
        You are an image quality inspector. Analyze the provided image based on these criteria and respond with a JSON object.
        1. human_present: Is a human hand or body part clearly visible? (true/false)
        2. watermark_present: Is a logo or watermark visible that is not part of the product itself? (true/false)
        3. background_cluttered: Is the background irrelevant or distracting? (true/false)
        4. is_blurry: Is the image low quality or blurry? (true/false)
        5. is_screenshot: Does the image appear to be a screenshot with UI elements? (true/false)
        Analyze the image and provide only the JSON response.
        """
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(st.session_state.image_bytes).decode('utf-8')}"}},
            ]
        )
        response_content = invoke_text_model_with_tracking(llm, message)
        quality_results = safe_json_parse(response_content)

        if quality_results:
            issues = [key for key, value in quality_results.items() if value]
            if not issues:
                st.success("Image quality check passed!")

                if st.session_state.get("is_branded_flow"):
                    st.session_state.step = "ask_branded_sku_questions"

                else:
                    st.session_state.step = "get_critical_attribute"
                st.rerun()
            else:
                st.session_state.quality_issues_list = issues 
                enhanceable_issues = {"is_blurry", "watermark_present", "background_cluttered"}
                
                if any(issue in enhanceable_issues for issue in issues):
                    st.session_state.step = "offer_enhancement"
                else:
                    st.session_state.quality_issues = ", ".join(issues).replace('_', ' ')
                    st.session_state.step = "quality_fail"
                st.rerun()
        else:
            st.session_state.quality_issues = "The AI model could not analyze the image."
            st.session_state.step = "quality_fail"
            st.rerun()

print("Image_Check_Done")

# --- NEW STEP: Offer Enhancement for Flawed Images ---
if st.session_state.step == "offer_enhancement":
    issue_str = ", ".join(st.session_state.quality_issues_list).replace('_', ' ')
    st.warning(f"Image Quality Warning: The image appears to have some issues: **{issue_str}**.")
    st.info("I can use AI to try and fix these issues and generate a clean, B2B-standard product image. Would you like to proceed?")

    col1, col2 = st.columns(2)
    if col1.button("✅ Yes, Attempt AI Enhancement", use_container_width=True):
        st.session_state.step = "perform_enhancement"
        st.rerun()
    
    if col2.button("🔄 No, I'll Upload a New Image", use_container_width=True):
        reset_session_state()
        st.rerun()

# --- NEW STEP: Perform the Image Enhancement ---
if st.session_state.step == "perform_enhancement":
    with st.spinner("Enhancing image with AI... This may take a moment."):
        # Dynamically build the instructions for the prompt based on detected flaws
        flaw_instructions_map = {
            "is_blurry": "The image is blurry; regenerate it with sharp focus and clear details.",
            "watermark_present": "A watermark or logo is present; remove it completely, intelligently filling in the area.",
            "background_cluttered": "The background is cluttered; replace it with a clean, solid light gray (#f0f0f0) background.",
            "low_resolution": "The image resolution is low; regenerate it as a high-resolution image (e.g., 1024x1024) with sharp, clear details."
            }
        
        instructions = [flaw_instructions_map[issue] for issue in st.session_state.quality_issues_list if issue in flaw_instructions_map]
        instruction_str = " ".join(instructions)

        enhancement_prompt = f"""
        You are a professional product photographer and digital retoucher for a high-end B2B e-commerce platform.

        Your task is to regenerate the provided product image to meet our strict catalog standards. The original image has the following quality issues: {instruction_str}

        Follow these critical rules for the regeneration:
        1. **Fix the specified flaws:** Execute the instructions precisely to correct the issues.
        2. **Maintain Product Integrity:** Do NOT change the product's design, color, shape, texture, or orientation. The output must be a photorealistic representation of the exact same product.
        3. **Maintain Content Integrity:** Do NOT change/miss the content of the Image. The output must include exact content of the Image except any watermark/human hand.
        4. **No Watermarks or Logos:** Ensure that the final image is free of any watermarks, logos, or branding elements.
        5. **Ensure B2B Standard Background:** The background must be a clean, non-distracting, solid light gray (#f0f0f0) or pure white (#ffffff). Remove all shadows or props unless they are integral to the product itself.
        6. **Photorealistic Output:** The final result must be a high-resolution, photorealistic image, not a drawing, illustration, or artistic interpretation.

        The final output should be only the regenerated image file.
        """

        enhancement_message = HumanMessage(
            content=[
                {"type": "text", "text": enhancement_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(st.session_state.image_bytes).decode('utf-8')}"}},
            ]
        )

        # Invoke the powerful image generation model
        generated_images = invoke_image_model_with_tracking(image_enhancer_llm, enhancement_message)
        
        if generated_images and len(generated_images) >= 1:
            # Extract the raw base64 data
            st.session_state.enhanced_image_bytes = generated_images[0]
            st.session_state.step = "confirm_enhancement"
            st.rerun()
        else:
            st.error("Image enhancement failed. The model did not return an image. Please try again with a new upload.")
            st.session_state.step = "quality_fail"
            st.rerun()


# --- NEW STEP: Confirm the Enhanced Image ---
if st.session_state.step == "confirm_enhancement":
    st.success("✅ AI Enhancement Complete!")
    st.write("Please review the result. If you are satisfied, we will proceed using the enhanced image.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before")
        st.image(st.session_state.uploaded_image, use_container_width=True)
    with col2:
        st.subheader("After (AI Enhanced)")
        st.image(st.session_state.enhanced_image_bytes, use_container_width=True)

    colA, colB = st.columns(2)
    if colA.button("👍 Use Enhanced Image", use_container_width=True, type="primary"):
        st.session_state.image_bytes = st.session_state.enhanced_image_bytes
        st.session_state.image_mime_type = "image/png" # Generated images are typically PNG
        if st.session_state.get("is_branded_flow"):
            st.session_state.step = "ask_branded_sku_questions"
        else:
            st.session_state.step = "get_critical_attribute"
        st.success("Great! Proceeding with the clean image...")
        st.rerun()

    if colB.button("🔄 Start Over with a New Image", use_container_width=True):
        reset_session_state()
        st.rerun()

if st.session_state.step == "quality_fail":
    # Display the persistent error messages
    st.error("ERROR: The uploaded image did not pass the quality check.")
    st.warning(f"Detected Issues: **{st.session_state.get('quality_issues', 'Unknown')}**")
    st.info("You can upload a different image to try again.")

    # Display the button to restart the process
    if st.button("Upload a New Image"):
        reset_session_state()
        st.rerun()



# --- Step 3: Critical Attribute Input ---
if st.session_state.step == "get_critical_attribute":
    with st.spinner("Step 3: Analyzing image to determine necessary information..."):
        # NEW PROMPT: Asks for up to two questions based on the image itself.
        prompt = f"""
        Analyze the provided image of a '{st.session_state.selected_product}'.
        Based on what you can see, what are the most critical attributes a B2B buyer would need to know that are likely not visible?
        Formulate a maximum of two concise questions to ask the user for this information.

        Examples:
        - For a transformer image: ["What is the rated capacity (e.g., 100 kVA)?", "What is the primary voltage (e.g., 480V)?"]
        - For a pipe fitting image: ["What is the connection size/type (e.g., 1/2\" NPT)?"]

        Respond with only a JSON object containing a list of questions, like: {{"questions": ["Question 1?", "Question 2?"]}}
        If only one question is necessary, return a list with one item. If no questions are needed, return an empty list.
        """

        message = HumanMessage(content=[{"type": "text", "text": prompt}, 
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(st.session_state.image_bytes).decode('utf-8')}"}},
        ])
        response_content = invoke_text_model_with_tracking(llm, message)
        question_data = safe_json_parse(response_content)

        # NEW: Handle a list of questions or an empty list.
        if question_data and "questions" in question_data and question_data["questions"]:
            st.session_state.critical_questions = question_data["questions"] # Store the list of questions
            st.session_state.step = "ask_user"
            st.rerun()
        else:
            # If the model fails or returns no questions, skip this step.
            st.warning("Could not determine critical questions, or none were needed. Proceeding without additional user input.")
            st.session_state.critical_attribute = "Not provided"
            st.session_state.step = "generate_listing"
            st.rerun()



if st.session_state.step == "ask_user":
    st.subheader("Prompt:")
    st.write("Please provide the following critical attributes for an accurate listing:")

    with st.form("attribute_form"):
        # NEW: Dynamically create a text input for each question from the list.
        user_inputs = {}
        for i, question in enumerate(st.session_state.critical_questions):
            user_inputs[question] = st.text_input(question, key=f"q_{i}")

        submitted = st.form_submit_button("Submit")
        if submitted:
            # NEW: Check if all generated questions have been answered.
            all_inputs_provided = all(user_inputs.values())
            if all_inputs_provided:
                # Format the answers into a single, readable string for the next step.
                formatted_answers = []
                for question, answer in user_inputs.items():
                    # Extract the attribute from the question for clean formatting.
                    # e.g., "What is the rated capacity (e.g., 100 kVA)?" -> "rated capacity"
                    attribute_name = question.split('(')[0].replace("What is the", "").strip()
                    formatted_answers.append(f"{attribute_name.title()}: {answer}")

                st.session_state.critical_attribute = ", ".join(formatted_answers)
                st.session_state.step = "ask_customization_yes_no"
                st.rerun()
            else:
                st.warning("Please answer all questions.")

if st.session_state.step == "ask_branded_sku_questions":
    with st.spinner(f"Analyzing {st.session_state.brand_name} product to identify key specifications..."):
        prompt = SKU_QUESTION_GENERATION_PROMPT.format(
            brand_name=st.session_state.brand_name,
            product_name=st.session_state.selected_product
        )
        message = HumanMessage(content=[{"type": "text", "text": prompt}, 
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(st.session_state.image_bytes).decode('utf-8')}"}},])
        response_content = invoke_text_model_with_tracking(llm, message)
        question_data = safe_json_parse(response_content)

        if question_data and "questions" in question_data and question_data["questions"]:
            st.session_state.sku_questions = question_data["questions"]
            st.session_state.step = "collect_branded_sku_answers"
            st.rerun()
        else:
            # If AI determines no questions are needed, we can skip straight to customization
            st.warning("AI determined all critical specifications are visible. Proceeding.")
            st.session_state.critical_attribute = "All specifications inferred from image."
            st.session_state.step = "ask_customization_yes_no"
            st.rerun()

# --- NEW STEP 2A.2: Collect Branded SKU Answers from User ---
if st.session_state.step == "collect_branded_sku_answers":
    st.subheader(f"Help Identify the Exact {st.session_state.brand_name} SKU")
    st.write("Please provide the values for the following critical specifications:")

    user_answers = {}
    for i, q in enumerate(st.session_state.sku_questions):
        st.markdown(f"**{q['description']}**")
        options = q['options'] + ["Other (enter manually)"]

        radio_key = f"radio_{i}"
        text_key = f"text_{i}"

        # Make radio interactive (reruns on change)
        radio_choice = st.radio("", options, key=radio_key, label_visibility="collapsed")

        # Show text_input only when Other is selected
        other_text_value = ""
        if radio_choice == "Other (enter manually)":
            other_text_value = st.text_input(
                "Please specify:",
                key=text_key,
                placeholder="Enter your custom value"
            )
        else:
            # Clear stale other text if the user changed selection away from Other
            if text_key in st.session_state:
                st.session_state[text_key] = ""

        user_answers[q['spec_name']] = {
            "choice": st.session_state.get(radio_key),
            "other_value": st.session_state.get(text_key, "").strip()
        }

    if st.button("Submit Specifications"):
        formatted_answers = []
        all_answered = True
        for spec_name, answer_data in user_answers.items():
            choice = answer_data["choice"]
            other_value = answer_data["other_value"]
            if choice == "Other (enter manually)":
                if other_value:
                    formatted_answers.append(f"{spec_name}: {other_value}")
                else:
                    all_answered = False
            else:
                formatted_answers.append(f"{spec_name}: {choice}")

        if all_answered:
            st.session_state.critical_attribute = ", ".join(formatted_answers)
            st.session_state.step = "ask_customization_yes_no"
            st.rerun()
        else:
            st.warning("Please provide an answer for all specifications, including any 'Other' fields you have selected.")



if st.session_state.step == "ask_customization_yes_no":
    st.subheader("Customization / Upgradation")
    st.write("Do you provide any customization or upgradation for this product?")

    col1, col2, col3 = st.columns([1, 1, 2]) # Give some space

    if col1.button("✅ Yes, I do", use_container_width=True):
        st.session_state.step = "ask_customization_details"
        st.rerun()

    if col2.button("❌ No", use_container_width=True):
        st.session_state.customization_details = None # Ensure it's cleared
        st.session_state.step = "generate_listing" # Skip the details step
        st.rerun()

# --- NEW STEP 3.2: Get Customization Details ---
if st.session_state.step == "ask_customization_details":
    st.subheader("Customization Details")
    with st.form("customization_form"):
        # Use st.text_area for potentially longer, multi-line input
        user_input = st.text_area(
            "What kind of customization or upgradation have you done?",
            placeholder="e.g., Custom logo branding, specific color options, increased power capacity..."
        )
        submitted = st.form_submit_button("Submit Details")
        if submitted and user_input:
            st.session_state.customization_details = user_input
            st.session_state.step = "generate_listing" # Proceed to final generation
            st.rerun()
        elif submitted and not user_input:
            st.warning("Please describe the customization.")


# --- Steps 4, 5, 6: Final Listing Generation ---
if st.session_state.step == "generate_listing":
    with st.spinner("Generating product name, specs, and description..."):
        
        customization_prompt_injection = ""
        if st.session_state.get("customization_details"):
            customization_info = st.session_state.customization_details
            customization_prompt_injection = f"""
            IMPORTANT CUSTOMIZATION NOTE: The user provides customization for this product. You MUST incorporate the following details:
            - In the 'specifications' list, add a new attribute exactly like this after correcting any spelling/grammatical errors from user: {{"attribute": "Customisable / Value Addition", "value": "{customization_info}"}}.
            - At the very end of the 'description', you MUST append this exact sentence after correcting any spelling/ grammatical mistake from user: "This product can also be customized or upgraded: {customization_info}."
            """
        
        brand_context = ""
        if st.session_state.get("is_branded_flow") and st.session_state.get("brand_name"):
            brand_context = f"- Brand: {st.session_state.brand_name}"


        final_prompt = f"""
        You are an expert B2B product cataloguer. Using the provided image and the following information, generate a complete product listing.

        - Confirmed Product: {st.session_state.selected_product} {brand_context}
        - User-Provided Specification: {st.session_state.critical_attribute}
        - {customization_prompt_injection}

        Generate the output as a single JSON object with these exact keys: "product_name", "specifications", "primary_keyword", "description".

        Follow these strict rules:
        1. product_name: Create a precise B2B-friendly name including 2-3 key specs inferred from the image and user input (e.g., material, type, size).
        2. specifications: Extract 3-8 key attributes and their values into a list of JSON objects, like [{{"attribute": "Material", "value": "Stainless Steel 304"}}]. Infer from the image and user input.
        3. primary_keyword: Derive one singular, industry-specific keyword from the product name.
        4. description: Write a 100-120 word SEO-friendly description. It must start with 'A' or 'The'. Do not repeat the product name in the body. Highlight benefits, durability, and applications.

        Provide only the final JSON object as your response.
        """
        message = HumanMessage(
            content=[
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(st.session_state.image_bytes).decode('utf-8')}"}},
            ]
        )
        response_content = invoke_text_model_with_tracking(llm, message)
        listing_data = safe_json_parse(response_content)

        if listing_data:
            st.session_state.final_listing = listing_data
            
            product_name = listing_data.get("product_name")
            specs = listing_data.get("specifications")
            new_images = generate_b2b_catalog_images(product_name, specs)

            main_image = st.session_state.image_bytes

            if new_images:
                st.session_state.final_image_bytes_list = [main_image] + new_images
            else:
                st.session_state.final_image_bytes_list = [main_image]

            if st.session_state.create_all_flow:
                current_result = {
                    "listing_data": listing_data,
                    "final_image_bytes_list": st.session_state.final_image_bytes_list,
                    "image_mime_type": st.session_state.image_mime_type
                }
                st.session_state.all_final_listings.append(current_result)

                current_index = st.session_state.processing_index
                total_products = len(st.session_state.products_to_process)

                if (current_index + 1) < total_products:
                    st.session_state.step = "confirm_single_product_creation"
                else:
                    st.session_state.step = "display_all_results"
            else:
                st.session_state.step = "display_results"
            st.rerun()
        else:
            st.error("Failed to generate the final listing. The model's response was not in the expected format. Please try again.")
            st.write("Model Response:", response_content) # For debugging


if st.session_state.step == "generate_additional_images":
    listing = st.session_state.final_listing
    product_name = listing.get("product_name")
    specs = listing.get("specifications")
    
    new_images = generate_b2b_catalog_images(product_name, specs)

    main_image = st.session_state.image_bytes
    
    if new_images:
        st.session_state.final_image_bytes_list = [main_image] + new_images
    else:
        st.session_state.final_image_bytes_list = [main_image]

    # --- CRITICAL FIX: Smarter Routing Logic ---
    if st.session_state.create_all_flow:
        # If we are in the "Create All" flow, store the result.
        current_result = {
            "listing_data": listing,
            "final_image_bytes_list": st.session_state.final_image_bytes_list,
            "image_mime_type": st.session_state.image_mime_type
        }
        st.session_state.all_final_listings.append(current_result)

        # Now, check if this was the last product in the list.
        current_index = st.session_state.processing_index
        total_products = len(st.session_state.products_to_process)

        if (current_index + 1) < total_products:
            st.session_state.step = "confirm_single_product_creation"
        else:
            st.session_state.step = "display_all_results"
    else:
        # If we are in a single product flow, go to the single display page.
        st.session_state.step = "display_results"
        
    st.rerun()

# --- NEW STEP: Intermediate Confirmation for "Create All" Flow ---
if st.session_state.step == "confirm_single_product_creation":
    current_index = st.session_state.processing_index
    total_products = len(st.session_state.products_to_process)
    product_name = st.session_state.products_to_process[current_index]
    
    render_product_listing(
        product_id="single_product", # A unique string for this case
        listing_data=st.session_state.final_listing, 
        image_bytes_list=st.session_state.final_image_bytes_list, 
        image_mime_type=st.session_state.image_mime_type
    )
    
    st.markdown("---")
    st.subheader("What's next?")

    col1, col2 = st.columns(2)

    # Check if there are more products to process
    if (current_index + 1) < total_products:
        if col1.button("➡️ Proceed with Next Product", use_container_width=True, type="primary"):
            st.session_state.processing_index += 1
            st.session_state.step = "extract_selected_product" # Loop back to the start
            st.rerun()
    else:
        # This was the last product
        if col1.button("✅ Finish and View All Products", use_container_width=True, type="primary"):
            st.session_state.step = "display_all_results"
            st.rerun()

    if col2.button("🔄 Recreate This Product Again", use_container_width=True):
        # Don't increment the index, just re-run the process for the same item
        st.session_state.all_final_listings.pop() # Remove the last (bad) result
        st.session_state.step = "extract_selected_product"
        st.rerun()


if st.session_state.step == "display_results":
    render_product_listing(
        product_id="single_product", # A unique string for this case
        listing_data=st.session_state.final_listing, 
        image_bytes_list=st.session_state.final_image_bytes_list, 
        image_mime_type=st.session_state.image_mime_type
    )

# --- NEW FINAL PAGE: Display All Generated Products ---
if st.session_state.step == "display_all_results":
    st.success("## 🚀 All Products Generated Successfully!")
    st.write("Below are all the product listings created during this session.")
    
    for i, result in enumerate(st.session_state.all_final_listings):
        st.markdown("---")
        render_product_listing(
        product_id=i, # The loop index is a perfect unique ID
        listing_data=result["listing_data"],
        image_bytes_list=result["final_image_bytes_list"],
        image_mime_type=result["image_mime_type"]
        )
