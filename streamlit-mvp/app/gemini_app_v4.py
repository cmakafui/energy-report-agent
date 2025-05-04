# gemini_app_v4.py
import streamlit as st
import os
import tempfile
import io
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

# AGNO imports
from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools

# Load environment variables (expects GOOGLE_API_KEY)
load_dotenv()

# Configure the Streamlit page
st.set_page_config(page_title="Energy Rating Analysis", layout="wide")

# Initialize session state variables
if "rgb_image_data" not in st.session_state:
    st.session_state.rgb_image_data = []
if "thermal_image_data" not in st.session_state:
    st.session_state.thermal_image_data = []
if "annotated_rgb_data" not in st.session_state:
    st.session_state.annotated_rgb_data = []
if "annotated_thermal_data" not in st.session_state:
    st.session_state.annotated_thermal_data = []
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "image_analyses" not in st.session_state:
    st.session_state.image_analyses = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None

# Create a temp directory for our annotated images
TEMP_DIR = tempfile.mkdtemp()

# Custom CSS
st.markdown(
    """
    <style>
    .energy-rating {
        font: bold 48px sans-serif;
        color: white;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }
    .rating-A {background-color: #2c6e49;}
    .rating-B {background-color: #4d9355;}
    .rating-C {background-color: #6f9b45;}
    .rating-D {background-color: #9b9b45;}
    .rating-E {background-color: #9b7f45;}
    .rating-F {background-color: #9b6245;}
    .rating-G {background-color: #9b4545;}
    
    /* Severity indicators */
    .severity-minor, .severity-moderate, .severity-severe {
        font-weight: bold;
    }
    .severity-minor {color: #6f9b45;}
    .severity-moderate {color: #9b7f45;}
    .severity-severe {color: #9b4545;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Define bounding box drawing function
def draw_bounding_boxes(image_data, anomalies):
    """
    Draw bounding boxes on an image for identified anomalies
    
    Args:
        image_data (bytes): Image data as bytes
        anomalies (list): List of dictionaries with 'coordinates' and 'severity' keys
        
    Returns:
        bytes: Image data with bounding boxes
    """
    try:
        # Open image from bytes
        img = PILImage.open(io.BytesIO(image_data))
        draw = ImageDraw.Draw(img)
        
        # Get image dimensions
        width, height = img.size
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw each anomaly
        for i, anomaly in enumerate(anomalies):
            # Extract coordinates and convert from normalized to pixel values
            try:
                x1, y1, x2, y2 = anomaly['coordinates']
                x1, y1 = int(x1 * width), int(y1 * height)
                x2, y2 = int(x2 * width), int(y2 * height)
                
                # Verify coordinates
                if x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
                    # Set color based on severity
                    if anomaly['severity'].lower() == 'minor':
                        color = "#6f9b45"  # Green
                    elif anomaly['severity'].lower() == 'moderate':
                        color = "#9b7f45"  # Yellow
                    else:  # severe
                        color = "#9b4545"  # Red
                        
                    # Draw rectangle with thick line
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                    
                    # Add a label with background
                    text = f"{i+1}: {anomaly['severity'].upper()}"
                    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
                    # Draw text background
                    draw.rectangle(
                        [x1, max(0, y1-text_height-4), x1 + text_width + 4, y1],
                        fill=color
                    )
                    # Draw text
                    draw.text((x1+2, y1-text_height-2), text, fill="white", font=font)
            except Exception as e:
                st.warning(f"Failed to draw anomaly {i+1}: {e}")
                continue
        
        # Convert back to bytes
        output = io.BytesIO()
        img.save(output, format="JPEG")
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error drawing bounding boxes: {e}")
        return image_data


# Setup the analysis agent with Google Gemini
def get_analysis_agent():
    return Agent(
        model=Gemini(id="gemini-2.5-flash-preview-04-17"),
        agent_id="energy-analysis",
        name="Energy Analysis Agent",
        tools=[ReasoningTools()],
        instructions=[
            "You are an AI agent that analyzes RGB and thermal image pairs to identify energy efficiency issues",
            "Identify thermal anomalies, insulation problems, air leaks, and moisture issues",
            "Provide detailed findings with severity ratings (minor/moderate/severe)",
            "For each anomaly, provide the coordinates as [x1, y1, x2, y2] representing the normalized top-left and bottom-right corners of a bounding box around the problem area",
            "Coordinates must be between 0.0 and 1.0 with the format [0.1, 0.2, 0.3, 0.4]",
            "Format findings in clear, understandable markdown language",
        ],
        markdown=True,
    )


def get_report_agent():
    return Agent(
        model=Gemini(id="gemini-2.5-flash-preview-04-17"),
        agent_id="energy-report",
        name="Energy Report Agent",
        tools=[ReasoningTools()],
        instructions=[
            "You are an AI agent that generates comprehensive energy efficiency reports",
            "Analyze the findings from multiple image pair analyses",
            "Provide an executive summary and EU Energy Rating (A-G)",
            "Generate priority recommendations with costs and savings",
            "Provide a conclusion with potential rating improvements",
        ],
        markdown=True,
    )


# Function to get agent analysis results
def get_agent_analysis(agent, prompt, images=None):
    """Use agent to analyze images and return response content as string"""
    try:
        response = agent.run(prompt, images=images)
        
        if response is None:
            return None

        if hasattr(response, "content") and response.content is not None:
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return None


# Function to render an energy rating badge
def render_energy_rating(rating):
    return f'<div class="energy-rating rating-{rating}">{rating}</div>'


# Function to render a severity label
def render_severity(severity):
    if severity.lower() == "minor":
        return '<span class="severity-minor">MINOR</span>'
    elif severity.lower() == "moderate":
        return '<span class="severity-moderate">MODERATE</span>'
    elif severity.lower() == "severe":
        return '<span class="severity-severe">SEVERE</span>'
    return severity


# Function to extract anomalies from analysis text
def extract_anomalies_from_analysis(analysis_text):
    """Extract anomaly information from the analysis text"""
    anomalies = []
    
    # Split by sections to find anomalies
    sections = analysis_text.split("###")
    
    for section in sections:
        if "Anomaly" in section and "- Severity:" in section:
            try:
                # Default values
                anomaly_data = {"severity": "unknown", "coordinates": [0.1, 0.1, 0.3, 0.3]}
                
                # Extract severity
                severity_lines = [line for line in section.split("\n") if "- Severity:" in line]
                if severity_lines:
                    severity = severity_lines[0].split("- Severity:")[1].strip()
                    anomaly_data["severity"] = severity
                
                # Extract coordinates
                coord_lines = [line for line in section.split("\n") if "- Coordinates:" in line]
                if coord_lines:
                    coords_str = coord_lines[0].split("- Coordinates:")[1].strip()
                    
                    # Handle various format possibilities
                    coords_str = coords_str.strip("[]")
                    coords = []
                    for coord in coords_str.split(","):
                        try:
                            coords.append(float(coord.strip()))
                        except ValueError:
                            # Skip invalid coordinates
                            pass
                    
                    # Ensure we have 4 coordinates
                    if len(coords) == 4:
                        # Ensure coordinates are in range [0,1]
                        coords = [max(0.0, min(c, 1.0)) for c in coords]
                        anomaly_data["coordinates"] = coords
                
                # Add this anomaly to our list
                anomalies.append(anomaly_data)
                
            except Exception as e:
                st.warning(f"Error parsing anomaly: {e}")
                continue
    
    if not anomalies:
        st.warning("No anomalies with valid coordinates found in the analysis.")
        
    return anomalies


# Header with title
st.title("Energy Rating Report Generator")
st.caption("Upload RGB and thermal image pairs for energy efficiency analysis")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📸 Upload Images", "🔍 Analysis Results", "📊 Final Report"])

# Building Information form in a sidebar
with st.sidebar:
    st.header("Building Information")
    building_year = st.number_input("Construction Year", min_value=1900, max_value=2025, value=1980)
    building_type = st.selectbox("Building Type", ["Residential", "Commercial", "Industrial"])
    location = st.text_input("Location", "Helsinki")
    
    st.divider()
    st.markdown("### About")
    st.markdown("This app analyzes RGB and thermal image pairs to identify energy efficiency issues.")

# Tab 1: Upload Images
with tab1:
    st.subheader("Step 1: Upload RGB Images")
    rgb_images = st.file_uploader(
        "Upload RGB images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    st.subheader("Step 2: Upload Thermal Images")
    thermal_images = st.file_uploader(
        "Upload thermal images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    start_analysis = st.button("Generate Analysis", type="primary")

# Process images and generate analysis
if start_analysis:
    if not rgb_images or not thermal_images:
        st.warning("Please upload both RGB and thermal images.")
    elif len(rgb_images) != len(thermal_images):
        st.error("Number of RGB and thermal images must match.")
    else:
        # Clear previous results
        st.session_state.image_analyses = []
        st.session_state.final_report = None
        st.session_state.rgb_image_data = []
        st.session_state.thermal_image_data = []
        st.session_state.annotated_rgb_data = []
        st.session_state.annotated_thermal_data = []
        
        with tab2:
            st.subheader("Processing Images...")
            progress_bar = st.progress(0)
            
            # Create a simple temporary directory structure
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded images to temp files for processing
                for i, (rgb_img, thermal_img) in enumerate(zip(rgb_images, thermal_images)):
                    # Save image data to session state
                    rgb_data = rgb_img.getvalue()
                    thermal_data = thermal_img.getvalue()
                    
                    st.session_state.rgb_image_data.append(rgb_data)
                    st.session_state.thermal_image_data.append(thermal_data)
                    
                    # Save to disk for AGNO processing
                    rgb_path = os.path.join(temp_dir, f"rgb_{i}.jpg")
                    thermal_path = os.path.join(temp_dir, f"thermal_{i}.jpg")
                    
                    with open(rgb_path, "wb") as f:
                        f.write(rgb_data)
                    with open(thermal_path, "wb") as f:
                        f.write(thermal_data)
                
                progress_bar.progress(0.2)
                
                # Create the analysis agent
                agent = get_analysis_agent()
                
                # Process each image pair
                for i, (rgb_img, thermal_img) in enumerate(zip(rgb_images, thermal_images)):
                    st.write(f"#### Analyzing Image Pair {i+1}")
                    
                    # Prepare the paths
                    rgb_path = os.path.join(temp_dir, f"rgb_{i}.jpg")
                    thermal_path = os.path.join(temp_dir, f"thermal_{i}.jpg")
                    
                    # Create the prompt for analysis
                    prompt = f"""
                    Analyze this RGB-thermal image pair for energy efficiency issues:

                    Building Info:
                    - Year: {building_year}
                    - Type: {building_type}
                    - Location: {location}

                    Identify anomalies, severity (minor/moderate/severe), recommendations, and cost estimates.
                    For each anomaly, provide the exact coordinates as [x1, y1, x2, y2] representing the normalized 
                    top-left and bottom-right corners of a bounding box around the problem area.
                    
                    Each coordinate must be a number between 0.0 and 1.0.
                    
                    Format your response in this structure:
                    
                    ## Summary
                    [Brief summary of findings]
                    
                    ## Thermal Anomalies
                    
                    ### Anomaly 1
                    - Location: [where the issue is]
                    - Severity: [minor/moderate/severe]
                    - Coordinates: [0.1, 0.2, 0.3, 0.4]
                    - Description: [detailed description]
                    - Temperature Difference: [if available]
                    
                    ### Anomaly 2
                    ...
                    
                    ## Recommendations
                    
                    ### Recommendation 1
                    - Title: [clear title]
                    - Description: [detailed explanation]
                    - Estimated Cost: [cost range]
                    - Estimated Savings: [annual savings]
                    - Priority: [low/medium/high]
                    - Implementation Time: [time estimate]
                    
                    ### Recommendation 2
                    ...
                    """
                    
                    with st.spinner(f"Analyzing image pair {i+1}..."):
                        # Run analysis with Agno images
                        rgb_image = Image(filepath=Path(rgb_path))
                        thermal_image = Image(filepath=Path(thermal_path))
                        
                        analysis = get_agent_analysis(
                            agent, prompt, images=[rgb_image, thermal_image]
                        )
                        
                        if analysis:
                            st.session_state.image_analyses.append(analysis)
                            
                            # Extract anomalies from the analysis text
                            anomalies = extract_anomalies_from_analysis(analysis)
                            
                            # Draw bounding boxes on both images
                            if anomalies and len(anomalies) > 0:
                                st.success(f"Found {len(anomalies)} anomalies to highlight")
                                
                                # Process RGB image
                                rgb_annotated = draw_bounding_boxes(
                                    st.session_state.rgb_image_data[i], anomalies
                                )
                                st.session_state.annotated_rgb_data.append(rgb_annotated)
                                
                                # Process thermal image
                                thermal_annotated = draw_bounding_boxes(
                                    st.session_state.thermal_image_data[i], anomalies
                                )
                                st.session_state.annotated_thermal_data.append(thermal_annotated)
                            else:
                                st.warning("No anomalies with valid coordinates found.")
                                # Use the original images if no anomalies found
                                st.session_state.annotated_rgb_data.append(st.session_state.rgb_image_data[i])
                                st.session_state.annotated_thermal_data.append(st.session_state.thermal_image_data[i])
                        else:
                            st.error(f"Analysis failed for image pair {i+1}")
                            # Use the original images if analysis fails
                            st.session_state.annotated_rgb_data.append(st.session_state.rgb_image_data[i])
                            st.session_state.annotated_thermal_data.append(st.session_state.thermal_image_data[i])
                    
                    progress_bar.progress(0.2 + 0.6 * (i + 1) / len(rgb_images))
                
                # Generate the final report
                if len(st.session_state.image_analyses) > 0:
                    with st.spinner("Generating final report..."):
                        report_agent = get_report_agent()
                        
                        # Combine all analyses
                        analyses_text = "\n\n".join(st.session_state.image_analyses)
                        
                        final_prompt = f"""
                        Generate a comprehensive energy efficiency report:
                        - Year: {building_year}
                        - Type: {building_type}
                        - Location: {location}
                        - Date: {datetime.now():%B %d, %Y}

                        Based on the following analyses:
                        {analyses_text}

                        Create a structured report with:
                        
                        # Energy Efficiency Report
                        
                        ## Executive Summary
                        [Concise summary of findings and recommendations]
                        
                        ## Energy Rating
                        [Assign an EU Energy Rating from A to G based on findings]
                        
                        ## Main Findings
                        [List the main findings]
                        
                        ## Priority Recommendations
                        [List recommendations with costs and savings]
                        
                        ## Potential Improvements
                        [Specify potential rating improvements]
                        
                        ## Financial Summary
                        [Provide financial analysis]
                        
                        ## Conclusion
                        [Final assessment and next steps]
                        """
                        
                        final_report = get_agent_analysis(report_agent, final_prompt)
                        if final_report:
                            st.session_state.final_report = final_report
                            st.session_state.analysis_complete = True
                        
                        progress_bar.progress(1.0)

# Tab 2: Analysis Results
with tab2:
    if st.session_state.analysis_complete and st.session_state.image_analyses:
        st.header("Image Analysis Results")

        # Display each analysis in an expandable section
        for i, analysis in enumerate(st.session_state.image_analyses):
            # Extract a summary if available
            summary = "Analysis results"
            for line in analysis.split("\n"):
                if line.strip().startswith("##") and "Summary" in line:
                    next_lines = analysis.split("\n")[analysis.split("\n").index(line)+1:]
                    for next_line in next_lines:
                        if next_line.strip() and not next_line.startswith("#"):
                            summary = next_line.strip()
                            break
                    break

            with st.expander(f"Image Pair {i + 1}: {summary}", expanded=True):
                # Add a two-column layout for images and analysis
                img_col, text_col = st.columns([1, 2])
                
                with img_col:
                    # Check if we have all the required data
                    if (i < len(st.session_state.rgb_image_data) and 
                        i < len(st.session_state.thermal_image_data) and
                        i < len(st.session_state.annotated_rgb_data) and
                        i < len(st.session_state.annotated_thermal_data)):
                        
                        # Create tabs for viewing original vs annotated images
                        orig_tab, annot_tab = st.tabs(["Original", "Annotated"])
                        
                        with orig_tab:
                            st.image(st.session_state.rgb_image_data[i], caption="RGB Image", use_container_width=True)
                            st.image(st.session_state.thermal_image_data[i], caption="Thermal Image", use_container_width=True)
                        
                        with annot_tab:
                            st.image(st.session_state.annotated_rgb_data[i], caption="Annotated RGB", use_container_width=True)
                            st.image(st.session_state.annotated_thermal_data[i], caption="Annotated Thermal", use_container_width=True)
                
                with text_col:
                    # Display the full markdown analysis
                    st.markdown(analysis)

# Tab 3: Final Report
with tab3:
    if st.session_state.analysis_complete and st.session_state.final_report:
        # Extract energy rating from the report
        energy_rating = "D"  # Default value
        for line in st.session_state.final_report.split("\n"):
            if "Energy Rating" in line and ":" in line:
                rating_text = line.split(":")[-1].strip()
                if rating_text and rating_text[0] in "ABCDEFG":
                    energy_rating = rating_text[0]
                break

        # Header with Energy Rating
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(render_energy_rating(energy_rating), unsafe_allow_html=True)
        with col2:
            st.header("Energy Efficiency Report")
            st.subheader(f"{building_type} in {location}, built {building_year}")

        # Display image gallery
        st.subheader("Analyzed Images")
        viz_type = st.radio("View", ["Original", "Annotated"], horizontal=True)
        
        for i in range(len(st.session_state.rgb_image_data)):
            cols = st.columns(2)
            
            if viz_type == "Original":
                with cols[0]:
                    st.image(st.session_state.rgb_image_data[i], caption=f"RGB Image {i+1}", use_container_width=True)
                with cols[1]:
                    st.image(st.session_state.thermal_image_data[i], caption=f"Thermal Image {i+1}", use_container_width=True)
            else:
                with cols[0]:
                    st.image(st.session_state.annotated_rgb_data[i], caption=f"RGB Image {i+1} with Anomalies", use_container_width=True)
                with cols[1]:
                    st.image(st.session_state.annotated_thermal_data[i], caption=f"Thermal Image {i+1} with Anomalies", use_container_width=True)
            
            if i < len(st.session_state.rgb_image_data) - 1:
                st.divider()

        # Display the report
        st.markdown(st.session_state.final_report)

        # Download button for report
        st.download_button(
            "Download Full Report",
            data=st.session_state.final_report,
            file_name=f"energy_report_{location}_{datetime.now():%Y%m%d}.md",
            mime="text/markdown",
        )