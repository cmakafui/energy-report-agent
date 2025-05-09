# gemini_app_v5.py
import streamlit as st
import os
import tempfile
import io
import re
from datetime import datetime
from pathlib import Path
import base64
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    ListItem,
    ListFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

# AGNO imports
from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure the Streamlit page
st.set_page_config(page_title="Thermal Imaging Analysis", layout="wide")

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
    
    .report-header {
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
    }
    .report-section {
        margin-top: 1.5rem;
    }
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
                x1, y1, x2, y2 = anomaly["coordinates"]
                x1, y1 = int(x1 * width), int(y1 * height)
                x2, y2 = int(x2 * width), int(y2 * height)

                # Verify coordinates
                if x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
                    # Set color based on severity
                    if anomaly["severity"].lower() == "minor":
                        color = "#6f9b45"  # Green
                    elif anomaly["severity"].lower() == "moderate":
                        color = "#9b7f45"  # Yellow
                    else:  # severe
                        color = "#9b4545"  # Red

                    # Draw rectangle with thick line
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

                    # Add a label with background
                    text = f"{i + 1}: {anomaly['severity'].upper()}"
                    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
                    # Draw text background
                    draw.rectangle(
                        [x1, max(0, y1 - text_height - 4), x1 + text_width + 4, y1],
                        fill=color,
                    )
                    # Draw text
                    draw.text(
                        (x1 + 2, y1 - text_height - 2), text, fill="white", font=font
                    )
            except Exception as e:
                st.warning(f"Failed to draw anomaly {i + 1}: {e}")
                continue

        # Convert back to bytes
        output = io.BytesIO()
        img.save(output, format="JPEG")
        return output.getvalue()

    except Exception as e:
        st.error(f"Error drawing bounding boxes: {e}")
        return image_data


# Enhanced markdown parser function for PDF
def parse_markdown_to_pdf_elements(markdown_text, styles):
    """
    Convert markdown text to ReportLab elements

    Args:
        markdown_text (str): Text in markdown format
        styles (dict): ReportLab styles dictionary

    Returns:
        list: List of ReportLab elements
    """
    elements = []

    # Create custom styles
    title_style = ParagraphStyle(
        name="CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceBefore=12,
        spaceAfter=12,
    )
    heading1_style = ParagraphStyle(
        name="CustomHeading1",
        parent=styles["Heading1"],
        fontSize=16,
        spaceBefore=10,
        spaceAfter=10,
    )
    heading2_style = ParagraphStyle(
        name="CustomHeading2",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=8,
        spaceAfter=8,
    )
    heading3_style = ParagraphStyle(
        name="CustomHeading3",
        parent=styles["Heading3"],
        fontSize=12,
        spaceBefore=6,
        spaceAfter=6,
    )
    normal_style = ParagraphStyle(
        name="CustomNormal",
        parent=styles["Normal"],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=3,
    )
    bullet_style = ParagraphStyle(
        name="BulletStyle",
        parent=styles["Normal"],
        fontSize=10,
        leftIndent=20,
        spaceBefore=2,
        spaceAfter=2,
    )

    # Split into lines and process
    lines = markdown_text.split("\n")
    i = 0

    current_list_items = []
    in_list = False
    bullet_pattern = re.compile(r"^\s*[\*\-]\s+(.+)$")
    numbered_pattern = re.compile(r"^\s*\d+\.\s+(.+)$")

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            if in_list and current_list_items:
                # End of list detected, add the list to elements
                list_flowable = ListFlowable(current_list_items, bulletType="bullet")
                elements.append(list_flowable)
                current_list_items = []
                in_list = False
            continue

        # Headers
        if line.startswith("# "):
            elements.append(Paragraph(line[2:], title_style))
        elif line.startswith("## "):
            elements.append(Paragraph(line[3:], heading1_style))
        elif line.startswith("### "):
            elements.append(Paragraph(line[4:], heading2_style))
        elif line.startswith("#### "):
            elements.append(Paragraph(line[5:], heading3_style))

        # Bullet lists
        elif bullet_pattern.match(line) or numbered_pattern.match(line):
            in_list = True

            if bullet_pattern.match(line):
                text = bullet_pattern.match(line).group(1)
            else:  # numbered list
                text = numbered_pattern.match(line).group(1)

            # Process formatting within list items
            text = process_inline_formatting(text)

            current_list_items.append(ListItem(Paragraph(text, bullet_style)))

        # Regular paragraph
        else:
            if in_list and current_list_items:
                # End of list detected, add the list to elements
                list_flowable = ListFlowable(current_list_items, bulletType="bullet")
                elements.append(list_flowable)
                current_list_items = []
                in_list = False

            # Join multi-line paragraphs
            paragraph_lines = [line]
            j = i + 1
            while (
                j < len(lines)
                and lines[j].strip()
                and not lines[j].strip().startswith(("#", "*", "-", "1."))
            ):
                paragraph_lines.append(lines[j].strip())
                j += 1
            i = j - 1  # Adjust the index for the next iteration

            # Process the paragraph with inline formatting
            paragraph_text = " ".join(paragraph_lines)
            paragraph_text = process_inline_formatting(paragraph_text)

            elements.append(Paragraph(paragraph_text, normal_style))

        i += 1

    # Handle any remaining list items
    if in_list and current_list_items:
        list_flowable = ListFlowable(current_list_items, bulletType="bullet")
        elements.append(list_flowable)

    return elements


def process_inline_formatting(text):
    """Process inline markdown formatting like bold, italic, etc."""

    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)

    # Italic: *text* or _text_
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = re.sub(r"_(.*?)_", r"<i>\1</i>", text)

    return text


# Create PDF report function
def create_pdf_report(final_report, building_info, image_data_pairs, output_path):
    """Generate a PDF report based on markdown content and images"""

    # Create a temporary directory for images
    temp_img_dir = tempfile.mkdtemp()
    temp_img_files = []

    try:
        # Setup document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        # Styles
        styles = getSampleStyleSheet()

        # Set up color for energy rating
        rating_colors = {
            "A": colors.HexColor("#2c6e49"),
            "B": colors.HexColor("#4d9355"),
            "C": colors.HexColor("#6f9b45"),
            "D": colors.HexColor("#9b9b45"),
            "E": colors.HexColor("#9b7f45"),
            "F": colors.HexColor("#9b6245"),
            "G": colors.HexColor("#9b4545"),
        }

        # Extract energy rating from the report
        energy_rating = "D"  # Default value
        for line in final_report.split("\n"):
            if "Energy Rating" in line and ":" in line:
                rating_text = line.split(":")[-1].strip()
                if rating_text and rating_text[0] in "ABCDEFG":
                    energy_rating = rating_text[0]
                    break

        # Create elements list for the PDF
        elements = []

        # Add title and building info
        title_style = ParagraphStyle(
            name="ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=12
        )
        elements.append(Paragraph("Energy Efficiency Report", title_style))
        elements.append(Spacer(1, 0.2 * inch))

        # Add building information table
        building_data = [
            ["Building Type:", f"{building_info['type']}"],
            ["Location:", f"{building_info['location']}"],
            ["Year of Construction:", f"{building_info['year']}"],
            ["Inspection Date:", f"{datetime.now():%B %d, %Y}"],
            ["Energy Rating:", f"{energy_rating}"],
        ]

        t = Table(building_data, colWidths=[2 * inch, 3.5 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("TEXTCOLOR", (0, 0), (-1, -2), colors.black),
                    ("TEXTCOLOR", (0, -1), (0, -1), colors.black),
                    (
                        "TEXTCOLOR",
                        (1, -1),
                        (1, -1),
                        rating_colors.get(energy_rating, colors.black),
                    ),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, -1), (1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (1, -1), (1, -1), 14),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        elements.append(t)
        elements.append(Spacer(1, 0.5 * inch))

        # Process markdown report using enhanced parser
        markdown_elements = parse_markdown_to_pdf_elements(final_report, styles)
        elements.extend(markdown_elements)

        # Add image gallery
        gallery_title = ParagraphStyle(
            name="GalleryTitle", parent=styles["Heading1"], fontSize=16, spaceAfter=10
        )
        elements.append(Paragraph("Analyzed Images", gallery_title))
        elements.append(Spacer(1, 0.2 * inch))

        # Add image pairs (up to 3 per page)
        for i, (rgb_data, thermal_data) in enumerate(image_data_pairs):
            if i > 0 and i % 3 == 0:
                elements.append(Spacer(1, 0.5 * inch))

            # Save images to temporary files that ReportLab can read
            rgb_img_path = os.path.join(temp_img_dir, f"rgb_{i}.jpg")
            thermal_img_path = os.path.join(temp_img_dir, f"thermal_{i}.jpg")

            # Convert bytes to images and save to disk
            rgb_img = PILImage.open(io.BytesIO(rgb_data))
            thermal_img = PILImage.open(io.BytesIO(thermal_data))

            # Resize if needed
            max_dim = 1000  # Maximum dimension for PDF images
            if rgb_img.width > max_dim or rgb_img.height > max_dim:
                ratio = min(max_dim / rgb_img.width, max_dim / rgb_img.height)
                new_size = (int(rgb_img.width * ratio), int(rgb_img.height * ratio))
                rgb_img = rgb_img.resize(new_size, PILImage.Resampling.LANCZOS)

            if thermal_img.width > max_dim or thermal_img.height > max_dim:
                ratio = min(max_dim / thermal_img.width, max_dim / thermal_img.height)
                new_size = (
                    int(thermal_img.width * ratio),
                    int(thermal_img.height * ratio),
                )
                thermal_img = thermal_img.resize(new_size, PILImage.Resampling.LANCZOS)

            # Save to disk
            rgb_img.save(rgb_img_path, format="JPEG")
            thermal_img.save(thermal_img_path, format="JPEG")

            # Keep track of files to clean up
            temp_img_files.extend([rgb_img_path, thermal_img_path])

            # Calculate display dimensions based on aspect ratio
            max_width = 2.5 * inch
            rgb_aspect = rgb_img.width / rgb_img.height
            thermal_aspect = thermal_img.width / thermal_img.height

            # Create table with two images side by side
            image_data = [
                [
                    RLImage(
                        rgb_img_path, width=max_width, height=max_width / rgb_aspect
                    ),
                    RLImage(
                        thermal_img_path,
                        width=max_width,
                        height=max_width / thermal_aspect,
                    ),
                ]
            ]

            caption_data = [[f"RGB Image {i + 1}", f"Thermal Image {i + 1}"]]

            # Images
            img_table = Table(image_data, colWidths=[3 * inch, 3 * inch])
            img_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            elements.append(img_table)

            # Captions
            cap_table = Table(caption_data, colWidths=[3 * inch, 3 * inch])
            cap_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Oblique"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ]
                )
            )
            elements.append(cap_table)
            elements.append(Spacer(1, 0.3 * inch))

        # Build the PDF
        doc.build(elements)
        return output_path
    finally:
        # Clean up temporary files
        for file_path in temp_img_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

        try:
            if os.path.exists(temp_img_dir):
                os.rmdir(temp_img_dir)
        except Exception:
            pass


# Setup the analysis agent with Google Gemini
def get_analysis_agent():
    return Agent(
        model=Gemini(id="gemini-2.5-flash-preview-04-17"),
        agent_id="energy-analysis",
        name="Energy Analysis Agent",
        tools=[ReasoningTools()],
        instructions=[
            "You are a professional thermographic engineer analyzing RGB and thermal image pairs to identify energy efficiency issues",
            "Follow the pattern of professional thermography reports by identifying thermal anomalies with precision",
            "For each anomaly, provide coordinates as [x1, y1, x2, y2] representing normalized bounding box coordinates (values between 0.0-1.0)",
            "Rate each issue's severity as 'minor', 'moderate', or 'severe' based on temperature differential and impact",
            "Provide detailed, technically sound descriptions of each issue and its underlying causes",
            "Estimate potential energy loss and repair costs in euro ranges",
            "Format findings in professional, concise markdown language suitable for engineering reports",
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
            "You are a professional energy efficiency consultant generating comprehensive reports",
            "Analyze the findings from multiple thermal image analyses to create a cohesive report",
            "Follow the structure of professional thermography reports with an executive summary, findings, and recommendations",
            "Provide a justified EU Energy Rating (A-G) based on the number and severity of issues found",
            "Generate prioritized recommendations with accurate cost ranges and expected energy savings",
            "Include implementation timeframes and ROI calculations for recommendations",
            "Use technical but accessible language appropriate for property owners and facility managers",
            "Format the report professionally using markdown with appropriate headers and sections",
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
                anomaly_data = {
                    "severity": "unknown",
                    "coordinates": [0.1, 0.1, 0.3, 0.3],
                }

                # Extract severity
                severity_lines = [
                    line for line in section.split("\n") if "- Severity:" in line
                ]
                if severity_lines:
                    severity = severity_lines[0].split("- Severity:")[-1].strip()
                    anomaly_data["severity"] = severity

                # Extract coordinates
                coord_lines = [
                    line for line in section.split("\n") if "- Coordinates:" in line
                ]
                if coord_lines:
                    coords_str = coord_lines[0].split("- Coordinates:")[-1].strip()

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
st.title("Thermal Imaging Analysis")
st.caption("Upload RGB and thermal image pairs to identify energy efficiency issues")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(
    ["📸 Upload Images", "🔍 Analysis Results", "📊 Final Report"]
)

# Building Information form in a sidebar
with st.sidebar:
    st.header("Building Information")
    building_year = st.number_input(
        "Construction Year", min_value=1900, max_value=2025, value=1980
    )
    building_type = st.selectbox(
        "Building Type", ["Residential", "Commercial", "Industrial"]
    )
    location = st.text_input("Location", "Helsinki")

    st.divider()
    st.markdown("### About")
    st.markdown("""
    This application analyzes RGB and thermal image pairs to identify energy 
    efficiency issues. It creates detailed reports with annotated images, 
    anomaly detection, and recommendations for improvements.
    
    Based on professional thermography standards, the analysis provides:
    - Thermal anomaly detection
    - Severity classification
    - Recommended repairs
    - Energy savings estimates
    """)

# Tab 1: Upload Images
with tab1:
    st.subheader("Step 1: Upload RGB Images")
    st.caption("Upload standard photographs of the building areas you want to analyze")
    rgb_images = st.file_uploader(
        "Upload RGB images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    st.subheader("Step 2: Upload Thermal Images")
    st.caption("Upload matching thermal images of the same areas")
    thermal_images = st.file_uploader(
        "Upload thermal images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    col1, col2 = st.columns(2)
    with col1:
        start_analysis = st.button("Generate Analysis", type="primary")
    with col2:
        if st.session_state.final_report:
            if st.button("Reset Analysis", type="secondary"):
                # Clear previous results
                st.session_state.image_analyses = []
                st.session_state.final_report = None
                st.session_state.rgb_image_data = []
                st.session_state.thermal_image_data = []
                st.session_state.annotated_rgb_data = []
                st.session_state.annotated_thermal_data = []
                st.session_state.analysis_complete = False
                st.experimental_rerun()

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
                for i, (rgb_img, thermal_img) in enumerate(
                    zip(rgb_images, thermal_images)
                ):
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
                for i, (rgb_img, thermal_img) in enumerate(
                    zip(rgb_images, thermal_images)
                ):
                    st.write(f"#### Analyzing Image Pair {i + 1}")

                    # Prepare the paths
                    rgb_path = os.path.join(temp_dir, f"rgb_{i}.jpg")
                    thermal_path = os.path.join(temp_dir, f"thermal_{i}.jpg")

                    # Create the prompt for analysis
                    prompt = f"""
                    Analyze this RGB-thermal image pair following professional thermography reporting standards:

                    Building Information:
                    - Year of Construction: {building_year}
                    - Building Type: {building_type}
                    - Location: {location}
                    - Analysis Date: {datetime.now():%B %d, %Y}

                    Expected Format:

                    ## Summary
                    [Brief, technical summary of findings - 2-3 sentences]

                    ## Thermal Anomalies

                    ### Anomaly 1
                    - Location: [precise location description]
                    - Severity: [minor/moderate/severe]
                    - Coordinates: [0.1, 0.2, 0.3, 0.4]
                    - Description: [technical description of the thermal pattern]
                    - Temperature Differential: [estimated °C difference]
                    - Probable Cause: [professional assessment]

                    ### Anomaly 2
                    ...

                    ## Recommendations

                    ### Recommendation 1
                    - Description: [technical explanation of required repair/improvement]
                    - Estimated Cost: [€XXX - €XXX]
                    - Estimated Annual Energy Savings: [kWh and €XXX]
                    - Priority: [immediate/high/medium/low]
                    - Implementation Time: [X days/weeks]

                    ### Recommendation 2
                    ...

                    Identify all visible thermal anomalies including: air leakage, insulation deficiencies, thermal bridges, moisture issues, and mechanical/electrical faults.
                    """

                    with st.spinner(f"Analyzing image pair {i + 1}..."):
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
                                st.success(
                                    f"Found {len(anomalies)} anomalies to highlight"
                                )

                                # Process RGB image
                                rgb_annotated = draw_bounding_boxes(
                                    st.session_state.rgb_image_data[i], anomalies
                                )
                                st.session_state.annotated_rgb_data.append(
                                    rgb_annotated
                                )

                                # Process thermal image
                                thermal_annotated = draw_bounding_boxes(
                                    st.session_state.thermal_image_data[i], anomalies
                                )
                                st.session_state.annotated_thermal_data.append(
                                    thermal_annotated
                                )
                            else:
                                st.warning("No anomalies with valid coordinates found.")
                                # Use the original images if no anomalies found
                                st.session_state.annotated_rgb_data.append(
                                    st.session_state.rgb_image_data[i]
                                )
                                st.session_state.annotated_thermal_data.append(
                                    st.session_state.thermal_image_data[i]
                                )
                        else:
                            st.error(f"Analysis failed for image pair {i + 1}")
                            # Use the original images if analysis fails
                            st.session_state.annotated_rgb_data.append(
                                st.session_state.rgb_image_data[i]
                            )
                            st.session_state.annotated_thermal_data.append(
                                st.session_state.thermal_image_data[i]
                            )

                    progress_bar.progress(0.2 + 0.6 * (i + 1) / len(rgb_images))

                # Generate the final report
                if len(st.session_state.image_analyses) > 0:
                    with st.spinner("Generating final report..."):
                        report_agent = get_report_agent()

                        # Combine all analyses
                        analyses_text = "\n\n".join(st.session_state.image_analyses)

                        final_prompt = f"""
                        Generate a professional energy efficiency report that follows thermography industry standards:

                        Building Information:
                        - Year: {building_year}
                        - Type: {building_type}
                        - Location: {location}
                        - Inspection Date: {datetime.now():%B %d, %Y}

                        Based on the following detailed analyses:
                        {analyses_text}

                        Create a structured report with the following sections:

                        # Energy Efficiency Report

                        ## Executive Summary
                        [Professional summary of key findings and recommendations - 3-4 sentences]

                        ## Building Information
                        [Summarize the building specifications and analysis conditions]

                        ## Energy Rating
                        [Assign an EU Energy Rating from A to G based on findings, with clear justification for the rating]

                        ## Summary of Findings
                        [Concise overview of the main thermal anomalies detected]

                        ## Priority Recommendations
                        [List recommendations in order of urgency, with costs, energy savings, and ROI]

                        ## Financial Analysis
                        [Summarize total estimated costs, energy savings, and payback periods]

                        ## Conclusion
                        [Final assessment with next steps and potential rating improvements]
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
                    next_lines = analysis.split("\n")[
                        analysis.split("\n").index(line) + 1 :
                    ]
                    for next_line in next_lines:
                        if next_line.strip() and not next_line.startswith("#"):
                            summary = next_line.strip()
                            break
                    break

            with st.expander(f"Image Pair {i + 1}: {summary}", expanded=i == 0):
                # Add a two-column layout for images and analysis
                img_col, text_col = st.columns([1, 2])

                with img_col:
                    # Check if we have all the required data
                    if (
                        i < len(st.session_state.rgb_image_data)
                        and i < len(st.session_state.thermal_image_data)
                        and i < len(st.session_state.annotated_rgb_data)
                        and i < len(st.session_state.annotated_thermal_data)
                    ):
                        # Create tabs for viewing original vs annotated images
                        orig_tab, annot_tab = st.tabs(["Original", "Annotated"])

                        with orig_tab:
                            st.image(
                                st.session_state.rgb_image_data[i],
                                caption="RGB Image",
                                use_container_width=True,
                            )
                            st.image(
                                st.session_state.thermal_image_data[i],
                                caption="Thermal Image",
                                use_container_width=True,
                            )

                        with annot_tab:
                            st.image(
                                st.session_state.annotated_rgb_data[i],
                                caption="Annotated RGB",
                                use_container_width=True,
                            )
                            st.image(
                                st.session_state.annotated_thermal_data[i],
                                caption="Annotated Thermal",
                                use_container_width=True,
                            )

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

        # Show images in a grid layout, 2 images per row
        img_rows = (len(st.session_state.rgb_image_data) + 1) // 2
        for r in range(img_rows):
            cols = st.columns(2)
            for j in range(2):
                i = r * 2 + j
                if i < len(st.session_state.rgb_image_data):
                    if viz_type == "Original":
                        with cols[j]:
                            st.image(
                                st.session_state.rgb_image_data[i],
                                caption=f"RGB Image {i + 1}",
                                use_container_width=True,
                            )
                            st.image(
                                st.session_state.thermal_image_data[i],
                                caption=f"Thermal Image {i + 1}",
                                use_container_width=True,
                            )
                    else:
                        with cols[j]:
                            st.image(
                                st.session_state.annotated_rgb_data[i],
                                caption=f"RGB Image {i + 1} with Anomalies",
                                use_container_width=True,
                            )
                            st.image(
                                st.session_state.annotated_thermal_data[i],
                                caption=f"Thermal Image {i + 1} with Anomalies",
                                use_container_width=True,
                            )

        # Display the report
        st.markdown("---")
        st.markdown(st.session_state.final_report)

        # Create columns for download options
        col1, col2 = st.columns(2)

        # Download button for markdown report
        with col1:
            st.download_button(
                "Download Markdown Report",
                data=st.session_state.final_report,
                file_name=f"energy_report_{location}_{datetime.now():%Y%m%d}.md",
                mime="text/markdown",
            )

        # PDF report generation and download
        with col2:
            if st.button("Generate PDF Report", type="primary"):
                with st.spinner("Generating PDF..."):
                    # Create temporary file for PDF
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_file:
                        pdf_path = tmp_file.name

                    # Generate PDF
                    building_info = {
                        "type": building_type,
                        "location": location,
                        "year": building_year,
                    }

                    # Create image pairs for PDF
                    image_pairs = []
                    for i in range(len(st.session_state.annotated_rgb_data)):
                        image_pairs.append(
                            (
                                st.session_state.annotated_rgb_data[i],
                                st.session_state.annotated_thermal_data[i],
                            )
                        )

                    # Generate PDF
                    create_pdf_report(
                        st.session_state.final_report,
                        building_info,
                        image_pairs,
                        pdf_path,
                    )

                    # Read PDF and provide download link
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()

                    # Clean up temporary file
                    try:
                        os.unlink(pdf_path)
                    except Exception:
                        pass

                    # Create download link for PDF
                    b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="energy_report_{location}_{datetime.now():%Y%m%d}.pdf">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
