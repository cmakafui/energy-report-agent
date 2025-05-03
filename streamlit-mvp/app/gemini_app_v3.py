# gemini_app_v3.py
import streamlit as st
import os
import tempfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# AGNO imports
from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools

# Load environment variables (expects GOOGLE_API_KEY)
load_dotenv()

# Configure the Streamlit page with a nice theme
st.set_page_config(
    page_title="Energy Rating Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://docs.agno.com",
        "Report a bug": "https://github.com/agno-ai/agno",
        "About": "# Energy Rating Analysis App\nBuilt with Streamlit and Agno.",
    },
)

# Custom CSS with more subtle colors
st.markdown(
    """
<style>
    /* Basic layout */
    .main .block-container {
        padding: 2rem 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(240, 242, 246, 0.1);
        border-radius: 4px 4px 0 0;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(230, 242, 255, 0.15);
    }
    
    /* Button styling */
    div.stButton > button:first-child {
        background-color: #2c6e49;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
    }
    div.stButton > button:hover {
        background-color: #3a7d57;
    }
    
    /* Energy rating badges */
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
    
    /* Rating colors - more muted palette */
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
    
    /* Utility classes */
    .savings-tag {
        background-color: rgba(230, 242, 255, 0.1);
        color: #81a4cd;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .info-box {
        background-color: rgba(248, 249, 250, 0.1);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #2c6e49;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Setup the analysis agent with Google Gemini
@st.cache_resource
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
            "Generate practical recommendations with estimated costs",
            "Format findings in clear, understandable markdown language",
        ],
        markdown=True,
        debug_mode=True,
    )


@st.cache_resource
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
        debug_mode=True,
    )


# Function to get agent analysis results
def get_agent_analysis(agent, prompt, images=None):
    """Use agent to analyze images and return response content as string"""
    try:
        response = agent.run(prompt, images=images)

        # Check if response is None
        if response is None:
            return None

        # Extract the content from RunResponse object
        # Different ways to access the content based on the response type
        if hasattr(response, "content") and response.content is not None:
            return response.content
        elif isinstance(response, str):
            return response
        else:
            # Convert the response object to string if none of the above work
            return str(response)
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return None


# Function to render a severity label with appropriate styling
def render_severity(severity):
    if severity.lower() == "minor":
        return '<span class="severity-minor">MINOR</span>'
    elif severity.lower() == "moderate":
        return '<span class="severity-moderate">MODERATE</span>'
    elif severity.lower() == "severe":
        return '<span class="severity-severe">SEVERE</span>'
    return severity


# Function to render an energy rating badge
def render_energy_rating(rating):
    return f'<div class="energy-rating rating-{rating}">{rating}</div>'


# Function to extract severity from text
def extract_severity(text):
    text = text.lower()
    if "minor" in text:
        return "minor"
    elif "moderate" in text:
        return "moderate"
    elif "severe" in text:
        return "severe"
    return "unknown"


# Initialize session state variables for storing image data
if "rgb_image_data" not in st.session_state:
    st.session_state.rgb_image_data = []
if "thermal_image_data" not in st.session_state:
    st.session_state.thermal_image_data = []
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "image_analyses" not in st.session_state:
    st.session_state.image_analyses = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None


# Header with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    # Placeholder for logo - you can replace with actual image
    st.markdown("🏠")
with col2:
    st.title("Energy Rating Report Generator")
    st.caption("Upload RGB and thermal image pairs for energy efficiency analysis")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(
    ["📸 Upload Images", "🔍 Analysis Results", "📊 Final Report"]
)

# Building Information form in a sidebar
with st.sidebar:
    st.header("Building Information")
    st.info("Fill in these details to get more accurate analysis", icon="ℹ️")

    building_year = st.number_input(
        "Construction Year", min_value=1900, max_value=2025, value=1980
    )
    building_type = st.selectbox(
        "Building Type", ["Residential", "Commercial", "Industrial"]
    )
    location = st.text_input("Location", "Helsinki")

    st.divider()
    st.markdown("### About")
    st.markdown(
        "This app analyzes RGB and thermal image pairs to identify energy efficiency issues."
    )
    st.markdown("Built with Streamlit and Agno framework")

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

    st.markdown(
        """
    <div class="info-box">
    <h4>How it works</h4>
    <p>This tool analyzes pairs of RGB and thermal images to identify energy efficiency issues in buildings. The AI algorithm detects thermal anomalies, insulation problems, air leaks, and moisture issues.</p>
    <ol>
        <li>Upload an equal number of RGB and thermal images</li>
        <li>Fill in building information in the sidebar</li>
        <li>Click "Generate Analysis" to process the images</li>
        <li>Review results and download the full report</li>
    </ol>
    </div>
    """,
        unsafe_allow_html=True,
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

        with tab2:
            st.subheader("Processing Images...")
            progress_bar = st.progress(0)

            with tempfile.TemporaryDirectory() as temp_dir:
                image_pairs = []
                for i, (rgb_img, thermal_img) in enumerate(
                    zip(rgb_images, thermal_images)
                ):
                    # Save image data to session state
                    st.session_state.rgb_image_data.append(rgb_img.getvalue())
                    st.session_state.thermal_image_data.append(thermal_img.getvalue())
                    
                    rgb_path = os.path.join(temp_dir, f"rgb_{i}.jpg")
                    thermal_path = os.path.join(temp_dir, f"thermal_{i}.jpg")
                    with open(rgb_path, "wb") as f:
                        f.write(rgb_img.getvalue())
                    with open(thermal_path, "wb") as f:
                        f.write(thermal_img.getvalue())

                    image_pairs.append(
                        {"rgb_path": rgb_path, "thermal_path": thermal_path}
                    )
                    progress_bar.progress(
                        (i + 1) / (len(rgb_images) * 2)
                    )  # First half for file processing

                agent = get_analysis_agent()

                for i, pair in enumerate(image_pairs):
                    st.write(f"#### Analyzing Pair {i + 1}")
                    prompt = f"""
                    Analyze this RGB-thermal image pair for energy efficiency issues:

                    Building Info:
                    - Year: {building_year}
                    - Type: {building_type}
                    - Location: {location}

                    Identify anomalies, severity (minor/moderate/severe), recommendations, and cost estimates.
                    Format your response in this structure:
                    
                    ## Summary
                    [Brief summary of findings]
                    
                    ## Thermal Anomalies
                    
                    ### Anomaly 1
                    - Location: [where the issue is]
                    - Severity: [minor/moderate/severe]
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

                    with st.spinner(f"Analyzing pair {i + 1}…"):
                        # Create Image objects from file paths
                        rgb_image = Image(filepath=Path(pair["rgb_path"]))
                        thermal_image = Image(filepath=Path(pair["thermal_path"]))

                        # Get structured analysis
                        analysis = get_agent_analysis(
                            agent, prompt, images=[rgb_image, thermal_image]
                        )

                        if analysis:
                            st.session_state.image_analyses.append(analysis)
                        else:
                            st.error("Analysis failed.")

                    progress_bar.progress(
                        0.5 + (i + 1) / (len(rgb_images) * 2)
                    )  # Second half for analysis

                # Generate final report if we have analyses
                if st.session_state.image_analyses:
                    with st.spinner("Generating final report…"):
                        report_agent = get_report_agent()

                        # Join all analyses for the prompt
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
                        - [Finding 1]
                        - [Finding 2]
                        ...
                        
                        ## Priority Recommendations
                        
                        ### Recommendation 1
                        - Title: [clear title]
                        - Description: [detailed explanation]
                        - Estimated Cost: [cost range]
                        - Estimated Savings: [annual savings]
                        - Priority: [low/medium/high]
                        - Implementation Time: [time estimate]
                        
                        ### Recommendation 2
                        ...
                        
                        ## Potential Improvements
                        - Current Rating: [current rating]
                        - Potential Rating: [improved rating after recommendations]
                        
                        ## Financial Summary
                        - Total Investment: [sum of all recommendations]
                        - Annual Savings: [total annual savings]
                        
                        ## Conclusion
                        [Final assessment and next steps]
                        """

                        # No images needed for final report
                        final_report = get_agent_analysis(report_agent, final_prompt)

                        if final_report:
                            st.session_state.final_report = final_report
                            st.session_state.analysis_complete = True
                        else:
                            st.error("Report generation failed.")

# Tab 2: Analysis Results
with tab2:
    if st.session_state.analysis_complete and st.session_state.image_analyses:
        st.header("Image Analysis Results")

        # Display each analysis in an expandable section
        for i, analysis in enumerate(st.session_state.image_analyses):
            # Extract the summary if available
            summary = "Analysis results"
            for line in analysis.split("\n"):
                if line.strip().startswith("##") and "Summary" in line:
                    # Get the next non-empty line
                    summary_index = analysis.split("\n").index(line) + 1
                    while (
                        summary_index < len(analysis.split("\n"))
                        and not analysis.split("\n")[summary_index].strip()
                    ):
                        summary_index += 1
                    if summary_index < len(analysis.split("\n")):
                        summary = analysis.split("\n")[summary_index].strip()
                    break

            with st.expander(f"Image Pair {i + 1}: {summary}", expanded=True):
                # Add a two-column layout for images and analysis
                img_col, text_col = st.columns([1, 2])
                
                with img_col:
                    # Display RGB and thermal images from session state
                    if i < len(st.session_state.rgb_image_data) and i < len(st.session_state.thermal_image_data):
                        st.subheader("RGB Image")
                        st.image(st.session_state.rgb_image_data[i], use_container_width=True)
                        
                        st.subheader("Thermal Image")
                        st.image(st.session_state.thermal_image_data[i], use_container_width=True)
                
                with text_col:
                    # Display the full markdown analysis
                    st.markdown(analysis)

# Tab 3: Final Report
with tab3:
    if st.session_state.analysis_complete and st.session_state.final_report:
        # Extract energy rating from the report if available
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
            st.caption(f"Generated on {datetime.now():%B %d, %Y}")

        # Add a gallery of analyzed images
        st.subheader("Analyzed Images")
        for i in range(len(st.session_state.rgb_image_data)):
            cols = st.columns(2)
            with cols[0]:
                st.image(st.session_state.rgb_image_data[i], caption=f"RGB Image {i+1}", width=300)
            with cols[1]:
                st.image(st.session_state.thermal_image_data[i], caption=f"Thermal Image {i+1}", width=300)
            
            # Add a separator between image pairs
            if i < len(st.session_state.rgb_image_data) - 1:
                st.divider()

        # Display the full markdown report
        st.markdown(st.session_state.final_report)

        # Download button
        st.download_button(
            "Download Full Report",
            data=st.session_state.final_report,
            file_name=f"energy_report_{location}_{datetime.now():%Y%m%d}.md",
            mime="text/markdown",
        )