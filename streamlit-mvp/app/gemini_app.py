# gemini_app_v1.py
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

st.set_page_config(page_title="Energy Rating Analysis", layout="wide")
st.title("Energy Rating Report Generator")
st.subheader("Upload RGB and thermal image pairs for energy efficiency analysis")


# Setup the analysis agent with Google Gemini
@st.cache_resource
def get_analysis_agent():
    return Agent(
        model=Gemini(
            id="gemini-2.5-flash-preview-04-17"
        ),  # Keep the model version you're using
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


# Function to get agent analysis results with proper content extraction
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


# Streamlit UI for file uploads and parameters
st.write("### Step 1: Upload RGB Images")
rgb_images = st.file_uploader(
    "Upload RGB images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

st.write("### Step 2: Upload Thermal Images")
thermal_images = st.file_uploader(
    "Upload thermal images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

st.write("### Step 3: Building Information")
building_year = st.number_input(
    "Construction Year", min_value=1900, max_value=2025, value=1980
)
building_type = st.selectbox(
    "Building Type", ["Residential", "Commercial", "Industrial"]
)
location = st.text_input("Location", "Helsinki")

if st.button("Generate Energy Report"):
    if not rgb_images or not thermal_images:
        st.warning("Please upload both RGB and thermal images.")
    elif len(rgb_images) != len(thermal_images):
        st.error("Number of RGB and thermal images must match.")
    else:
        st.write("### Processing Images...")
        progress_bar = st.progress(0)
        cols = st.columns(len(rgb_images))

        with tempfile.TemporaryDirectory() as temp_dir:
            image_pairs = []
            for i, (rgb_img, thermal_img) in enumerate(zip(rgb_images, thermal_images)):
                rgb_path = os.path.join(temp_dir, f"rgb_{i}.jpg")
                thermal_path = os.path.join(temp_dir, f"thermal_{i}.jpg")
                with open(rgb_path, "wb") as f:
                    f.write(rgb_img.getvalue())  # noqa: E701
                with open(thermal_path, "wb") as f:
                    f.write(thermal_img.getvalue())  # noqa: E701

                with cols[i]:
                    st.image(rgb_img, caption=f"RGB {i + 1}", width=150)
                    st.image(thermal_img, caption=f"Thermal {i + 1}", width=150)

                image_pairs.append({"rgb_path": rgb_path, "thermal_path": thermal_path})
                progress_bar.progress((i + 1) / len(rgb_images))

            st.write("### Analysis Results")
            agent = get_analysis_agent()
            all_findings = []

            for i, pair in enumerate(image_pairs):
                st.write(f"#### Analyzing Pair {i + 1}")
                prompt = f"""
                Analyze this RGB-thermal image pair for energy efficiency issues:

                Building Info:
                - Year: {building_year}
                - Type: {building_type}
                - Location: {location}

                Identify anomalies, severity (minor/moderate/severe), recommendations, and cost estimates.
                """
                with st.spinner(f"Analyzing pair {i + 1}…"):
                    # Create Image objects from file paths
                    rgb_image = Image(filepath=Path(pair["rgb_path"]))
                    thermal_image = Image(filepath=Path(pair["thermal_path"]))

                    # Get analysis using the revised function that properly extracts string content
                    analysis = get_agent_analysis(
                        agent, prompt, images=[rgb_image, thermal_image]
                    )

                    if analysis:
                        st.markdown(analysis)
                        all_findings.append(analysis)  # Now this will be a string
                    else:
                        st.error("Analysis failed.")

            st.write("### Final Energy Efficiency Report")
            with st.spinner("Generating final report…"):
                if all_findings:
                    # Join the string findings - this will now work correctly
                    findings_text = "\n\n".join(all_findings)
                    final_prompt = f"""
                    Generate a comprehensive energy efficiency report:
                    - Year: {building_year}
                    - Type: {building_type}
                    - Location: {location}
                    - Date: {datetime.now():%B %d, %Y}

                    Findings:
                    {findings_text}

                    Include:
                    1. Executive Summary
                    2. EU Energy Rating (A–G)
                    3. Priority Recommendations with costs & savings
                    4. Conclusion on rating improvements
                    """
                    # No images needed for final report
                    final_report = get_agent_analysis(agent, final_prompt)
                    if final_report:
                        st.markdown(final_report)
                        st.download_button(
                            "Download Report",
                            data=final_report,
                            file_name=f"energy_report_{location}_{datetime.now():%Y%m%d}.md",
                            mime="text/markdown",
                        )
                    else:
                        st.error("Report generation failed.")
                else:
                    st.error("No findings to generate a report.")
