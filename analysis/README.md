# Energy Rating Analysis with Gemini AI

This repository contains a Jupyter notebook for analyzing building energy efficiency using RGB and thermal image pairs. The notebook uses Google's Gemini 2.5 model through the Agno framework to identify thermal anomalies, insulation problems, and air leaks, then generates comprehensive energy efficiency reports with cost estimates and recommendations.

## Features

- **Image Pair Analysis**: Analyze RGB and thermal image pairs to identify energy efficiency issues
- **AI-Powered Recognition**: Utilizes Google's Gemini 2.5 model to detect thermal anomalies
- **Detailed Reports**: Generates structured energy efficiency reports with EU energy ratings (A-G)
- **Cost Estimation**: Provides estimated costs and savings for recommended improvements
- **Visualizations**: Creates visualizations for energy ratings, costs vs. savings, and ROI analysis
- **Batch Processing**: Support for analyzing multiple image pairs for whole-building assessment

## Prerequisites

- Python 3.9+
- Jupyter Notebook environment
- Google API key for Gemini access
- Agno framework

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/cmakafui/energy-report-agent.git
   cd analysis
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Google API key:
   - Create a `.env` file in the root directory
   - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage

1. Start Jupyter Notebook:

   ```bash
   jupyter notebook notebooks
   ```

2. Open the notebook in your browser

3. Run the cells sequentially to:
   - Load and display your RGB and thermal image pairs
   - Analyze images for energy efficiency issues
   - Generate detailed recommendations with cost estimates
   - Create a comprehensive energy efficiency report
   - Visualize key metrics and findings

## Example Data

The notebook is configured to use example images located at:

```
../images/basement_door/rgb.jpg
../images/basement_door/thermal.jpg
```

You can add your own images by modifying the image paths in the notebook.

## File Structure

```
analysis/
├── .env                       # Environment variables (create this file)
├── README.md                  # This README
├── requirements.txt           # Python dependencies
├── images/                    # Example images
│   └── basement_door/
│       ├── rgb.jpg            # RGB image
│       └── thermal.jpg        # Thermal image
└── notebooks/
    └── agent_exp_01.ipynb     # Main Jupyter notebook
```

## Key Components

- **Analysis Agent**: Analyzes RGB-thermal image pairs to identify energy efficiency issues
- **Report Agent**: Generates comprehensive energy efficiency reports based on analysis results
- **Visualization Tools**: Creates charts to visualize findings and cost-benefit analysis
- **Batch Processing**: Processes multiple image pairs for comprehensive building assessment

## Agno Framework Integration

This notebook leverages the Agno framework to interact with Google's Gemini 2.5 model. The framework provides:

- Simple API access to advanced AI models
- Tools for reasoning and analysis
- Structured output formatting
- Image processing capabilities

## Acknowledgements

- Google Gemini 2.5 model
- Agno framework
