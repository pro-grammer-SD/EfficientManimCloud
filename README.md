# 🎬 EfficientManim Cloud

EfficientManim Cloud is a web-based, node-centric editor for [Manim](https://www.manim.community/), built with Streamlit. It allows users to visually construct mathematical animations by connecting Mobjects and Animations, providing a more intuitive workflow than writing raw Python code.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Manim](https://img.shields.io/badge/Manim-333333?style=for-the-badge&logo=python&logoColor=white)
![Google GenAI](https://img.shields.io/badge/Google%20GenAI-4285F4?style=for-the-badge&logo=google&logoColor=white)

## ✨ Features

- **Visual Node Editor**: Map out your scene logic using a node-based interface. Connect objects to animations seamlessly.
- **AI Assistant**: Leverages Google's GenAI to assist in script generation and solving complex Manim logic.
- **Instant Preview**: Render low-quality single frames (PNG) to verify your scene's layout without waiting for full video renders.
- **Full Video Rendering**: Export high-quality MP4 videos with customizable FPS and quality settings.
- **Auto-Code Generation**: Automatically translates your visual graph into clean, executable Manim Python code.
- **Live Logging**: Real-time console logs within the UI to track rendering progress and debug errors.

## 🛠️ Prerequisites

Before running the project, ensure you have the following system dependencies installed (required by Manim):

### System Dependencies (Linux/Debian)
```bash
sudo apt update
sudo apt install -y ffmpeg libcairo2-dev libpango1.0-dev \
    texlive texlive-latex-extra texlive-fonts-recommended \
    graphviz
```

### Windows
- Install [FFmpeg](https://ffmpeg.org/download.html).
- Install [MiKTeX](https://miktex.org/) or TeX Live for LaTeX support.
- Install [Manim](https://docs.manim.community/en/stable/installation.html) following the official guide.

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/EfficientManimCloud.git
   cd EfficientManimCloud
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python -m streamlit run main.py
   ```

## 🖥️ Usage

1. **Configure API**: (Optional) Enter your Google GenAI API key in the sidebar for AI assistance.
2. **Add Nodes**: Use the sidebar to add Mobjects (like Circle, Square, Text) and Animations (like Create, FadeIn).
3. **Connect**: Link your Animation nodes to the Mobject nodes they should act upon.
4. **Preview**: Click 'Render Preview' to see the current state.
5. **Export**: Click 'Render Video' to generate the final animation.

## 🧰 Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Animation Engine**: [Manim Community Edition](https://www.manim.community/)
- **AI Integration**: [Google Generative AI](https://ai.google.dev/)
- **Graphing**: Graphviz (for dependency visualization)
- **Environment**: Python 3.9+

---

*Note: This project is currently in active development. Features and UI are subject to change.*
