import streamlit as st
import os
from google import genai
from google.genai import types
import subprocess
import tempfile
import uuid
import re
import inspect
import urllib.parse
import requests
from pathlib import Path
from datetime import datetime
from enum import Enum, auto

# ==============================================================================
# 0. CONFIG & SETUP
# ==============================================================================

st.set_page_config(
    page_title="EfficientManim Web",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "nodes" not in st.session_state:
    st.session_state.nodes = {}
if "connections" not in st.session_state:
    st.session_state.connections = [] # List of {"start": node_id, "end": node_id}
if "assets" not in st.session_state:
    st.session_state.assets = {}
if "logs" not in st.session_state:
    st.session_state.logs = []
if "temp_dir" not in st.session_state:
    # Create a persistent temp dir for the session
    st.session_state.temp_dir = Path(tempfile.mkdtemp(prefix="eff_manim_"))
if "selected_node_id" not in st.session_state:
    st.session_state.selected_node_id = None
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

# Constants
APP_VERSION = "0.1.2-web"
TEMP_DIR = st.session_state.temp_dir

# Manim Availability
try:
    import manim
    from manim import *
    from manim.utils.color import ManimColor, ParsableManimColor
    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False
    st.error("CRITICAL: Manim library not found. Rendering will be disabled.")

# ==============================================================================
# 1. UTILS & HELPERS
# ==============================================================================

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {level}: {msg}"
    st.session_state.logs.append(entry)
    # Keep log size manageable
    if len(st.session_state.logs) > 100:
        st.session_state.logs.pop(0)

class NodeType(Enum):
    MOBJECT = auto()
    ANIMATION = auto()

class NodeData:
    def __init__(self, name, n_type, cls_name):
        self.id = str(uuid.uuid4())
        self.name = name
        self.type = n_type
        self.cls_name = cls_name
        self.params = {}
        self.param_metadata = {}
        self.audio_asset_id = None
        self.is_ai_generated = False
        self.ai_source = None
        
        # Auto-load defaults if possible
        if MANIM_AVAILABLE:
            self.load_defaults()

    def load_defaults(self):
        try:
            cls = getattr(manim, self.cls_name, None)
            if not cls: return
            sig = inspect.signature(cls.__init__)
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'args', 'kwargs', 'mobject'): continue
                if param.default is inspect.Parameter.empty: continue
                # Basic default loading
                val = param.default
                if isinstance(val, (int, float, str, bool)):
                    self.params[param_name] = val
        except:
            pass

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "type": self.type.name,
            "cls_name": self.cls_name, "params": self.params,
            "audio_asset_id": self.audio_asset_id,
            "is_ai_generated": self.is_ai_generated
        }

    @staticmethod
    def from_dict(d):
        n = NodeData(d["name"], NodeType[d["type"]], d["cls_name"])
        n.id = d["id"]
        n.params = d["params"]
        n.audio_asset_id = d.get("audio_asset_id")
        n.is_ai_generated = d.get("is_ai_generated", False)
        return n
    
    def is_param_enabled(self, param_name):
        return self.param_metadata.get(param_name, {}).get("enabled", True)

    def set_param_enabled(self, param_name, enabled):
        if param_name not in self.param_metadata: self.param_metadata[param_name] = {}
        self.param_metadata[param_name]["enabled"] = enabled
        
    def should_escape_string(self, param_name):
        return self.param_metadata.get(param_name, {}).get("escape", False)
        
    def set_escape_string(self, param_name, escape):
        if param_name not in self.param_metadata: self.param_metadata[param_name] = {}
        self.param_metadata[param_name]["escape"] = escape

class Asset:
    def __init__(self, name, path, kind):
        self.id = str(uuid.uuid4())
        self.name = name
        self.current_path = path
        self.kind = kind # image, video, audio

class TypeSafeParser:
    @staticmethod
    def is_asset_param(param_name):
        n = param_name.lower()
        if "filename" in n: return True
        if "file" in n or "image" in n or "sound" in n or "svg" in n:
             if "fill" in n or "profile" in n: return False
             return True
        return False

    @staticmethod
    def is_color_param(param_name):
        kw = {'color', 'fill_color', 'stroke_color', 'background_color'}
        return any(k in param_name.lower() for k in kw)

    @staticmethod
    def is_numeric_param(param_name):
        kw = {'radius', 'width', 'height', 'scale', 'factor', 'size', 'thickness', 
              'stroke_width', 'font_size', 'length', 'rate', 'opacity', 'alpha',
              'x', 'y', 'z', 'angle'}
        if TypeSafeParser.is_asset_param(param_name): return False
        return any(k in param_name.lower() for k in kw)

    @staticmethod
    def parse_numeric(value):
        try:
            return float(value)
        except:
            return 0.0

# ==============================================================================
# 2. LOGIC: GRAPH COMPILER & RENDERER
# ==============================================================================

def get_asset_path(asset_id):
    if asset_id in st.session_state.assets:
        return st.session_state.assets[asset_id].current_path
    return None

def format_param_value(param_name, value, node_data):
    try:
        # Asset ID -> Path
        if isinstance(value, str) and value in st.session_state.assets:
            path = get_asset_path(value)
            return f'r"{path}"' if path else '""'
        
        # Mobject Reference (UUID)
        if isinstance(value, str) and len(value) == 36 and value in st.session_state.nodes:
            return f"m_{value[:6]}"
            
        # Raw String Literal (LaTeX)
        if isinstance(value, str) and value.startswith('r"""') and value.endswith('"""'):
            return value

        # Escaping
        if node_data.should_escape_string(param_name):
             return str(value).strip("'\"")

        # Color
        if TypeSafeParser.is_color_param(param_name):
             return f'"{value}"'
        
        # Numeric
        if TypeSafeParser.is_numeric_param(param_name):
            return str(TypeSafeParser.parse_numeric(value))
            
        # String fallback
        if isinstance(value, str):
            return f'"{value}"'
            
        return str(value)
    except Exception:
        return f'"{value}"'

def compile_graph():
    """Generates Manim Python code from session state."""
    code = "from manim import *\nimport numpy as np\n"
    try:
        import pydub
        code += "from pydub import AudioSegment\n"
        has_pydub = True
    except:
        has_pydub = False
        
    code += "\nclass EfficientScene(Scene):\n    def construct(self):\n"

    nodes = st.session_state.nodes
    conns = st.session_state.connections

    # 1. Instantiate Mobjects
    mobjects = [n for n in nodes.values() if n.type == NodeType.MOBJECT]
    m_vars = {}
    
    for m in mobjects:
        args = []
        # Construct args
        for k, v in m.params.items():
            if not m.is_param_enabled(k): continue
            val_str = format_param_value(k, v, m)
            args.append(f"{k}={val_str}")
            
        var_name = f"m_{m.id[:6]}"
        m_vars[m.id] = var_name
        code += f"        {var_name} = {m.cls_name}({', '.join(args)})\n"
        code += f"        self.add({var_name})\n"

    # 2. Animations
    animations = [n for n in nodes.values() if n.type == NodeType.ANIMATION]
    # Simple dependency resolution: just sequential for this demo
    for anim in animations:
        # Find target mobjects linked to this animation
        targets = []
        for c in conns:
            if c['end'] == anim.id: # Wire goes TO animation (Input)
                src_id = c['start']
                if src_id in m_vars:
                    targets.append(m_vars[src_id])
        
        args = targets.copy()
        
        # Audio logic
        if anim.audio_asset_id and has_pydub:
            path = get_asset_path(anim.audio_asset_id)
            if path:
                clean_path = path.replace("\\", "/")
                audio_var = f"audio_{anim.id[:6]}"
                code += f"        # Voiceover\n"
                code += f"        self.add_sound(r'{clean_path}')\n"
                code += f"        {audio_var} = AudioSegment.from_file(r'{clean_path}')\n"
                # Override run_time
                anim.params['run_time'] = f"{audio_var}.duration_seconds"

        for k, v in anim.params.items():
            if not anim.is_param_enabled(k): continue
            if k == 'run_time' and "duration_seconds" in str(v):
                args.append(f"{k}={v}")
            else:
                val_str = format_param_value(k, v, anim)
                args.append(f"{k}={val_str}")
        
        if targets: # Only play if we have a target
            code += f"        self.play({anim.cls_name}({', '.join(args)}))\n"
            code += f"        self.wait(0.5)\n"

    st.session_state.generated_code = code
    return code

def render_manim(code, mode="image", fps=15, quality="l"):
    """Runs manim cli."""
    if not MANIM_AVAILABLE:
        st.error("Manim is not installed.")
        return None

    script_path = TEMP_DIR / "render_script.py"
    with open(script_path, "w") as f:
        f.write(code)

    # Output paths
    output_dir = TEMP_DIR
    
    # Flags
    cmd = ["manim", "-s" if mode=="image" else "", "--disable_caching"]
    cmd.append(f"-q{quality}")
    cmd.append(f"--fps={fps}")
    if mode == "image":
        cmd.append("--format=png")
    else:
        cmd.append("--format=mp4")
        
    cmd += [str(script_path), "EfficientScene"]

    # Run
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(output_dir)
        )
        if result.returncode != 0:
            log(f"Render Error: {result.stderr}", "ERROR")
            st.error(f"Render Failed:\n{result.stderr}")
            return None
        
        # Find Output
        media_dir = output_dir / "media"
        if mode == "image":
            files = list(media_dir.rglob("*.png"))
        else:
            files = list(media_dir.rglob("*.mp4"))
            
        if files:
            return max(files, key=os.path.getmtime)
        else:
            log("No output file found.", "WARN")
            return None

    except Exception as e:
        log(f"Execution Error: {e}", "ERROR")
        return None

# ==============================================================================
# 3. AI INTEGRATION
# ==============================================================================

def generate_ai_code(prompt, model_name, api_key):
    if not api_key:
        st.error("API Key missing. Go to Settings.")
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        sys_prompt = (
            "You are a Manim expert. Generate production-ready Python Manim code.\n"
            "Output ONLY the python code block inside ```python ... ```.\n"
            "Use 'EfficientScene(Scene)' class.\n"
            "Define explicit variables like circle_1, square_2.\n"
            "Include self.play() animations."
        )
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text=sys_prompt)]),
                types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
            ]
        )
        return response.text
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

def parse_ai_nodes(code):
    """
    Rudimentary parser to convert AI code back to nodes. 
    Replacing the full PySide logic with a regex extraction.
    """
    new_nodes = {}
    new_conns = []
    
    # Reset existing (Simplification for web app)
    st.session_state.nodes = {}
    st.session_state.connections = []

    # 1. Extract Mobjects: var = Class(...)
    mob_pattern = r'(\w+)\s*=\s*([A-Z]\w+)\((.*?)\)'
    mobs_found = {}
    
    for match in re.finditer(mob_pattern, code):
        var_name, cls_name, params_raw = match.groups()
        if hasattr(manim, cls_name) and issubclass(getattr(manim, cls_name), manim.Mobject):
            node = NodeData(var_name, NodeType.MOBJECT, cls_name)
            node.is_ai_generated = True
            
            # Simple param extraction
            p_parts = params_raw.split(',')
            for p in p_parts:
                if '=' in p:
                    k, v = p.split('=', 1)
                    node.params[k.strip()] = v.strip().strip("'\"")
            
            st.session_state.nodes[node.id] = node
            mobs_found[var_name] = node.id
            
    # 2. Extract Animations: self.play(Anim(var))
    play_pattern = r'self\.play\((.*?)\)'
    anim_pattern = r'([A-Z]\w+)\((.*?)\)'
    
    for match in re.finditer(play_pattern, code):
        content = match.group(1)
        # Find animation calls inside play
        for am in re.finditer(anim_pattern, content):
            acls, aargs = am.groups()
            if hasattr(manim, acls):
                # Check if it's an animation
                is_anim = False
                try:
                    if issubclass(getattr(manim, acls), manim.Animation) or "Animation" in acls:
                        is_anim = True
                except: pass
                
                if is_anim:
                    anode = NodeData(acls, NodeType.ANIMATION, acls)
                    anode.is_ai_generated = True
                    st.session_state.nodes[anode.id] = anode
                    
                    # Try to link to mobject
                    for var in mobs_found:
                        if var in aargs:
                            st.session_state.connections.append({
                                "start": mobs_found[var],
                                "end": anode.id
                            })
                            
    log(f"AI Merge: {len(st.session_state.nodes)} nodes created.")

# ==============================================================================
# 4. SIDEBAR & SETTINGS
# ==============================================================================

with st.sidebar:
    st.title("EfficientManim 🕸️")
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Settings", expanded=False):
        api_key = st.text_input("Gemini API Key", type="password", key="gemini_key")
        os.environ["GEMINI_API_KEY"] = api_key
        
        gemini_model = st.selectbox("AI Model", ["gemini-2.0-flash", "gemini-1.5-pro"])
        render_quality = st.selectbox("Quality", ["Low (ql)", "Medium (qm)", "High (qh)"])
        fps = st.number_input("FPS", 15, 60, 15)
        
    st.markdown("---")
    
    # Asset Manager
    st.subheader("📂 Assets")
    uploaded_file = st.file_uploader("Upload Image/Audio", type=['png','jpg','mp3','wav'])
    if uploaded_file:
        file_path = TEMP_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Determine kind
        kind = "unknown"
        if uploaded_file.type.startswith("image"): kind = "image"
        elif uploaded_file.type.startswith("audio"): kind = "audio"
        
        # Register
        asset = Asset(uploaded_file.name, str(file_path), kind)
        st.session_state.assets[asset.id] = asset
        st.success(f"Added: {asset.name}")

    # Asset List
    if st.session_state.assets:
        for aid, asset in st.session_state.assets.items():
            st.markdown(f"- {asset.name} ({asset.kind})")

# ==============================================================================
# 5. MAIN TABS
# ==============================================================================

tab_graph, tab_props, tab_ai, tab_latex, tab_voice, tab_render = st.tabs([
    "🕸️ Graph Editor", "🧩 Properties", "🤖 AI Gen", "✒️ LaTeX", "🎙️ Voiceover", "🎬 Render"
])

# --- TAB 1: GRAPH EDITOR ---
with tab_graph:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Add Node")
        with st.form("add_node_form"):
            n_cat = st.radio("Category", ["Mobject", "Animation"], horizontal=True)
            
            # Simple list of common Manim classes
            if n_cat == "Mobject":
                opts = ["Circle", "Square", "Rectangle", "Text", "MathTex", "Line", "Dot", "Star", "Triangle"]
            else:
                opts = ["FadeIn", "FadeOut", "Create", "Write", "Rotate", "Transform", "MoveTo", "ScaleInPlace"]
            
            n_cls = st.selectbox("Class", opts)
            n_name = st.text_input("Name (optional)", value=n_cls)
            
            if st.form_submit_button("Create Node"):
                ntype = NodeType.MOBJECT if n_cat=="Mobject" else NodeType.ANIMATION
                node = NodeData(n_name, ntype, n_cls)
                st.session_state.nodes[node.id] = node
                st.success(f"Created {n_name}")
                st.rerun()

    with col2:
        st.subheader("Graph Structure")
        
        # 1. List Mobjects
        st.markdown("#### 📦 Mobjects")
        mobs = [n for n in st.session_state.nodes.values() if n.type == NodeType.MOBJECT]
        for m in mobs:
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.info(f"**{m.name}** ({m.cls_name})")
            if c2.button("Edit", key=f"edit_{m.id}"):
                st.session_state.selected_node_id = m.id
                # Switch tab hack not possible in pure streamlit, user must click tab
                st.toast(f"Selected {m.name}. Go to Properties tab.")
            if c3.button("🗑️", key=f"del_{m.id}"):
                del st.session_state.nodes[m.id]
                # Cleanup connections
                st.session_state.connections = [c for c in st.session_state.connections 
                                              if c['start'] != m.id and c['end'] != m.id]
                st.rerun()

        st.markdown("#### 🎬 Animations")
        anims = [n for n in st.session_state.nodes.values() if n.type == NodeType.ANIMATION]
        
        # Connection Manager
        for a in anims:
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{a.name}** ({a.cls_name})")
                if c2.button("🗑️", key=f"del_{a.id}"):
                    del st.session_state.nodes[a.id]
                    st.session_state.connections = [c for c in st.session_state.connections if c['end'] != a.id]
                    st.rerun()
                
                # Show existing links
                current_links = [c['start'] for c in st.session_state.connections if c['end'] == a.id]
                
                # Multiselect for inputs
                mob_options = {m.id: m.name for m in mobs}
                selected = st.multiselect(
                    f"Targets for {a.name}", 
                    options=mob_options.keys(),
                    format_func=lambda x: mob_options[x],
                    default=current_links,
                    key=f"link_{a.id}"
                )
                
                # Update connections logic
                # First remove old ones for this anim
                st.session_state.connections = [c for c in st.session_state.connections if c['end'] != a.id]
                # Add new ones
                for source_id in selected:
                    st.session_state.connections.append({"start": source_id, "end": a.id})

# --- TAB 2: PROPERTIES ---
with tab_props:
    nid = st.session_state.selected_node_id
    if nid and nid in st.session_state.nodes:
        node = st.session_state.nodes[nid]
        st.subheader(f"Edit: {node.name} ({node.cls_name})")
        
        # Name
        new_name = st.text_input("Node Name", node.name)
        if new_name != node.name:
            node.name = new_name
        
        # Auto-detect params if we have Manim, else generic list
        st.markdown("##### Parameters")
        
        # Display current params dict
        keys_to_delete = []
        for k, v in node.params.items():
            c1, c2, c3 = st.columns([1, 2, 0.5])
            c1.text(k)
            
            # Value Editor
            if TypeSafeParser.is_color_param(k):
                try:
                    val = st.color_picker("", v, key=f"p_{nid}_{k}")
                    node.params[k] = val
                except:
                    node.params[k] = c2.text_input("", str(v), key=f"p_{nid}_{k}")
            elif isinstance(v, bool):
                node.params[k] = c2.checkbox("", v, key=f"p_{nid}_{k}")
            else:
                node.params[k] = c2.text_input("", str(v), key=f"p_{nid}_{k}")
                
            if c3.button("x", key=f"del_p_{nid}_{k}"):
                keys_to_delete.append(k)
        
        for k in keys_to_delete:
            del node.params[k]
            st.rerun()
            
        # Add new param
        with st.expander("Add Parameter"):
            with st.form(f"add_param_{nid}"):
                pk = st.text_input("Name")
                pv = st.text_input("Value")
                if st.form_submit_button("Add"):
                    if pk:
                        node.params[pk] = pv
                        st.rerun()
                        
    else:
        st.info("Select a node from the Graph Editor to edit properties.")

# --- TAB 3: AI GEN ---
with tab_ai:
    st.header("🤖 AI Generation")
    
    prompt = st.text_area("Prompt", placeholder="Create a blue circle that transforms into a square.")
    if st.button("Generate Code"):
        with st.spinner("Thinking..."):
            code = generate_ai_code(prompt, gemini_model, api_key)
            if code:
                # Extract code block
                match = re.search(r"```python(.*?)```", code, re.DOTALL)
                if match:
                    code = match.group(1).strip()
                st.session_state.ai_code_cache = code
    
    if "ai_code_cache" in st.session_state:
        st.code(st.session_state.ai_code_cache, language="python")
        if st.button("Merge to Graph"):
            parse_ai_nodes(st.session_state.ai_code_cache)
            st.success("Merged AI nodes into graph!")
            st.rerun()

# --- TAB 4: LaTeX ---
with tab_latex:
    st.header("LaTeX to Image")
    tex_input = st.text_area("LaTeX Equation", r"E = mc^2")
    if st.button("Render LaTeX"):
        # Use MathPad API
        try:
            url = f"https://mathpad.ai/api/v1/latex2image?latex={urllib.parse.quote(tex_input)}&format=png&scale=4"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                # Save as asset
                fname = f"latex_{uuid.uuid4().hex[:6]}.png"
                fpath = TEMP_DIR / fname
                with open(fpath, "wb") as f:
                    f.write(resp.content)
                
                asset = Asset(f"LaTeX: {tex_input[:10]}", str(fpath), "image")
                st.session_state.assets[asset.id] = asset
                st.image(resp.content)
                st.success("Saved to Assets!")
            else:
                st.error("API Error")
        except Exception as e:
            st.error(f"Error: {e}")

# --- TAB 5: Voiceover ---
with tab_voice:
    st.header("TTS")
    tts_text = st.text_area("Script", "Hello world.")
    tts_voice = st.selectbox("Voice", ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Zephyr"])
    
    if st.button("Generate Audio"):
        if not api_key:
            st.error("No API Key")
        else:
            try:
                client = genai.Client(api_key=api_key)
                # Config
                config = types.GenerateContentConfig(
                    response_modalities=["audio"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=tts_voice)
                        )
                    )
                )
                
                # Generate
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp", # Using a known model with TTS support or fallback
                    contents=types.Content(parts=[types.Part.from_text(tts_text)]),
                    config=config
                )
                
                # Extract Audio (Simplified: assuming inline data)
                # In real scenario, handling streams/parts is needed. 
                # This is a mock-up of the logic seen in PySide.
                
                # Since 'gemini-2.0-flash-exp' TTS isn't standard in the python SDK publicly yet (preview),
                # We will simulate success or implement if supported.
                # Assuming standard generation returns base64 or bytes in parts.
                
                # Fallback check
                if response.candidates and response.candidates[0].content.parts:
                    part = response.candidates[0].content.parts[0]
                    audio_data = part.inline_data.data
                    
                    fname = f"tts_{uuid.uuid4().hex[:6]}.wav"
                    fpath = TEMP_DIR / fname
                    with open(fpath, "wb") as f:
                        f.write(audio_data)
                        
                    asset = Asset(f"TTS: {tts_text[:10]}", str(fpath), "audio")
                    st.session_state.assets[asset.id] = asset
                    st.audio(audio_data)
                    st.success("Generated & Saved!")
                    
            except Exception as e:
                st.error(f"TTS Failed (Model might not support it yet): {e}")

# --- TAB 6: RENDER ---
with tab_render:
    st.header("🎬 Render Scene")
    
    # 1. Compile
    code = compile_graph()
    with st.expander("View Generated Code"):
        st.code(code, language="python")
        
    # 2. Controls
    r_mode = st.radio("Mode", ["Preview (Image)", "Render (Video)"], horizontal=True)
    
    if st.button("Start Render"):
        mode_str = "image" if "Preview" in r_mode else "video"
        
        # Parse Quality
        q_map = {"Low (ql)": "l", "Medium (qm)": "m", "High (qh)": "h"}
        q_char = q_map.get(render_quality, "l")
        
        with st.spinner("Rendering... (This may take a while)"):
            out_path = render_manim(code, mode_str, fps, q_char)
            
            if out_path:
                st.success("Render Complete!")
                if mode_str == "image":
                    st.image(str(out_path))
                else:
                    st.video(str(out_path))
            else:
                st.error("Render failed. Check logs.")

# Logs Footer
with st.expander("System Logs"):
    st.write(st.session_state.logs)
    