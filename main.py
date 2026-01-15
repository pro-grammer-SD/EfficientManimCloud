import re
import streamlit as st
import manim
from manim import *
import os
import tempfile
import json
import uuid
import inspect
import subprocess
import graphviz
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types

# ==============================================================================
# 1. CORE CONFIGURATION & STATE
# ==============================================================================

st.set_page_config(
    page_title="EfficientManim Cloud",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TEMP_DIR = Path(tempfile.gettempdir()) / "EfficientManim_Session"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Session State Initialization
def init_state():
    defaults = {
        "nodes": {},          # id -> NodeDict
        "connections": [],    # List of {"from": id, "to": id}
        "logs": [],           # List of strings
        "project_name": "MyProject",
        "generated_code": "",
        "last_preview": None,
        "last_video": None,
        "api_key": "",
        "ai_chat_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# ==============================================================================
# 2. LOGGING & UTILS
# ==============================================================================

def log(message, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {level}: {message}"
    st.session_state.logs.append(entry)
    # Keep log size manageable
    if len(st.session_state.logs) > 100:
        st.session_state.logs.pop(0)

def clear_logs():
    st.session_state.logs = []

def get_manim_classes(base_class):
    """Reflect on Manim library to get available classes."""
    classes = []
    for name in dir(manim):
        if name.startswith("_"): continue
        try:
            obj = getattr(manim, name)
            if inspect.isclass(obj) and issubclass(obj, base_class) and obj is not base_class:
                classes.append(name)
        except:
            pass
    return sorted(classes)

# ==============================================================================
# 3. DATA MODELS
# ==============================================================================

class NodeManager:
    @staticmethod
    def create_node(cls_name, node_type="MOBJECT"):
        node_id = str(uuid.uuid4())[:8]
        
        # Get default params via inspection
        params = {}
        try:
            cls = getattr(manim, cls_name)
            sig = inspect.signature(cls.__init__)
            for name, param in sig.parameters.items():
                if name in ['self', 'args', 'kwargs', 'mobject']: continue
                
                default = param.default
                if default is inspect.Parameter.empty:
                    default = None
                
                # Simplify types for JSON serialization
                if isinstance(default, (np.ndarray, list, tuple)):
                    default = str(list(default))
                elif isinstance(default, (ManimColor, ParsableManimColor)):
                    default = str(default)
                
                params[name] = default
        except Exception as e:
            log(f"Error inspecting {cls_name}: {e}", "WARN")

        node = {
            "id": node_id,
            "name": f"{cls_name}_{node_id}",
            "cls_name": cls_name,
            "type": node_type,
            "params": params,
            "active": True
        }
        st.session_state.nodes[node_id] = node
        log(f"Created node: {node['name']}")
        return node_id

    @staticmethod
    def delete_node(node_id):
        if node_id in st.session_state.nodes:
            del st.session_state.nodes[node_id]
            # Remove connections
            st.session_state.connections = [
                c for c in st.session_state.connections 
                if c["from"] != node_id and c["to"] != node_id
            ]
            log(f"Deleted node {node_id}")

    @staticmethod
    def connect(from_id, to_id):
        # Prevent duplicates
        for c in st.session_state.connections:
            if c["from"] == from_id and c["to"] == to_id:
                return
        st.session_state.connections.append({"from": from_id, "to": to_id})
        log(f"Connected {from_id} -> {to_id}")

# ==============================================================================
# 4. ENGINE LAYER (CODE GEN & RENDER)
# ==============================================================================

class Engine:
    @staticmethod
    def generate_code():
        """Topological sort-ish generation of Manim script."""
        nodes = st.session_state.nodes
        conns = st.session_state.connections
        
        code = "from manim import *\nimport numpy as np\n\nclass GeneratedScene(Scene):\n    def construct(self):\n"
        
        # 1. Define Mobjects
        mobjects = [n for n in nodes.values() if n['type'] == 'MOBJECT']
        m_map = {} # id -> var_name
        
        for m in mobjects:
            var_name = f"m_{m['id']}"
            m_map[m['id']] = var_name
            
            # Param formatting
            args = []
            for k, v in m['params'].items():
                if v is None or v == "": continue
                # Basic type inference for string representation
                val_str = str(v)
                # If it looks like a number, keep it; if color/string, quote it
                # This is a basic heuristics for the prototype
                if val_str.replace('.','',1).isdigit():
                    pass 
                elif val_str.startswith("[") or "np." in val_str or "array" in val_str:
                    pass
                elif val_str.upper() in ["RED", "BLUE", "GREEN", "WHITE", "BLACK"]:
                    pass
                else:
                    if not (val_str.startswith('"') or val_str.startswith("'")):
                        val_str = f"'{val_str}'"
                
                args.append(f"{k}={val_str}")
            
            code += f"        {var_name} = {m['cls_name']}({', '.join(args)})\n"
            code += f"        self.add({var_name})\n"
            
        # 2. Define Animations
        anims = [n for n in nodes.values() if n['type'] == 'ANIMATION']
        
        # Simple sequencing based on creation order for now (Topological sort in full version)
        for anim in anims:
            # Find targets (Mobjects connected TO this animation)
            # Actually logic: Mobject -> Animation (in desktop app logic)
            # Or Animation -> Mobject? 
            # Let's assume standard: Animation wraps Mobject.
            # In our UI, we will link Animation -> Mobject (Target)
            
            targets = []
            for c in conns:
                if c["from"] == anim['id']: # Animation node connects to Mobject
                    target_id = c["to"]
                    if target_id in m_map:
                        targets.append(m_map[target_id])
            
            if targets:
                # Param formatting
                args = targets.copy() # First args are usually mobjects
                for k, v in anim['params'].items():
                    if v is None or v == "": continue
                    args.append(f"{k}={v}")
                
                anim_code = f"{anim['cls_name']}({', '.join(args)})"
                code += f"        self.play({anim_code})\n"
        
        st.session_state.generated_code = code
        return code

    @staticmethod
    def render_frame(node_id=None):
        """Render a single frame preview."""
        code = Engine.generate_code()
        
        # If previewing specific node, isolate it (Simple logic: render whole scene for now)
        # To strictly optimize, we would filter the code graph.
        
        script_path = TEMP_DIR / "preview_script.py"
        with open(script_path, "w") as f:
            f.write(code)
            
        # Run Manim
        # -ql = Low Quality
        # -s = Save last frame only
        cmd = ["manim", "-ql", "-s", "--format=png", "--disable_caching", str(script_path), "GeneratedScene"]
        
        try:
            log("Starting preview render...")
            result = subprocess.run(
                cmd, 
                cwd=str(TEMP_DIR), 
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                log(f"Render Error: {result.stderr}", "ERROR")
                return None
            
            # Find output
            media_dir = TEMP_DIR / "media" / "images" / "GeneratedScene"
            pngs = list(media_dir.glob("*.png"))
            if pngs:
                latest = max(pngs, key=os.path.getmtime)
                st.session_state.last_preview = str(latest)
                log("Preview rendered successfully.")
                return str(latest)
            return None
            
        except Exception as e:
            log(f"Subprocess failed: {e}", "ERROR")
            return None

    @staticmethod
    def render_video(fps=15, quality="l"):
        code = st.session_state.generated_code
        if not code:
            code = Engine.generate_code()
            
        script_path = TEMP_DIR / "video_script.py"
        with open(script_path, "w") as f:
            f.write(code)
            
        # Quality flags
        q_flag = f"-q{quality}"
        
        cmd = ["manim", q_flag, f"--fps={fps}", "--format=mp4", "--disable_caching", str(script_path), "GeneratedScene"]
        
        try:
            log("Starting video render (this may take time)...")
            result = subprocess.run(
                cmd,
                cwd=str(TEMP_DIR),
                capture_output=True,
                text=True,
                timeout=120 # 2 min limit for cloud
            )
            
            if result.returncode != 0:
                log(f"Video Error: {result.stderr}", "ERROR")
                return None
                
            # Find output
            # Manim structure: media/videos/video_script/quality/GeneratedScene.mp4
            # We search recursively in media/videos
            media_dir = TEMP_DIR / "media" / "videos"
            mp4s = list(media_dir.rglob("*.mp4"))
            
            if mp4s:
                latest = max(mp4s, key=os.path.getmtime)
                st.session_state.last_video = str(latest)
                log(f"Video rendered: {latest.name}")
                return str(latest)
            
        except Exception as e:
            log(f"Video render failed: {e}", "ERROR")
        return None

# ==============================================================================
# 5. UI COMPONENTS
# ==============================================================================

def sidebar_ui():
    with st.sidebar:
        st.title("⚙️ Controls")
        
        # Project Management
        with st.expander("📁 Project", expanded=True):
            st.text_input("Project Name", key="project_name")
            
            col1, col2 = st.columns(2)
            if col1.button("Save JSON"):
                project_data = {
                    "nodes": st.session_state.nodes,
                    "connections": st.session_state.connections,
                    "meta": {"name": st.session_state.project_name}
                }
                json_str = json.dumps(project_data, indent=2)
                st.download_button("Download", json_str, file_name=f"{st.session_state.project_name}.mmp", mime="application/json")
            
            uploaded = col2.file_uploader("Load", type=["mmp"], label_visibility="collapsed")
            if uploaded:
                try:
                    data = json.load(uploaded)
                    st.session_state.nodes = data.get("nodes", {})
                    st.session_state.connections = data.get("connections", [])
                    st.session_state.project_name = data.get("meta", {}).get("name", "Imported")
                    st.rerun()
                except Exception as e:
                    st.error(f"Load failed: {e}")

        # Render Settings
        with st.expander("🎬 Render Settings"):
            fps = st.slider("FPS", 10, 60, 15)
            quality = st.select_slider("Quality", options=["l", "m", "h"], value="l")
            if st.button("Render Final Video"):
                with st.spinner("Rendering..."):
                    Engine.render_video(fps, quality)
        
        # AI Settings
        with st.expander("🤖 Gemini AI"):
            st.text_input("API Key", key="api_key", type="password", help="Get from Google AI Studio")
            st.caption("Model: gemini-3-flash-preview")

def editor_tab():
    col_list, col_props, col_graph = st.columns([1, 1, 2])
    
    # 1. NODE LIST & ADDER
    with col_list:
        st.subheader("Nodes")
        
        # Add Node
        add_type = st.selectbox("Type", ["Mobject", "Animation"])
        base_cls = Mobject if add_type == "Mobject" else Animation
        available = get_manim_classes(base_cls)
        
        # Searchable dropdown
        selected_cls = st.selectbox("Class", available)
        if st.button("Add Node", use_container_width=True):
            NodeManager.create_node(selected_cls, add_type.upper())
            st.rerun()
            
        st.divider()
        
        # List Existing
        node_ids = list(st.session_state.nodes.keys())
        if not node_ids:
            st.info("No nodes added.")
            selected_node_id = None
        else:
            # Create labels for selection
            labels = {nid: st.session_state.nodes[nid]['name'] for nid in node_ids}
            selected_node_id = st.radio(
                "Select to Edit:", 
                node_ids, 
                format_func=lambda x: labels[x]
            )
            
            if st.button("🗑️ Delete Selected", type="primary"):
                NodeManager.delete_node(selected_node_id)
                st.rerun()

    # 2. PROPERTIES INSPECTOR
    with col_props:
        st.subheader("Inspector")
        if selected_node_id and selected_node_id in st.session_state.nodes:
            node = st.session_state.nodes[selected_node_id]
            st.markdown(f"**{node['cls_name']}** (`{node['id']}`)")
            
            # Name Editor
            new_name = st.text_input("Name", node['name'])
            if new_name != node['name']:
                node['name'] = new_name
            
            st.divider()
            
            # Dynamic Params
            params = node['params']
            for k, v in params.items():
                val = v
                # Determine widget type based on value
                if isinstance(v, bool) or str(v).lower() in ['true', 'false']:
                    val = st.checkbox(k, value=(str(v).lower()=='true'))
                elif str(v).replace('.','',1).isdigit():
                    # Number
                    val = st.text_input(k, value=str(v))
                else:
                    # String / Color / Array
                    val = st.text_input(k, value=str(v) if v is not None else "")
                
                params[k] = val
                
            # Connections (Linker)
            st.divider()
            st.markdown("**Output To (Target)**")
            
            # Logic: We connect THIS node TO another node
            # Filter potential targets (prevent self-loops)
            targets = [n for nid, n in st.session_state.nodes.items() if nid != selected_node_id]
            target_ids = [n['id'] for n in targets]
            target_labels = {n['id']: n['name'] for n in targets}
            
            target_sel = st.selectbox(
                "Connect to...", 
                ["(None)"] + target_ids,
                format_func=lambda x: target_labels[x] if x != "(None)" else "(Select Node)"
            )
            
            if st.button("🔗 Link") and target_sel != "(None)":
                NodeManager.connect(selected_node_id, target_sel)
                st.rerun()

    # 3. VISUAL GRAPH
    with col_graph:
        st.subheader("Graph View")
        
        if st.session_state.nodes:
            graph = graphviz.Digraph()
            graph.attr(rankdir='LR')
            
            for nid, node in st.session_state.nodes.items():
                shape = 'box' if node['type'] == 'MOBJECT' else 'ellipse'
                color = 'lightblue' if node['type'] == 'MOBJECT' else 'lightyellow'
                style = 'filled'
                if nid == selected_node_id:
                    color = 'orange'
                
                graph.node(nid, node['name'], shape=shape, style=style, fillcolor=color)
            
            for c in st.session_state.connections:
                graph.edge(c["from"], c["to"])
                
            st.graphviz_chart(graph)
        else:
            st.markdown("*Add nodes to see the graph.*")
            
        # Preview Trigger
        if st.button("📸 Render Preview Frame", use_container_width=True):
            with st.spinner("Rendering Preview..."):
                Engine.render_frame()

def preview_tab():
    st.subheader("Preview & Code")
    c1, c2 = st.columns(2)
    
    with c1:
        if st.session_state.last_preview:
            st.image(st.session_state.last_preview, caption="Latest Preview Frame")
        else:
            st.info("No preview generated yet.")
            
        if st.session_state.last_video:
            st.video(st.session_state.last_video)
            st.success("Video Rendered!")
            
    with c2:
        st.markdown("### Generated Python Code")
        code = Engine.generate_code()
        st.code(code, language="python")

def ai_tab():
    st.subheader("🤖 Gemini AI Generator")
    
    if not st.session_state.api_key:
        st.warning("Please enter your Gemini API Key in the Sidebar settings.")
        return

    prompt = st.text_area("Describe your scene:", placeholder="Create a red circle that transforms into a blue square...")

    if st.button("Generate Nodes"):
        try:
            client = genai.Client(api_key=st.session_state.api_key or os.environ.get("GEMINI_API_KEY"))

            sys_prompt = """
    You are a Manim expert.
    Output ONLY raw Python code wrapped inside ```python fences.
    Do not define a Scene class.
    Just write variable declarations and animation calls.
    """

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=sys_prompt + "\nRequest: " + prompt)],
                )
            ]

            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH")
            )

            stream_container = st.empty()
            full_text = ""

            for chunk in client.models.generate_content_stream(
                model="gemini-3-flash-preview",
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    full_text += chunk.text
                    match = re.search(r"```python\s*(.*?)\s*```", full_text, re.DOTALL)
                    if match:
                        stream_container.code(match.group(1), language="python")

            st.success("Code generated. Extracted clean. No excuses left.")

        except Exception as e:
            st.error(f"AI Error: {e}")

# ==============================================================================
# 6. MAIN APP LOOP
# ==============================================================================

def main():
    sidebar_ui()
    
    tab_editor, tab_prev, tab_ai, tab_logs = st.tabs(["Editor", "Preview / Export", "AI Assistant", "Logs"])
    
    with tab_editor:
        editor_tab()
        
    with tab_prev:
        preview_tab()
        
    with tab_ai:
        ai_tab()
        
    with tab_logs:
        st.subheader("System Logs")
        if st.button("Clear Logs"):
            clear_logs()
            st.rerun()
        
        for entry in reversed(st.session_state.logs):
            color = "red" if "ERROR" in entry else "orange" if "WARN" in entry else "grey"
            st.markdown(f":{color}[{entry}]")

if __name__ == "__main__":
    main()
    