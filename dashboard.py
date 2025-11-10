
import streamlit as st
import subprocess
import signal
import os
import time
import sys 

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Vehicle Safety & Automation Suite ",
  
    layout="wide"
)

# --- Initialize Session State ---
if 'lka_process' not in st.session_state:
    st.session_state.lka_process = None
if 'aeb_process' not in st.session_state:
    st.session_state.aeb_process = None
if 'acc_process' not in st.session_state:
    st.session_state.acc_process = None
if 'v2v_1_process' not in st.session_state:
    st.session_state.v2v_1_process = None
if 'v2v_2_process' not in st.session_state:
    st.session_state.v2v_2_process = None

def stop_process(process_key):
    """Helper function to safely terminate a running process."""
    process = st.session_state.get(process_key)
    if process and process.poll() is None: 
        try:

            process.terminate() 
            
            process.wait(timeout=5) # Wait 5 seconds for it to close
            st.success(f"Stopped process {process_key} (PID: {process.pid})")
        except subprocess.TimeoutExpired:
            # If it's still running, force kill it
            process.kill()
            st.warning(f"Process {process_key} did not respond, forcing kill.")
        st.session_state[process_key] = None
    elif process:
        st.session_state[process_key] = None 

def check_process_status(process_key):
    """Checks if a process is still running and updates state."""
    process = st.session_state.get(process_key)
    if process and process.poll() is not None:
        st.session_state[process_key] = None
        return False
    return process is not None

# --- Dashboard Title ---
st.title("Intelligent Vehicle Safety & Automation Suite")
st.markdown("A professional dashboard to demonstrate 4 core ADAS systems. **Please ensure CARLA is running** before launching any module.")
st.info("Remember to launch your MQTT broker (like Mosquitto) before running the V2V demo.")
st.markdown("---")

# --- Create the 4 Tabs ---
tab_lka, tab_aeb, tab_acc, tab_v2v = st.tabs([
    "1. Lane Keep Assist (LKA)", 
    "2. Sensor Fusion AEB", 
    "3. Adaptive Cruise Control (ACC)", 
    "4. V2V Data Sharing"
])


# --- Tab 1: Lane Keep Assist (LKA) ---

with tab_lka:
    st.header("1. Lane Keep Assist (LKA) System")
    st.markdown("This module upgrades the **Lane Departure Warning** into a full control system. It uses OpenCV to find lane deviation, then feeds that error into a **PID Controller** to actively steer the vehicle and keep it centered.")
    
    if os.path.exists("ldw.png"):
        st.image("ldw.png", caption="LKA system actively steering the vehicle.")
    else:
        st.info("Add an image named 'ldw.png' to this folder to see it here.")
    
    col1, col2 = st.columns(2)
    is_lka_running = check_process_status('lka_process')
    with col1:
        if st.button(" Launch LKA Module", key="lka_launch", use_container_width=True, disabled=is_lka_running):
            st.session_state.lka_process = subprocess.Popen([sys.executable, "lka_standalone.py"])
            st.success(f"LKA Module launched! (PID: {st.session_state.lka_process.pid}).")
            st.experimental_rerun()
    with col2:
        if st.button(" Stop LKA Module", key="lka_stop", use_container_width=True, disabled=not is_lka_running):
            stop_process('lka_process')
            st.experimental_rerun()
    if is_lka_running:
        st.success(f"LKA Module is running (PID: {st.session_state.lka_process.pid})...")


# --- Tab 2: Sensor Fusion AEB (Upgraded AEB) ---

with tab_aeb:
    st.header("2. Sensor Fusion Autonomous Emergency Braking (AEB)")
    st.markdown("This module fuses 2D **YOLOv5** object detection with 3D **Depth Camera** data. Braking is now based on precise distance in meters, not 2D pixel size, making it a robust, professional-grade system.")
    
    if os.path.exists("aeb.png"):
        st.image("aeb.png", caption="AEB fusing YOLO (boxes) and Depth (distance) data.")
    else:
        st.info("Add an image named 'aeb.png' to this folder to see it here.")
    
    col1, col2 = st.columns(2)
    is_aeb_running = check_process_status('aeb_process')
    with col1:
        if st.button(" Launch Fusion AEB Module", key="aeb_launch", use_container_width=True, disabled=is_aeb_running):
            st.session_state.aeb_process = subprocess.Popen([sys.executable, "aeb_fusion_standalone.py"])
            st.success(f"Fusion AEB Module launched! (PID: {st.session_state.aeb_process.pid}).")
            st.experimental_rerun()
    with col2:
        if st.button(" Stop Fusion AEB Module", key="aeb_stop", use_container_width=True, disabled=not is_aeb_running):
            stop_process('aeb_process')
            st.experimental_rerun()
    if is_aeb_running:
        st.success(f"Fusion AEB Module is running (PID: {st.session_state.aeb_process.pid})...")


# --- Tab 3: Adaptive Cruise Control (Focused ACC) ---

with tab_acc:
    st.header("3. Adaptive Cruise Control (ACC) System")
    st.markdown("This module demonstrates a focused and robust ACC system. It uses simulator 'ground truth' data for perfect perception, managing a traffic environment, and running a state machine for **following and cruising** using PID control.")
    
    if os.path.exists("acc.png"):
        st.image("acc.png", caption="ACC maintaining a safe following distance.")
    else:
        st.info("Add an image named 'acc.png' to this folder to see it here.")
    
    col1, col2 = st.columns(2)
    is_acc_running = check_process_status('acc_process')
    with col1:
        if st.button(" Launch ACC Module", key="acc_launch", use_container_width=True, disabled=is_acc_running):
            st.session_state.acc_process = subprocess.Popen([sys.executable, "acc_standalone.py"])
            st.success(f"ACC Module launched! (PID: {st.session_state.acc_process.pid}).")
            st.experimental_rerun()
    with col2:
        if st.button(" Stop ACC Module", key="acc_stop", use_container_width=True, disabled=not is_acc_running):
            stop_process('acc_process')
            st.experimental_rerun()
    if is_acc_running:
        st.success(f"ACC Module is running (PID: {st.session_state.acc_process.pid})...")


# --- Tab 4: Cooperative V2V Data Sharing ---

with tab_v2v:
    st.header("4. V2V Data Sharing Monitor")
    st.markdown("This module demonstrates your original V2V concept. It launches **two** vehicles, `car1` and `car2`, both on autopilot. Each car publishes its **full kinematic metadata** (speed, location, acceleration, heading) via MQTT. This demo opens **two GUI windows** to monitor each car, showing the *live data being received* from the other vehicle.")
    
    if os.path.exists("v2v.png"):
        st.image("v2v.png", caption="V2V Monitor showing data received from another vehicle.")
    else:
        st.info("Add an image named 'v2v.png' to this folder to see it here.")
    
    col1, col2 = st.columns(2)
    is_v2v_running = check_process_status('v2v_1_process') or check_process_status('v2v_2_process')
    with col1:
        if st.button(" Launch V2V Demo (2-Cars)", key="v2v_launch", use_container_width=True, disabled=is_v2v_running):
            st.session_state.v2v_1_process = subprocess.Popen([sys.executable, "v2v_data_share_final.py", "--role", "car1", "--follow"])
            time.sleep(2) # Stagger spawns
            st.session_state.v2v_2_process = subprocess.Popen([sys.executable, "v2v_data_share_final.py", "--role", "car2"])
            st.success("VV Module launched! Check for TWO new GUI windows: 'V2V Data Monitor (car1)' and '(car2)'.")
            st.experimental_rerun()
    with col2:
        if st.button(" Stop V2V Demo", key="v2v_stop", use_container_width=True, disabled=not is_v2v_running):
            stop_process('v2v_1_process')
            stop_process('v2v_2_process')
            st.experimental_rerun()
    if is_v2v_running:
        st.success("V2V Module is running...")