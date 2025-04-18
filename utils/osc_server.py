from pythonosc import dispatcher
from pythonosc import osc_server

# Import handlers and config - relative paths might need adjustment based on final structure
# Assuming handlers are defined elsewhere and config holds the OSC addresses
from student_ui.osc_handler import OSCHandler
from dashboard.osc_handler import DashboardOSCHandler
from utils.config import (
    OSC_TOGGLE_ADDRESS,
    OSC_DISPLAY_ADDRESS,
    OSC_SET_CONCENTRATION,
    DASHBOARD_STATUS_ADDRESS,
    DASHBOARD_TRIGGER_ADDRESS
)

def run_osc_server(ip, port, handler, server_name="OSC Server"):
    """Generic OSC server runner."""
    disp = dispatcher.Dispatcher()

    # Dynamically map based on handler type
    if isinstance(handler, OSCHandler): # Student UI Listener
        print(f"Mapping Student UI OSC addresses for {server_name}...")
        disp.map(OSC_TOGGLE_ADDRESS, handler.handle_message)
        disp.map(OSC_DISPLAY_ADDRESS, handler.handle_message)
        disp.map(OSC_SET_CONCENTRATION, handler.handle_message)
    elif isinstance(handler, DashboardOSCHandler): # Dashboard Listener
        print(f"Mapping Dashboard OSC addresses for {server_name}...")
        # Use specific handlers from DashboardOSCHandler
        disp.map(DASHBOARD_STATUS_ADDRESS, handler.handle_status, needs_reply_address=False)
        disp.map(DASHBOARD_TRIGGER_ADDRESS, handler.handle_trigger, needs_reply_address=False)
    else:
        print(f"Error: Unknown handler type ({type(handler)}) for OSC server {server_name}")
        return

    try:
        server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
        print(f"{server_name} OSC Server serving on {server.server_address}")
        server.serve_forever()
    except OSError as e:
         print(f"Error starting {server_name} OSC Server on {ip}:{port} - {e}")
         print("Please check if the port is already in use or if permissions are sufficient.")
    except Exception as e:
        print(f"An unexpected error occurred in {server_name} OSC server: {e}") 