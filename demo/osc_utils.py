from pythonosc import dispatcher
from pythonosc import osc_server

from config import (
    OSC_TOGGLE_ADDRESS, OSC_DISPLAY_ADDRESS, OSC_SET_CONCENTRATION,
    DASHBOARD_STATUS_ADDRESS, DASHBOARD_TRIGGER_ADDRESS
)
from osc_handler import OSCHandler
from dashboard_osc_handler import DashboardOSCHandler

def run_osc_server(ip, port, handler, server_name="OSC Server"):
    """Generic OSC server runner."""
    disp = dispatcher.Dispatcher()
    
    # Print server info
    print(f"Setting up {server_name} OSC Server on {ip}:{port}")
    
    # Dynamically map based on handler type
    if isinstance(handler, OSCHandler):  # Student UI Listener
        print(f"Configuring handler for Student UI at addresses: {OSC_TOGGLE_ADDRESS}, {OSC_DISPLAY_ADDRESS}, {OSC_SET_CONCENTRATION}")
        disp.map(OSC_TOGGLE_ADDRESS, handler.handle_message)
        disp.map(OSC_DISPLAY_ADDRESS, handler.handle_message)
        disp.map(OSC_SET_CONCENTRATION, handler.handle_message)
    elif isinstance(handler, DashboardOSCHandler):  # Dashboard Listener
        # Use specific handlers from DashboardOSCHandler
        print(f"Configuring handler for Dashboard at addresses: {DASHBOARD_STATUS_ADDRESS}, {DASHBOARD_TRIGGER_ADDRESS}")
        disp.map(DASHBOARD_STATUS_ADDRESS, handler.handle_status)
        disp.map(DASHBOARD_TRIGGER_ADDRESS, handler.handle_trigger)
        
        # Verify the handlers are callable
        print(f"Status handler is {handler.handle_status}")
        print(f"Trigger handler is {handler.handle_trigger}")
    else:
        print(f"Error: Unknown handler type for OSC server {server_name}")
        return

    try:
        server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
        print(f"{server_name} OSC Server now serving on {server.server_address}")
        server.serve_forever()
    except OSError as e:
        print(f"Error starting {server_name} OSC Server on {ip}:{port} - {e}")
    except Exception as e:
        print(f"An unexpected error occurred in {server_name} OSC server: {e}") 