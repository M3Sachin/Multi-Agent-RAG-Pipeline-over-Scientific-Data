"""
Run script to start both backend API and Streamlit UI.
Usage:
    python run.py          # Start both
    python run.py api     # Start only API
    python run.py ui      # Start only UI
"""

import argparse
import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def start_api():
    """Start the FastAPI backend server."""
    print("=" * 50)
    print("Starting API server on http://localhost:8000")
    print("=" * 50)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--reload",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=SCRIPT_DIR,
    )


def start_ui():
    """Start the Streamlit UI."""
    print("=" * 50)
    print("Starting Streamlit UI on http://localhost:8501")
    print("=" * 50)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            "8501",
        ],
        cwd=SCRIPT_DIR,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Scientific RAG Pipeline")
    parser.add_argument(
        "service",
        nargs="?",
        choices=["api", "ui", "both"],
        default="both",
        help="Which service to start (default: both)",
    )
    args = parser.parse_args()

    if args.service == "api":
        start_api()
    elif args.service == "ui":
        start_ui()
    else:
        # Start both in parallel using subprocess
        print("Starting both services...")
        print("")

        # Start API in background
        api_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        # Wait for API to start
        print("Waiting for API to start...")
        time.sleep(3)

        # Start UI
        ui_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "ui/app.py",
                "--server.address",
                "0.0.0.0",
                "--server.port",
                "8501",
            ],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        try:
            # Wait for both processes
            api_process.wait()
            ui_process.wait()
        except KeyboardInterrupt:
            print("\nStopping services...")
            api_process.terminate()
            ui_process.terminate()
            api_process.wait()
            ui_process.wait()
            print("Services stopped.")


if __name__ == "__main__":
    main()
