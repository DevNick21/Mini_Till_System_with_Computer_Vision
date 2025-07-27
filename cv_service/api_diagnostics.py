#!/usr/bin/env python
"""
Diagnostic tool for the BetFred Classification API system
Checks system status, model availability, and components
"""
import os
import sys
import argparse
import time
from pathlib import Path


def check_system_status():
    """Check the current status of the classification system"""
    current_dir = Path(__file__).parent
    model_dir = current_dir / "trained_models"

    # Check for required components
    api_exists = os.path.exists(current_dir / "classification_api.py")
    run_api_exists = os.path.exists(current_dir / "run_api.py")
    train_exists = os.path.exists(current_dir / "train.py")
    
    # Check for model files
    models = {
        "resnet": os.path.exists(model_dir / "best_resnet_classifier.pth"),
        "efficientnet": os.path.exists(model_dir / "best_efficientnet_classifier.pth"),
        "densenet": os.path.exists(model_dir / "best_densenet_classifier.pth"),
        "ensemble_weights": os.path.exists(model_dir / "ensemble_weights.pth")
    }

    # Check if API is active
    api_active = False
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['cmdline'] and any('uvicorn' in cmd for cmd in proc.info['cmdline']):
                cmdline = ' '.join(proc.info['cmdline'])
                if 'classification_api' in cmdline:
                    api_active = True
                    break
    except ImportError:
        pass

    # Calculate if system is ready
    # System is ready if at least EfficientNet or DenseNet model is available
    models_available = models["efficientnet"] or models["densenet"]
    
    system_ready = api_exists and models_available
    
    return {
        "system_ready": system_ready,
        "models": models,
        "api_exists": api_exists,
        "run_api_exists": run_api_exists,
        "train_exists": train_exists,
        "api_active": api_active
    }


def print_status(status):
    """Print the system status in a readable format"""
    print("\n✨ BETFRED EFFICIENTNET CLASSIFICATION SYSTEM DIAGNOSTIC ✨")
    print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overall System Status
    if status["system_ready"]:
        print("\n✅ SYSTEM STATUS: READY")
    else:
        print("\n⚠️ SYSTEM STATUS: NOT READY")
    
    # Service Status
    if status["api_active"]:
        print("🟢 API Service: Running")
    else:
        print("⚫ API Service: Stopped")
    
    # Model Status
    available_models = sum(1 for exists in status["models"].values() if exists)
    total_models = len(status["models"])
    print(f"📊 Models: {available_models}/{total_models} available")
    
    # Components Status
    components_available = sum([
        status["api_exists"], 
        status["run_api_exists"],
        status["train_exists"]
    ])
    print(f"🧩 Components: {components_available}/3 available")

    # Detailed Status
    print("\n=== DETAILED STATUS ===")
    
    # Models Section
    print("\n📊 MODEL FILES:")
    for model, exists in status["models"].items():
        icon = "✅" if exists else "❌"
        status_text = "Available" if exists else "Missing"
        print(f"{icon} {model.ljust(16)} {status_text}")
    
    # Components Section  
    print("\n🧩 SYSTEM COMPONENTS:")
    api_icon = "✅" if status["api_exists"] else "❌"
    api_status = "Available" if status["api_exists"] else "Missing"
    print(f"{api_icon} API Implementation    {api_status}")
    
    run_api_icon = "✅" if status["run_api_exists"] else "❌"
    run_api_status = "Available" if status["run_api_exists"] else "Missing"
    print(f"{run_api_icon} API Launcher         {run_api_status}")
    
    train_icon = "✅" if status["train_exists"] else "❌"
    train_status = "Available" if status["train_exists"] else "Missing"
    print(f"{train_icon} Training Script      {train_status}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not status["system_ready"]:
        if not any(status["models"].values()):
            print("• Models missing - Run training first:")
            print("  > python train.py")
        elif not status["models"]["ensemble_weights"]:
            print("• Ensemble weights missing - Run training:")
            print("  > python train.py")
        elif not status["api_exists"]:
            print("• API implementation missing")
    else:
        if status["api_active"]:
            print("• System is running correctly")
            print("• API endpoints available at http://localhost:8001")
        else:
            print("• System is ready but not running")
            print("• Start API with: python run_api.py")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic tool for the BetFred Classification API")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Display detailed information")
    parser.add_argument("--refresh", "-r", type=int, default=0,
                        help="Refresh status every N seconds (0 to disable)")
    args = parser.parse_args()

    # Function to display current status
    def display_status():
        status = check_system_status()
        print_status(status)
        
        # Add detailed information if verbose flag is set
        if args.verbose:
            print("\n📋 DETAILED INFORMATION:")
            print("• API URL:        http://localhost:8001")
            print("• Health Check:   http://localhost:8001/health")
            print("• Documentation:  http://localhost:8001/docs")
            
            # System component paths
            print("\n📁 SYSTEM COMPONENT PATHS:")
            current_dir = Path(__file__).parent
            print(f"• API Path:       {current_dir / 'classification_api.py'}")
            print(f"• Launcher Path:  {current_dir / 'run_api.py'}")
            print(f"• Training Path:  {current_dir / 'train.py'}")
            print(f"• Models Path:    {current_dir / 'trained_models'}")
            
            # Help information
            print("\n💁 HELP INFORMATION:")
            print("• To start API:   python run_api.py")
            print("• To train model: python train.py")
    
    # Single display or continuous refresh
    if args.refresh <= 0:
        display_status()
    else:
        try:
            while True:
                # Clear screen and display updated status
                os.system('cls' if os.name == 'nt' else 'clear')
                display_status()
                print(f"\n🔄 Auto-refreshing every {args.refresh} seconds. Press Ctrl+C to stop.")
                time.sleep(args.refresh)
        except KeyboardInterrupt:
            print("\n✅ Monitoring stopped.")


if __name__ == "__main__":
    main()
