import shutil
import os

def complete_cleanup():
    """Bersihkan SEMUA file generated sebelumnya"""
    print("🧹 COMPLETE CLEANUP STARTED...")
    
    # Delete semua model files
    model_files = [
        "models/vectorizer.pkl",
        "models/classifier.pkl", 
        "models/"
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"✅ Deleted directory: {file_path}")
            else:
                os.remove(file_path)
                print(f"✅ Deleted file: {file_path}")
    
    # Delete output files
    output_dirs = [
        "output/",
        "data/processed/"
    ]
    
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"✅ Deleted directory: {dir_path}")
    
    # Recreate directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/reports", exist_ok=True)
    os.makedirs("output/visualizations", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("✅ COMPLETE CLEANUP FINISHED!")
    print("   All models, cache, and output files deleted")
    print("   Fresh directories created")

# RUN CLEANUP
complete_cleanup()