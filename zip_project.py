import os
import zipfile
import datetime

def zip_project():
    # Get current timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"project_backup_{timestamp}.zip"
    
    # Folders to exclude (Junk/Heavy files)
    EXCLUDE_DIRS = {
        'node_modules',   # React dependencies (Heavy!)
        '__pycache__',    # Python compiled files
        '.git',           # Git history
        '.vscode',        # Editor settings
        'venv',           # Python Virtual Env
        'env',
        'build',          # React build artifacts
        'dist'
    }
    
    # Extensions to exclude
    EXCLUDE_EXT = {'.pyc', '.zip', '.log'}

    print(f"--- 📦 Starting Backup: {zip_filename} ---")
    
    count = 0
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through all files in current directory
            for root, dirs, files in os.walk('.'):
                # Modify dirs in-place to skip excluded folders
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check extensions
                    _, ext = os.path.splitext(file)
                    if ext in EXCLUDE_EXT or file == zip_filename:
                        continue
                        
                    # Create relative path for the zip structure
                    arcname = os.path.relpath(file_path, start='.')
                    
                    # Print progress every 10 files
                    count += 1
                    if count % 10 == 0:
                        print(f"Zipping... {count} files processed", end='\r')
                        
                    zipf.write(file_path, arcname)
                    
        print(f"\n\n✅ Success! Project zipped to:")
        print(f"   {os.path.abspath(zip_filename)}")
        print(f"   Final Size: {os.path.getsize(zip_filename) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ Error creating zip: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    # Ensure script runs from the directory it's located in
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        zip_project()
    except Exception as e:
        print(e)