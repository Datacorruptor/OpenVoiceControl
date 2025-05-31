import os
import requests
import zipfile
import shutil  # Added for directory removal

MODELS = {
    'tiny': {
        'url': 'https://github.com/Datacorruptor/OpenVoiceControl/releases/download/misc/tiny.zip',
        'multipart': False
    },
    'small': {
        'url': 'https://github.com/Datacorruptor/OpenVoiceControl/releases/download/misc/small.zip',
        'multipart': False
    },
    'medium': {
        'url': 'https://github.com/Datacorruptor/OpenVoiceControl/releases/download/misc/medium.zip',
        'multipart': False
    },
    'large-v3-turbo': {
        'url': 'https://github.com/Datacorruptor/OpenVoiceControl/releases/download/misc/large-v3-turbo.zip',
        'multipart': False
    },
    'large-v3': {
        'url_template': 'https://github.com/Datacorruptor/OpenVoiceControl/releases/download/misc/large-v3.zip.{}',
        'parts': 2,
        'multipart': True
    },
}

def main():
    
    conf_model_name = open("settings\\config.txt").read().split("model_name:")[1].split("\n")[0] #janky way to carve out model size from config to download it
    
    
    print(f"Selected model to download {conf_model_name}")
    selected_model = conf_model_name
    
    if selected_model not in MODELS:
        print("No url for specified model, you are on your own")
        return
    
    model_info = MODELS[selected_model]
    models_dir = 'models'
    target_dir = models_dir
    check_dir = os.path.join(models_dir, selected_model)
    
    # Check if model already exists
    if os.path.exists(check_dir):
        print("Model already installed, download canceled.")
        return

    os.makedirs(models_dir, exist_ok=True)
    combined_zip_path = os.path.join(models_dir, 'combined.zip')

    try:
        with open(combined_zip_path, 'wb') as f:
            if model_info['multipart']:
                print(f"Downloading {selected_model} multipart model...")
                for part_num in range(1, model_info['parts'] + 1):
                    part_url = model_info['url_template'].format(str(part_num).zfill(3))
                    print(f"Downloading part {part_num}/{model_info['parts']}")
                    response = requests.get(part_url, stream=True)
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                print(f"Downloading {selected_model} model...")
                response = requests.get(model_info['url'], stream=True)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Download completed. Extracting...")
        with zipfile.ZipFile(combined_zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)  # Extract to model-specific directory
        
        print(f"Extraction completed successfully. Files are in {target_dir}/")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Clean up partial extraction if any errors occurred
        if os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
                print("Cleaned up partially extracted files")
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")
    finally:
        if os.path.exists(combined_zip_path):
            os.remove(combined_zip_path)
            print("Temporary combined zip file removed.")

if __name__ == "__main__":
    main()