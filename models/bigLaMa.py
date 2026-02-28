"""
BigLaMa Architecture Loader
Pipeline Stage: Architecture Setup & Transfer Learning Initialization
"""
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_pretrained_lama(target_dir="models/weights"):
    """
    Downloads the official BigLaMa pre-trained checkpoint (Places2) 
    if it doesn't already exist locally.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Define paths for both the zip and the extracted model
    zip_path = os.path.join(target_dir, "big-lama.zip")
    pt_path = os.path.join(target_dir, "big-lama.pt")
    
    # Official Hugging Face mirror
    lama_url = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
    
    # Check if the extracted file already exists
    if os.path.exists(pt_path):
        logging.info("Pre-trained BigLaMa weights (.pt) found locally.")
        return True
        
    # Check if the zip file exists
    if not os.path.exists(zip_path):
        logging.info("Downloading pre-trained BigLaMa (Places2) weights...")
        try:
            # Using wget to bypass potential urllib bot-blocking
            os.system(f"wget -qO {zip_path} {lama_url}")
            
            # Verify the download actually created the file
            if os.path.exists(zip_path):
                logging.info(f"BigLaMa weights archive saved to {zip_path}")
            else:
                logging.warning("wget failed to download the file.")
                
        except Exception as e:
            logging.warning(f"Could not fetch weights automatically: {e}")
            logging.info("We will load the hub model dynamically in Phase 4.")
    else:
        logging.info("Pre-trained BigLaMa weights archive (.zip) found locally.")
        
    return True
