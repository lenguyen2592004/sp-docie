import os
import time
import uuid

class Config:
    output_dir = "outputs"
    wandb_project = "docie-ner-re-pipeline"
    wandb_key = "372ca6a40e9fa5baf88db35b6d6dd619f33bcfbc"
    
    @staticmethod
    def get_unique_run_name(base_name):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        short_hash = str(uuid.uuid4())[:8]
        return f"{base_name}_{timestamp}_{short_hash}"
