import os
import torchio as tio
import argparse

def convert_mhd_to_nii(input_dir, output_dir):
    """
    Convert all MHD files in input directory to NII.GZ format and save them to output directory.
    
    Args:
        input_dir (str): Directory containing MHD files
        output_dir (str): Directory to save converted NII.GZ files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all MHD files in input directory
    mhd_files = [f for f in os.listdir(input_dir) if f.endswith('.mhd')]
    
    for mhd_file in mhd_files:
        input_path = os.path.join(input_dir, mhd_file)
        output_filename = os.path.splitext(mhd_file)[0] + '.nii.gz'
        output_path = os.path.join(output_dir, output_filename)
        
        # Load MHD file and save as NII.GZ
        image = tio.ScalarImage(input_path)
        image.save(output_path)
        print(f"Converted {mhd_file} to {output_filename}")

if __name__ == "__main__":
    input_dir = "/disk/cyq/2025/diffusion_seg_compare/logs/cas_IS/predict-2025-04-06/22-27-13/pred_file"
    output_dir = "/disk/cyq/2025/diffusion_seg_compare/logs/cas_IS/predict-2025-04-06/22-27-13/pred_file"

    convert_mhd_to_nii(input_dir, output_dir)
