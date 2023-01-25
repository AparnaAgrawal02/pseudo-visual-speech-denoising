#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH -w gnode033
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
 
python inference.py --lipsync_student_model_path=checkpoints/lipsync_student.pth --checkpoint_path=checkpoints/denoising.pt --input=/ssd_scratch/cvit/jaya/output/noisehome_tamil/0 --result_dir="/ssd_scratch/cvit/jaya/denoised/noise_home_tamil/snr0"

python inference.py --lipsync_student_model_path=checkpoints/lipsync_student.pth --checkpoint_path=checkpoints/denoising.pt --input=/ssd_scratch/cvit/jaya/output/noisehome_tamil/minus_8 --result_dir="/ssd_scratch/cvit/jaya/denoised/noise_home_tamil/snr8"



python inference.py --lipsync_student_model_path=checkpoints/lipsync_student.pth --checkpoint_path=checkpoints/denoising.pt --input=/ssd_scratch/cvit/jaya/output/noisehome_telegu/0 --result_dir="/ssd_scratch/cvit/jaya/denoised/noise_home_telegu/snr0"

python inference.py --lipsync_student_model_path=checkpoints/lipsync_student.pth --checkpoint_path=checkpoints/denoising.pt --input=/ssd_scratch/cvit/jaya/output/noisehome_telegu/minus_8 --result_dir="/ssd_scratch/cvit/jaya/denoised/noise_home_telegu/snr8"



python inference.py --lipsync_student_model_path=checkpoints/lipsync_student.pth --checkpoint_path=checkpoints/denoising.pt --input=/ssd_scratch/cvit/jaya/output/noisehome_hindi/0 --result_dir="/ssd_scratch/cvit/jaya/denoised/noise_home_hindi/snr0"


python inference.py --lipsync_student_model_path=checkpoints/lipsync_student.pth --checkpoint_path=checkpoints/denoising.pt --input=/ssd_scratch/cvit/jaya/output/noisehome_hindi/minus_8 --result_dir="/ssd_scratch/cvit/jaya/denoised/noise_home_hindi/snr8"


