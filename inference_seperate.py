import os, sys, argparse
import numpy as np
import subprocess
import librosa, face_detection
from tqdm import tqdm
import cv2
import audio.audio_utils as audio
import audio.hparams as hp 
from models import *
import torch

# Initialize the global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 	
sampling_rate = 16000

def load_wav(path,video_file):
	video = os.path.join(path,video_file)
	wav_file  = 'tmp.wav';

	subprocess.call('ffmpeg -hide_banner -loglevel panic -threads 1 -y -i %s -async 1 -ac 1 -vn \
					-acodec pcm_s16le -ar 16000 %s' % (video, wav_file), shell=True)
	
	wav = audio.load_wav(wav_file, sampling_rate)

	os.remove("tmp.wav")

	return wav


def get_spec(wav):

	# Compute STFT using librosa
	stft = librosa.stft(y=wav, n_fft=hp.hparams.n_fft_den, hop_length=hp.hparams.hop_size_den, win_length=hp.hparams.win_size_den).T
	stft = stft[:-1]														# Tx257

	# Decompose into magnitude and phase representations
	mag = np.abs(stft)
	mag = audio.db_from_amp(mag)
	phase = audio.angle(stft)

	# Normalize the magnitude and phase representations
	norm_mag = audio.normalize_mag(mag)
	norm_phase = audio.normalize_phase(phase)
	
	# Concatenate the magnitude and phase representations
	spec_ip = np.concatenate((norm_mag, norm_phase), axis=1)				# Tx514

	return spec_ip


def crop_mels(start_idx, noisy_wav):
    
    end_idx = start_idx + 3200

    # Get the segmented wav (0.2 second)
    noisy_seg_wav = noisy_wav[start_idx : end_idx]
    if len(noisy_seg_wav) != 3200: 
        return None
    
    # Compute the melspectrogram using librosa
    spec = audio.melspectrogram(noisy_seg_wav, hp.hparams).T          		# 16x80
    spec = spec[:-1] 

    return spec

def get_segmented_mels(start_idx, noisy_wav):

    mels = []
    if start_idx - 1280 < 0: 
        return None

    # Get the overlapping continuous segments of noisy mels
    for i in range(start_idx, start_idx + hp.hparams.wav_step_size, 640):
        m = crop_mels(i - 1280, noisy_wav)
        if m is None or m.shape[0] != hp.hparams.mel_step_size:
            return None
        mels.append(m.T)

    mels = np.asarray(mels)                                             	

    return mels

def generate_video(stft, args,root,f):

	# Reconstruct the predicted wav
	mag = stft[:257, :]
	phase = stft[257:, :]

	denorm_mag = audio.unnormalize_mag(mag)
	denorm_phase = audio.unnormalize_phase(phase)
	recon_mag = audio.amp_from_db(denorm_mag)
	complex_arr = audio.make_complex(recon_mag, denorm_phase)
	wav = librosa.istft(complex_arr, hop_length=hp.hparams.hop_size_den, win_length=hp.hparams.win_size_den)
	print(wav.shape,"generated")
	base = os.path.basename(args.input)
	copy = root
	copy =  os.path.basename(os.path.dirname(copy))
	# Create the folder to save the results
	result_dir =os.path.join(args.result_dir, os.path.basename(root))
	result_dir =os.path.join(result_dir,copy)
	
	print(result_dir,args.result_dir)
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	# Save the wav file

	audio_output = os.path.join(result_dir, 'result.wav')
	librosa.output.write_wav(audio_output, wav, sampling_rate)

	print("Saved the denoised audio file:", audio_output)

	# Save the video output file if the input is a video file
	if f.split('.')[1] in ['wav', 'mp3']:
		audio_ = os.path.join(result_dir,f.split(".")[0]+".wav")
		librosa.output.write_wav(audio_, wav, sampling_rate)
		os.remove(audio_output)
		return
	else:
		#print("Hi")
		no_sound_video = os.path.join(result_dir, 'result_nosouund.mp4')
		subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s -c copy -an -strict -2 %s' % (os.path.join(root,f), no_sound_video), shell=True)

		video_output_mp4 = os.path.join(result_dir, f)
		if os.path.exists(video_output_mp4):
			os.remove(video_output_mp4)
		
		subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -strict -2 -q:v 1 %s' % 
						(audio_output, no_sound_video, video_output_mp4), shell=True)

		os.remove(no_sound_video)

		print("Saved the denoised video file:", video_output_mp4)
		os.remove(audio_output)
		return
def load_model(args):

	model = Model()
	print("Loaded model from: ", args.checkpoint_path)

	if not torch.cuda.is_available():
		checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
	else:
		checkpoint = torch.load(args.checkpoint_path)

	ckpt = {}
	for key in checkpoint['state_dict'].keys():
		if key.startswith('module.'):
			k = key.split('module.', 1)[1]
		else:
			k = key
		ckpt[k] = checkpoint['state_dict'][key]
	model.load_state_dict(ckpt)	
	model = model.to(device)
	return model.eval()

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def predict(args):
	print("input:",args.input)
	# Load the student lipsync model
	#lipsync_student = load_lipsync_model(args)
	# Load the model
	model = load_model(args)
	for root, dirs, files in os.walk(args.input):
		print("root:",root)
		print("dirs:",dirs)
		#print("files:",files)
		if len(files)==0:
			continue
		for f in files:
			try:
				inp_wav = load_wav(root,f)
				video_stream = cv2.VideoCapture(f)
				fps = video_stream.get(cv2.CAP_PROP_FPS)

				print("Input wav: ", inp_wav.shape)
				print('Reading video frames...')

				full_frames = []
				while 1:
					still_reading, frame = video_stream.read()
					if not still_reading:
						video_stream.release()
						break
					if args.resize_factor > 1:
						frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

					if args.rotate:
						frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

					y1, y2, x1, x2 = args.crop
					if x2 == -1: x2 = frame.shape[1]
					if y2 == -1: y2 = frame.shape[0]

					frame = frame[y1:y2, x1:x2]

					full_frames.append(frame)
				print ("Number of frames available for inference: "+str(len(full_frames)))

			except:
				continue
			
			total_steps = inp_wav.shape[0]
			# Get the windows of 1 second wav step segments with a small overlap 0===1280
			id_windows = [range(i, i + hp.hparams.wav_step_size) for i in range(1280, total_steps, 
						  hp.hparams.wav_step_size - hp.hparams.wav_step_overlap) if (i + hp.hparams.wav_step_size <= total_steps)]

			print("id_windows:",id_windows)			

			generated_stft = None
			all_spec_batch = []
			all_mel_batch = []
			skip=False
			for i, window in enumerate(id_windows):
				#print("within for:",i)
				start_idx = window[0]
				end_idx = start_idx + hp.hparams.wav_step_size 
				
				# Segment the wav (1 second window)
				wav = inp_wav[start_idx : end_idx]

				# Get the corresponding input noisy melspectrograms
				spec_window = get_spec(wav) 
				if(spec_window.shape[0] != hp.hparams.spec_step_size):
					skip=True
					print("skip:",skip)
					break
				all_spec_batch.append(spec_window)
				
				# Get the melspectrogram for lipsync model
				#mel_window = get_segmented_mels(start_idx, inp_wav)
				#if(mel_window is None):
				#	skip=True
				#	print("mel_window skip:",skip)
				#	break

				#mel_window = np.expand_dims(mel_window, axis=1)
				#all_mel_batch.append(mel_window)


			#if skip==True or len(all_spec_batch)==0 or len(all_mel_batch)==0:
			#	print("return skip:",skip)
			#	continue

			#all_spec_batch = np.array(all_spec_batch)

			#all_mel_batch = np.array(all_mel_batch)

			#if all_spec_batch.shape[0] != all_mel_batch.shape[0]:
			#	continue

			#print("Total input segment windows: ", all_spec_batch.shape[0])

			pred_stft = []
			for i in tqdm(range(0, all_spec_batch.shape[0], args.batch_size)):
				print("last for:",i)
				mel_batch = all_mel_batch[i:i+args.batch_size]
				spec_batch = all_spec_batch[i:i+args.batch_size]

				# Convert to torch tensors
				inp_mel = torch.FloatTensor(mel_batch).to(device)
				inp_stft = torch.FloatTensor(spec_batch).to(device)

				# Predict the faces using the student lipsync model
				#with torch.no_grad(): 
					#faces = lipsync_student(inp_mel)

				# Predict the spectrograms for the corresponding window
				with torch.no_grad():
					pred = model(inp_stft, faces)

				# Detach from gpu
				pred = pred.cpu().numpy()

				pred_stft.extend(pred)

			print("Successfully predicted for all the windows")
			
			# Convert to numpy array
			pred_stft = np.array(pred_stft)

			# Concatenate all the predictions 
			steps = int(hp.hparams.spec_step_size - ((hp.hparams.wav_step_overlap/640) * 4))

			if pred_stft.shape[0] == 1:
				generated_stft = pred_stft[0].T
			else:
				generated_stft = pred_stft[0].T[:, :steps]
			
			for i in range(1, pred_stft.shape[0]):
				# Last batch
				if i==pred_stft.shape[0]-1:
					generated_stft = np.concatenate((generated_stft, pred_stft[i].T), axis=1)
				else:
					generated_stft = np.concatenate((generated_stft, pred_stft[i].T[:, :steps]), axis=1)


			if generated_stft is not None:
				print("generated_stft:",generated_stft.shape)
				generate_video(generated_stft, args,root,f)
			else:
				print("Oops! Couldn't denoise the input file!")


if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--lipsync_student_model_path', type=str, required=True, help='Path of the lipgan model to generate frames')
	parser.add_argument('--checkpoint_path', type=str,  required=True, help='Path of the saved checkpoint to load weights from')
	#parser.add_argument('--input', type=str, required=True, help='Filepath of input noisy audio/video')
	parser.add_argument('--input', help='video directory', required=True)
	parser.add_argument('--batch_size', type=int, default=32, required=False, help='Batch size for the model')
	parser.add_argument('--result_dir', default='results', required=False, help='Path of the directory to save the results')

	args = parser.parse_args()

	predict(args)
