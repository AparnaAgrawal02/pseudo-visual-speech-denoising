import os
import sys
import argparse
import numpy as np
import subprocess
import librosa
from tqdm import tqdm
import audio.audio_utils as audio
import audio.hparams as hp
from models import *
import matplotlib.image
import torch
import cv2
import uuid

# Initialize the global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sampling_rate = 16000


def load_wav(path, video_file):
    video = os.path.join(path, video_file)
    wav_file = 'tmp.wav'

    subprocess.call('ffmpeg -hide_banner -loglevel panic -threads 1 -y -i %s -async 1 -ac 1 -vn \
					-acodec pcm_s16le -ar 16000 %s' % (video, wav_file), shell=True)

    wav = audio.load_wav(wav_file, sampling_rate)

    os.remove("tmp.wav")

    return wav


def get_spec(wav):

    # Compute STFT using librosa
    stft = librosa.stft(y=wav, n_fft=hp.hparams.n_fft_den,
                        hop_length=hp.hparams.hop_size_den, win_length=hp.hparams.win_size_den).T
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
    noisy_seg_wav = noisy_wav[start_idx: end_idx]
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


def generate_video(stft, args, root, f, name):

    # Reconstruct the predicted wav
    mag = stft[:257, :]
    phase = stft[257:, :]

    denorm_mag = audio.unnormalize_mag(mag)
    denorm_phase = audio.unnormalize_phase(phase)
    recon_mag = audio.amp_from_db(denorm_mag)
    complex_arr = audio.make_complex(recon_mag, denorm_phase)
    wav = librosa.istft(complex_arr, hop_length=hp.hparams.hop_size_den,
                        win_length=hp.hparams.win_size_den)
    print(wav.shape, "generated")
    base = os.path.basename(args.input)
    copy = root
    copy = os.path.basename(os.path.dirname(copy))
    # Create the folder to save the results
    result_dir = os.path.join(args.result_dir, os.path.basename(root))
    result_dir = os.path.join(result_dir, copy)

    print(result_dir, args.result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Save the wav file

    audio_output = os.path.join(
        result_dir, result_dir, f.split(".")[0]+"_result.wav")
    librosa.output.write_wav(audio_output, wav, sampling_rate)

    print("Saved the denoised audio file:", audio_output)

    # Save the video output file if the input is a video file
    if f.split('.')[1] in ['wav', 'mp3']:
        audio_ = os.path.join(result_dir, f.split(".")[0]+".wav")
        librosa.output.write_wav(audio_, wav, sampling_rate)

        return
    else:
        # print("Hi")
        no_sound_video = os.path.join(result_dir, 'result_nosouund.mp4')
        subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s -c copy -an -strict -2 %s' %
                        (os.path.join(root, f), no_sound_video), shell=True)

        video_output_mp4 = os.path.join(result_dir, f)
        if os.path.exists(video_output_mp4):
            os.remove(video_output_mp4)

        subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -strict -2 -q:v 1 %s' %
                        (audio_output, no_sound_video, video_output_mp4), shell=True)
        subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -strict -2 -q:v 1 %s' %
                        (audio_output, name, video_output_mp4.split(".")[0]+"_faces.mp4"), shell=True)

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


def load_lipsync_model(args):

    lipsync_student = Lipsync_Student()

    if not torch.cuda.is_available():
        lipsync_student_checkpoint = torch.load(
            args.lipsync_student_model_path, map_location='cpu')
    else:
        lipsync_student_checkpoint = torch.load(
            args.lipsync_student_model_path)

    ckpt = {}
    for key in lipsync_student_checkpoint['state_dict'].keys():
        if key.startswith('module.'):
            k = key.split('module.', 1)[1]
        else:
            k = key
        ckpt[k] = lipsync_student_checkpoint['state_dict'][key]
    lipsync_student.load_state_dict(ckpt)
    lipsync_student = lipsync_student.to(device)

    return lipsync_student.eval()


def predict(args):
    print("input:", args.input)
    # Load the student lipsync model
    lipsync_student = load_lipsync_model(args)
    # Load the model
    model = load_model(args)
    for root, dirs, files in os.walk(args.input):
            # print("files:",files)
        if len(files) == 0:
            continue
        for f in files:
                # Load the input wav
            # print("file:",os.path.join(root,f))
            try:
                inp_wav = load_wav(root, f)
                print("Input wav: ", inp_wav.shape)
            except:
                continue
            total_steps = inp_wav.shape[0]
            # print("inp_wav.shape:",inp_wav.shape)

            # Get the windows of 1 second wav step segments with a small overlap 0===1280
            id_windows = [range(i, i + hp.hparams.wav_step_size) for i in range(1280, total_steps,
                                                                                hp.hparams.wav_step_size - hp.hparams.wav_step_overlap) if (i + hp.hparams.wav_step_size <= total_steps)]
            print("id_windows:", id_windows)
            #get faces from video
            video_cap = cv2.VideoCapture(args.face_path)
            fps = int(video_cap.get(cv2.CAP_PROP_FPS))

            success, image = video_cap.read()
            faces =[]
            while success:
                faces.append(image)
                success, image = video_cap.read()
            faces = np.array(faces)
            generated_stft = None
            all_spec_batch = []
            all_mel_batch = []
            skip = False
            
            frame_idx =0
            for i, window in enumerate(id_windows):
                
            #print("within for:",i)
                start_idx = window[0]
                end_idx = start_idx + hp.hparams.wav_step_size
                

                # Segment the wav (1 second window)
                wav = inp_wav[start_idx: end_idx]
                #faces of 1 second window
                face1sec = faces[frame_idx:frame_idx+fps]
                frame_idx = frame_idx+fps
                print("face1sec:",face1sec.shape)

                # Get the corresponding input noisy melspectrograms
                spec_window = get_spec(wav)
                if(spec_window.shape[0] != hp.hparams.spec_step_size):
                    skip = True
                    print("skip:", skip)
                    break
                all_spec_batch.append(spec_window)

                # Get the melspectrogram for lipsync model
                mel_window = get_segmented_mels(start_idx, inp_wav)
                if(mel_window is None):
                    skip = True
                    print("mel_window skip:", skip)
                    break
                mel_window = np.expand_dims(mel_window, axis=1)
                all_mel_batch.append(mel_window)

            if skip == True or len(all_spec_batch) == 0 or len(all_mel_batch) == 0:
                print("return skip:", skip)
                continue

            all_spec_batch = np.array(all_spec_batch)

            all_mel_batch = np.array(all_mel_batch)

            if all_spec_batch.shape[0] != all_mel_batch.shape[0]:
                continue

            print("Total input segment windows: ", all_spec_batch.shape[0])
            out = None
            pred_stft = []
            for i in tqdm(range(0, all_spec_batch.shape[0], args.batch_size)):
                print("last for:", i)
                mel_batch = all_mel_batch[i:i+args.batch_size]
                spec_batch = all_spec_batch[i:i+args.batch_size]

                # Convert to torch tensors
                inp_mel = torch.FloatTensor(mel_batch).to(device)
                inp_stft = torch.FloatTensor(spec_batch).to(device)
                face1sec = torch.FloatTensor(face1sec).to(device)

                """ # Predict the faces using the student lipsync model
                with torch.no_grad():
                    faces = lipsync_student(inp_mel)
                """

                print(faces.shape)

                # Predict the spectrograms for the corresponding window
                with torch.no_grad():
                    pred = model(inp_stft, face1sec)

                # Detach from gpu
                pred = pred.cpu().numpy()

                pred_stft.extend(pred)

            print("Successfully predicted for all the windows")

            # Convert to numpy array
            pred_stft = np.array(pred_stft)

            # Concatenate all the predictions
            steps = int(hp.hparams.spec_step_size -
                        ((hp.hparams.wav_step_overlap/640) * 4))

            if pred_stft.shape[0] == 1:
                generated_stft = pred_stft[0].T
            else:
                generated_stft = pred_stft[0].T[:, :steps]

            for i in range(1, pred_stft.shape[0]):
                # Last batch
                if i == pred_stft.shape[0]-1:
                    generated_stft = np.concatenate(
                        (generated_stft, pred_stft[i].T), axis=1)
                else:
                    generated_stft = np.concatenate(
                        (generated_stft, pred_stft[i].T[:, :steps]), axis=1)

            if generated_stft is not None:
                print("generated_stft:", generated_stft.shape)
                generate_video(generated_stft, args, root, f, args.face_path)
            else:
                print("Oops! Couldn't denoise the input file!")
            out.release()


'''
def predict(args):
	# Load the student lipsync model
        lipsync_student = load_lipsync_model(args)
        # Load the model
        model = load_model(args)
        print("input:",args.input)
        for root, dirs, files in os.walk(args.input):
                print(root)
                print(dirs)
                print(files)
                if len(files)==0:
                    continue
                for f in files:
			# Load the input wav
                        inp_wav = load_wav(root,f)
                        print("Input wav: ", inp_wav.shape)
                        total_steps = inp_wav.shape[0]

			# Get the windows of 1 second wav step segments with a small overlap
                        id_windows = [range(i, i + hp.hparams.wav_step_size) for i in range(1280, total_steps, hp.hparams.wav_step_size - hp.hparams.wav_step_overlap) if (i + hp.hparams.wav_step_size <= total_steps)]
                        print("id_windows:",id_windows)

                        generated_stft = None
			all_spec_batch = []
			all_mel_batch = []
			skip=False
			for i, window in enumerate(id_windows):

				start_idx = window[0]
				end_idx = start_idx + hp.hparams.wav_step_size 
				
				# Segment the wav (1 second window)
				wav = inp_wav[start_idx : end_idx]

				# Get the corresponding input noisy melspectrograms
				spec_window = get_spec(wav) 
				if(spec_window.shape[0] != hp.hparams.spec_step_size):
					skip=True
					break
				all_spec_batch.append(spec_window)
				
				# Get the melspectrogram for lipsync model
				mel_window = get_segmented_mels(start_idx, inp_wav)
				if(mel_window is None):
					skip=True
					break

				mel_window = np.expand_dims(mel_window, axis=1)
				all_mel_batch.append(mel_window)


			if skip==True or len(all_spec_batch)==0 or len(all_mel_batch)==0:
				print( skip==True, len(all_spec_batch)==0 , len(all_mel_batch)==0,f)
				continue

			all_spec_batch = np.array(all_spec_batch)

			all_mel_batch = np.array(all_mel_batch)

			if all_spec_batch.shape[0] != all_mel_batch.shape[0]:
				print("all_spec_batch.shape[0] != all_mel_batch.shape[0]",f)
				continue

			print("Total input segment windows: ", all_spec_batch.shape[0])

			pred_stft = []
			for i in tqdm(range(0, all_spec_batch.shape[0], args.batch_size)):

				mel_batch = all_mel_batch[i:i+args.batch_size]
				spec_batch = all_spec_batch[i:i+args.batch_size]

				# Convert to torch tensors
				inp_mel = torch.FloatTensor(mel_batch).to(device)
				inp_stft = torch.FloatTensor(spec_batch).to(device)

				# Predict the faces using the student lipsync model
				with torch.no_grad(): 
					faces = lipsync_student(inp_mel)

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
				generate_video(generated_stft, args,root,f)
			else:
				print("Oops! Couldn't denoise the input file!")

'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lipsync_student_model_path', type=str,
                        required=True, help='Path of the lipgan model to generate frames')
    parser.add_argument('--checkpoint_path', type=str,  required=True,
                        help='Path of the saved checkpoint to load weights from')
    #parser.add_argument('--input', type=str, required=True, help='Filepath of input noisy audio/video')
    parser.add_argument('--face_path', help='video directory', required=True)
    parser.add_argument('--input', help='video directory', required=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        required=False, help='Batch size for the model')
    parser.add_argument('--result_dir', default='results', required=False,
                        help='Path of the directory to save the results')

    args = parser.parse_args()

    predict(args)
