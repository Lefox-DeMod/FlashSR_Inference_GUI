import torch

from TorchJaekwon.Util.UtilAudio import UtilAudio
from TorchJaekwon.Util.UtilData import UtilData

from FlashSR.FlashSR import FlashSR

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

student_ldm_ckpt_path:str = './ModelWeights/student_ldm.pth'
sr_vocoder_ckpt_path:str = './ModelWeights/sr_vocoder.pth'
vae_ckpt_path:str = './ModelWeights/vae.pth'
flashsr = FlashSR( student_ldm_ckpt_path, sr_vocoder_ckpt_path, vae_ckpt_path)
flashsr = flashsr.to(device)

audio_path:str = './Assets/ExampleInput/music.wav'

# resample audio to 48kHz
# audio.shape = [batch, time] ex) [4, 245760]
audio, sr = UtilAudio.read(audio_path, sample_rate = 48000)
# currently, the model only supports 245760 samples (5.12 seconds of audio)
audio = UtilData.fix_length(audio, 245760)
audio = audio.to(device)
pred_hr_audio = flashsr(audio)
UtilAudio.write('./output.wav', pred_hr_audio, 48000)
print('')