# RMS-ControlNet
[![arXiv](https://img.shields.io/badge/arXiv-2408.11915-red.svg?style=flat-square)](https://www.arxiv.org/abs/2408.11915) [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://jnwnlee.github.io/video-foley-demo/)

![Figure](./Figure.png)

This is a PyTorch implementation of RMS-ControlNet, a submodule of Video-Foley.

If you find this repository helpful, please consider citing it.
```bibtex
@article{lee2024video,
  title={Video-Foley: Two-Stage Video-To-Sound Generation via Temporal Event Condition For Foley Sound},
  author={Lee, Junwon and Im, Jaekwon and Kim, Dabin and Nam, Juhan},
  journal={arXiv preprint arXiv:2408.11915},
  year={2024}
}
```
## Set up
### Clone the repository.
```
git clone https://github.com/jakeoneijk/RMS_ControlNet_Inference.git
```
```
cd RMS_ControlNet_Inference
```

### Make conda env (If you don't want to use conda env, you may skip this)
```
source ./Script/0_conda_env_setup.sh
```

### Install pytorch and pytorch lightning. You should check your CUDA Version and install compatible version.
```
source ./Script/1_pytorch_install.sh
```

### Setup this repository
```
source ./Script/2_setup.sh
```

### Download pretrained weights. 

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/datasets/jakeoneijk/RMS_ControlNet_weights/tree/main)

## Use
### Please check Test.py

You can import the module from anywhere
```python
from AudioLDMControlNetInfer.AudioLDMControlNet import AudioLDMControlNet

pretrained_path:str = './ControlNetstep300000.pth' #path of weights you downloaded from Hugging Face

audio_ldm_controlnet = AudioLDMControlNet(
    control_net_pretrained_path = pretrained_path,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
```

Read the audio file for RMS reference and extract the RMS values
```python
from Util.RMS import RMS

rms_ref_audio_path:str = './Examples/rms_ref.wav'
sample_rate:int = 16000
duration_sec:float = 10.24

rms_ref_audio:torch.Tensor = load_audio(rms_ref_audio_path)
rms:torch.Tensor = torch.from_numpy(RMS.get_rms_fit_to_audio_ldm_mel(audio=rms_ref_audio))
```

You can generate audio conditioned on both RMS and text prompts
```python
generated_audio_by_text:np.ndarray = audio_ldm_controlnet.generate(
    text_prompt='dog bark loud',
    rms=rms
)
```

You can also use audio prompts to condition the timbre
```
timb_ref_audio = load_audio('./Examples/timb_ref(footstep).mp3')
generated_audio_by_waveform:np.ndarray = audio_ldm_controlnet.generate(
    waveform=timb_ref_audio,
    rms=rms
)
```

## References
- [AudioLDM](https://github.com/haoheliu/AudioLDM-training-finetuning)(provided by the authors: please follow the CC-BY-NC-SA 4.0 license)
    - AudioLDM-s: [```audioldm-s-full.ckpt```](https://zenodo.org/records/7600541/files/audioldm-s-full)
    - Others (CLAP, VAE, HifiGAN): [```checkpoints.tar```](https://drive.google.com/file/d/1T6EnuAHIc8ioeZ9kB1OZ_WGgwXAVGOZS/view?usp=drive_link)
- [ControlNet](https://github.com/lllyasviel/ControlNet)