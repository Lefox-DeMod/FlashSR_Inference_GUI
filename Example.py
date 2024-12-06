from FlashSR.FlashSR import FlashSR

student_ldm_ckpt_path:str = './ModelWeights/student_ldm.pth'
sr_vocoder_ckpt_path:str = './ModelWeights/sr_vocoder.pth'
vae_ckpt_path:str = './ModelWeights/vae.pth'
flashsr = FlashSR( student_ldm_ckpt_path, sr_vocoder_ckpt_path, vae_ckpt_path)