import os
import threading
import math
import numpy as np
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from CTkToolTip import CTkToolTip
import torch
from FlashSR.FlashSR import FlashSR
from PIL import Image, ImageTk
import warnings

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
            "or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)."
)

ctk.set_default_color_theme("dark-blue")


class FlashSRGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FlashSR GUI")

        try:
            icon_img = Image.open(os.path.join("Assets/logo.png"))
            self.icon_ref = ImageTk.PhotoImage(icon_img)
            self.iconphoto(False, self.icon_ref)
        except Exception as e:
            print("Icon load error:", e)

        # allow size dynamics and auto-adjustment by content
        self.resizable(True, True)

        # Unified Padding
        self.spacing = {"padx": 5, "pady": 5}

        # Default path to models
        self.student_ckpt = tk.StringVar(value="./ModelWeights/student_ldm.pth")
        self.vocoder_ckpt = tk.StringVar(value="./ModelWeights/sr_vocoder.pth")
        self.vae_ckpt = tk.StringVar(value="./ModelWeights/vae.pth")

        # Device Selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flashsr = None

        # Variables GUI
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.out_sr_choice = tk.StringVar(value="48000")
        self.overlap_sec = tk.StringVar(value="0.5")
        self.lowpass_input = tk.BooleanVar(value=False)
        self.force_lr_sr = tk.StringVar(value="")
        self.batch_mode = tk.BooleanVar(value=False)

        # Default Sampling Rate
        self.process_sr = 48000
        self.chunk_size = 245_760  # 5.12 s @ 48 kHz

        # Font
        self.font = ctk.CTkFont(family="Red Hat Display", size=14)
        self.font_bold = ctk.CTkFont(family="Red Hat Display", size=13, weight="bold")

        # interrupt flag
        self.interrupt_flag = tk.BooleanVar(value=False)

        # load assets
        try:
            self.folder_icon = ctk.CTkImage(light_image=Image.open(os.path.join("Assets/file.png")), size=(24, 24))
        except Exception:
            self.folder_icon = None
        try:
            self.stop_icon = ctk.CTkImage(light_image=Image.open(os.path.join("Assets/stop.png")), size=(24, 24))
        except Exception:
            self.stop_icon = None
        try:
            self.logo = ctk.CTkImage(light_image=Image.open(os.path.join("Assets/logo_text.png")), size=(607, 243))
        except Exception:
            self.logo = None

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        # Tabview without fixed sizes
        self.tab_view = ctk.CTkTabview(self, corner_radius=10)
        self.tab_view.pack(fill="both", expand=True, **self.spacing)
        try:
            self.tab_view._segmented_button.configure(font=self.font_bold)
        except Exception:
            pass

        # Tables
        self.tab_view.add("About")
        self.tab_view.add("Path")
        self.tab_view.add("Model Path")
        self.tab_view.add("Processing")
        self.tab_view.add("Log")

        # === About ===
        about_frame = ctk.CTkFrame(self.tab_view.tab("About"))
        about_frame.pack(fill="both", expand=True, **self.spacing)

        logo_label = ctk.CTkLabel(about_frame, image=self.logo, text="")
        #  avoid the conflict of the double pady, we set only padx from spacing
        logo_label.pack(pady=10, padx=self.spacing["padx"])

        version_label = ctk.CTkLabel(about_frame, text="Version: 0.3.1 (29/09/25)", font=self.font_bold)
        version_label.pack(**self.spacing)

        # === Path ===
        frame_paths = ctk.CTkFrame(self.tab_view.tab("Path"))
        frame_paths.pack(fill="x", **self.spacing)
        # Stretching the middle column (Entry)
        frame_paths.grid_columnconfigure(1, weight=1)

        self.in_lbl = ctk.CTkLabel(frame_paths, text="Input audio:", font=self.font)
        self.in_lbl.grid(row=1, column=0, sticky="w", **self.spacing)

        in_entry = ctk.CTkEntry(frame_paths, textvariable=self.input_path, font=self.font)
        in_entry.grid(row=1, column=1, sticky="nsew", **self.spacing)

        browse_in_btn = ctk.CTkButton(frame_paths, image=self.folder_icon, text="", command=self._browse_input, width=34, font=self.font)
        browse_in_btn.grid(row=1, column=2, **self.spacing)

        self.out_lbl = ctk.CTkLabel(frame_paths, text="Output file:", font=self.font)
        self.out_lbl.grid(row=2, column=0, sticky="w", **self.spacing)

        out_entry = ctk.CTkEntry(frame_paths, textvariable=self.output_path, font=self.font)
        out_entry.grid(row=2, column=1, sticky="nsew", **self.spacing)

        save_out_btn = ctk.CTkButton(frame_paths, image=self.folder_icon, text="", command=self._save_output, width=34, font=self.font)
        save_out_btn.grid(row=2, column=2, **self.spacing)

        chk_batch = ctk.CTkCheckBox(frame_paths, text="Batch mode", variable=self.batch_mode, command=self._toggle_batch_mode, font=self.font)
        chk_batch.grid(row=1, column=3, **self.spacing)

        CTkToolTip(self.in_lbl, message="Select the low-resolution audio file to process.", font=self.font)
        CTkToolTip(self.out_lbl, message="Choose where to save the super-resolved output (WAV).", font=self.font)
        CTkToolTip(chk_batch, message="Enable to process all audio files in a folder.", font=self.font)

        # === Model Path ===
        frame_weights = ctk.CTkFrame(self.tab_view.tab("Model Path"))
        frame_weights.pack(fill="x", **self.spacing)
        frame_weights.grid_columnconfigure(1, weight=1)

        lbl1 = ctk.CTkLabel(frame_weights, text="Student LDM ckpt:", font=self.font)
        lbl1.grid(row=1, column=0, sticky="w", **self.spacing)
        entry1 = ctk.CTkEntry(frame_weights, textvariable=self.student_ckpt, font=self.font)
        entry1.grid(row=1, column=1, sticky="nsew", **self.spacing)
        browse1 = ctk.CTkButton(frame_weights, image=self.folder_icon, text="", command=lambda: self._browse_ckpt(self.student_ckpt), width=34, font=self.font)
        browse1.grid(row=1, column=2, **self.spacing)
        CTkToolTip(lbl1, message="Main generative block that reconstructs high-frequency details.", font=self.font)

        lbl2 = ctk.CTkLabel(frame_weights, text="SR Vocoder ckpt:", font=self.font)
        lbl2.grid(row=2, column=0, sticky="w", **self.spacing)
        entry2 = ctk.CTkEntry(frame_weights, textvariable=self.vocoder_ckpt, font=self.font)
        entry2.grid(row=2, column=1, sticky="nsew", **self.spacing)
        browse2 = ctk.CTkButton(frame_weights, image=self.folder_icon, text="", command=lambda: self._browse_ckpt(self.vocoder_ckpt), width=34, font=self.font)
        browse2.grid(row=2, column=2, **self.spacing)
        CTkToolTip(lbl2, message="Vocoder that converts latent features back into waveform audio.", font=self.font)

        lbl3 = ctk.CTkLabel(frame_weights, text="VAE ckpt:", font=self.font)
        lbl3.grid(row=3, column=0, sticky="w", **self.spacing)
        entry3 = ctk.CTkEntry(frame_weights, textvariable=self.vae_ckpt, font=self.font)
        entry3.grid(row=3, column=1, sticky="nsew", **self.spacing)
        browse3 = ctk.CTkButton(frame_weights, image=self.folder_icon, text="", command=lambda: self._browse_ckpt(self.vae_ckpt), width=34, font=self.font)
        browse3.grid(row=3, column=2, **self.spacing)
        CTkToolTip(lbl3, message="Variational Autoencoder used for encoding/decoding spectral features.", font=self.font)

        # === Processing ===
        frame_proc = ctk.CTkFrame(self.tab_view.tab("Processing"))
        frame_proc.pack(fill="x", **self.spacing)
        frame_proc.grid_columnconfigure(1, weight=1)
        frame_proc.grid_columnconfigure(3, weight=0)

        lbl_sr = ctk.CTkLabel(frame_proc, text="Output SR:", font=self.font)
        lbl_sr.grid(row=1, column=0, sticky="w", **self.spacing)

        sr_combo = ctk.CTkComboBox(frame_proc, variable=self.out_sr_choice, values=["44100", "48000"], state="readonly", font=self.font)
        sr_combo.grid(row=1, column=1, sticky="w", **self.spacing)
        CTkToolTip(lbl_sr, message="Select the output sampling rate.\n48 kHz is native; 44.1 kHz for music workflows.", font=self.font)

        lbl_ov = ctk.CTkLabel(frame_proc, text="Overlap (seconds):", font=self.font)
        lbl_ov.grid(row=1, column=2, sticky="w", **self.spacing)

        ov_entry = ctk.CTkEntry(frame_proc, textvariable=self.overlap_sec, width=120, font=self.font)
        ov_entry.grid(row=1, column=3, sticky="w", **self.spacing)
        CTkToolTip(lbl_ov, message="Amount of overlap between chunks in seconds.\nRecommended: 0.25–0.5 s.", font=self.font)

        chk_lp = ctk.CTkCheckBox(frame_proc, text="Lowpass input", variable=self.lowpass_input, font=self.font)
        chk_lp.grid(row=2, column=0, sticky="w", **self.spacing)
        CTkToolTip(chk_lp, message="If checked: assume input is artificially low-pass filtered.\nIf unchecked: use real low-resolution audio.", font=self.font)

        lbl_force = ctk.CTkLabel(frame_proc, text="Force LR SR (optional):", font=self.font)
        lbl_force.grid(row=2, column=2, sticky="w", **self.spacing)

        force_entry = ctk.CTkEntry(frame_proc, textvariable=self.force_lr_sr, width=120, font=self.font)
        force_entry.grid(row=2, column=3, sticky="w", **self.spacing)
        CTkToolTip(lbl_force, message="Manually set input sample rate (e.g., 22050).\nLeave empty to auto-detect from file header.", font=self.font)

        frame_run = ctk.CTkFrame(self.tab_view.tab("Processing"))
        frame_run.pack(fill="x", **self.spacing)

        self.run_btn = ctk.CTkButton(frame_run, text="Run FlashSR", command=self._run_async, font=self.font_bold)
        self.run_btn.pack(side="left", **self.spacing)

        self.status = tk.StringVar(value="Ready.")
        status_lbl = ctk.CTkLabel(frame_run, textvariable=self.status, font=self.font)
        status_lbl.pack(side="left", **self.spacing)

        self.progress = ctk.CTkProgressBar(frame_run, mode="determinate")
        self.interrupt_btn = ctk.CTkButton(frame_run, image=self.stop_icon, text="", command=self._interrupt, width=34, font=self.font)

        # === Log ===
        log_frame = ctk.CTkFrame(self.tab_view.tab("Log"))
        log_frame.pack(fill="both", expand=True, **self.spacing)

        self.log_text = ctk.CTkTextbox(log_frame, height=320, width=600, state="disabled", font=self.font)
        self.log_text.pack(fill="both", expand=True, **self.spacing)

    def _toggle_batch_mode(self):
        if self.batch_mode.get():
            self.in_lbl.configure(text="Input folder:")
            self.out_lbl.configure(text="Output folder:")
            CTkToolTip(self.in_lbl, message="Select the folder containing low-resolution audio files to process.", font=self.font)
            CTkToolTip(self.out_lbl, message="Choose the folder to save the super-resolved outputs (WAV).", font=self.font)
        else:
            self.in_lbl.configure(text="Input audio:")
            self.out_lbl.configure(text="Output file:")
            CTkToolTip(self.in_lbl, message="Select the low-resolution audio file to process.", font=self.font)
            CTkToolTip(self.out_lbl, message="Choose where to save the super-resolved output (WAV).", font=self.font)

    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        self.update_idletasks()

    def _interrupt(self):
        self.interrupt_flag.set(True)

    def _browse_input(self):
        if self.batch_mode.get():
            folder = filedialog.askdirectory()
            if folder:
                self.input_path.set(folder)
        else:
            filename = filedialog.askopenfilename(
                filetypes=[
                    ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac"),
                    ("All files", "*.*")
                ]
            )
            if filename:
                self.input_path.set(filename)

    def _save_output(self):
        if self.batch_mode.get():
            folder = filedialog.askdirectory()
            if folder:
                self.output_path.set(folder)
        else:
            filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[
                    ("WAV files", "*.wav"),
                    ("All files", "*.*")
                ]
            )
            if filename:
                self.output_path.set(filename)

    def _browse_ckpt(self, var):
        filename = filedialog.askopenfilename(
            filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)

    def show_custom_message(self, title, message, error=False):
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.grab_set()  # modal

        # Centering relative to the main window
        self.update_idletasks()
        w, h = 360, 160
        x = self.winfo_rootx() + (self.winfo_width() // 2) - (w // 2)
        y = self.winfo_rooty() + (self.winfo_height() // 2) - (h // 2)
        win.geometry(f"{w}x{h}+{x}+{y}")

        lbl = ctk.CTkLabel(win, text=message, font=self.font)
        lbl.pack(padx=20, pady=20)

        btn = ctk.CTkButton(win, text="OK", command=win.destroy,
                            fg_color=("red" if error else "green"),
                            font=self.font_bold)
        btn.pack(pady=10)

    def _load_model(self):
        try:
            s = self.student_ckpt.get().strip()
            v = self.vocoder_ckpt.get().strip()
            a = self.vae_ckpt.get().strip()
            if all(os.path.exists(p) for p in [s, v, a]):
                print(f"Using device: {self.device}")
                self.flashsr = FlashSR(s, v, a)
                self.flashsr.to(self.device)
                self.flashsr.eval()
                self.show_custom_message("Success", "Model loaded successfully!")
            else:
                self.show_custom_message("Error", "Model weights not found! Check paths in Model Weights section.", error=True)
        except Exception as e:
            self.show_custom_message("Error", f"Failed to load model: {str(e)}", error=True)

    def _overlap_add(self, segments, overlap_samples, total_length):
        """
        Overlap-add without crossfade. Assumes equal length of segments == chunk_size.
        """
        output = np.zeros(total_length, dtype=np.float32)
        hop_length = self.chunk_size - overlap_samples
        for i, seg in enumerate(segments):
            start = i * hop_length
            end = min(start + len(seg), total_length)
            if end <= start:
                break
            seg_use = seg[:end - start]
            output[start:end] += seg_use
        return output

    def _infer_chunk(self, chunk_np, lowpass):
        """
        Inference with enforced chunk size length.
        """
        chunk_tensor = torch.from_numpy(chunk_np).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            pred_chunk = self.flashsr(chunk_tensor, lowpass_input=lowpass)
        pred = pred_chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)

        if len(pred) > self.chunk_size:
            pred = pred[:self.chunk_size]
        elif len(pred) < self.chunk_size:
            pred = np.pad(pred, (0, self.chunk_size - len(pred)), mode="constant")
        return pred

    def _run_async(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        try:
            self.run_btn.configure(state="disabled")
            self.status.set("Running...")
            # Show progress bar and inerrupt button
            self.progress.pack(side="left", **self.spacing)
            self.interrupt_btn.pack(side="left", **self.spacing)
            self.interrupt_flag.set(False)
            self.progress.set(0)
            self.log("Starting processing...")

            # Options
            target_sr = int(self.out_sr_choice.get())
            try:
                overlap_sec = float(self.overlap_sec.get())
            except ValueError:
                overlap_sec = 0.5
            overlap_samples = int(max(0.0, overlap_sec) * self.process_sr)
            overlap_samples = min(overlap_samples, self.chunk_size - 1)

            force_sr_str = self.force_lr_sr.get().strip()
            force_sr = int(force_sr_str) if force_sr_str else None
            lowpass = self.lowpass_input.get()

            if self.batch_mode.get():
                input_dir = self.input_path.get().strip()
                output_dir = self.output_path.get().strip()
                if not input_dir or not os.path.isdir(input_dir):
                    raise ValueError("Please select a valid input folder.")
                if not output_dir:
                    raise ValueError("Please choose an output folder.")
                if self.flashsr is None:
                    raise RuntimeError("Model is not loaded.")

                os.makedirs(output_dir, exist_ok=True)

                extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac')
                files = [f for f in os.listdir(input_dir) if f.lower().endswith(extensions)]
                num_files = len(files)
                self.log(f"Found {num_files} audio files to process.")

                for file_idx, file in enumerate(files):
                    if self.interrupt_flag.get():
                        self.log("Batch processing interrupted by user.")
                        break

                    inp = os.path.join(input_dir, file)
                    outp = os.path.join(output_dir, os.path.splitext(file)[0] + '.wav')

                    self.log(f"Processing file {file_idx+1}/{num_files}: {file}")

                    audio_lr, orig_sr = librosa.load(inp, sr=force_sr, mono=True)

                    self.log("Resampling to 48kHz...")
                    audio_48 = librosa.resample(audio_lr, orig_sr=orig_sr, target_sr=self.process_sr)
                    total_length = len(audio_48)

                    hop_size = self.chunk_size - overlap_samples
                    num_chunks = math.ceil(total_length / hop_size)
                    self.log(f"Processing {num_chunks} chunks...")

                    segments = []
                    for i in range(num_chunks):
                        if self.interrupt_flag.get():
                            self.log(f"Interrupted during chunk {i+1} of file {file}.")
                            break
                        start = i * hop_size
                        end = start + self.chunk_size
                        chunk = audio_48[start:end]
                        if len(chunk) < self.chunk_size:
                            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode='constant')

                        self.log(f"Inferring chunk {i+1}/{num_chunks}...")
                        pred_chunk = self._infer_chunk(chunk, lowpass)
                        segments.append(pred_chunk)

                        progress_value = (i + 1) / num_chunks
                        self.progress.set(progress_value)
                        self.update_idletasks()

                    if self.interrupt_flag.get():
                        continue

                    self.log("Performing overlap-add...")
                    sr_audio = self._overlap_add(segments, overlap_samples, total_length)

                    self.log("Normalizing audio...")
                    peak = np.max(np.abs(sr_audio)) if len(sr_audio) else 1.0
                    if peak > 0:
                        sr_audio = sr_audio / peak * 0.95

                    if target_sr != self.process_sr:
                        self.log(f"Resampling to {target_sr} Hz...")
                        sr_audio = librosa.resample(sr_audio, orig_sr=self.process_sr, target_sr=target_sr)

                    self.log(f"Saving to {outp}...")
                    sf.write(outp, sr_audio, target_sr)

                    self.log(f"File {file} processed successfully.")

                if self.interrupt_flag.get():
                    self.status.set("Interrupted.")
                    self.show_custom_message("Interrupted", "Batch processing was interrupted.", error=True)
                else:
                    self.status.set("Done.")
                    self.show_custom_message("Success", "Batch super-resolution completed!")
                    self.log("Batch processing completed.")

            else:
                inp = self.input_path.get().strip()
                outp = self.output_path.get().strip()
                if not inp or not os.path.isfile(inp):
                    raise ValueError("Please select a valid input audio file.")
                if not outp:
                    raise ValueError("Please choose an output path.")
                if self.flashsr is None:
                    raise RuntimeError("Model is not loaded.")

                self.log("Loading audio...")
                audio_lr, orig_sr = librosa.load(inp, sr=force_sr, mono=True)

                self.log("Resampling to 48kHz...")
                audio_48 = librosa.resample(audio_lr, orig_sr=orig_sr, target_sr=self.process_sr)
                total_length = len(audio_48)

                hop_size = self.chunk_size - overlap_samples
                num_chunks = math.ceil(total_length / hop_size)
                self.log(f"Processing {num_chunks} chunks...")
                segments = []
                for i in range(num_chunks):
                    if self.interrupt_flag.get():
                        self.log("Processing interrupted by user.")
                        break
                    start = i * hop_size
                    end = start + self.chunk_size
                    chunk = audio_48[start:end]
                    if len(chunk) < self.chunk_size:
                        chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode='constant')

                    self.log(f"Processing chunk {i+1}/{num_chunks}...")
                    pred_chunk = self._infer_chunk(chunk, lowpass)
                    segments.append(pred_chunk)

                    progress_value = (i + 1) / num_chunks
                    self.progress.set(progress_value)
                    self.update_idletasks()

                if self.interrupt_flag.get():
                    self.status.set("Interrupted.")
                    self.show_custom_message("Interrupted", "Processing was interrupted.", error=True)
                else:
                    self.log("Performing overlap-add...")
                    sr_audio = self._overlap_add(segments, overlap_samples, total_length)

                    self.log("Normalizing audio...")
                    peak = np.max(np.abs(sr_audio)) if len(sr_audio) else 1.0
                    if peak > 0:
                        sr_audio = sr_audio / peak * 0.95

                    if target_sr != self.process_sr:
                        self.log(f"Resampling to {target_sr} Hz...")
                        sr_audio = librosa.resample(sr_audio, orig_sr=self.process_sr, target_sr=target_sr)

                    self.log("Saving output...")
                    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
                    sf.write(outp, sr_audio, target_sr)

                    self.status.set(f"Done. Saved: {outp}")
                    self.show_custom_message("Success", f"Super-resolution completed!\nSaved to:\n{outp}")
                    self.log("Processing completed.")

        except Exception as e:
            self.status.set("Error.")
            self.log(f"Error: {str(e)}")
            self.show_custom_message("Error", f"Inference failed: {str(e)}", error=True)
        finally:
            self.run_btn.configure(state="normal")
            self.progress.pack_forget()
            self.interrupt_btn.pack_forget()
            self.interrupt_flag.set(False)


if __name__ == "__main__":
    app = FlashSRGUI()
    app.mainloop()
