import os
import threading
import numpy as np
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from FlashSR.FlashSR import FlashSR


# --- Tooltip helper ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        try:
            x, y, cx, cy = self.widget.bbox("insert")
        except Exception:
            x, y, cx, cy = (0, 0, 0, 0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("tahoma", 8, "normal")
        )
        label.pack(ipadx=4, ipady=2)

    def hide_tip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class FlashSRGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FlashSR GUI")
        self.geometry("840x640")

        # Default Model paths
        self.student_ckpt = tk.StringVar(value="./ModelWeights/student_ldm.pth")
        self.vocoder_ckpt = tk.StringVar(value="./ModelWeights/sr_vocoder.pth")
        self.vae_ckpt = tk.StringVar(value="./ModelWeights/vae.pth")

        # Device/model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flashsr = None

        # GUI variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.out_sr_choice = tk.StringVar(value="48000")
        self.overlap_sec = tk.StringVar(value="0.5")
        self.lowpass_input = tk.BooleanVar(value=False)
        self.force_lr_sr = tk.StringVar(value="")  # optional forced SR (e.g., 22050)

        # Processing constants
        self.process_sr = 48000
        self.chunk_size = 245_760  # 5.12 s at 48 kHz

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # Paths
        frame_paths = ttk.LabelFrame(self, text="Paths")
        frame_paths.pack(fill="x", **pad)

        in_lbl = ttk.Label(frame_paths, text="Input audio:")
        in_lbl.grid(row=0, column=0, sticky="w")
        in_entry = ttk.Entry(frame_paths, textvariable=self.input_path, width=64)
        in_entry.grid(row=0, column=1, sticky="we")
        ttk.Button(frame_paths, text="Browse", command=self._browse_input).grid(row=0, column=2)
        ToolTip(in_lbl, "Select the low-resolution audio file to process.")

        out_lbl = ttk.Label(frame_paths, text="Output file:")
        out_lbl.grid(row=1, column=0, sticky="w")
        out_entry = ttk.Entry(frame_paths, textvariable=self.output_path, width=64)
        out_entry.grid(row=1, column=1, sticky="we")
        ttk.Button(frame_paths, text="Save as", command=self._save_output).grid(row=1, column=2)
        ToolTip(out_lbl, "Choose where to save the super-resolved output (WAV).")

        # Model weights
        frame_weights = ttk.LabelFrame(self, text="Model Weights")
        frame_weights.pack(fill="x", **pad)

        lbl1 = ttk.Label(frame_weights, text="Student LDM ckpt:")
        lbl1.grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_weights, textvariable=self.student_ckpt, width=64).grid(row=0, column=1, sticky="we")
        ttk.Button(frame_weights, text="Browse", command=lambda: self._browse_ckpt(self.student_ckpt)).grid(row=0, column=2)
        ToolTip(lbl1, "Main generative block that reconstructs high-frequency details.")

        lbl2 = ttk.Label(frame_weights, text="SR Vocoder ckpt:")
        lbl2.grid(row=1, column=0, sticky="w")
        ttk.Entry(frame_weights, textvariable=self.vocoder_ckpt, width=64).grid(row=1, column=1, sticky="we")
        ttk.Button(frame_weights, text="Browse", command=lambda: self._browse_ckpt(self.vocoder_ckpt)).grid(row=1, column=2)
        ToolTip(lbl2, "Vocoder that converts latent features back into waveform audio.")

        lbl3 = ttk.Label(frame_weights, text="VAE ckpt:")
        lbl3.grid(row=2, column=0, sticky="w")
        ttk.Entry(frame_weights, textvariable=self.vae_ckpt, width=64).grid(row=2, column=1, sticky="we")
        ttk.Button(frame_weights, text="Browse", command=lambda: self._browse_ckpt(self.vae_ckpt)).grid(row=2, column=2)
        ToolTip(lbl3, "Variational Autoencoder used for encoding/decoding spectral features.")

        # Processing
        frame_proc = ttk.LabelFrame(self, text="Processing")
        frame_proc.pack(fill="x", **pad)

        lbl_sr = ttk.Label(frame_proc, text="Output SR:")
        lbl_sr.grid(row=0, column=0, sticky="w")
        sr_combo = ttk.Combobox(frame_proc, textvariable=self.out_sr_choice, values=["44100", "48000"], state="readonly", width=10)
        sr_combo.grid(row=0, column=1, sticky="w")
        ToolTip(lbl_sr, "Select the output sampling rate.\n48 kHz is native; 44.1 kHz for music workflows.")

        lbl_ov = ttk.Label(frame_proc, text="Overlap (seconds):")
        lbl_ov.grid(row=0, column=2, sticky="w")
        ov_entry = ttk.Entry(frame_proc, textvariable=self.overlap_sec, width=10)
        ov_entry.grid(row=0, column=3, sticky="w")
        ToolTip(lbl_ov, "Amount of overlap between chunks in seconds.\nRecommended: 0.25–0.5 s.")

        chk_lp = ttk.Checkbutton(frame_proc, text="Lowpass input", variable=self.lowpass_input)
        chk_lp.grid(row=1, column=0, sticky="w")
        ToolTip(chk_lp, "If checked: assume input is artificially low-pass filtered.\nIf unchecked: use real low-resolution audio.")

        lbl_force = ttk.Label(frame_proc, text="Force LR SR (optional):")
        lbl_force.grid(row=1, column=2, sticky="w")
        force_entry = ttk.Entry(frame_proc, textvariable=self.force_lr_sr, width=10)
        force_entry.grid(row=1, column=3, sticky="w")
        ToolTip(lbl_force, "Manually set input sample rate (e.g., 22050).\nLeave empty to auto-detect from file header.")

        frame_run = ttk.Frame(self)
        frame_run.pack(fill="x", **pad)
        self.run_btn = ttk.Button(frame_run, text="Run FlashSR", command=self._run_async)
        self.run_btn.pack(side="left")
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(frame_run, textvariable=self.status).pack(side="left", padx=12)

    def _browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac"), ("All files", "*.*")])
        if filename:
            self.input_path.set(filename)

    def _save_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if filename:
            self.output_path.set(filename)

    def _browse_ckpt(self, var):
        filename = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")])
        if filename:
            var.set(filename)

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
                messagebox.showinfo("Success", "Model loaded successfully!")
            else:
                messagebox.showerror("Error", "Model weights not found! Check paths in Model Weights section.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def _overlap_add(self, segments, overlap_samples, total_length):
        """
        overlap-add without crossfade.
        Assumes each segment has identical length == chunk_size.
        """
        output = np.zeros(total_length, dtype=np.float32)
        hop_length = self.chunk_size - overlap_samples
        for i, seg in enumerate(segments):
            start = i * hop_length
            end = min(start + len(seg), total_length)
            if end <= start:
                break
            # trim segment if it would exceed total_length
            seg_use = seg[:end - start]
            output[start:end] += seg_use
        return output

    def _infer_chunk(self, chunk_np, lowpass):
        """
        Inference with strict length enforcement to chunk_size.
        """
        chunk_tensor = torch.from_numpy(chunk_np).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            pred_chunk = self.flashsr(chunk_tensor, lowpass_input=lowpass)
        pred = pred_chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Enforce exact chunk_size length to avoid size mismatch
        if len(pred) > self.chunk_size:
            pred = pred[:self.chunk_size]
        elif len(pred) < self.chunk_size:
            pred = np.pad(pred, (0, self.chunk_size - len(pred)), mode="constant")
        return pred

    def _run_async(self):
        t = threading.Thread(target=self._run)
        t.daemon = True
        t.start()

    def _run(self):
        try:
            self.run_btn.config(state="disabled")
            self.status.set("Running...")

            inp = self.input_path.get().strip()
            outp = self.output_path.get().strip()
            if not inp or not os.path.isfile(inp):
                raise ValueError("Please select a valid input audio file.")
            if not outp:
                raise ValueError("Please choose an output path.")
            if self.flashsr is None:
                raise RuntimeError("Model is not loaded.")

            # Options
            target_sr = int(self.out_sr_choice.get())
            try:
                overlap_sec = float(self.overlap_sec.get())
            except ValueError:
                overlap_sec = 0.5
            overlap_samples = int(max(0.0, overlap_sec) * self.process_sr)
            overlap_samples = min(overlap_samples, self.chunk_size - 1)  # ensure hop > 0

            # Load audio (optionally forcing SR)
            force_sr_str = self.force_lr_sr.get().strip()
            force_sr = int(force_sr_str) if force_sr_str else None
            audio_lr, orig_sr = librosa.load(inp, sr=force_sr, mono=True)

            # Resample to 48kHz
            audio_48 = librosa.resample(audio_lr, orig_sr=orig_sr, target_sr=self.process_sr)
            total_length = len(audio_48)

            # Chunking
            hop_size = self.chunk_size - overlap_samples
            segments = []
            for start in range(0, total_length, hop_size):
                end = start + self.chunk_size
                chunk = audio_48[start:end]
                if len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode='constant')

                pred_chunk = self._infer_chunk(chunk, lowpass=self.lowpass_input.get())
                segments.append(pred_chunk)

            # Overlap-add
            sr_audio = self._overlap_add(segments, overlap_samples, total_length)

            # Normalize to prevent clipping
            peak = np.max(np.abs(sr_audio)) if len(sr_audio) else 1.0
            if peak > 0:
                sr_audio = sr_audio / peak * 0.95

            # Resample to target if needed
            if target_sr != self.process_sr:
                sr_audio = librosa.resample(sr_audio, orig_sr=self.process_sr, target_sr=target_sr)

            # Save
            os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
            sf.write(outp, sr_audio, target_sr)

            self.status.set(f"Done. Saved: {outp}")
            messagebox.showinfo("Success", f"Super-resolution completed! Saved to {outp}")

        except Exception as e:
            self.status.set("Error.")
            messagebox.showerror("Error", f"Inference failed: {str(e)}")
        finally:
            self.run_btn.config(state="normal")


if __name__ == "__main__":
    app = FlashSRGUI()
    app.mainloop()
