import os
import subprocess
import time
import queue
import threading
import numpy as np
import pygame
import psutil
import glob
import gc
import csv
import random
import cv2
from scipy.io.wavfile import write
from scipy.signal import spectrogram
from datetime import datetime
import nidaqmx
from nidaqmx.constants import AcquisitionType, RegenerationMode
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg


import config  # Import all your constants


# === Utility Functions ===

def high_priority():   
    try:
        p = psutil.Process(os.getpid())
        p.nice(-20)
    except Exception:
        pass

def define_buffer(n_channels, chunk_samples):
    return np.zeros((n_channels, chunk_samples))

def is_above_threshold(data, channel_idx, thresh):
    x = data[channel_idx][::10]
    x -= np.mean(x)
    x = np.abs(x)
    value = np.quantile(x, 0.8)
    print("Trigger value obtained:", value)
    return value > thresh  # Always trigger for now

def is_within_playback_time_window(start, end):
    now = datetime.now().time()
    return True

# === Playback Thread ===

def playback_thread(trigger_queue, playback_queue, stop_event):
    print("🔊 Playback thread started.")
    pygame.mixer.init()

    wav_files = glob.glob(os.path.join(config.playback_folder, "*.wav"))

    print(f"🎶 Found {len(wav_files)} audio files: {[os.path.basename(w) for w in wav_files]}")
    audio_list = wav_files * config.m

    while not stop_event.is_set():
        try:
            trigger = trigger_queue.get() 
        except queue.Empty:
            continue

        if trigger is None:
            break

        if trigger == 1:
            print("▶️ Playback triggered.")            
            play_list = list(audio_list)
            random.shuffle(play_list)
            print(f"🔀 Shuffled order: {[os.path.basename(a) for a in play_list]}")
            playback_queue.put({"order": [os.path.basename(a) for a in play_list]})
            
            if config.delay > 0:
                time.sleep(config.delay)

            for audio in play_list:
                if stop_event.is_set():
                    break

                fname = os.path.basename(audio)
                print(f"🎵 Playing: {fname}")
                pygame.mixer.music.load(audio)
                pygame.mixer.music.play()
                playback_queue.put(fname)

                while pygame.mixer.music.get_busy() and not stop_event.is_set():
                    time.sleep(0.05)

                if not stop_event.is_set() and config.interval_s > 0:
                    time.sleep(config.interval_s)

            print("✅ Playback session completed.")

    print("🛑 Playback thread terminated.")

# === Acquisition Thread ===

def acquisition_thread(data_queue, save_queue, video_trigger_queue, trigger_queue, stop_event):
    high_priority()
    gc.collect()
    gc.disable()

    total_samples = int(config.fs * config.T_total)
    total_chunks = total_samples // config.chunk_samples
    i = 0
    buffer = define_buffer(len(config.channels), config.chunk_samples)
    start_time = time.perf_counter()

    print(f"🎙️ Starting acquisition for {config.T_total} seconds ({total_chunks} chunks)")
    print(f"💡 LED will flash at sample {config.LED_FLASH_SAMPLE} in EVERY chunk")

    led_output = np.zeros(config.chunk_samples, dtype=np.float64)
    
    dur_samples = config.led_duration_samples  # duración en muestras

    for t_s in config.flash_times_s:
        start = int(t_s * config.fs)
        end = min(start + dur_samples, config.chunk_samples)
        if start < config.chunk_samples:
            led_output[start:end] = config.led_level

    with nidaqmx.Task() as ai_task, nidaqmx.Task() as ao_task:
        for ch in config.channels:
            ai_task.ai_channels.add_ai_voltage_chan(
                f"{config.device}/{ch}", terminal_config=nidaqmx.constants.TerminalConfiguration(-1))
        ai_task.timing.cfg_samp_clk_timing(
            rate=config.fs,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=config.chunk_samples
        )

        ao_task.ao_channels.add_ao_voltage_chan(f"{config.device}/ao0")
        ao_task.timing.cfg_samp_clk_timing(
            rate=config.fs,
            source=f"/{config.device}/ai/SampleClock",
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=config.chunk_samples
        )

        reader = AnalogMultiChannelReader(ai_task.in_stream)
        writer = AnalogSingleChannelWriter(ao_task.out_stream, auto_start=False)

        ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        writer.write_many_sample(led_output)
        ao_task.start()
        ai_task.start()

        for chunk_idx in range(total_chunks):
            measure_time = time.perf_counter()
            human_time = time.strftime('%d-%m-%Y_%H-%M-%S')
            video_trigger_queue.put(measure_time)

            if (config.enable_playback and
                is_within_playback_time_window(config.time_init_playback, config.time_end_playback) and
                chunk_idx in config.playback_chunks):
                trigger_queue.put(1)
            else:
                trigger_queue.put(0)

            reader.read_many_sample(
                buffer,
                number_of_samples_per_channel=config.chunk_samples,
                timeout=config.chunk_duration * 2
            )


            data_queue.put((i, buffer.copy(), measure_time, human_time))
            save_queue.put((i, buffer.copy(), measure_time, human_time))
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Saved chunk {i}")
            i += 1


            target_time = start_time + (chunk_idx + 1) * config.chunk_duration
            while time.perf_counter() < target_time:
                time.sleep(0.001)

    stop_event.set()
    video_trigger_queue.put(None)
    trigger_queue.put(None)
    print(f"\n🟩 Acquisition finished. Chunks acquired: {i}")
    gc.enable()

# === Video Capture Thread ===

def video_capture_thread(video_queue, video_trigger_queue, stop_event):
    if not config.video_enabled:
        print("🎥 Video module disabled")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.video_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video_resolution[1])
    cap.set(cv2.CAP_PROP_FPS, config.video_fps)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print(f"🎥 Camera configured: {config.video_resolution[0]}x{config.video_resolution[1]} at {config.video_fps}fps")

    while not stop_event.is_set():
        try:
            trigger_time = video_trigger_queue.get(timeout=None)
            if trigger_time is None:
                break

            start_time = time.perf_counter()
            frame_interval = 1.0 / config.video_fps
            next_frame_time = start_time + frame_interval
            max_video_duration = config.chunk_duration - 3  # segundos

            # Nombre de archivo temporal
            human_time = time.strftime('%d-%m-%Y_%H.%M.%S')
            filename = os.path.join(config.video_output_dir, f"Video_{config.birdname}_{human_time}.mp4")

            # Escribir directamente a disco
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, config.video_fps, config.video_resolution)

            if not out.isOpened():
                print(f"❌ No se pudo crear el archivo de video: {filename}")
                continue

            frame_count = 0
            while (time.perf_counter() - start_time) < max_video_duration:
                while time.perf_counter() < next_frame_time:
                    pass
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Error leyendo frame de cámara.")
                    break
                # Asegurar tamaño correcto y escribir
                if frame.shape[:2] != config.video_resolution[::-1]:
                    frame = cv2.resize(frame, config.video_resolution)
                out.write(frame)
                frame_count += 1
                next_frame_time += frame_interval

            out.release()
            gc.collect()

            # Calcular rendimiento real
            elapsed = time.perf_counter() - start_time
            real_fps = frame_count / elapsed
            print(f"📸 Capturados {frame_count} frames en {elapsed:.2f}s → {real_fps:.2f} fps efectivos")
            
            # Encolar solo la metadata (ya no los frames)
            video_queue.put((filename, start_time, human_time, frame_count))

        except queue.Empty:
            continue

    cap.release()
    print("🎥 Video capture thread finished.")

def video_save_thread(video_queue, stop_event):
    if not config.video_enabled:
        return

    while not stop_event.is_set() or not video_queue.empty():
        try:
            # Ahora la queue contiene (filename, measure_time, human_time, frame_count)
            filename, measure_time, human_time, frame_count = video_queue.get_nowait()

            if not os.path.exists(filename):
                print(f"⚠️ Archivo no encontrado: {filename}")
                continue

            # --- Convertir a H264 silenciosamente ---
            output_filename = filename[:-4] + "_h265.mp4"
            try:
                cmd = [
                        'ffmpeg',
                        '-i', filename,
                        '-c:v', 'libx265',   # códec H.265 / HEVC
                        '-preset', 'medium', # velocidad de compresión (puede ser ultrafast, fast, slow, etc.)
                        '-crf', '28',        # calidad (más alto = menor calidad; rango típico 18–30)
                        '-y',                 # sobrescribir salida sin preguntar
                        output_filename
                     ]

                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )

                if result.returncode == 0:
                    print(f"✅ Video convertido a H265: {output_filename}")
                    try:
                        os.remove(filename)
                        print(f"🗑️  Archivo original eliminado: {filename}")
                    except Exception as delete_error:
                        print(f"⚠️  No se pudo eliminar el archivo original: {delete_error}")
                else:
                    print(f"❌ Error en la conversión H265 (código: {result.returncode})")

            except Exception as e:
                print(f"❌ Error ejecutando ffmpeg: {e}")

            # --- Guardar metadata ---
            meta_filename = os.path.join(
                config.video_output_dir,
                f"Video_{config.birdname}_{human_time}_meta.csv"
            )
            with open(meta_filename, "w", newline="") as metafile:
                writer_csv = csv.writer(metafile)
                writer_csv.writerow(["first_frame_perf_time", "frame_count", "fps"])
                writer_csv.writerow([measure_time, frame_count, config.video_fps])

            print(f"🎥 Video procesado y metadata guardada: {output_filename}")

        except queue.Empty:
            time.sleep(0.05)
            continue

    print("🎥 Video save thread finished.")

def save_thread(save_queue, stop_event):
    while not stop_event.is_set() or not save_queue.empty():
        try:
            i, data, measure_time, human_time = save_queue.get_nowait()
            data = data.T
            row_stats = []
            headers = []

            max_val = np.max(np.abs(data))
            if max_val == 0:
                scaled = np.zeros_like(data, dtype=np.int16)
                avg_all = np.zeros(data.shape[1])
                ampl_all = np.zeros(data.shape[1])
            else:
                avg_all = np.mean(data, axis=0)
                centered = data - avg_all
                ampl_all = np.max(np.abs(centered), axis=0)
                scaled = (centered / ampl_all * 32767).astype(np.int16)
                ampl_final = ampl_all / 32767

            for n in range(len(config.channels)):
                wav_filename = os.path.join(
                    config.output_dir,
                    f"{config.channel_names[n]}_{config.birdname}_{human_time}.wav"
                )
                write(wav_filename, config.fs, scaled[:, n])
                headers.extend([f"{config.channel_names[n]}_avg", f"{config.channel_names[n]}_ampl"])
                row_stats.extend([avg_all[n], ampl_final[n]])

            headers.append("measure_time")
            row_stats.append(measure_time)
            headers.append("Pass treshold or not?")
            row_stats.append(is_above_threshold(data.T, config.spectro_channel_idx, config.threshold))
            csv_filename = os.path.join(config.output_dir, f"{config.birdname}_{human_time}.csv")
            with open(csv_filename, "w", newline="") as csvfile:
                writer_csv = csv.writer(csvfile)
                writer_csv.writerow(headers)
                writer_csv.writerow(row_stats)

        except queue.Empty:
            time.sleep(0.1)

# === Plotting Thread ===

def plotting_thread(data_queue, stop_event):
    app = QApplication.instance() or QApplication([])
    win = pg.GraphicsLayoutWidget(title="DAQ Live Viewer")
    win.resize(1000, 600)

    plot_waveform = win.addPlot(title=f"{config.channel_names[config.spectro_channel_idx]} waveform")
    curve_waveform = plot_waveform.plot(pen='y')

    win.nextRow()
    plot_spectrogram = win.addPlot(title=f"{config.channel_names[config.spectro_channel_idx]} spectrogram")
    img = pg.ImageItem()
    plot_spectrogram.addItem(img)
    win.show()

    while not stop_event.is_set() or not data_queue.empty():
        try:
            i, data, measure_time, human_time = data_queue.get_nowait()
        except queue.Empty:
            QtWidgets.QApplication.processEvents()
            continue

        y = data[config.spectro_channel_idx][::10]
        x = np.linspace(0, config.chunk_duration, len(y))
        curve_waveform.setData(x, y)

        f_spec, t_spec, Sxx = spectrogram(data[config.spectro_channel_idx][::2],
                                          fs=config.fs / 2, nperseg=256, noverlap=128)
        Sxx_dB = 10 * np.log10(Sxx + 1e-12)
        img.setImage(Sxx_dB.T, levels=(Sxx_dB.max(), Sxx_dB.min()))
        img.resetTransform()
        dx = t_spec[1] - t_spec[0]
        dy = f_spec[1] - f_spec[0]
        img.setTransform(QTransform().scale(dx, dy))
        img.setPos(0, 0)

        QtWidgets.QApplication.processEvents()

    print("🛑 Plotting thread finished.")
    time.sleep(config.chunk_duration)
    win.close()
    QApplication.quit()

