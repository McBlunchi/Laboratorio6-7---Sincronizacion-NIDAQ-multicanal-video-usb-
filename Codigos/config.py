import os
import time
from datetime import time as dtime

# --- DAQ Config ---
fs = 44150
chunk_duration = 180
#chunk_duration = 10
chunk_samples = int(chunk_duration * fs)
T_total = chunk_duration * 20 * 15   + 1
#T_total = chunk_duration * 2   + 1
#T_total = chunk_duration * 2 + 1
threshold = 0.01100
channels = ["ai0", "ai1"]
channel_names = ["Electrodos", "Sonido"]
device = "Dev2"
spectro_channel_idx = 1
birdname = 'Test_29_10_dia_6'

LED_FLASH_SAMPLE = fs * 5
led_duration_samples = int(1 * fs)
flash_times_s = [4,10,20,30,60,70,80,90,100,110,120,130,140,150,160,170] # Tipo campanadas, el 1ª 1, el 2ª 2, el 3ª 3
led_level = 5.0  # voltaje del flash

# --- Video Config ---
video_enabled = True
video_resolution = (1280, 720)
#video_resolution = (854, 480)
video_fps = 30
video_chunk_duration = chunk_duration - 1

# --- Playback Config ---
enable_playback = True
playback_folder = r"C:\Users\tesistas\Desktop\Rolla_Urriste_Labo_6\canto_fran"
time_init_playback = dtime(19, 00)
time_end_playback = dtime(18, 00)
playback_repeats = 1
playback_chunks = [0,100,101,102,103, 140,141,142,143, 180,181,182,183] 
delay = 3
m = 6
interval_s = 10.0

# --- Paths ---"
Route = r'C:\Users\tesistas\Desktop\Rolla_Urriste_Labo_6\29-10'
base_dir = birdname
today_str = time.strftime('%d-%m-%Y')
output_dir = os.path.join(Route, base_dir, today_str)
video_output_dir = os.path.join(output_dir, "video")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(video_output_dir, exist_ok=True)
