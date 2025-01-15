# ASML ASR Demo
## Installation
```bash
cd /path/to/gradio_asr/tools
bash install_virtualenv.sh
bash install_whisper_modules.sh
bash install_modules.sh
```

## Preparation
### Prepare faster-whisper models
In convert_whisper_to_faster.sh \
Set --model to whisper model saved directory \
Set --output_dir to the desired output directory
```bash
cd /path/to/gradio_asr
bash convert_whisper_to_faster.sh
mkdir -p models
cp -r /path/to/output_dir models/
```

## Run
### Run asr server
In the remote server to run ASR transcription,
```bash
cd /path/to/gradio_asr
. tools/activate_python.sh
python asr_server.py
```

### Run gradio server
In the remote server to run Gradio demo,
```bash
cd /path/to/gradio_asr
. tools/activate_python.sh
python gradio_server.py
```

# ASR DEMO video
## Speaker video & demo video
<p align="center">
<img src="https://github.com/user-attachments/assets/e7e61370-efc0-4ed1-b168-052abb33a612" alt="Demo Video Thumbnail">
</p>
<p align="center">
<img src="https://github.com/user-attachments/assets/5121176f-3cca-41cb-86f0-57dc5b1111b6" alt="Demo Video Thumbnail">
</p>
