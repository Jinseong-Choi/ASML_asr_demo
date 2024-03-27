# Gradio demo for Whisper ASR
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