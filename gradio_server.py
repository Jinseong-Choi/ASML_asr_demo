import gradio as gr
from PIL import Image
import grpc

import asr_pb2
import asr_pb2_grpc

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

languages = list(LANGUAGES.values())

def clear_interface():
    return None, ""

def request(filepath, language):
    channel = grpc.insecure_channel('localhost:50051')
    stub = asr_pb2_grpc.ASRServiceStub(channel)
    request = asr_pb2.ASRRequest()
    request.filepath = filepath
    request.language = language

    # TODO: Asynchronous call
    response = stub.Transcribe(request)

    return response.transcription

def create_interface_for_microphone():
    with gr.Column():
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources="microphone", label='Input Speech', type="filepath")
                language = gr.Dropdown(label='Language', choices=languages, value="korean")
            transcribed_text = gr.Textbox(label='Transcribed Text')
        with gr.Row():
            transcribe_button = gr.Button('Transcribe')
            transcribe_button.click(fn=request, inputs=[audio_input, language], outputs=transcribed_text)
            clear_button = gr.Button('Clear')
            clear_button.click(fn=clear_interface, inputs=[], outputs=[audio_input, transcribed_text])

def create_interface_for_file_upload():
    with gr.Column():
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources="upload", label='Input Speech', type="filepath")
                language = gr.Dropdown(label='Language', choices=languages, value="korean")
            transcribed_text = gr.Textbox(label='Transcribed Text')
        with gr.Row():
            transcribe_button = gr.Button('Transcribe')
            transcribe_button.click(fn=request, inputs=[audio_input, language], outputs=transcribed_text)
            clear_button = gr.Button('Clear')
            clear_button.click(fn=clear_interface, inputs=[], outputs=[audio_input, transcribed_text])

with gr.Blocks() as demo:
    gr.Markdown("# KEP ASR Demo")
    with gr.Tab("Transcribe from audio file"):
        create_interface_for_file_upload()
    with gr.Tab("Transcribe from microphone"):
        create_interface_for_microphone()

demo.launch(server_name='0.0.0.0', share=True)