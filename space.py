import sys
import os

sys.path.append(os.path.dirname(__file__))

from gradio_app import create_gradio_interface

demo = create_gradio_interface()
demo.launch(server_name="0.0.0.0", server_port=7860)