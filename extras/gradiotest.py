import gradio as gr

def hello(name):
    return f"Hello {name}!"

demo = gr.Interface(fn=hello, inputs="text", outputs="text")
demo.launch(debug=True)
