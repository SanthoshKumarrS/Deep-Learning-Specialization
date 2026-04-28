from fastai.vision.all import *
import gradio as gr
import pathlib

def is_cat(x): return x[0].isupper()

learn = load_learner(r'C:/Users/santh/catndog/model.pkl')

categories = ('Dog','Cat')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories,map(float,probs)))


image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
example = ['dog.jfif','cat.jfif','dunno.jfif']

intf = gr.Interface(fn = classify_image, inputs= image, outputs= label, examples= example)
intf.launch(inline=False)
