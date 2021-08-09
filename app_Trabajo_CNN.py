import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output, ClientsideFunction, State
import plotly.graph_objects as go
import os
import base64
from PIL import Image
import requests
from io import BytesIO as _BytesIO
from torchvision.io.image import read_image
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

import numpy as np
import pandas as pd

import torch
import torchvision
from torchcam.cams import SmoothGradCAMpp

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
app.config.suppress_callback_exceptions = True

tabs_styles = {
    'height': '50px',
    'padding': '6px'
}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',    
}

#Cargamos el modelo guardado
my_resnet18 = torchvision.models.resnet18(pretrained = True)
my_resnet18.fc = torch.nn.Linear(in_features = 512, out_features = 4)
my_resnet18.load_state_dict(torch.load(r'D:\(30) Postgrado PUCP IA\5. Visión Artificial\Trabajo_final\Detector_resnet\data\my_resnet18.pth'))

my_googlenet = torchvision.models.googlenet(pretrained = True)
my_googlenet.fc = torch.nn.Linear(in_features = 1024, out_features = 4)
my_googlenet.load_state_dict(torch.load(r'D:\(30) Postgrado PUCP IA\5. Visión Artificial\Trabajo_final\Detector_resnet\data\my_gnet.pth'))

my_resnet50 = torchvision.models.resnet50(pretrained = True)
my_resnet50.fc = torch.nn.Linear(in_features = 2048, out_features = 4)
my_resnet50.load_state_dict(torch.load(r'D:\(30) Postgrado PUCP IA\5. Visión Artificial\Trabajo_final\Detector_resnet\data\my_resnet50.pth'))

#my_googlenetcuda = torchvision.models.googlenet(pretrained = True)
#my_googlenetcuda.fc = torch.nn.Linear(in_features = 1024, out_features = 4)
#my_googlenetcuda.load_state_dict(torch.load(r'D:\(30) Postgrado PUCP IA\5. Visión Artificial\Trabajo_final\Detector_resnet\data\my_gnetcuda.pth',map_location =torch.device('cpu')))

import torchvision.transforms as T
from PIL import Image

def transformar_imagen(name):
  '''
  Pre procesamiento de la imagen
  '''
  image = Image.open(name).convert('RGB')  
  transf = T.Compose([T.Resize(size = (224, 224)),
                      #T.CenterCrop(224),
                      T.ToTensor(), # entre 0 y 1
                      T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]) # medias y desviacion estandar (Welford's method)
  #image = Image.open(name)
  # Tamaño (3,224,224) => (1, 3, 224, 224)
  return transf(image).unsqueeze(0) # tensor singleton

def prediccion(network, name):
  '''
  Predicción con grado de confidencia
  '''
  tensor = transformar_imagen(name)
  network.eval()
  output = network(tensor)
  #output = F.softmax(pred, dim=-1)
  _, pred = torch.max(output,1)
  return pred

def parse_contents(contents,name):
    #save_file(name,contents)
    return html.Div([
        
        html.Img(src=contents,
                 style={
                    'maxWidth': '100%',
                    'maxHeight': '100%',
                    'marginLeft': 'auto',
                    'marginRight': 'auto'
            }),
        html.Hr(),             
        
    ])

UPLOAD_DIRECTORY = 'D:/(30) Postgrado PUCP IA/5. Visión Artificial/Trabajo_final/Detector_resnet/data/'
def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("DESARROLLO EN APLICACIONES CON VISIÓN ARTIFICIAL"),
            html.H3("Detección de COVID-19 en imágenes de rayos X de tórax"),
            html.Br(),
            html.Div(                
                children="Seleccione modelo",
            ),            

            dcc.RadioItems(
                id='id_modelo',
                options=[
                    {'label': 'Resnet18', 'value': 'model1'},
                    {'label': 'GoogleNet', 'value': 'model2'},
                    {'label': 'Resnet50', 'value': 'model3'},
                    {'label': 'GoogleNet_cuda', 'value': 'model4'}
                    
                ],
                value='model1'
            ),            

            html.Br(),
            html.Div(                
                children="Ingrese URL de imagen",
            ),

            dcc.Input(id="input-url",
                      placeholder='Ingrese URL y clic en Run',
                    ),
            
            html.Button('Run', id='btn-run', n_clicks=0),

            html.Br(),
            html.Br(),
            html.Div(                
                children="Ingrese radiografía",
            ),


        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        #id="control-card",
        children=[
                        
            dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop o ',
                        html.A('Seleccione archivos')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '2px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
            ),                     
            
        ],
    )


app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("iapucp.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]            
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[                       
                
                # Scatter plot por distrito
                html.Div([
                    html.Div(
                        id="prediction",                        
                        className="six columns", style={'display': 'inline-block'}
                    ),

                    html.Div([
                        dcc.Graph(id="bar_softmaxo")       
                        
                    ],className="six columns", style={'display': 'inline-block'})          
                    
                ],className="twelve columns"),  

                html.Br(), 

                html.Div([

                    html.Div([
                        dcc.Graph(id="output-image-upload")       
                        
                    ],className="six columns", style={'display': 'inline-block'}),  

                    html.Div([
                        dcc.Graph(id="output-gradcam-upload")       
                        
                    ],className="six columns", style={'display': 'inline-block'})
                    
                ],className="twelve columns")           

            ],
        ),
    ],
)

@app.callback(Output('bar_softmaxo', 'figure'),
              [Input('id_modelo','value'),
              Input('upload-image', 'contents'),
              Input("btn-run", "n_clicks"),
              Input("input-url", "n_submit"),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'),              
              State("input-url", "value")])

def update_output_(model_input,contents,n_clicks,n_submit, name, list_of_dates,url):

    class_names = ['Normal','Viral', 'Covid', 'Opacity']
    if model_input=='model1':
        model = my_resnet18
        model_text = 'Modelo ResNet18'
    elif model_input=='model2':
        model = my_googlenet
        model_text = 'Modelo GoogleNet'
    elif model_input == 'model3':
        model = my_resnet50
        model_text = 'Modelo ResNet50'
    #elif model_input == 'model4':
    #    model = my_googlenetcuda
    #    model_text = 'Modelo GoogleNet-Cuda'


    scores_fig = px.bar(
            x=[0,0,0,0],
            y=class_names,
            labels=dict(x="Confidence", y="Classes"),
            title="Cargue imagen para predicción",
            orientation="h",
        )

    if contents is not None:
        imagen_test = 'D:/(30) Postgrado PUCP IA/5. Visión Artificial/Trabajo_final/Detector_resnet/data/' +  name
        tensor = transformar_imagen(imagen_test)
        model.eval()
        output = model(tensor)
        preds = torch.softmax(model(tensor), dim=1)
        probas = preds.tolist()[0]
                
        scores_fig = px.bar(
            x=probas,
            y=class_names,
            labels=dict(x="Confidence", y="Classes"),
            title="Probabilidad de predicción" + model_text,
            height=300,
            orientation="h",
        )

    scores_fig.update_layout(
        margin=dict(l=1, r=1, t=30, b=30),     
    )   

    return scores_fig
        



@app.callback(Output('output-image-upload', 'figure'),
              Input('upload-image', 'contents'),
              Input("btn-run", "n_clicks"),
              Input("input-url", "n_submit"),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'),              
              State("input-url", "value"))

def update_output(contents,n_clicks,n_submit, name, list_of_dates,url):

    fig = px.scatter(
            x=[0,0,0,0],
            y=[0,0,0,0])

    if n_clicks:
        image = Image.open(requests.get(url, stream=True).raw)
        #buff = _BytesIO()
        #image.save(buff, format='png')
        #contents = base64.b64encode(buff.getvalue()).decode("utf8")
        
        #contents = base64.b64encode(im_bytes).decode('utf8')
        #children = [parse_contents(image,url)]
        fig = px.imshow(image, title="Imagen Original")
        

    if contents is not None:
        #children = [parse_contents(contents,name)]
        #print(type(contents))
        #print(list_of_contents)
        

        fig = go.Figure(go.Image(source=contents))
    
    fig.update_layout(margin=dict(l=1, r=1, t=30, b=30))    
    return fig
        

@app.callback(Output('output-gradcam-upload', 'figure'),
              Input('id_modelo','value'),
              Input('upload-image', 'contents'),
              Input("btn-run", "n_clicks"),
              Input("input-url", "n_submit"),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'),              
              State("input-url", "value"))

def update_output(model_input,contents,n_clicks,n_submit, name, list_of_dates,url):

    fig = px.scatter(
            x=[0,0,0,0],
            y=[0,0,0,0])

    #Cargamos el modelo guardado

    if model_input=='model1':
        model = my_resnet18
        model_text = 'Modelo ResNet18'
    elif model_input=='model2':
        model = my_googlenet
        model_text = 'Modelo GoogleNet'
    elif model_input == 'model3':
        model = my_resnet50
        model_text = 'Modelo ResNet50'
    #elif model_input == 'model4':
    #    model = my_googlenetcuda
    #    model_text = 'Modelo GoogleNet-Cuda'
    
    cam_extractor = SmoothGradCAMpp(model)
    
    if contents is not None:
        imagen_test = 'D:/(30) Postgrado PUCP IA/5. Visión Artificial/Trabajo_final/Detector_resnet/data/' +  name
        img = read_image(imagen_test)
        input_tensor = transformar_imagen(imagen_test)

        out = model(input_tensor)        
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        image = Image.open(imagen_test).convert('RGB')

        # Resize the CAM and overlay it
        result = overlay_mask(image, to_pil_image(activation_map, mode='F'), alpha=0.5)
        #children = [parse_contents(result,name)]
        fig = px.imshow(result, title="Cam Extractor")
    
    fig.update_layout(margin=dict(l=1, r=1, t=30, b=30)) 
    return fig
        #return children

        #pred = prediccion(model,imagen_test)
    


@app.callback(Output('prediction', 'children'),
              Input('id_modelo','value'),
              Input('upload-image', 'contents'),
              Input("btn-run", "n_clicks"),
              Input("input-url", "n_submit"),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'),
              State("input-url", "value"))


def update_output(model_input,
                  contents,
                  n_clicks,
                  n_submit,
                  filename,
                  list_of_dates,
                  url
                  ):

    class_names = ['Normal','Viral', 'Covid', 'Opacity'] 

    if model_input=='model1':
        model = my_resnet18
        model_text = 'Modelo ResNet18'
    elif model_input=='model2':
        model = my_googlenet
        model_text = 'Modelo GoogleNet'
    elif model_input == 'model3':
        model = my_resnet50
        model_text = 'Modelo ResNet50'
    #elif model_input == 'model4':
    #    model = my_googlenetcuda
    #    model_text = 'Modelo GoogleNet-Cuda'

    if contents is not None:
        imagen_test = 'D:/(30) Postgrado PUCP IA/5. Visión Artificial/Trabajo_final/Detector_resnet/data/' + filename

        pred = prediccion(model,imagen_test) 
        print('Content no vacío')        

        children = [
            html.H3('Resultado: '),
            html.H5(class_names[int(pred.numpy())]),
            html.H3('Nombre de archivo: '),
            html.H5(filename),
            html.H3('Modelo usado: '),
            html.H5(model_text)
        ]        
        return children

    elif n_clicks:
        print('Se presionó RUN')
        transf = T.Compose([T.Resize(size = (224, 224)),
                      #T.CenterCrop(224),
                      T.ToTensor(), # entre 0 y 1
                      T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        imagen_test = Image.open(requests.get(url, stream=True).raw)
        #print(imagen_test)     

        tensor = transf(imagen_test).unsqueeze(0) 

        #tensor = transformar_imagen(name)
        model.eval()
        output = model(tensor)
        #output = F.softmax(pred, dim=-1)
        _, pred = torch.max(output,1)                
        
        #pred = prediccion(model,imagen_test)         

        children = [
            html.H3('Resultado: '),
            html.H5(class_names[int(pred.numpy())]),
            html.H3('Ruta de imagen: '),
            html.H5(url),
            html.H3('Modelo usado: '),
            html.H5(model_text)
        ]        
        return children


if __name__ == '__main__':
    app.run_server(
        debug='True',
        port=8050,
        host='127.0.0.1'
    )