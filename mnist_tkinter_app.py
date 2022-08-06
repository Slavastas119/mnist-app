import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color
from skimage.transform import resize
from tensorflow.keras.models import load_model
import joblib
import xgboost as xgb
import sys

def load_models(sklearn_file='sklearn_models.joblib', 
                xgboost_file='mnist_xgboost.json', 
                xgboost_pca_file='mnist_xgboost_pca.json', 
                cnn_file='mnist_cnn.h5', 
                dnn_file='mnist_dnn.h5'):
    # SKLearn models
    models = joblib.load(sklearn_file)
    # XGBoost models
    clf = xgb.XGBClassifier()
    models['xgboost'] = xgb.XGBClassifier()
    models['xgboost'].load_model(xgboost_file)
    models['xgboost_pca'] = xgb.XGBClassifier()
    models['xgboost_pca'].load_model(xgboost_pca_file)
    # Keras models
    models['cnn'] = load_model(cnn_file)
    models['dnn'] = load_model(dnn_file)
    
    return models

def load_df_accuracy(file='test_accuracy.joblib'):
    df_accuracy = pd.DataFrame.from_dict(joblib.load(file), orient='index') \
                                .rename(columns={0: 'Accuracy (test)'}) \
                                .mul(100).round(2)
    df_accuracy.index.name = 'Model'
    
    return df_accuracy

def make_df_predictions(models=load_models(), img_file='temp.jpg'):
    img = plt.imread(img_file)
    img_gray = color.rgb2gray(img)
    image_resized = resize(img_gray, (28, 28), anti_aliasing=True)
    
    df_proba = pd.DataFrame.from_dict({'cnn': models['cnn'].predict(image_resized.reshape(1, *image_resized.shape, 1), verbose=0)[0],
                                       'logistic_regression': models['logistic_regression'].predict_proba(image_resized.reshape(1, image_resized.size))[0],
                                       'logistic_regression_pca': models['logistic_regression_pca'].predict_proba(models['pca'].transform(image_resized.reshape(1, image_resized.size)))[0],
                                       'xgboost': models['xgboost'].predict_proba(image_resized.reshape(1, image_resized.size))[0],
                                       'xgboost_pca': models['xgboost_pca'].predict_proba(models['pca'].transform(image_resized.reshape(1, image_resized.size)))[0]},
                                      orient='index') \
                                .mul(100).round(2)
    df_pred = pd.DataFrame(df_proba.apply(np.argmax, axis=1)).rename(columns={0: 'Predicted number'})
    
    return df_pred, df_proba
    

def enlarge_image(image_resized, factor=10):
    img = []
    for row in image_resized:
        new_row = np.repeat(row, factor).tolist()
        for _ in range(factor):
            img.append(new_row)
    return np.array(img)



class ImageGenerator:
    def __init__(self, parent, posx, posy, *kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 280
        self.sizey = 280
        
        self.models = load_models()
        self.df_accuracy = load_df_accuracy()

        self.penColor = 'white'  #  (255, 255, 255)
        self.backColor = 'black'  #  (0, 0, 0)
        self.penWidth = 20  # 
        self.drawing_area = tk.Canvas(
            self.parent, width=self.sizex, height=self.sizey, bg=self.backColor
        )
        self.drawing_area.place(x=self.posx, y=self.posy)
        self.drawing_area.bind('<B1-Motion>', self.motion)
        self.resized_image = tk.Canvas(
            self.parent, width=self.sizex, height=self.sizey, bg=self.backColor
        )
        self.resized_image.place(x=self.posx+self.sizex+20, y=self.posy)
        self.pca_image = tk.Canvas(
            self.parent, width=self.sizex, height=self.sizey, bg=self.backColor
        )
        self.pca_image.place(x=self.posx+(self.sizex+20)*2, y=self.posy)
        self.dnn_image = tk.Canvas(
            self.parent, width=self.sizex, height=self.sizey, bg=self.backColor
        )
        self.dnn_image.place(x=self.posx+(self.sizex+20)*3, y=self.posy)
        self.label = tk.Label(
            self.parent, text='Drawing window (280x280)', fg='black', font=('Arial', 10)
        )
        self.label.place(x=self.posx, y=10)
        self.label = tk.Label(
            self.parent, text='Reshaped image (28x28)', fg='black', font=('Arial', 10)
        )
        self.label.place(x=self.posx+(self.sizex+20), y=10)
        self.label = tk.Label(
            self.parent, text='PCA result (5x5)', fg='black', font=('Arial', 10)
        )
        self.label.place(x=self.posx+(self.sizex+20)*2, y=10)
        self.label = tk.Label(
            self.parent, text='Deconvolutional image (28x28)', fg='black', font=('Arial', 10)
        )
        self.label.place(x=self.posx+(self.sizex+20)*3, y=10)
        self.label_models = tk.Label(
            self.parent, text='\n'+'\n'.join(self.df_accuracy.index), fg='black', font=('Arial', 10)
        )
        self.label_models.place(x=self.sizex / 7, y=self.sizey + 100)
        self.label_accuracy = tk.Label(
            self.parent, text=self.df_accuracy.to_string(index=False), fg='black', font=('Arial', 10)
        )
        self.label_accuracy.place(x=self.sizex / 7 + 150, y=self.sizey + 100)
        self.label_pred_text = tk.StringVar()
        self.label_pred_text.set('Predicted number')
        self.label_pred = tk.Label(
            self.parent, textvariable=self.label_pred_text, fg='black', font=('Arial', 10)
        )
        self.label_pred.place(x=self.sizex / 7 + 300, y=self.sizey + 100)
        self.label_proba_text = tk.StringVar()
        self.label_proba_text.set('Predicted probabilities')
        self.label_proba = tk.Label(
            self.parent, textvariable=self.label_proba_text, fg='black', font=('Arial', 10)
        )
        self.label_proba.place(x=self.sizex / 7 + 450, y=self.sizey + 100)
        self.button = tk.Button(
            self.parent, text='Done', width=10, bg='white', command=self.save
        )
        self.button.place(x=self.sizex / 7, y=self.sizey + self.posy + 20)
        self.button1 = tk.Button(
            self.parent, text='Clear', width=10, bg='white', command=self.clear
        )
        self.button1.place(x=(self.sizex / 7) + 80, y=self.sizey + self.posy + 20)

        self.image = Image.new('RGB', (self.sizex, self.sizey), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

    def save(self):
        filename = 'temp.jpg'
        self.image.save(filename)
        
        img = plt.imread(filename)
        img_gray = color.rgb2gray(img)
        image_resized = resize(img_gray, (28, 28), anti_aliasing=True)
        
        self.df_pred, self.df_proba = make_df_predictions(models=self.models, img_file=filename)
        
        df = pd.concat([self.df_accuracy, self.df_pred, self.df_proba], axis=1)#.to_string(col_space=25, max_colwidth=25)
        self.label_pred_text.set(df[self.df_pred.columns].to_string(index=False))
        text = '\t'.join([str(col) for col in self.df_proba.columns])
        for row in df[self.df_proba.columns].values:
            text += '\n' + '\t'.join([str(el) for el in row])
        self.label_proba_text.set(text)

        self.img = ImageTk.PhotoImage(Image.fromarray((self._resize_image(image_resized)*255).astype('uint8')))
        self.resized_image.create_image(0, 0, anchor=tk.NW, image=self.img)
        self.resized_image.image = self.img
        
        self.img_pca = ImageTk.PhotoImage(
            Image.fromarray(
                (self._resize_image(
                    self.models['pca'].transform(image_resized.reshape(1, image_resized.size)).reshape(5, 5),
                    factor=int(self.sizex/5)
                )*255).astype('uint8')
            )
        )
        self.pca_image.create_image(0, 0, anchor=tk.NW, image=self.img_pca)
        self.pca_image.image = self.img_pca
        
        self.img_dnn = ImageTk.PhotoImage(
            Image.fromarray(
                (self._resize_image(
                    self.models['dnn'].predict(self.df_pred.loc['cnn',].values.reshape(-1, 1), verbose=0)[0]
                )*255).astype('uint8')
            )
        )
        self.dnn_image.create_image(0, 0, anchor=tk.NW, image=self.img_dnn)
        self.dnn_image.image = self.img_dnn
        
        #self.parent.destroy()
    
    def _resize_image(self, image_resized, factor=10):
        return enlarge_image(image_resized, factor)

    def clear(self):
        # image
        self.drawing_area.delete('all')
        self.image = Image.new('RGB', (self.sizex, self.sizey), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        
        self.resized_image.delete('all')
        self.pca_image.delete('all')
        self.dnn_image.delete('all')
        
        self.label_pred_text.set('Predicted number')
        self.label_proba_text.set('Predicted probabilities')

    def motion(self, event):
        # image
        self.drawing_area.create_oval(
            event.x,
            event.y,
            event.x + self.penWidth,
            event.y + self.penWidth,
            fill=self.penColor,
            outline=self.penColor,
        )  # 

        self.draw.ellipse(
            (
                (event.x, event.y),
                (event.x + self.penWidth, event.y + self.penWidth),
            ),
            fill=self.penColor,
            outline=self.penColor,
            width=self.penWidth,
        )  # ,point、line


if __name__ == '__main__':
    root = tk.Tk()
    root.wm_geometry(f'%dx%d+%d+%d' % (1200, 600, 10, 10))
    root.config(bg='black')
    ImageGenerator(root, 10, 30)
    root.mainloop()