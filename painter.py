from tkinter import *
from PIL import Image, ImageDraw
import path_utils
import train_network
import data_loader
import image_compresser
import grad_descent

CANVAS_WIDTH = 500
CANVAS_HEIGHT = 500

ANN = train_network.load_network()
LogReg = grad_descent.load_model()


def get_max(array):
    max = 0
    index = 0
    for i in range(len(array[0])):
        if max < array[0][i]:
            max = array[0][i]
            index = i
    return index


class PaintApp:
    drawing_tool = "pencil"

    # State of left mouse button
    left_button = "up"

    image_name, main_frame, drawing_area, save_button, clear_button, save_image, predict_neural_net, predict_logistic = None, None, \
                                                                                                                        None, None, None, None, None, None
    predict_button = None
    image = None
    draw = None
    # Positions of the mouse
    xpos, ypos = None, None

    def __init__(self, root):
        self.main_frame = Frame(root, width=900, height=900, bg='white')

        # self.image_name = Entry(self.main_frame)
        self.image_class = Entry(self.main_frame)
        self.predicted_text = Entry(self.main_frame)
        self.setup_canvas()

        self.focused_text = ''

        self.setup_text_entries()

        self.save_button.bind("<ButtonPress-1>", self.save_image)
        self.clear_button.bind("<ButtonPress-1>", self.clear_canvas)
        self.predict_button.bind("<ButtonPress-1>", self.predict_image)
        self.predict_neural_net.bind("<ButtonPress-1>", self.neural_net)
        self.predict_logistic.bind("<ButtonPress-1>", self.logistic)
        self.drawing_area.pack(side=LEFT)
        # self.image_name.pack(anchor="w", fill=X)
        self.image_class.pack(anchor="w", fill=X)
        self.save_button.pack(anchor="w", fill=X)
        self.clear_button.pack(anchor="w", fill=X)
        self.predict_button.pack(anchor="w", fill=X)
        self.predict_neural_net.pack(anchor="w", fill=X)
        self.predict_logistic.pack(anchor="w", fill=X)
        self.predicted_text.pack(anchor="w", fill=X)

        self.main_frame.pack()
        self.image = Image.new("RGB", (CANVAS_HEIGHT, CANVAS_WIDTH), 'white')
        self.draw = ImageDraw.Draw(self.image)
        # Catching events
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.left_button_down)
        self.drawing_area.bind("<ButtonRelease-1>", self.left_button_up)

    def neural_net(self, event=None):
        self.predict_image(event)
        arr = data_loader.get_numpy_from_path('./prediction_compressed/predict.jpg')
        arr = arr.reshape(1, 256, 256)
        prediction = ANN.predict(arr)
        a = get_max(prediction)
        print(a)
        self.predicted_text.delete(0, END)
        self.predicted_text.insert(0, a)

    def logistic(self, event=None):
        self.predict_image(event)
        arr = data_loader.get_numpy_from_path('./prediction_compressed/predict.jpg')
        prediction = grad_descent.predict_image(arr, LogReg)
        print(prediction)

    def text_focus_out(self, event):
        if event.widget.get() == '':
            if event.widget == self.image_name:
                event.widget.insert(0, 'Enter name')
            else:
                event.widget.insert(0, 'Enter class name')

    def text_focus(self, event):
        self.focused_text = event.widget.get()
        if event.widget.get() == 'Enter name' or event.widget.get() == 'Enter class name':
            event.widget.delete(0, END)

    # Predicts image
    def predict_image(self, event):
        path_to_file = 'predict.jpg'
        self.image.save(path_to_file)
        print('Saving file:' + path_to_file)
        image_compresser.rescale_image_and_save(path_to_file, original_path='', compression_path='./prediction_compressed/')


    def setup_canvas(self):
        self.drawing_area = Canvas(self.main_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        self.save_button = Button(self.main_frame, text="Save image")
        self.clear_button = Button(self.main_frame, text="Clear canvas")
        self.predict_button = Button(self.main_frame, text='Predict image')
        self.predict_neural_net = Button(self.main_frame, text="ANN Predict")
        self.predict_logistic = Button(self.main_frame, text="LogReg Predict")

    def setup_text_entries(self):
        self.focused_text = ''

        self.image_class.insert(0, 'Enter class name')

        self.image_class.bind("<FocusIn>", self.text_focus)
        self.image_class.bind("<FocusOut>", self.text_focus_out)

    def clear_canvas(self, event):
        self.drawing_area.delete("all")
        self.image = Image.new("RGB", (CANVAS_HEIGHT, CANVAS_WIDTH), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def save_image(self, event):
        # Name of Image
        # name = self.image_name.get()
        image_class_name = self.image_class.get()
        # Saving image
        path_to_file = path_utils.uniquify('./data/' + image_class_name + '_.jpg')
        self.image.save(path_to_file)
        print('Saving file:' + path_to_file)

    def left_button_down(self, event=None):
        self.left_button = "down"

    def left_button_up(self, event=None):
        self.left_button = "up"
        self.xpos = None
        self.ypos = None

    def motion(self, event=None):
        if self.drawing_tool == "pencil":
            self.pencil_draw(event)

    # Drawing with pencil
    def feedforward(self, event=None):
        return

    def pencil_draw(self, event=None):
        if self.left_button == "down":
            if self.xpos is not None and self.ypos is not None:
                event.widget.create_line(self.xpos, self.ypos, event.x, event.y,
                                         smooth=TRUE, width=15)
                self.draw.line([self.xpos, self.ypos, event.x, event.y], fill=(0, 0, 0))
            self.xpos = event.x
            self.ypos = event.y


root = Tk()
paint_app = PaintApp(root)
root.mainloop()
