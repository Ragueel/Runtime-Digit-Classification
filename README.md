# Runtime-Digit-Classification
Simple digit classification where you could draw digits and program will classify which number it is.

I created this simple project to know, how much, different models perform on the same task. 3 models were used here, which are: 
SVC, Logistic Regression, ANN.

Project has UI where you could draw numbers(600x600) and press predict button to classify which number it is(according to machine at least).
Training dataset contains more than 600 digits size of 200x200 pixels (everything was handwritten).

Every image later converted to MNIST like array and later used in training.

Sadly model suffered from underfitting, if you want experiment with it, feel free to download it and contact me if you have any question.

run painter.py to start program

Libraries used:<br>
tensorflow<br>
sklearn<br>
pillow (PIL)
