from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import data_loader
import pickle

# Logistic regression implementation

FILE_NAME = 'regression.sav'  # Save name


def train_model():
    (data, label) = data_loader.get_data()
    data = data.reshape((len(data), 256 * 256))

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=20)

    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')

    clf.fit(x_train, y_train)
    save_model(clf)


def predict_image(image, clf):
    x = image.reshape((1, 256 * 256))

    pred = clf.predict(x)
    return pred


def save_model(model):
    pickle.dump(model, open(FILE_NAME, 'wb'))


def load_model():
    loaded_model = pickle.load(open(FILE_NAME, 'rb'))
    return loaded_model
