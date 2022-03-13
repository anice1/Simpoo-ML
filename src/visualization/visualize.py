import matplotlib.pyplot as plt
import numpy as np

def plot_lr(history, experiment_name:str, lr=3e-1):
    """plots the learning rate curve of our model

    Args:
        history ([type]): [description]
        lr (learning_rate, optional): The set learning rate of your model, if not set, defaults to Adam optimizer's learning rate 3e-1.
    """

    loss = history.history["loss"]
    learning_rate = lr * 10 ** (np.arange(len(loss))/20)
    plt.plot(lr, loss)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title(experiment_name)

def decision_boundary(clf, X_test, y_test):
    """Plots a decision boudary

    Args:
        clf (model): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
    """
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig("test.png")
