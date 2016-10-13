import numpy as np
import random
import os, subprocess
import copy
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from numpy import genfromtxt


class Perceptron:
    def __init__(self, N):
        # Random linearly separated data
        self.X = self.generate_points(N)

    def generate_points(self, N):
        X, y = self.make_blobs()
        bX = []
        for k in range(0, N):
            bX.append((np.concatenate(([1], X[k, :])), y[k]))

        # this will calculate linear regression at this point
        X = np.concatenate((np.ones((N, 1)), X), axis=1);  # adds the 1 constant
        self.linRegW = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)  # lin reg
        return bX

    def make_blobs(self):
        dataset = genfromtxt('features.csv', delimiter=' ')
        y = dataset[:, 0]
        X = dataset[:, 1:]
        y[y<>0] = -1
        y[y==0] = +1
        c0 = plt.scatter(X[y==-1,0], X[y==-1,1], s=20, color='r', marker='x')
        c1 = plt.scatter(X[y==1,0], X[y==1,1], s=20, color='b', marker='o')
        ctrs = 3 * np.random.normal(0, 1, (2, 2))

        plt.legend((c0, c1), ('All other Numbers 1', 'Numbers Zero +1'), loc='upper right', scatterpoints=1, fontsize=11)

        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(r'Intensisty and symmetry of digits')
        plt.savefig('midterm.plot.pdf', bbox_inches='tight')
        plt.show()

        #X, y = make_blobs(n_samples=100, centers=ctrs, n_features=2, cluster_std=1.0, shuffle=False, random_state=0)

        #y[y == 0] = -1

        #c0 = plt.scatter(X[y == -1, 0], X[y == -1, 1], s=20, color='r', marker='x')
        #c1 = plt.scatter(X[y == 1, 0], X[y == 1, 1], s=20, color='b', marker='o')
        return X, y


    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5, 5))
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        l = np.linspace(-1.5, 2.5)
        V = self.linRegW
        plt.plot(V)
        a, b = -V[1] / V[2], -V[0] / V[2]
        plt.plot(l, a * l + b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x, s in self.X:
            plt.plot(x[1], x[2], cols[s] + '.')
        if mispts:
            for x, s in mispts:
                plt.plot(x[1], x[2], cols[s] + 'x')
        if vec.size:
            aa, bb = -vec[1] / vec[2], -vec[0] / vec[2]
            plt.plot(l, aa * l + bb, 'g-', lw=2)
       # if save:
        #    if not mispts:
         #       plt.title('N = %s' % (str(len(self.X))))
        #    else:
        #        plt.title('N = %s with %s test points' \
        #                  % (str(len(self.X)), str(len(mispts))))
        #    plt.savefig('p_N%s' % (str(len(self.X))), \
        #                dpi=200, bbox_inches='tight')



    def classification_error(self, vec, pts=None):
        # Error defined as fraction of misclassified points
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0

        for x, s in pts:

            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)

        return error

    def choose_miscl_point(self, vec):
        # Choose a random point among the misclassified
        pts = self.X
        mispts = []
        for x, s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0, len(mispts))]

    def pla(self, save=False, line=False):
        # Initialize the weigths to zeros

        # Initialize the weights to solution of linear regression
        if line:
            w = self.linRegW
        else:
            w = np.zeros(3)
        w0 = np.zeros(3)
        bestW = []
        pocketError = []
        # Reassign variables
        X, N = self.X, len(self.X)

        count = 0
        it = 0
        # Iterate until all points are correctly classified
        while self.classification_error(w) != 0:
            it += 1
            # Pick random misclassified point
            x, y = self.choose_miscl_point(w0)
            # Update weights
            w0 += y * x
            # Only update if the new line weight is better
            pocketError.append(self.classification_error(w0))
            bestW.append(self.classification_error(w))
            if self.classification_error(w) > self.classification_error(w0):
                w = copy.deepcopy(w0)
                count = 0
            else:
                count += 1
            # break after 500 iterations with the same wieghts
            if count > 500:
                break
            # breakafter 30 iterations overall
            if it > 30:
                break
            if save:
                self.plot(vec=w)
                plt.title('N = %s, Iteration %s\n' \
                          % (str(N), str(it)))
                plt.savefig('p_N%s_it%s' % (str(N), str(it)), \
                            dpi=200, bbox_inches='tight')
        self.w = w
        print pocketError
        print bestW

        if save:
            self.plot(vec=w)
            plt.title('N = %s, Iteration %s\n' \
                      % (str(N), str(it)))
            plt.savefig('p_N%s_it%s' % (str(N), str(it)), \
                        dpi=200, bbox_inches='tight')

        return it

    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)


def main():
    it = np.zeros(1)
    for x in range(0, 1):
        p = Perceptron(7291)
        it[x] = p.pla(save=True, line=False)
        print it

        # n, bins, patches = plt.hist(it, 50, normed=1, facecolor='green', alpha=0.75)
        # plt.show()


        # p.plot()


main()















