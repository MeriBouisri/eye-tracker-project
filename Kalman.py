import cv2
import numpy as np

class Kalman(object):
    def __init__(self, dt, point):
        self.dt = dt

        # Vecteur d'état initial
        self.E = np.matrix([[point[0]], [point[1]], [0], [0]])  # Prend les points donnés au début, les vitesses restent à 0.

        # Matrice de transition
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # Matrice d'observation, on observe que x et y
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])


        # Matrice bruit (matrice de covariance)
        self.Q = np.matrix([[0.2, 0, 0, 0],
                            [0, 0.2, 0, 0],
                            [0, 0, 0.2, 0],
                            [0, 0, 0, 0.2]])  # Plus valeur petite, plus la reduction du bruit est grande

        self.R = np.matrix([[1, 0],
                            [0, 1]])


        self.P = np.eye(self.A.shape[1])


    def predict(self):
        self.E = np.dot(self.A, self.E)
        # Estimation de la covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        x = float(self.E[0, 0])
        y = float(self.E[0, -1])
        return x, y


    def update(self, z):
        # Calcul du gain de Kalman : s'affine au fil du temps
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Vecteur z contient les mesures
        # Nouveau E corrigé/innové
        self.E = np.round(self.E + np.dot(K, (z-np.dot(self.H, self.E))))
        I = np.eye(self.H.shape[1])
        self.P = (I-(K*self.H))*self.P
        #x = self.E[0,1]
        #y = self.E[1,1]

        return self.E


