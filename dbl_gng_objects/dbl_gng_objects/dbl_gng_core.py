"""DBL-GNG core (numpy only, no ROS, no display).

Port of examples/DBL_GNG.py from https://github.com/CornerSiow/DBL-GNG (MIT).
Algorithm logic is kept as in the reference. Deviations, all non-algorithmic:
  * addNewNode used the module-level global `gng`; it now uses `self`.
  * print() calls are gated by `verbose` (default False).
  * display/data-loading code of the reference script is not included.
"""

import random

import numpy as np


class DBL_GNG:
    def __init__(self, feature_number, maxNodeLength, L1=0.5, L2=0.01,
                 errorNodeFactor=0.5, newNodeFactor=0.5, verbose=False):
        self.feature_number = feature_number
        self.M = maxNodeLength
        self.alpha = L1
        self.beta = L2
        self.delta = errorNodeFactor
        self.rho = newNodeFactor
        self.verbose = verbose

        self.eps = 1e-04

    def resetBatch(self):
        self.Delta_W_1 = np.zeros_like(self.W)
        self.Delta_W_2 = np.zeros_like(self.W)
        self.A_1 = np.zeros(len(self.W))
        self.A_2 = np.zeros(len(self.W))
        self.S = np.zeros((len(self.W), len(self.W)))

    def initializeDistributedNode(self, data, number_of_starting_points=1):
        data = data[:, :self.feature_number].copy()
        np.random.shuffle(data)

        nodeList = np.empty((0, self.feature_number), dtype=np.float32)
        edgeList = np.empty((0, 2), dtype=int)

        # copy the data, ready for crop
        tempData = data.copy()

        # define the batch size
        batchSize = len(data) // number_of_starting_points

        for i in range(number_of_starting_points):
            idx = np.arange(len(tempData), dtype=int)

            # randomly select a node
            selectedIndex = np.random.choice(idx[-batchSize:])
            currentNode = tempData[selectedIndex]

            nodeList = np.append(nodeList, [currentNode], axis=0)

            # distance from all data to the current node
            y2 = np.sum(np.square(tempData), axis=1)
            dot_product = 2 * np.matmul(currentNode, tempData.T)
            dist = y2 - dot_product

            idx = np.argsort(dist)

            # third closest node as neighbor, leaving some space in between
            neighborNode = tempData[idx[2]]
            nodeList = np.append(nodeList, [neighborNode], axis=0)

            edgeList = np.append(edgeList, [[i * 2, i * 2 + 1]], axis=0)

            # remove the current area, so it won't repeat in the follow search
            idx = idx[batchSize:]
            tempData = tempData[idx]

        self.W = nodeList
        self.C = edgeList
        self.E = np.zeros(len(self.W))

    # input Batch, Feature
    def batchLearning(self, X):
        X = X[:, :self.feature_number]

        i_adj = np.eye(len(self.W))

        adj = np.zeros((len(self.W), len(self.W)))
        adj[self.C[:, 0], self.C[:, 1]] += 1
        adj[self.C[:, 1], self.C[:, 0]] += 1
        adj[adj > 0] = 1

        batchIndices = np.arange(len(X))

        # distance
        x2 = np.sum(np.square(X), axis=1)
        y2 = np.sum(np.square(self.W), axis=1)
        dot_product = 2 * np.matmul(X, self.W.T)

        dist = np.clip(np.expand_dims(x2, axis=1) + y2 - dot_product, a_min=0, a_max=None)
        dist = np.sqrt(dist + self.eps)

        # first and second winner nodes
        tempDist = dist.copy()
        s1 = np.argmin(tempDist, axis=1)
        tempDist[batchIndices, s1] = 99999
        s2 = np.argmin(tempDist, axis=1)

        # add error to s1
        self.E += np.sum(i_adj[s1] * dist, axis=0) * self.alpha

        # update s1 position
        self.Delta_W_1 += (np.matmul(i_adj[s1].T, X) - (self.W.T * np.sum(i_adj[s1], axis=0)).T) * self.alpha

        # update s1 neighbor position
        self.Delta_W_2 += (np.matmul(adj[s1].T, X) - np.multiply(self.W.T, adj[s1].sum(0)).T) * self.beta

        # s1 node activation
        self.A_1 += np.sum(i_adj[s1], axis=0)

        # neighbor node activation
        self.A_2 += np.sum(adj[s1], axis=0)

        # count the important edge (s1 and s2)
        connectedEdge = np.zeros_like(self.S)
        connectedEdge[s1, s2] = 1
        connectedEdge[s2, s1] = 1

        t = i_adj[s1] + i_adj[s2]
        connectedEdge *= np.matmul(t.T, t)

        self.S += connectedEdge

    def updateNetwork(self):
        self.W += (self.Delta_W_1.T * (1 / (self.A_1 + self.eps))).T + (self.Delta_W_2.T * (1 / (self.A_2 + self.eps))).T

        self.C = np.asarray(self.S.nonzero()).T

        self.removeIsolatedNodes()

        self.E *= self.delta

        if random.random() > 0.9:
            self.removeNonActivatedNodes()

    def removeIsolatedNodes(self):
        adj = np.zeros((len(self.W), len(self.W)))
        adj[self.C[:, 0], self.C[:, 1]] = 1
        adj[self.C[:, 1], self.C[:, 0]] = 1

        isolatedNodes = (np.sum(adj, axis=0) + np.sum(adj, axis=1) == 0).nonzero()[0]

        finalDelete = list(np.unique(isolatedNodes))

        if len(finalDelete) > 1:
            finalDelete.sort(reverse=True)

        for v in finalDelete:
            self.C = self.C[np.logical_not(np.logical_or(self.C[:, 0] == v, self.C[:, 1] == v))]

            self.C[self.C[:, 0] > v, 0] -= 1
            self.C[self.C[:, 1] > v, 1] -= 1

        if len(finalDelete) > 0:
            if self.verbose:
                print("Isolated", finalDelete)

            self.S = np.delete(self.S, finalDelete, axis=0)
            self.S = np.delete(self.S, finalDelete, axis=1)

            self.W = np.delete(self.W, finalDelete, axis=0)
            self.E = np.delete(self.E, finalDelete, axis=0)
            self.A_1 = np.delete(self.A_1, finalDelete, axis=0)
            self.A_2 = np.delete(self.A_2, finalDelete, axis=0)

    def removeNonActivatedNodes(self):
        nodeActivation = self.A_1
        nonActivatedNodes = (nodeActivation == 0).nonzero()[0]

        finalDelete = list(nonActivatedNodes)

        if len(finalDelete) > 1:
            finalDelete.sort(reverse=True)

        for v in finalDelete:
            self.C = self.C[np.logical_not(np.logical_or(self.C[:, 0] == v, self.C[:, 1] == v))]

            self.C[self.C[:, 0] > v, 0] -= 1
            self.C[self.C[:, 1] > v, 1] -= 1

        if len(finalDelete) > 0:
            if self.verbose:
                print("Non Activated", finalDelete)

            self.S = np.delete(self.S, finalDelete, axis=0)
            self.S = np.delete(self.S, finalDelete, axis=1)

            self.W = np.delete(self.W, finalDelete, axis=0)
            self.E = np.delete(self.E, finalDelete, axis=0)
            self.A_1 = np.delete(self.A_1, finalDelete, axis=0)
            self.A_2 = np.delete(self.A_2, finalDelete, axis=0)

    def addNewNode(self):
        g = np.sum(self.E > np.quantile(self.E, 0.85))

        for _ in range(g):
            if len(self.W) >= self.M:
                return

            q1 = np.argmax(self.E)
            if self.E[q1] <= 0:
                if self.verbose:
                    print("Zero error q1")
                return

            # connected nodes
            connectedNodes = np.unique(np.concatenate((self.C[self.C[:, 0] == q1, 1], self.C[self.C[:, 1] == q1, 0])))
            if len(connectedNodes) == 0:
                return

            # maximum error of neighbors
            q2 = connectedNodes[np.argmax(self.E[connectedNodes])]
            if self.E[q2] <= 0:
                if self.verbose:
                    print("Zero error q2")
                return

            # insert new node between q1 and q2
            q3 = len(self.W)
            new_w = (self.W[q1] + self.W[q2]) * 0.5
            self.W = np.vstack((self.W, new_w))
            self.E = np.concatenate((self.E, np.zeros(1)), axis=0)

            # update the error
            self.E[q1] *= self.rho
            self.E[q2] *= self.rho
            self.E[q3] = (self.E[q1] + self.E[q2]) * 0.5

            # remove the original edge
            self.C = self.C[np.logical_not(np.logical_and(self.C[:, 0] == q1, self.C[:, 1] == q2))]
            self.C = self.C[np.logical_not(np.logical_and(self.C[:, 0] == q2, self.C[:, 1] == q1))]

            # add the edges
            self.C = np.vstack((self.C, np.asarray([q1, q3])))
            self.C = np.vstack((self.C, np.asarray([q2, q3])))

            # add a col and row
            self.S = np.pad(self.S, pad_width=((0, 1), (0, 1)), mode='constant')

            self.S[q1, q2] = 0
            self.S[q2, q1] = 0

            self.S[q1, q3] = 1
            self.S[q3, q1] = 1

            self.S[q2, q3] = 1
            self.S[q3, q2] = 1

            self.A_1 = np.concatenate((self.A_1, np.ones(1)), axis=0)
            self.A_2 = np.concatenate((self.A_2, np.ones(1)), axis=0)

    def cutEdge(self):
        self.removeNonActivatedNodes()

        mask = self.S > 0

        filterV = np.quantile(self.S[mask], 0.15)

        temp = self.S.copy()
        temp[self.S < filterV] = 0
        self.C = np.asarray(temp.nonzero()).T

        self.removeIsolatedNodes()
