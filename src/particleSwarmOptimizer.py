import numpy as np
import math 
import random
import costfunc as cf
import autoEncoder as AE
from autoEncoder import AutoEncoder
import encodeData as ED

from sklearn.neighbors import KNeighborsClassifier

class Particle:
    
    def __init__(self, dimensions, bounds, w):
        self.n = dimensions
        # self.bounds = bounds
        # self.c1 = c1  
        # self.c2 = c2  
        self.inertia = w    

        self.positions = np.array([random.uniform(bounds[0], bounds[1]) for i in range(dimensions)])
        self.velocities = np.zeros((dimensions))
        self.best_positions = self.positions
        self.best_neighbor_positions = self.best_positions
        self.Us = [0,0]
        self.accelerations = [1,1]
        self.n = dimensions
        self.cost = 1

    def updateCost(self, p0, p1):
        self.cost = cf.cost(p0, p1)



    def stepAlgorithm(self, p0, p1):
        self.updateCost(p0, p1)
        
        for i in range(len(self.velocities)):
            inertia =  self.inertia*self.velocities[i] 
            cognitive = self.accelerations[0]*1*(self.best_positions[i]-self.positions[i])
            social = self.accelerations[1]*1*(self.best_neighbor_positions[i] - self.positions[i])

            self.velocities[i] = inertia + cognitive + social

            self.positions[i] += self.velocities[i]



    

class Swarm:
    def __init__(self, particleList, dimensions, bounds = [0,math.pi]):
        self.particles = particleList
        self.num_particles = len(particleList)
        self.dims = dimensions
        self.bounds = [bounds*self.dims]
        self.lowestCosts = []
        self.bestParticle = particleList[0]
        self.tolerance = 2
        self.epochTolerance = 10

    def find_closest_neighbors(self, particles, query_particle, n_neighbors):
        particle_positions = np.array([p.positions for p in particles])
        query_position = query_particle.positions
        
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(particle_positions, np.zeros(particle_positions.shape[0]))
        
        distances, indices = knn.kneighbors([query_position], n_neighbors=n_neighbors)
        
        closest_neighbors = [particles[i] for i in indices[0]]
        
        return closest_neighbors

    def updateBestNeighbors(self):
        for particle in self.particles:
    
            n_neighbors = min(self.num_particles, len(self.particles))
            neighbors = self.find_closest_neighbors(self.particles, particle, n_neighbors)
            
            
            costs = []
            for neighbor in neighbors: 
                cost = neighbor.cost
                costs.append(cost)
        
            best_neighbor_index = np.argmin(costs)
            best_neighbor = neighbors[best_neighbor_index]
            
             
            if best_neighbor != particle: 
                particle.best_neighbor_positions = best_neighbor.positions
                particle.cost = best_neighbor.cost
            

            self.lowestCosts.append(particle.cost)

            if len(self.lowestCosts) > self.epochTolerance:
                del self.lowestCosts[0]
            
    def lowestCost(self):
        pass
    

    def stepAlgorithm(self, normal_data, fraud_data):
        if len(self.lowestCosts) == self.epochTolerance and np.ptp(self.lowestCosts, axis=1) < self.tolerance:
            return min(self.particles, key = lambda k: k.cost).cost
            #return positions with the lowest cost
        
        self.updateBestNeighbors() 

        print(min(self.particles, key = lambda k: k.cost).cost)

        newUs = [np.diag(np.random.rand(self.dims)), np.diag(np.random.rand(self.dims))]

        for i in self.particles:
            i.Us = newUs

            p0 = AutoEncoder(normal_data, 0, self.dims, i.positions)
            p1 = AutoEncoder(fraud_data, 1, self.dims, i.positions)

            print(p0, p1)

            i.stepAlgorithm(1-p0[0], p1[0])

            if i.cost == 0:
                return i.positions
        
        return None
        

            