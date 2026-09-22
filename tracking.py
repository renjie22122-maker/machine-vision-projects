import numpy as np
import scipy as sp
from scipy import io
import cv2 as cv
import matplotlib.pyplot as plt
import pylab
import os

def computeLikelihood(image, template):
    methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR, cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]
    likelihood = cv.matchTemplate(image[:, :, 2], template, methods[0])
    pad_first = int(template.shape[0])
    pad_second = int(template.shape[1])
    pad_amounts = ((0, pad_first - 1), (0, pad_second - 1))
    likelihood = np.pad(likelihood, pad_amounts, 'constant')
    likelihood[likelihood < 0] = 0
    kernel = np.ones((10, 10), np.float32) / 100
    smoothed = cv.filter2D(likelihood, -1, kernel)
    return smoothed

def HW1_CornerTrack(corner):
    template = sp.io.loadmat(corner + '.mat')['pixelsTemplate']
    print('We are matching this template with shape: ', template.shape)
    plt.imshow(template)
    plt.show()
    images = []
    iFrame = 0
    folder = 'Pattern01/'
    lst = os.listdir(folder)
    lst.sort()
    for frameNum in lst:
        images.append(cv.imread(folder + frameNum))
        iFrame += 1
    plt.imshow(images[0])
    plt.show()
    imgHeight, imgWidth, colors = images[0].shape
    numParticles = 2000
    weight_of_samples = np.ones((numParticles, 1))
    weight_of_samples /= np.sum(weight_of_samples)
    samples_to_propagate = range(0, numParticles)
    numDims_w = 2
    particles_old = np.random.rand(numParticles, numDims_w)
    particles_old[:, 0] = particles_old[:, 0] * imgHeight
    particles_old[:, 1] = particles_old[:, 1] * imgWidth
    r = np.zeros((iFrame, numDims_w))
    for iTime in range(iFrame):
        print('Processing Frame', iTime)
        cum_hist_of_weights = np.cumsum(weight_of_samples)
        cum_hist_of_weights /= cum_hist_of_weights[-1]
        samples_to_propagate = np.zeros(numParticles, dtype=np.int32)
        some_threshes = np.random.rand(numParticles)
        for sampNum in range(numParticles):
            thresh = some_threshes[sampNum]
            for index in range(numParticles):
                if cum_hist_of_weights[index] > thresh:
                    break
            samples_to_propagate[sampNum] = index
        particles_new = np.zeros_like(particles_old)
        for particleNum in range(numParticles):
            std_dev = 20
            noise = np.random.normal(0, std_dev, (numParticles, numDims_w))
            particles_new = particles_old[samples_to_propagate, :] + noise
        likelihood = computeLikelihood(images[iTime], template)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(images[iTime])
        axarr[0].set_title('Particles')
        axarr[0].plot(particles_new[:, 1] + template.shape[1] / 2, particles_new[:, 0] + template.shape[0] / 2, 'rx')
        axarr[1].imshow(likelihood)
        axarr[1].set_title('Likelihood')
        for particleNum in range(numParticles):
            particle = particles_new[particleNum, :]
            inFrame = particle[0] >= 0.0 and particle[0] < imgHeight and (particle[1] >= 0.0) and (particle[1] < imgWidth)
            if inFrame:
                minX = particle[1]
                minY = particle[0]
                weight_of_samples[particleNum] = likelihood[int(minY), int(minX)]
            else:
                weight_of_samples[particleNum] = 0.0
        sum_w = np.sum(weight_of_samples)
        if sum_w == 0:
            weight_of_samples[:] = 1.0 / numParticles
        else:
            weight_of_samples /= sum_w
        indices = np.argsort(weight_of_samples, 0)
        bestScoringParticles = particles_new[np.squeeze(indices[-15:]), :]
        plt.plot(bestScoringParticles[-1:, 1], bestScoringParticles[-1:, 0], 'rx')
        r[iTime, :] = (bestScoringParticles[-1, 1] + template.shape[1] / 2, bestScoringParticles[-1, 0] + template.shape[0] / 2)
        print(r[iTime, :])
        plt.show()
        plt.imshow(images[iTime])
        plt.plot(r[iTime, 0], r[iTime, 1], 'rx')
        plt.show()
        particles_old = particles_new
    return r
