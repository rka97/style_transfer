import skimage.io as io
from typing import cast, List
import skimage.exposure as ex
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from sympy import symbols, diff

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 

def recursiveFilter (img, der, h):
    a= np.exp(-1*np.sqrt(2)/h)
    var= np.power(a,der)
    [l,m,n] = img.shape
    #print(var)
    im = np.copy(img)
    for i in range (2,m):
        for j in range (n):
             im[:,i,j] = im[:,i,j] + np.multiply(var[:,i] ,( im[:,i - 1,j] - im[:,i,j]) )
    for i in range (m-2,0,-1):
        for j in range (n):
             im[:,i,j] = im[:,i,j] + np.multiply(var[:,i+1] ,( im[:,i + 1,j] - im[:,i,j]) )
    return im
def transpose(img):
    #print('\n')
    [l,m,n] = img.shape
    im = np.zeros((m,l,n))
    #print (im.shape)
    #print('\n')
    for i in range(n):
        im[:,:,i]=np.transpose(img[:,:,i])
    return im

def denoising (img):
    [l,m,n] = img.shape    #img = (img.astype(float)-np.min(img.ravel()))/(np.max(img.ravel())-np.min(img.ravel()))
    sigma_r=0.4
    sigma_s=60
    #print(l,m,n)
    # using finite diffrence to get partial derivative
    dIdx=np.diff(img,1,1)  #diff(A,1,1) works on successive elements in the columns of A and returns a p-by-(m-1) difference matrix.
    dIdy=np.diff(img,1,0) #diff(A,1,2) works on successive elements in the rows of A and returns a (p-1)-by-m difference matrix.
    derx = np.zeros((l,m))
    dery = np.zeros((l,m))
    #print(dIdx.shape)
    #print(dIdy.shape)
    for i in range(n):   #abs(I'(x))
        derx[:,1:m] = derx[:,1:m] + abs( dIdx[:,:,i] )
        dery[1:l,:] = dery[1:l,:] + abs( dIdy[:,:,i] )
    #horizontal and vertical derivatives
    dhdx = (1 + sigma_s/sigma_r * derx);
    dvdy = (1 + sigma_s/sigma_r * dery);
    #to get ct, we integrate, not needed in case of using the recursive filter
    #cth = np.cumsum(dhdx,2)
    #ctv = np.cumsum(dvdy,1)
    dvdy=np.transpose(dvdy) 
    x = np.sqrt(3)
    y= np.sqrt(4**3-1)
    t_img = np.copy(img)
    for i in range (3):  # 3 is the no of iterations usually used, we could change it
        sigma_h = sigma_s * x * 2**(3-i-1) / y
       # print(t_img[:,:,0] )
       # print('\n')
       # print('\n')
       # print('\n')
        t_img = recursiveFilter(t_img,dhdx,sigma_h)
       # print(t_img[:,:,0] )
       # print('\n')
       # print('\n')
        t_img = transpose(t_img)#,l,m,n)
        t_img = recursiveFilter(t_img,dvdy,sigma_h)
       # print(t_img[1,2] )
        t_img = transpose(t_img)#,l,m,n)
       # print(t_img[1,1] )
       # print('\n')
    #print(t_img[:,:,0] )
    #print('\n')
    t_img = t_img*255.0 #need to handle cases where the image is binary...etc
    return t_img.astype(np.uint8)
    
def main():  
    img=io.imread('cow.jpg')
    original = np.copy(img)
    #the opencv function
    #dst = cv2.edgePreservingFilter(src, flags=1, sigma_s=60, sigma_r=0.4)
   #typ =np.iinfo(img.dtype).max  #in case of a binary image
    img = (img.astype(np.float))/255.0   #do i need rescaling??
    denoised_img = denoising(img)
    show_images([original,denoised_img])

main()